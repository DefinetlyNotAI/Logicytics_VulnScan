import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from vulnscan.config import TrainingConfig
from vulnscan.log import log


# ---------------- DATASET CLASS ----------------
# noinspection DuplicatedCode
class EmbeddingDataset(Dataset):
    def __init__(self, embed_cache_dir):
        self.files = sorted(os.listdir(embed_cache_dir))
        self.embed_cache_dir = embed_cache_dir
        self.current_batch_idx = -1
        self.current_batch = None
        self.cum_sizes = []
        total = 0
        for f in self.files:
            data = torch.load(os.path.join(embed_cache_dir, f))
            total += data['embeddings'].shape[0]
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1]

    def _load_batch(self, idx):
        path = os.path.join(self.embed_cache_dir, self.files[idx])
        self.current_batch = torch.load(path)
        self.current_batch_idx = idx

    def __getitem__(self, idx):
        for batch_idx, cum in enumerate(self.cum_sizes):
            if idx < cum:
                if batch_idx != self.current_batch_idx:
                    self._load_batch(batch_idx)
                rel_idx = idx if batch_idx == 0 else idx - self.cum_sizes[batch_idx - 1]
                return self.current_batch['embeddings'][rel_idx], self.current_batch['labels'][rel_idx]
        raise IndexError("Index out of range")


# ---------------- MODEL ----------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        return self.fc(x)


# ---------------- TRAINING ----------------
class Train:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = cfg.DEVICE

    # Compute a weighted sampler from model losses
    def create_sampler(self, dataset: EmbeddingDataset, model: SimpleNN):
        losses = []
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        model.eval()
        with torch.no_grad():
            for X, y in DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE):
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                batch_loss = criterion(outputs, y)
                losses.extend(batch_loss.view(-1).tolist())  # flatten before extending

        weights = torch.tensor(losses).float()
        weights /= weights.sum()  # normalize
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # Train for a single epoch
    def train_one_epoch(
            self, model: SimpleNN, loader: DataLoader,
            optimizer: torch.optim.Adam, criterion: nn.BCEWithLogitsLoss,
            scaler: torch.amp.GradScaler
    ):
        model.train()
        epoch_loss, all_preds, all_labels = 0, [], []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                                    enabled=(self.device == "cuda")):
                outputs = model(X)
                loss = criterion(outputs, y)
            if self.device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).round().cpu().detach().numpy())
            all_labels.extend(y.cpu().numpy())
        return epoch_loss / len(loader), all_preds, all_labels

    # Validate on a dataset
    def validate(self, model: SimpleNN, loader: DataLoader, criterion: nn.BCEWithLogitsLoss):
        model.eval()
        val_loss, val_preds, val_labels_list = 0, [], []
        with torch.no_grad():
            for X, y in tqdm(loader, desc="Validating", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).round().cpu().numpy())
                val_labels_list.extend(y.cpu().numpy())
        val_loss /= len(loader)
        metrics = {
            "accuracy": accuracy_score(val_labels_list, val_preds),
            "precision": precision_score(val_labels_list, val_preds, zero_division=0),
            "recall": recall_score(val_labels_list, val_preds, zero_division=0),
            "f1": f1_score(val_labels_list, val_preds, zero_division=0)
        }
        return val_loss, metrics

    # Save model checkpoint
    def save_checkpoint(self, model: SimpleNN, optimizer: torch.optim.Adam, scaler: torch.amp.GradScaler, epoch: int):
        round_dir = f"{self.cfg.CACHE_DIR}/{self.cfg.MODEL_NAME}/round_{self.cfg.MODEL_ROUND}"
        os.makedirs(round_dir, exist_ok=True)
        model_path = f"{round_dir}/{self.cfg.MODEL_NAME}_round{self.cfg.MODEL_ROUND}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch
        }, model_path)
        return model_path

    # Main training loop supporting TRAIN_LOOPS
    def model(self, model: SimpleNN, train_dataset: EmbeddingDataset, val_loader: DataLoader):
        history_loops = []

        for loop in range(self.cfg.TRAIN_LOOPS):
            log(f"Starting TRAIN_LOOP {loop + 1}/{self.cfg.TRAIN_LOOPS}", self.cfg)

            # Create sampler focusing on weak spots
            sampler = self.create_sampler(train_dataset, model)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.BATCH_SIZE,
                                      sampler=sampler)

            optimizer = optim.Adam(model.parameters(), lr=self.cfg.LR)
            criterion = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))

            # LR Jumpstarter
            max_lr = self.cfg.LR * self.cfg.LR_JUMP["MAX"]
            min_lr = self.cfg.LR * self.cfg.LR_JUMP["MIN"]
            patience_for_jump = self.cfg.JUMP_PATIENCE
            lr_decay_factor = self.cfg.LR_DECAY
            best_val_loss = float('inf')
            patience_counter = 0
            jump_counter = 0

            history = {
                "train_loss": [], "val_loss": [],
                "accuracy": [], "precision": [], "recall": [], "f1": []
            }

            for epoch in tqdm(range(self.cfg.MAX_EPOCHS), desc=f"Loop {loop+1}/{self.cfg.TRAIN_LOOPS} Epochs", leave=False):
                try:
                    log(message=f"Epoch {epoch + 1}/{self.cfg.MAX_EPOCHS}", cfg=self.cfg, silent=True)
                    train_loss, train_preds, train_labels = self.train_one_epoch(
                        model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, scaler=scaler
                    )

                    val_loss, val_metrics = self.validate(model=model, loader=val_loader, criterion=criterion)

                    # LR Jumpstarter logic
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        for g in optimizer.param_groups:
                            g['lr'] = max(g['lr'] * lr_decay_factor, min_lr)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience_for_jump:
                            jump_counter += 1
                            log(message=f"Validation stalled. Jumping LR (jump #{jump_counter})!", cfg=self.cfg, silent=True)
                            for g in optimizer.param_groups:
                                g['lr'] = min(g['lr'] * 3, max_lr)
                            patience_counter = 0

                    # Update history
                    history["train_loss"].append(train_loss)
                    history["val_loss"].append(val_loss)
                    for k in ["accuracy", "precision", "recall", "f1"]:
                        history[k].append(val_metrics[k])

                    log(message=f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}", cfg=self.cfg, silent=True)

                    # Early stopping
                    if patience_counter >= self.cfg.EARLY_STOPPING_PATIENCE and not self.cfg.AUTO_CONTINUE:
                        log("Early stopping triggered.", self.cfg)
                        break

                    # Save checkpoint each epoch
                    self.save_checkpoint(model, optimizer, scaler, epoch)
                except KeyboardInterrupt:
                    self.save_checkpoint(model, optimizer, scaler, epoch)
                    sys.exit("\nTraining interrupted. Model saved.")
            history_loops.append(history)
        return history_loops
