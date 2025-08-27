import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from vulnscan.config import cfg
from vulnscan.log import log


# ---------------- DATASET CLASS ----------------
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
def create_sampler(dataset: EmbeddingDataset, model: SimpleNN):
    losses = []
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    model.eval()
    with torch.no_grad():
        for X, y in DataLoader(dataset=dataset, batch_size=cfg.BATCH_SIZE):
            X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
            outputs = model(X)
            batch_loss = criterion(outputs, y)
            losses.extend(batch_loss.view(-1).tolist())  # flatten before extending

    weights = torch.tensor(losses).float()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def train_model(model: SimpleNN, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = cfg.MAX_EPOCHS):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.LR)

    # Jumpstarter scheduler config
    max_lr = cfg.LR * cfg.LR_JUMP["MAX"]
    min_lr = cfg.LR * cfg.LR_JUMP["MIN"]
    patience_for_jump = cfg.JUMP_PATIENCE
    lr_decay_factor = cfg.LR_DECAY
    best_val_loss = cfg.BEST_VAL_LOSS
    patience_counter = cfg.COUNTER["PATIENCE"]
    jump_counter = cfg.COUNTER["JUMP"]

    history = {
        "train_loss": [], "val_loss": [],
        "accuracy": [], "precision": [],
        "recall": [], "f1": []
    }

    # Use the new amp API
    scaler = torch.amp.GradScaler(enabled=(cfg.DEVICE == "cuda"))

    for epoch in range(max_epochs):
        try:
            log(f"Epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss, all_preds, all_labels = 0, [], []

            # --- Training Loop ---
            for X, y in tqdm(train_loader):
                X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda" if cfg.DEVICE == "cuda" else "cpu", enabled=(cfg.DEVICE == "cuda")):
                    outputs = model(X)
                    loss = criterion(outputs, y)

                if cfg.DEVICE == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                all_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            # --- Metrics ---
            acc = accuracy_score(y_true=all_labels, y_pred=all_preds)
            history["train_loss"].append(epoch_loss / len(train_loader))
            history["accuracy"].append(acc)

            # --- Validation Loop ---
            model.eval()
            val_loss, val_preds, val_labels_list = 0, [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    val_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                    val_labels_list.extend(y.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc = accuracy_score(y_true=val_labels_list, y_pred=val_preds)
            val_f1 = f1_score(y_true=val_labels_list, y_pred=val_preds, zero_division=0)
            history["val_loss"].append(val_loss)
            history["f1"].append(val_f1)

            # --- Jumpstarter LR logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * lr_decay_factor, min_lr)
            else:
                patience_counter += 1
                if patience_counter >= patience_for_jump:
                    jump_counter += 1
                    log(f"Validation stalled. Jumping LR (jump #{jump_counter})!")
                    for g in optimizer.param_groups:
                        g['lr'] = min(g['lr'] * 3, max_lr)
                    patience_counter = 0

            log(f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # --- Early Stopping ---
            if val_loss > best_val_loss:
                patience_counter += 1
                if patience_counter >= cfg.EARLY_STOPPING_PATIENCE and not cfg.AUTO_CONTINUE:
                    log("Early stopping triggered.")
                    break

            # --- Save checkpoint ---
            round_dir = f"{cfg.CACHE_DIR}/round_{cfg.MODEL_ROUND}"
            os.makedirs(name=round_dir, exist_ok=True)
            model_path = f"{round_dir}/{cfg.MODEL_NAME}_round{cfg.MODEL_ROUND}.pth"
            torch.save(model.state_dict(), model_path)
        except KeyboardInterrupt:
            log("Training interrupted by user early. Saving premature model.")
            torch.save(model.state_dict(), model_path)
            break

    return history
