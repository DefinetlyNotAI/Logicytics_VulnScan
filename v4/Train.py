import datetime
import json
import os
import random
import signal

import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from faker import Faker
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel

# ---------------- CONFIG ----------------

# Model / caching / logging
MODEL_NAME = "Model_Sense.4n1"
CACHE_DIR = f"cache/{MODEL_NAME}"
os.makedirs(CACHE_DIR, exist_ok=True)

# Auto-increment round based on existing folders, omitting any suffix like '-F'
existing_rounds = [
    int(f.split('_')[-1].split('-')[0])
    for f in os.listdir(CACHE_DIR)
    if f.startswith('round_') and f.split('_')[-1].split('-')[0].isdigit()
]
MODEL_ROUND = max(existing_rounds) + 1 if existing_rounds else 1

LOG_FILE = f"{CACHE_DIR}/training.log"
EMBED_CACHE_DIR = f"{CACHE_DIR}/round_{MODEL_ROUND}/embeddings"
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir=f"{CACHE_DIR}/round_{MODEL_ROUND}/tensorboard_logs")

# Training parameters
BATCH_SIZE: int = 16
MAX_EPOCHS: int = 35
EARLY_STOPPING_PATIENCE: int = 5
LR: float = 1e-3
LR_JUMP: dict[str, int] = {"MAX": 5, "MIN": 0.1}
COUNTER: dict[str, int] = {"PATIENCE": 0, "JUMP": 0}
JUMP_PATIENCE: int = 3
LR_DECAY: float = 0.9
BEST_VAL_LOSS: float = float("inf")
AUTO_CONTINUE: bool = False

# Dataset / data generation
DATASET_SIZE: int = 100
TEXT_MAX_LEN: int = 128
TEXT_MAX_LEN_JUMP_RANGE: int = 10
TRAIN_VAL_SPLIT: float = 0.8
SENSITIVE_PROB: float = 0.3
SENSITIVE_FIELDS: list[str] = ["ssn", "credit_card", "email", "phone_number", "address", "name"]

# Language / generation
MULTI_LANGUAGES: list[str] = ["english", "spanish", "french", "dutch", "arabic", "japanese"]
TOP_K: int = 30
TOP_P: float = 0.9
TEMPERATURE: float = 0.9
REP_PENALTY: float = 1.2
RETRY_LIMIT: int = 3

# Device / system
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
RAM_THRESHOLD: float = 0.85

# Misc / globals
STOP_TRAINING: bool = False
faker = Faker()


# ---------------- SIGNAL HANDLER ----------------
# noinspection PyUnusedLocal
def signal_handler(sig, frame):
    global STOP_TRAINING
    log("Keyboard Interrupt detected! Stopping training gracefully...")
    STOP_TRAINING = True


signal.signal(signal.SIGINT, signal_handler)


# ---------------- LOGGING ----------------
def log(message: str):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} | {message}\n")


# ---------------- SENSITIVE DATA ----------------
def generate_sensitive_text():
    field = random.choice(SENSITIVE_FIELDS)
    if field == "ssn":
        return f"SSN: {faker.ssn()}"
    elif field == "credit_card":
        return f"Credit Card: {faker.credit_card_number()}"
    elif field == "email":
        return f"Email: {faker.email()}"
    elif field == "phone_number":
        return f"Phone: {faker.phone_number()}"
    elif field == "address":
        return f"Address: {faker.address().replace(chr(10), ', ')}"
    elif field == "name":
        return f"Name: {faker.name()}"
    return "Sensitive info: [REDACTED]"


# ---------------- GPT TEXT GENERATION ----------------
log("Loading GPT-Neo model for text generation...")
gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(DEVICE)
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
log("Loading MiniLM for embeddings...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_gpt_text(lang: str, max_words: int = TEXT_MAX_LEN, max_word_range: int = TEXT_MAX_LEN_JUMP_RANGE,
                      retry_limit: int = RETRY_LIMIT):
    max_words += random.randint(-max_word_range, max_word_range)
    for _ in range(retry_limit):
        prompt = f"Write one short, simple, natural sentence in {lang} about daily life:"
        input_enc = gpt_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = input_enc.input_ids.to(DEVICE)
        attention_mask = input_enc.attention_mask.to(DEVICE)
        with torch.no_grad():
            output_ids = gpt_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_words + len(input_ids[0]),
                do_sample=True,
                top_k=TOP_K,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                pad_token_id=gpt_tokenizer.pad_token_id,
                eos_token_id=gpt_tokenizer.eos_token_id,
                repetition_penalty=REP_PENALTY
            )
        text = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        for p in ".!?":
            if p in text:
                text = text.split(p)[0].strip()
        if len(text.split()) > 1:
            return text
    return f"A short sentence in {lang}."


# ---------------- DATASET GENERATION ----------------
def generate_dataset(num_samples: int = DATASET_SIZE):
    dataset, labels = [], []
    log(f"Generating {num_samples} samples using GPT-Neo + Faker...")
    for _ in tqdm(range(num_samples)):
        if STOP_TRAINING:
            break
        sensitive = random.random() < SENSITIVE_PROB
        text = generate_sensitive_text() if sensitive else generate_gpt_text(random.choice(MULTI_LANGUAGES))
        dataset.append(text)
        labels.append(int(sensitive))
    return dataset, labels


# ---------------- EMBEDDINGS ----------------
def offload_embeddings(batch_embeddings: torch.Tensor, batch_labels: torch.Tensor, idx: int):
    path = f"{EMBED_CACHE_DIR}/batch_{idx}.pt"
    torch.save({'embeddings': batch_embeddings.cpu(), 'labels': batch_labels.cpu()}, path)


def generate_embeddings(texts: list[str], labels: list[int | float], tokenizer: BertTokenizer,
                        batch_size: int = BATCH_SIZE):
    bert = BertModel.from_pretrained('bert-base-uncased').to('cpu')
    bert.eval()
    batch_embeddings, batch_labels, batch_idx = [], [], 0
    for i in tqdm(range(0, len(texts), batch_size)):
        if STOP_TRAINING:
            break
        batch_texts = texts[i:i + batch_size]
        batch_lbls = labels[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding='max_length', truncation=True,
                            max_length=TEXT_MAX_LEN, return_tensors='pt')
        with torch.no_grad():
            emb = bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask']).pooler_output
        batch_embeddings.append(emb)
        batch_labels.append(torch.tensor(batch_lbls, dtype=torch.float32).unsqueeze(1))

        if psutil.virtual_memory().percent / 100 > RAM_THRESHOLD:
            offload_embeddings(torch.cat(batch_embeddings, dim=0), torch.cat(batch_labels, dim=0), batch_idx)
            batch_embeddings, batch_labels = [], []
            batch_idx += 1

    if batch_embeddings:
        offload_embeddings(torch.cat(batch_embeddings, dim=0), torch.cat(batch_labels, dim=0), batch_idx)


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
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


# ---------------- TRAINING ----------------
def create_sampler(dataset: EmbeddingDataset, model: SimpleNN):
    losses = []
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    model.eval()
    with torch.no_grad():
        for X, y in DataLoader(dataset, batch_size=BATCH_SIZE):
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            batch_loss = criterion(outputs, y)
            losses.extend(batch_loss.view(-1).tolist())  # flatten before extending

    weights = torch.tensor(losses).float()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_model(model: SimpleNN, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = MAX_EPOCHS):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Jumpstarter scheduler config
    max_lr = LR * LR_JUMP["MAX"]
    min_lr = LR * LR_JUMP["MIN"]
    patience_for_jump = JUMP_PATIENCE
    lr_decay_factor = LR_DECAY
    best_val_loss = BEST_VAL_LOSS
    patience_counter = COUNTER["PATIENCE"]
    jump_counter = COUNTER["JUMP"]

    history = {
        "train_loss": [], "val_loss": [],
        "accuracy": [], "precision": [],
        "recall": [], "f1": []
    }

    # Use the new amp API
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(max_epochs):
        if STOP_TRAINING:
            break

        log(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss, all_preds, all_labels = 0, [], []

        # --- Training Loop ---
        for X, y in tqdm(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", enabled=(DEVICE == "cuda")):
                outputs = model(X)
                loss = criterion(outputs, y)

            if DEVICE == "cuda":
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
        acc = accuracy_score(all_labels, all_preds)
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["accuracy"].append(acc)

        # --- Validation Loop ---
        model.eval()
        val_loss, val_preds, val_labels_list = 0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                val_labels_list.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
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
            if patience_counter >= EARLY_STOPPING_PATIENCE and not AUTO_CONTINUE:
                log("Early stopping triggered.")
                break

        # --- Save checkpoint ---
        round_dir = f"{CACHE_DIR}/round_{MODEL_ROUND}"
        os.makedirs(round_dir, exist_ok=True)
        model_path = f"{round_dir}/{MODEL_NAME}_round{MODEL_ROUND}.pth"
        torch.save(model.state_dict(), model_path)

    return history


# ---------------- PLOTTING ----------------
def plot_training(history: dict):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["accuracy"], label="Accuracy")
    plt.plot(history["f1"], label="F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Training Round {MODEL_ROUND}")
    plt.savefig(f"{CACHE_DIR}/round_{MODEL_ROUND}/training_plot.png")
    plt.close()
    log("Saved training plot.")


# ---------------- MAIN ----------------
def main():
    global STOP_TRAINING
    log("Starting advanced self-training sensitive data classifier...")

    # Generate dataset
    texts, labels = generate_dataset(DATASET_SIZE)
    split = int(len(texts) * TRAIN_VAL_SPLIT)
    train_texts, train_labels = texts[:split], labels[:split]
    val_texts, val_labels = texts[split:], labels[split:]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    log("Generating train embeddings...")
    generate_embeddings(train_texts, train_labels, tokenizer)
    log("Generating validation embeddings...")
    generate_embeddings(val_texts, val_labels, tokenizer)

    train_dataset = EmbeddingDataset(EMBED_CACHE_DIR)
    val_dataset = EmbeddingDataset(EMBED_CACHE_DIR)

    model_dummy = SimpleNN().to(DEVICE)
    sampler = create_sampler(train_dataset, model_dummy)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleNN(input_dim=768).to(DEVICE)
    history = train_model(model, train_loader, val_loader)
    plot_training(history)

    # Save history
    with open(f"{CACHE_DIR}/round_{MODEL_ROUND}/training_history.json", "w") as f:
        json.dump(history, f)
    log("Training complete. All data, plots, and model saved.")


if __name__ == "__main__":
    main()
