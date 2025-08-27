print("[INIT] Importing Libraries...")
import gc
import json
import os
import random

import torch
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# ------------------- CONFIG -------------------
print("[INIT] Setting Config...")
GEN_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
CLS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE_GEN = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMB = "cpu"  # use CPU for embeddings to save VRAM

OUTPUT_FILE = "generated_files.json"
OFFLOAD_FILE = "offloaded_data.json"

NUM_GENERATED = 100
MAX_DATASET_SIZE = 2000
RELOAD_THRESHOLD = 500
MAX_LENGTH = 64
TEMPERATURE = 0.9
BATCH_SIZE = 8
EPOCHS = 3
MAX_TRAINS = 5

config_table = [
    ["Generator Model", GEN_MODEL_NAME],
    ["Classifier Model", CLS_MODEL_NAME],
    ["Generation Device", DEVICE_GEN],
    ["Embedding Device", DEVICE_EMB],
    ["Output File", OUTPUT_FILE],
    ["Offload File", OFFLOAD_FILE],
    ["Num Generated per Cycle", NUM_GENERATED],
    ["Max Dataset Size", MAX_DATASET_SIZE],
    ["Reload Threshold", RELOAD_THRESHOLD],
    ["Max Length", MAX_LENGTH],
    ["Temperature", TEMPERATURE],
    ["Batch Size", BATCH_SIZE],
    ["Epochs per Train", EPOCHS],
    ["Max Training Cycles", MAX_TRAINS if MAX_TRAINS != -1 else "Infinite"]
]

print("[INFO] Configuration:")
print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))

# ------------------- MODELS -------------------
print("[INIT] Loading GPT-Neo generator...")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE_GEN)

print("[INIT] Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
embed_model = AutoModel.from_pretrained(CLS_MODEL_NAME).to(DEVICE_EMB)


# ------------------- CLASSIFIER -------------------
class Classifier(nn.Module):
    def __init__(self, input_dim=384, hidden=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


classifier = Classifier().to(DEVICE_GEN)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)


# ------------------- HELPERS -------------------
def generate_texts(num_samples=50):
    samples = []
    prompts = [
        "Generate a realistic sensitive file entry (email, password, API key) without repeating the instruction.",
        "Generate a realistic safe non-sensitive file entry (notes, reminders, general text) without repeating the instruction."
    ]
    for _ in tqdm(range(num_samples), desc="Generating"):
        label = random.choice(["sensitive", "non-sensitive"])
        prompt = prompts[0] if label == "sensitive" else prompts[1]
        inputs = gen_tokenizer(prompt, return_tensors="pt").to(DEVICE_GEN)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=gen_tokenizer.eos_token_id
            )
        text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        samples.append({"label": label, "text": text})
        torch.cuda.empty_cache()
    return samples


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def manage_offloading(dataset):
    if len(dataset) > MAX_DATASET_SIZE:
        print(f"[OFFLOAD] Dataset too big ({len(dataset)}), offloading...")
        offload = load_json(OFFLOAD_FILE)
        offload.extend(dataset[MAX_DATASET_SIZE:])
        save_json(OFFLOAD_FILE, offload)
        dataset = dataset[:MAX_DATASET_SIZE]
    elif len(dataset) < RELOAD_THRESHOLD and os.path.exists(OFFLOAD_FILE):
        print(f"[RELOAD] Dataset small ({len(dataset)}), reloading from offload...")
        offload = load_json(OFFLOAD_FILE)
        dataset.extend(offload[:RELOAD_THRESHOLD])
        save_json(OFFLOAD_FILE, offload[RELOAD_THRESHOLD:])
    return dataset


def embed_texts(texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = embed_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE_EMB)
        with torch.no_grad():
            out = embed_model(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(out)
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0).to(DEVICE_GEN)


def train_classifier(dataset, max_epochs=EPOCHS):
    texts = [d["text"] for d in dataset]
    labels = [0 if d["label"] == "non-sensitive" else 1 for d in dataset]

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2)
    X_train_emb = embed_texts(X_train)
    X_val_emb = embed_texts(X_val)
    y_train = torch.tensor(y_train, device=DEVICE_GEN)
    y_val = torch.tensor(y_val, device=DEVICE_GEN)

    for epoch in range(max_epochs):
        classifier.train()
        optimizer.zero_grad()
        preds = classifier(X_train_emb)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_preds = classifier(X_val_emb)
            val_loss = criterion(val_preds, y_val)
            acc = (val_preds.argmax(dim=1) == y_val).float().mean().item()
        print(
            f"[TRAIN] Epoch {epoch + 1}/{max_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}")

    return acc


# ------------------- MAIN LOOP -------------------
def main():
    dataset = load_json(OUTPUT_FILE)
    train_count = 0
    best_acc = 0.0

    while MAX_TRAINS == -1 or train_count < MAX_TRAINS:
        print(f"\n[CYCLE] ===== Training Cycle {train_count + 1} =====")

        new_data = generate_texts(NUM_GENERATED)
        dataset.extend(new_data)
        dataset = manage_offloading(dataset)
        save_json(OUTPUT_FILE, dataset)

        acc = train_classifier(dataset)
        train_count += 1

        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), "best_classifier.pth")
            print(f"[SAVE] New best model saved with accuracy {best_acc:.2f}")

        gc.collect()
        torch.cuda.empty_cache()

        if MAX_TRAINS != -1 and train_count >= MAX_TRAINS:
            print("[EXIT] Max training cycles reached.")
            break


if __name__ == "__main__":
    main()
