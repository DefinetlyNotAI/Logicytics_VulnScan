import glob
import os
import sys

import torch
from sentence_transformers import SentenceTransformer

from data import test_texts, test_labels
from vulnscan import SimpleNN

# ---------------- INIT ----------------
NAME = "Model_Sense.4n1"
ROUND = 5

# ---------------- LOAD MODEL + EMBEDDER ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def load_embeddings(folder_path, pattern):
    """Load all .pt files matching pattern and concatenate embeddings and labels"""
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    print("Found files:", files)
    if not files:
        sys.exit(f"No files found in {folder_path} matching {pattern}")
    all_embeddings = []
    all_labels = []
    for f in files:
        data = torch.load(f, map_location=device)
        all_embeddings.append(data["embeddings"])
        all_labels.append(data["labels"])
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


# Example paths
cache_dir = f"../cache/{NAME}/round_{ROUND}/embeddings"

# Load all train/test/val embeddings
train_embeddings, train_labels = load_embeddings(cache_dir, "train_*.pt")
test_embeddings, embed_test_labels = load_embeddings(cache_dir, "test_*.pt")
val_embeddings, val_labels = load_embeddings(cache_dir, "validation_*.pt")

# Initialize model
input_dim = train_embeddings.shape[1]
model = SimpleNN(input_dim=input_dim).to(device)
state = torch.load(
    f"../cache/{NAME}/round_{ROUND}/{NAME}_round{ROUND}.pth",
    map_location="cpu"
)
model.load_state_dict(state["model_state_dict"])
model.eval()

# Load SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ---------------- 1. TEST ON STORED EMBEDDINGS ----------------
with torch.no_grad():
    outputs = model(test_embeddings.to(device))
    preds = torch.sigmoid(outputs).squeeze()

print("\n=== TEST 1: Stored Embeddings (50% sample) ===")
print("Sample predictions vs true labels (first 50%):")
sample_size = len(preds) // 2
for i in range(sample_size):
    print(f"    Pred={preds[i].item():.3f} | Label={embed_test_labels[i].item()}")
# Calculate accuracy
pred_labels = (preds >= 0.5).long().view(-1).cpu()
true_labels = embed_test_labels.view(-1).cpu()
accuracy = (pred_labels == true_labels).float().mean().item()
print(f"Accuracy on stored embeddings: {accuracy*100:.2f}%")

# ---------------- 2. TEST ON NATURAL EXAMPLES ----------------
# Encode with SentenceTransformer
with torch.no_grad():
    test_embs = embed_model.encode(test_texts, convert_to_tensor=True, device=device)
    outputs = model(test_embs)
    preds = torch.sigmoid(outputs).squeeze()

print("\n\n=== TEST 2: Natural Examples ===")
print("Real-world samples predictions vs true labels:")
correct = 0
for i, (text, pred, label) in enumerate(zip(test_texts, preds, test_labels)):
    decision = 1 if pred.item() >= 0.5 else 0
    if decision == label:
        correct += 1
    print(f"    [{i + 1}] Pred={pred.item():.3f} | Label={label} | Text={text[:50]}...")

print(f"\nAccuracy on natural 100 samples: {correct}/{len(test_labels)} = {correct / len(test_labels) * 100:.2f}%")
