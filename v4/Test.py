import torch
from transformers import AutoTokenizer, AutoModel
import json
import torch.nn as nn

# ------------------- CONFIG -------------------
CLS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "../models/Model_Sense.1n1.pth"  # your saved classifier
TEST_FILE = "offloaded_data.json"  # JSON with {"label": ..., "text": ...} entries

# ------------------- DEFINE CLASSIFIER -------------------


class Classifier(nn.Module):
    def __init__(self, input_dim=384, hidden=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ------------------- LOAD MODELS -------------------
print("[INIT] Loading classifier...")
classifier = Classifier().to(DEVICE)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval()

print("[INIT] Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
embed_model = AutoModel.from_pretrained(CLS_MODEL_NAME).to(DEVICE)
embed_model.eval()

# ------------------- LOAD TEST DATA -------------------
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

texts = [d["text"] for d in test_data]
labels = [0 if d["label"] == "non-sensitive" else 1 for d in test_data]


# ------------------- EMBED TEXTS -------------------
def embed_texts(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = embed_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = embed_model(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(out)
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)


X_emb = embed_texts(texts)
y_true = torch.tensor(labels, device=DEVICE)

# ------------------- PREDICT & EVALUATE -------------------
with torch.no_grad():
    logits = classifier(X_emb)
    preds = torch.argmax(logits, dim=1)

accuracy = (preds == y_true).float().mean().item()
print(f"[RESULT] Accuracy on test set: {accuracy:.2%}")

# Optional: show misclassified examples
for i, (p, t) in enumerate(zip(preds, y_true)):
    if p != t:
        print(f"\n[WRONG] Text: {texts[i][:100]}...\nPredicted: {p.item()} | True: {t.item()}")
