import sys

import torch
from sentence_transformers import SentenceTransformer
from vulnscan import SimpleNN
import glob
import os
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
test_embeddings, test_labels = load_embeddings(cache_dir, "test_*.pt")
val_embeddings, val_labels = load_embeddings(cache_dir, "validation_*.pt")

# Initialize model
input_dim = train_embeddings.shape[1]
model = SimpleNN(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(
    f"../cache/{NAME}/round_{ROUND}/{NAME}_round{ROUND}.pth",
    map_location="cpu"
))
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
    print(f"    Pred={preds[i].item():.3f} | Label={test_labels[i].item()}")
# Calculate accuracy
pred_labels = (preds >= 0.5).long().cpu()
accuracy = (pred_labels == test_labels).sum().item() / len(test_labels)
print(f"\nAccuracy on stored embeddings: {accuracy * 100:.2f}%")

# ---------------- 2. TEST ON NATURAL EXAMPLES ----------------
sensitive_texts = [
    "My SSN is 123-45-6789",
    "Here is my credit card number: 4111 1111 1111 1111",
    "Password: hunter2",
    "My email is johndoe@gmail.com",
    "Bank account number: 987654321",
    "Private key: -----BEGIN RSA PRIVATE KEY-----",
    "The system IP is 192.168.1.1",
    "Contact me at +1-202-555-0147",
    "Visa card exp 12/25 CVV 123",
    "Secret API key: sk_test_4eC39HqLyjWDarjtT1zdp7dc",
    "My passport number is X1234567",
    "Driver's license ID: D12345678",
    "Social security info: 987-65-4321",
    "Encrypted key: 6f1e9a2c3d4b5e6f7a8b",
    "Database password: P@ssw0rd123",
    "Credit card CVV: 321",
    "Bank routing number: 021000021",
    "Private SSH key: -----BEGIN OPENSSH PRIVATE KEY-----",
    "My personal address: 123 Main St, Springfield",
    "Medical record ID: MR123456",
    "Tax ID: 123-45-6789",
    "Company login password: Admin@2025",
    "Phone PIN: 4321",
    "Wi-Fi password: mysecretwifi",
    "API token: abc123def456",
    "OAuth secret: 987zyx654",
    "PIN code: 2468",
    "Student ID: S123456789",
    "Mother's maiden name: Smith",
    "Credit card expiration: 10/27",
    "Bank account PIN: 5678",
    "Encrypted password hash: $2b$12$abcd1234...",
    "My vehicle VIN: 1HGCM82633A004352",
    "Passport expiration: 09/2030",
    "Private email password: qwerty123",
    "Debit card number: 5500 0000 0000 0004",
    "Bank security code: 789",
    "My date of birth: 1990-01-01",
    "Secret question answer: Blue",
    "Corporate VPN password: VPN@1234",
    "My home phone: +1-555-123-4567",
    "Bank card CVV2: 456",
    "Employee SSN: 234-56-7890",
    "Private notes: Login credentials",
    "Access key: AKIAIOSFODNN7EXAMPLE",
    "Credit card PIN: 1357",
    "Encrypted token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "My secondary email password: pass1234",
    "Personal tax record: TR987654321",
]
nonsensitive_texts = [
    "The sky is blue today",
    "I had pasta for lunch",
    "Cats are very playful animals",
    "The capital of France is Paris",
    "I enjoy playing video games",
    "Tomorrow will be sunny",
    "Mount Everest is the tallest mountain",
    "Water boils at 100 degrees Celsius",
    "I love listening to music",
    "Python is a programming language",
    "I walked in the park yesterday",
    "Coffee tastes great in the morning",
    "The train arrives at 9 AM",
    "My favorite color is green",
    "Birds are singing outside",
    "The museum opens at 10 AM",
    "I like reading science fiction novels",
    "The ocean waves are calming",
    "I bought a new notebook",
    "Chocolate ice cream is delicious",
    "The book is on the table",
    "I enjoy jogging in the evening",
    "The concert was amazing",
    "I prefer tea over coffee",
    "Clouds are moving quickly",
    "The cat is sleeping on the sofa",
    "I painted a landscape yesterday",
    "The sun sets in the west",
    "I visited the mountains last summer",
    "Reading improves vocabulary",
    "The flowers bloom in spring",
    "I wrote a poem today",
    "Dogs are loyal pets",
    "The bakery smells wonderful",
    "I attended a workshop on AI",
    "The painting has vibrant colors",
    "I love hiking in the forest",
    "The classroom is very bright",
    "I learned a new recipe today",
    "The movie was entertaining",
    "I played chess with a friend",
    "The library has many books",
    "I enjoy listening to jazz music",
    "The playground is full of kids",
    "I watched a documentary yesterday",
    "The stars are shining tonight",
    "I planted a tree in the backyard",
    "The festival was fun and lively",
    "I took a photography course",
    "The coffee shop is near my office",
]
test_texts = sensitive_texts + nonsensitive_texts
test_labels = [1] * len(sensitive_texts) + [0] * len(nonsensitive_texts)

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
