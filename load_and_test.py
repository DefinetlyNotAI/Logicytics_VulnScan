import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import os

from vulnscan import SimpleNN


# ---------------- CONFIGURATION ----------------
dir_location = "cache/Model_Sense.4n1/round_7/"
model_file = f"{dir_location}/Model_Sense.4n1_round7.pth"
emb_dir = f"{dir_location}/embeddings/"

# ---------------- LOAD MODEL ----------------
input_dim = 384  # <-- change this to match your embedding size
model = SimpleNN(input_dim)
model.load_state_dict(torch.load(model_file))
model.eval()

# ---------------- LOAD TEST DATA ----------------
test_embeddings_list = []
test_labels_list = []

found = False
for fname in os.listdir(emb_dir):
    if fname.startswith("test_") and fname.endswith(".pt"):
        data = torch.load(os.path.join(emb_dir, fname))
        test_embeddings_list.append(data["embedding"])
        test_labels_list.append(data["label"])
        found = True
if not found:
    raise FileNotFoundError(f"No test embeddings found in {emb_dir}")

test_embeddings = torch.stack(test_embeddings_list)
test_labels = torch.tensor(test_labels_list)

test_loader = DataLoader(
    TensorDataset(test_embeddings, test_labels.float()),
    batch_size=32, shuffle=False
)

# ---------------- EVALUATE ----------------
all_preds, all_labels = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.int().tolist())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1:", f1_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))
