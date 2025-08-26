import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# -------------------
# Configuration
# -------------------
BASE_BATCH_SIZE = 8
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EARLY_STOPPING_PATIENCE = 5
JSONDataFile = "./v4/generated_files.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Naming, check README.md for details
TypeOfModel: str = "SenseNano"
VersionNum: int = 1
RepeatNum: int = 1
ModelID: str = "n"


# -------------------
# Dataset
# -------------------
class FileDataset(Dataset):
    def __init__(self, data_file_):
        with open(data_file_, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings_cache = {}

    def get_embedding(self, text):
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        vector = self.model.encode(text)
        self.embeddings_cache[text] = vector
        return vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.tensor(self.get_embedding(item["content"]), dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.float32)
        return embedding, label


# -------------------
# Neural Network
# -------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# -------------------
# Training
# -------------------
def naming_convention() -> str:
    Version = f"{VersionNum}{ModelID}{RepeatNum}"
    ModelName = f"Model_{TypeOfModel}.{Version}"

    AllowedModelTypes = ["SenseNano", "SenseMini", "Sense", "SenseMacro"]
    AllowedModelIDs = ["b", "dt", "et", "g", "l", "n", "nb", "r", "lr", "v", "x"]
    if TypeOfModel not in AllowedModelTypes:
        raise ValueError(f"TypeOfModel must be one of [{AllowedModelTypes}]")
    if ModelID not in AllowedModelIDs:
        raise ValueError(f"ModelID must be one of [{AllowedModelIDs}]")

    return ModelName


def train(data_file: str, generated_file_name: str):
    dataset = FileDataset(data_file)

    # Dynamic batch size
    batch_size = min(BASE_BATCH_SIZE, max(1, len(dataset) // 10))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(dataset[0][0])
    model = SimpleClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}, Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{generated_file_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"No improvement for {EARLY_STOPPING_PATIENCE} epochs. Stopping early.")
                break

    print(f"Training complete, best model saved to {generated_file_name}.pth")


if __name__ == "__main__":
    name = naming_convention()
    train(data_file=JSONDataFile, generated_file_name=name)
