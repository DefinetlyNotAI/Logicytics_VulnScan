import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchviz import make_dot
from tqdm import tqdm

from data import test_texts, test_labels

# ----------------- Setup -----------------
NAME = "Model_SenseMacro.4n1"
ROUND = 7
OUTPUT_DIR = f"../{NAME}_Data_Visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EMBEDDINGS_DIR = f"../cache/{NAME}/round_{ROUND}/embeddings"  # your embeddings directory
MODEL_PATH = f"../cache/{NAME}/round_{ROUND}/{NAME}_round{ROUND}.pth"  # path to saved SimpleNN state_dict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Dataset -----------------
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


# ----------------- Model -----------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim_):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim_, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


# ----------------- Visualizations -----------------

def visualize_weight_distribution(model_, filename="Weight_Distribution.png"):
    layer = model_.fc[0]
    weights = layer.weight.detach().cpu().numpy()
    plt.hist(weights.flatten(), bins=50)
    plt.title("Weight Distribution - First Layer")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def visualize_activations(model_, sample_input_, filename="Visualize_Activation.png"):
    activations = []

    # noinspection PyUnusedLocal
    def hook_fn(module_, input_, output):
        activations.append(output)

    model_.fc[0].register_forward_hook(hook_fn)
    _ = model_(sample_input_.to(DEVICE))
    act = activations[0].detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.bar(range(act.shape[1]), act[0])
    plt.title("Activation Values - First Layer")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Value")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def visualize_tsne_custom(model_, embedder_, texts_, labels_, filename="Visualize_tSNE_custom.png",
                          use_penultimate=True):
    """
    Visualize t-SNE for a few custom real-world texts.

    Args:
        model_ : trained SimpleNN classifier
        embedder_ : embedding model (e.g., MiniLM)
        texts_ (list[str]) : input sentences
        labels_ (list[int]) : class labels (0=Non-sensitive, 1=Sensitive)
        filename (str) : output file
        use_penultimate (bool) : take penultimate layer features
    """
    model_.eval()
    all_features = []

    with torch.no_grad():
        for text in texts_:
            # get embeddings from MiniLM (or your embedder)
            embedding = embedder_.encode(text, convert_to_tensor=True).unsqueeze(0).to(DEVICE)

            if use_penultimate:
                # pass through all but the last Linear layer
                feat = model_.fc[:-1](embedding)
            else:
                feat = model_(embedding)

            all_features.append(feat.cpu().numpy())

    all_features = np.vstack(all_features)
    labels_ = np.array(labels_)
    n_samples, n_features = all_features.shape
    n_components = min(2, n_samples, n_features)

    tsne = TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=max(1, min(30, n_samples - 1))
    )
    reduced = tsne.fit_transform(all_features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_, cmap="viridis", s=80, edgecolors="k")
    plt.colorbar(scatter, label="Class (0=Non-sensitive, 1=Sensitive)")
    for i, txt in enumerate(texts_):
        plt.annotate(f"{labels_[i]}:{i}", (reduced[i, 0] + 1, reduced[i, 1] + 1), fontsize=8)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE of Custom Real-World Samples")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def visualize_tsne(model_, dataloader_, filename="Visualize_tSNE.png", use_penultimate=True):
    model_.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader_:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if use_penultimate:  # use output of second-to-last layer for richer representation
                feat = model_.fc[:-1](X)
            else:
                feat = model_(X)
            all_features.append(feat.cpu().numpy())
            all_labels.append(y.cpu().numpy().ravel())
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels, axis=0)
    n_samples, n_features = all_features.shape
    n_components = min(2, n_samples, n_features)  # adapt automatically
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=max(1, min(30, n_samples - 1)))
    reduced = tsne.fit_transform(all_features)
    plt.figure(figsize=(10, 8))
    if n_components == 1:
        plt.scatter(range(n_samples), reduced[:, 0], c=all_labels, cmap='viridis', alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("t-SNE Dim 1")
    else:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=all_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Class (0=Non-sensitive,1=Sensitive)")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.title("t-SNE Visualization of Features")
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()


def visualize_feature_importance(input_dim_, filename="Feature_Importance.svg"):
    tokens = [f"feat_{i}" for i in range(input_dim_)]
    importance = np.random.rand(input_dim_)
    plt.figure(figsize=(len(tokens) * 0.5, 6))
    # use a single color to avoid the Seaborn palette warning
    sns.barplot(x=tokens[:1000], y=importance[:1000], color="steelblue")
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def plot_loss_landscape_3d(model_, dataloader_, criterion_, grid_size=None, epsilon=0.01,
                           filename="Loss_Landscape_3D.html", device="cpu"):
    model_.eval()
    model_.to(device)

    # Flatten all parameters into a single vector
    params = torch.cat([p.view(-1) for p in model_.parameters()])

    # Create random directions u, v in parameter space
    u = epsilon * torch.randn_like(params)
    v = epsilon * torch.randn_like(params)
    u /= torch.norm(u)
    v /= torch.norm(v)

    if grid_size is None:
        param_count = sum(p.numel() for p in model_.parameters())
        grid_size = max(10, min(50, param_count // 10))
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    loss_values = np.zeros((grid_size, grid_size))

    # Store original parameters
    orig_params = params.clone()

    with torch.no_grad():
        for i, dx in enumerate(tqdm(x, desc="dx")):
            for j, dy in enumerate(y):
                # Perturbed parameter vector
                new_params = orig_params + dx * u + dy * v

                # Load new parameters into the model temporarily
                idx = 0
                for p in model_.parameters():
                    numel = p.numel()
                    p.copy_(new_params[idx:idx + numel].view_as(p))
                    idx += numel

                # Compute loss
                total_loss = 0
                for X, yb in dataloader_:
                    X, yb = X.to(device), yb.to(device)
                    yb = yb.float().view(-1, 1)
                    out = model_(X)
                    total_loss += criterion_(out, yb).item()
                loss_values[i, j] = total_loss

        # Restore original parameters
        idx = 0
        for p in model_.parameters():
            numel = p.numel()
            p.copy_(orig_params[idx:idx + numel].view_as(p))
            idx += numel

    # Plot
    X_grid, Y_grid = np.meshgrid(x, y)
    fig = go.Figure(data=[go.Surface(z=loss_values, x=X_grid, y=Y_grid, colorscale="Viridis")])
    fig.update_layout(title="Loss Landscape", scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="Loss"))

    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    fig.write_html(filename)
    print(f"3D loss landscape saved to {filename}")


def save_model_state_dict(model_, filename="Model_State_Dict.txt"):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        for name, tensor in model_.state_dict().items():
            f.write(f"{name}: {tensor.size()}\n")


def generate_model_visualization(model_, input_dim_, filename="Model_Visualization.png"):
    dummy_input = torch.randn(1, input_dim_).to(DEVICE)
    dot = make_dot(model_(dummy_input), params=dict(model_.named_parameters()))
    dot.format = "png"
    dot.render(filename=os.path.join(OUTPUT_DIR, filename), format="png")


def save_graph(model_, filename="Neural_Network_Nodes_Graph.gexf"):
    G = nx.DiGraph()
    threshold = 0.1
    for name, param in model_.named_parameters():
        if 'weight' in name:
            W = param.detach().cpu().numpy()
            rows, cols = np.where(np.abs(W) > threshold)
            for r, c in zip(rows, cols):
                G.add_edge(f"{name}_in_{c}", f"{name}_out_{r}", weight=W[r, c])
    nx.write_gexf(G, os.path.join(OUTPUT_DIR, filename))


def save_model_summary(model_, filename="Model_Summary.txt"):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        f.write(str(model_))


# ----------------- Main -----------------
dataset = EmbeddingDataset(EMBEDDINGS_DIR)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Determine input_dim from dataset
sample_X, _ = dataset[0]
input_dim = sample_X.shape[0]

# Load model
model = SimpleNN(input_dim).to(DEVICE)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Sample input for activation visualization
sample_input = sample_X.unsqueeze(0).to(DEVICE)

# Criterion for loss landscape
criterion = nn.BCEWithLogitsLoss()

# Run all visualizations
print("Running visualize_weight_distribution...")
visualize_weight_distribution(model)
print("Running visualize_activations...")
visualize_activations(model, sample_input)
print("Preparing texts and labels for t-SNE custom visualization...")

print("Loading SentenceTransformer embedder...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Running visualize_tsne...")
visualize_tsne(model, dataloader)
print("Running visualize_tsne_custom...")
visualize_tsne_custom(model, embedder, test_texts, test_labels)
print("Running visualize_feature_importance...")
visualize_feature_importance(input_dim)
print("Saving model state dict...")
save_model_state_dict(model)
print("Generating model visualization...")
generate_model_visualization(model, input_dim)
print("Saving graph...")
save_graph(model)
print("Saving model summary...")
save_model_summary(model)
print("Running plot_loss_landscape_3d...")
DEVICE = "cpu"  # or "cuda" if you want GPU
model_cpu = model.to(DEVICE)
plot_loss_landscape_3d(
    model_=model_cpu,
    dataloader_=dataloader,
    criterion_=criterion,
    filename="Loss_Landscape_3D.html",
    device=DEVICE
)
print("All visualizations completed. Files saved in 'data/' directory.")
