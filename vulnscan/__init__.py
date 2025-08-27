import matplotlib.pyplot as plt

from vulnscan.log import log
from vulnscan.genData import generate_dataset, generate_embeddings
from vulnscan.config import cfg
from vulnscan.train import create_sampler, train_model, SimpleNN, EmbeddingDataset


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
    plt.title(f"Training Round {cfg.MODEL_ROUND}")
    plt.savefig(f"{cfg.CACHE_DIR}/round_{cfg.MODEL_ROUND}/training_plot.png")
    plt.close()
    log("Saved training plot.")
