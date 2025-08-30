import matplotlib.pyplot as plt

from vulnscan.log import log
from vulnscan.genData import DataGen
from vulnscan.config import TrainingConfig
from vulnscan.train import Train, SimpleNN, EmbeddingDataset


# ---------------- PLOTTING ----------------
def plot_training(cfg, history_loops: list):
    plt.figure(figsize=(12, 6))

    for i, history in enumerate(history_loops):
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], label=f"Train Loss (Loop {i+1})")
        plt.plot(epochs, history["val_loss"], label=f"Val Loss (Loop {i+1})")
        plt.plot(epochs, history["accuracy"], label=f"Accuracy (Loop {i+1})")
        plt.plot(epochs, history["f1"], label=f"F1 Score (Loop {i+1})")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title(f"Training Progress - Round {cfg.MODEL_ROUND}")
    plt.legend()
    save_path = f"{cfg.CACHE_DIR}/{cfg.MODEL_NAME}/round_{cfg.MODEL_ROUND}/training_plot.png"
    plt.savefig(save_path)
    plt.close()
    log(message=f"Saved training plot at {save_path.replace('\\', '/')}.", cfg=cfg)
