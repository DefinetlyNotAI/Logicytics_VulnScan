import json
import os

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from vulnscan import log, Train, plot_training, SimpleNN, EmbeddingDataset, TrainingConfig, DataGen


# ---------------- MAIN ----------------
def main():
    log(message="Loading GPT-Neo model for text generation...", cfg=cfg)
    gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(cfg.DEVICE)
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    log(message="Loading MiniLM for embeddings...", cfg=cfg)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    log(message="Starting advanced self-training sensitive data classifier...", cfg=cfg)
    generate = DataGen(cfg=cfg)

    # Generate dataset
    dataset_path = f"{cfg.DATA_CACHE_DIR}/dataset_{cfg.DATASET_SIZE}.pt"
    if os.path.exists(dataset_path):
        data = torch.load(dataset_path)
        texts, labels = data["texts"], data["labels"]
    else:
        texts, labels = generate.dataset(gpt_tokenizer=gpt_tokenizer, gpt_model=gpt_model)
        torch.save({"texts": texts, "labels": labels}, dataset_path)

    train_split = int(len(texts) * cfg.TRAIN_VAL_SPLIT)
    val_split = int(len(texts) * cfg.VAL_SPLIT)

    train_texts, train_labels = texts[:train_split], labels[:train_split]
    val_texts, val_labels = texts[train_split:val_split], labels[train_split:val_split]
    test_texts, test_labels = texts[val_split:], labels[val_split:]

    log(message="Generating test embeddings...", cfg=cfg)
    generate.embeddings(embed_model=embed_model, texts=test_texts, labels=test_labels, split="test")
    log(message="Generating train embeddings...", cfg=cfg)
    generate.embeddings(embed_model=embed_model, texts=train_texts, labels=train_labels, split="train")
    log(message="Generating validation embeddings...", cfg=cfg)
    generate.embeddings(embed_model=embed_model, texts=val_texts, labels=val_labels, split="validation")

    train_dataset = EmbeddingDataset(cfg.EMBED_CACHE_DIR)
    val_dataset = EmbeddingDataset(cfg.EMBED_CACHE_DIR)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    train = Train(cfg=cfg)
    model = SimpleNN(input_dim=384).to(cfg.DEVICE)

    # Run training (handles TRAIN_LOOPS internally)
    history_loops = train.model(model=model, train_dataset=train_dataset, val_loader=val_loader)

    # Plot + save history for each loop
    for i, history in enumerate(history_loops):
        plot_training(cfg=cfg, history_loops=history_loops)
        with open(f"{cfg.CACHE_DIR}/round_{cfg.MODEL_ROUND}/training_history_loop{i+1}.json", "w") as f:
            json.dump(history, f)

    log(message="Training complete. All data, plots, and model saved.", cfg=cfg)


if __name__ == "__main__":
    cfg = TrainingConfig()
    cfg.update({
        # Model / caching / logging
        "MODEL_NAME": "Model_Sense.4n1",  # Name of the model for identification and caching

        # Training parameters
        "BATCH_SIZE": 32,  # Number of samples per training batch
        "MAX_EPOCHS": 35,  # Maximum number of training epochs
        "TRAIN_LOOPS": 3,  # Number of training loops (full dataset passes, with improvement measures)
        "EARLY_STOPPING_PATIENCE": 5,  # Number of epochs to wait for improvement before premature stopping
        "LR": 1e-3,  # Initial learning rate
        "LR_JUMP": {"MAX": 5, "MIN": 0.1},  # Upper and lower limits for learning rate jumps
        "COUNTER": {"PATIENCE": 0, "JUMP": 0},  # Counters for early stopping patience and learning rate jumps
        "JUMP_PATIENCE": 3,  # Epochs to wait before applying a learning rate jump
        "LR_DECAY": 0.9,  # Factor to multiply learning rate after decay
        "AUTO_CONTINUE": False,  # Whether to automatically continue training and ignore EARLY_STOPPING_PATIENCE

        # Dataset / data generation
        "DATASET_SIZE": 25000,  # Number of samples to generate for training (not the same as for the training rounds themselves)
        "TEXT_MAX_LEN": 128,  # Maximum length of generated text samples
        "TEXT_MAX_LEN_JUMP_RANGE": 10,  # Range for random variation in text length
        "VAL_SPLIT": 0.85,  # Fraction of dataset used for training + validation (rest for testing)
        "TRAIN_VAL_SPLIT": 0.8,  # Fraction of dataset used for training (rest for validation)
        "SENSITIVE_PROB": 0.5,  # Probability that a sample contains sensitive data

        # Language / generation
        "TOP_K": 30,  # Top-K sampling: only consider this many top predictions
        "TOP_P": 0.9,  # Top-p (nucleus) sampling probability
        "TEMPERATURE": 0.9,  # Sampling temperature for randomness
        "REP_PENALTY": 1.2,  # Repetition penalty to reduce repeated tokens
        "RETRY_LIMIT": 3,  # Number of times to retry generation if it fails

        # Device / system
        "RAM_THRESHOLD": 0.85  # Maximum allowed fraction of RAM usage before halting generation and offloading
    })
    main()
