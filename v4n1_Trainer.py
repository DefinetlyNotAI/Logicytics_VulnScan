import json
import os
import sys

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from vulnscan import log, Train, plot_training, SimpleNN, EmbeddingDataset, TrainingConfig, DataGen


# ---------------- INIT ----------------
def init(config: TrainingConfig) -> dict:
    """Initialize static, config-free resources (only once)."""
    try:
        log("Loading GPT-Neo tokenizer/model (static init)...", cfg=config, only_console=True)
        gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        if gpt_tokenizer.pad_token is None:
            gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

        log("Loading MiniLM for embeddings (static init)...", cfg=config, only_console=True)
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        return {
            "gpt_tokenizer": gpt_tokenizer,
            "gpt_model": gpt_model,
            "embed_model": embed_model,
        }
    except KeyboardInterrupt:
        sys.exit("Interrupted by user in initialization.")
    except Exception as err:
        sys.exit(f"Error during initialization: {err}")


# ---------------- TRAIN ----------------
def train(config: TrainingConfig, resources: dict):
    part = "???"
    try:
        # Load resources from init
        part = "init resources loading"
        gpt_tokenizer = resources["gpt_tokenizer"]
        gpt_model = resources["gpt_model"].to(config.DEVICE)  # attach to device here
        embed_model = resources["embed_model"]

        # Initialise DataGen
        part = "initialising DataGen"
        log("Initialising DataGen with config...", cfg=config, silent=True)
        generate = DataGen(cfg=config)

        # Generate dataset
        part = "generating/loading the dataset"
        dataset_path = f"{config.DATASET_CACHE_DIR}/dataset_{config.DATASET_SIZE}.pt"
        if os.path.exists(dataset_path):
            log("Loading existing dataset...", cfg=config)
            data = torch.load(dataset_path)
            texts, labels = data["texts"], data["labels"]
        else:
            log("Dataset not found, generating", cfg=config)
            texts, labels = generate.dataset(gpt_tokenizer=gpt_tokenizer, gpt_model=gpt_model)
            torch.save({"texts": texts, "labels": labels}, dataset_path)

        # Split dataset
        part = "splitting the dataset"
        train_split = int(len(texts) * config.TRAIN_VAL_SPLIT)
        val_split = int(len(texts) * config.VAL_SPLIT)

        train_texts, train_labels = texts[:train_split], labels[:train_split]
        val_texts, val_labels = texts[train_split:val_split], labels[train_split:val_split]
        test_texts, test_labels = texts[val_split:], labels[val_split:]

        # Generate embeddings for all splits
        part = "generating the embeddings"
        log("Generating test embeddings...", cfg=config)
        generate.embeddings(embed_model=embed_model, texts=test_texts, labels=test_labels, split="test")
        log("Generating train embeddings...", cfg=config)
        generate.embeddings(embed_model=embed_model, texts=train_texts, labels=train_labels, split="train")
        log("Generating validation embeddings...", cfg=config)
        generate.embeddings(embed_model=embed_model, texts=val_texts, labels=val_labels, split="validation")

        # Prepare datasets and dataloaders
        part = "preparing datasets and dataloaders"
        train_dataset = EmbeddingDataset(config.EMBED_CACHE_DIR)
        val_dataset = EmbeddingDataset(config.EMBED_CACHE_DIR)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        train_ = Train(cfg=config)
        model = SimpleNN(input_dim=384).to(config.DEVICE)

        # Run training (handles TRAIN_LOOPS internally)
        part = "training the model"
        history_loops = train_.model(model=model, train_dataset=train_dataset, val_loader=val_loader)

        # Plot + save history for each loop
        part = "plotting and saving training history"
        for i, history in enumerate(history_loops):
            plot_training(cfg=config, history_loops=history_loops)
            with open(
                    f"{config.CACHE_DIR}/{config.MODEL_NAME}/round_{config.MODEL_ROUND}/training_history_loop{i + 1}.json",
                    "w") as f:
                json.dump(history, f)

        log("Training complete. All data, plots, and model saved.", cfg=config)
    except KeyboardInterrupt:
        sys.exit("Interrupted by user during training.")
    except Exception as err:
        sys.exit(f"Error during '{part}': {err}")


if __name__ == "__main__":
    # noinspection DuplicatedCode
    # ---------------- CONFIG ----------------
    cfg = TrainingConfig()
    cfg.update({
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

        # Number of samples to generate for training (not the same as for the training rounds themselves)
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
    train_init = init(cfg)

    # ----------------- RUN ------------------
    try:
        available_dataset = [10, 100, 1000, 5000, 10000, 17500, 25000]
        for loop_idx, dataset in enumerate(available_dataset, start=1):
            if dataset <= 1000:
                name = "SenseNano"
            elif 1000 < dataset <= 5000:
                name = "SenseMini"
            elif 5000 < dataset <= 10000:
                name = "Sense"
            else:
                name = "SenseMacro"
            model_round = loop_idx
            cfg.update({
                # Model / caching / logging
                "MODEL_NAME": f"Model_{name}.4n1",  # Name of the model for identification and caching
                "DATASET_SIZE": dataset,
                # Number of samples to generate for training (not the same as for the training rounds themselves)
                "MODEL_ROUND": model_round  # Current training round (auto-incremented)
            })
            log(message=f"Training 'Model_{name}.4n1/round_{model_round}/' with {dataset} dataset...", cfg=cfg)
            train(config=cfg, resources=train_init)
    except KeyboardInterrupt:
        sys.exit("Interrupted by user in main.")
    except Exception as e:
        sys.exit(f"Error during training: {e}")
