import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vulnscan import DataGen, TrainingConfig, log

# ---------------- CONFIG ----------------
# noinspection DuplicatedCode
cfg = TrainingConfig()
cfg.update({
    "MODEL_NAME": "Model_Sense.4n1",
    "BATCH_SIZE": 32,
    "MAX_EPOCHS": 35,
    "TRAIN_LOOPS": 3,
    "EARLY_STOPPING_PATIENCE": 5,
    "LR": 1e-3,
    "LR_JUMP": {"MAX": 5, "MIN": 0.1},
    "COUNTER": {"PATIENCE": 0, "JUMP": 0},
    "JUMP_PATIENCE": 3,
    "LR_DECAY": 0.9,
    "AUTO_CONTINUE": False,
    "DATASET_SIZE": 25000,
    "TEXT_MAX_LEN": 128,
    "TEXT_MAX_LEN_JUMP_RANGE": 10,
    "VAL_SPLIT": 0.85,
    "TRAIN_VAL_SPLIT": 0.8,
    "SENSITIVE_PROB": 0.5,
    "TOP_K": 30,
    "TOP_P": 0.9,
    "TEMPERATURE": 0.9,
    "REP_PENALTY": 1.2,
    "RETRY_LIMIT": 3,
    "RAM_THRESHOLD": 0.85
})

# ---------------- MODEL ----------------
log(message="Loading GPT-Neo model for text generation...", cfg=cfg)
gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(cfg.DEVICE)
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# ---------------- DATASET GENERATION ----------------
dataset_ranges = [10, 100, 1000, 5000, 10000, 17500, 25000]

for dr in dataset_ranges:
    dataset_path = f"{cfg.DATA_CACHE_DIR}/dataset_{dr}.pt"

    # Skip if already exists
    if os.path.exists(dataset_path):
        data = torch.load(dataset_path, map_location="cpu")
        log(f"Found existing dataset {dataset_path} with {len(data['texts'])} samples. Skipping generation.", cfg=cfg)
        continue

    # --- Find largest smaller dataset ---
    base_texts, base_labels = [], []
    smaller_existing = [r for r in dataset_ranges if r < dr]
    smaller_existing.sort(reverse=True)

    for sr in smaller_existing:
        candidate_path = f"{cfg.DATA_CACHE_DIR}/dataset_{sr}.pt"
        if os.path.exists(candidate_path):
            data = torch.load(candidate_path, map_location="cpu")
            base_texts, base_labels = data["texts"], data["labels"]
            log(f"Using {candidate_path} as base with {len(base_texts)} samples.", cfg=cfg)
            break

    # How many new we need
    remaining = dr - len(base_texts)
    if remaining <= 0:
        log(f"Already have enough samples for {dr}, just saving subset.", cfg=cfg)
        torch.save({"texts": base_texts[:dr], "labels": base_labels[:dr]}, dataset_path)
        continue

    cfg.update({"DATASET_SIZE": remaining})
    log(f"Generating {remaining} new samples for dataset {dr}...", cfg=cfg)
    generate = DataGen(cfg=cfg)

    try:
        new_texts, new_labels = generate.dataset(gpt_tokenizer=gpt_tokenizer, gpt_model=gpt_model)
    except KeyboardInterrupt:
        base_texts.extend(new_texts)
        base_labels.extend(new_labels)
        torch.save({"texts": base_texts, "labels": base_labels}, dataset_path)
        log(f"Interrupted. Saved {len(base_texts)} samples so far to {dataset_path}", cfg=cfg)
        raise

    # Save full dataset
    base_texts.extend(new_texts)
    base_labels.extend(new_labels)
    torch.save({"texts": base_texts, "labels": base_labels}, dataset_path)
    log(f"Saved dataset with {len(base_texts)} samples to {dataset_path}", cfg=cfg)
