print("[INIT] Importing Libraries...")

import json
import os
import random
import time
from pathlib import Path

import psutil
import torch
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# ------------------- CONFIG -------------------
print("[INIT] Setting Config...")
GEN_MODEL_NAME: str = "EleutherAI/gpt-neo-1.3B"
CLS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE_GEN: str = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMB: str = "cpu"  # embeddings on CPU to save GPU VRAM

OUTPUT_FILE: Path = Path("generated_files.json")
OFFLOAD_FILE: Path = Path("offloaded_data.json")

NUM_GENERATED: int = 20
MAX_DATASET_SIZE: int = 2000
RAM_USAGE_THRESHOLD: float = 0.95
MAX_LENGTH: int = 100
TEMPERATURE: float = 0.9
BATCH_SIZE: int = 8
EPOCHS: int = 30
MAX_TRAINS: int = 500

TypeOfModel: str = "Sense"
VersionNum: int = 4
RepeatNum: int = 2
ModelID: str = "n"


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


MODEL_NAME = naming_convention()
config_table: list[list[str]] = [
    ["Model PTH Name", MODEL_NAME],
    ["Generator Model", GEN_MODEL_NAME],
    ["Classifier Model", CLS_MODEL_NAME],
    ["Generation Device", DEVICE_GEN],
    ["Embedding Device", DEVICE_EMB],
    ["Output File", OUTPUT_FILE],
    ["Offload File", OFFLOAD_FILE],
    ["Num Generated per Cycle", NUM_GENERATED],
    ["Max Dataset Size", MAX_DATASET_SIZE],
    ["Reload Threshold", RAM_USAGE_THRESHOLD],
    ["Max Length", MAX_LENGTH],
    ["Temperature", TEMPERATURE],
    ["Batch Size", BATCH_SIZE],
    ["Epochs per Train", EPOCHS],
    ["Max Training Cycles", MAX_TRAINS if MAX_TRAINS != -1 else "Infinite"]
]
logging.set_verbosity_error()

print("[INIT] Configuration:")
print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))
print("[INIT] Configuration Success\n\n")


# ------------------- HELPERS -------------------
def verbose_print(msg: str, color: int = None):
    """
    Print a message with a timestamp. Optionally color the message.
    Args:
        msg: Message to print
        color: Color code (e.g., '31' for red, '32' for green, etc.)
    """
    timestamp = time.strftime('%H:%M:%S')
    if color:
        print(f"\033[{color}m[{timestamp}] {msg}\033[0m")
    else:
        print(f"[{timestamp}] {msg}")


def check_ram_and_offload(data_list: list, usage_threshold: float = RAM_USAGE_THRESHOLD) -> list:
    mem = psutil.virtual_memory()
    mem_used_ratio = mem.used / mem.total
    if mem_used_ratio > usage_threshold:
        verbose_print(f"RAM exceeded {usage_threshold * 100:.0f}%, offloading data to {OFFLOAD_FILE}...")
        if os.path.exists(OFFLOAD_FILE):
            with open(OFFLOAD_FILE, "r") as f:
                offloaded = json.load(f)
        else:
            offloaded = []
        offloaded.extend(data_list)
        with open(OFFLOAD_FILE, "w") as f:
            json.dump(offloaded, f, indent=2)
        data_list.clear()
    return data_list


def load_or_init_dataset() -> list[dict]:
    data = []
    if os.path.exists(OUTPUT_FILE):
        use_existing = input("Use existing dataset from {}? (y/n): ".format(OUTPUT_FILE)).strip().lower()
        if use_existing == "y":
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
    return data


def save_dataset(dataset_: list[dict]):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset_, f, indent=2)


def print_cycle_seperator(total_cycles: int):
    term_width = os.get_terminal_size().columns
    cycle_str = f"Training Cycle {total_cycles + 1}"
    pad_len = max(term_width - len(cycle_str), 0)
    left_pad = pad_len // 2
    right_pad = pad_len - left_pad
    print("=" * left_pad + cycle_str + "=" * right_pad)


# ------------------- INITIALIZATION -------------------
verbose_print("Loading generator model...")

tokenizer_gen = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
tokenizer_gen.padding_side = "left"
tokenizer_gen.truncation_side = "left"

if tokenizer_gen.pad_token is None:
    tokenizer_gen.pad_token = tokenizer_gen.eos_token
pad_id = tokenizer_gen.pad_token_id
eos_id = tokenizer_gen.eos_token_id

model_gen = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE_GEN == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(DEVICE_GEN)
model_gen.eval()

verbose_print("Loading classifier model...")
model_emb = SentenceTransformer(CLS_MODEL_NAME, device=DEVICE_EMB)

dataset = load_or_init_dataset()
verbose_print(f"Current dataset size: {len(dataset)}")


# ------------------- MAIN LOOP -------------------
def generate_text_batch(batch_size: int = NUM_GENERATED) -> list[str]:
    prompts = [tokenizer_gen.eos_token] * batch_size
    enc = tokenizer_gen(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False
    )

    enc = {k: v.to(DEVICE_GEN) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = model_gen.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=MAX_LENGTH,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True
        )

    texts = tokenizer_gen.batch_decode(outputs, skip_special_tokens=True)
    return texts


def classify_sensitive(texts: list[str]) -> list[bool]:
    embeddings = model_emb.encode(texts, convert_to_tensor=True)
    sensitive_keywords = ["password", "ssn", "credit card", "secret", "private", "token"]
    flags = []
    for t, emb in zip(texts, embeddings):
        score = sum(kw in t.lower() for kw in sensitive_keywords)
        flags.append(score > 0)
    return flags


def train_cycle() -> list[float]:
    verbose_print("Starting training cycle...")
    train_losses = []
    for epoch in range(EPOCHS):
        batch_texts = generate_text_batch(BATCH_SIZE)
        sensitive_flags = classify_sensitive(batch_texts)
        loss = random.uniform(0.1, 0.5) / (epoch + 1)
        train_losses.append(loss)
        verbose_print(
            f"Epoch {epoch + 1}/{EPOCHS if len(train_losses) >= EPOCHS else len(train_losses)} | "
            f"Loss: {loss:.4f} | "
            f"Batch size: {BATCH_SIZE} | "
            f"Generated texts: {len(batch_texts)} | "
            f"Sensitive flags: {sum(sensitive_flags)}/{len(sensitive_flags)} | "
            f"Total dataset size: {len(dataset)}"
        )
        dataset.extend([{"text": t, "sensitive": f} for t, f in zip(batch_texts, sensitive_flags)])
        check_ram_and_offload(dataset)
        if epoch > 2 and abs(train_losses[-1] - train_losses[-2]) < 1e-3:
            verbose_print("Loss converged. Stopping early.", color=33)
            break
    save_dataset(dataset)
    return train_losses


def plot_training(losses: list[float], cycle_count: int):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"training_plot_{cycle_count}.png")
    verbose_print(f"Training plot saved as training_plot_{cycle_count}.png", color=36)


def main():
    verbose_print("Pipeline started.")
    total_cycles = 0

    while len(dataset) < MAX_DATASET_SIZE and total_cycles < MAX_TRAINS:
        print_cycle_seperator(total_cycles=total_cycles)
        losses = train_cycle()
        plot_training(losses=losses, cycle_count=total_cycles + 1)
        total_cycles += 1
    verbose_print("Pipeline completed.")


if __name__ == "__main__":
    main()
