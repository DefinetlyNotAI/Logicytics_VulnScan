import json
import random
import time
from pathlib import Path

import psutil
import torch
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------- CONFIG -------------------
GEN_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
CLS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE_GEN = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMB = "cpu"  # embeddings on CPU to save GPU VRAM

OUTPUT_FILE = Path("generated_files.json")
OFFLOAD_FILE = Path("offloaded_data.json")

NUM_GENERATED = 20
MAX_DATASET_SIZE = 2000
RAM_USAGE_THRESHOLD = 0.9
MAX_LENGTH = 150
TEMPERATURE = 0.9
BATCH_SIZE = 8
EPOCHS = 30


# ------------------- HELPERS -------------------
def log(msg, color=None):
    timestamp = time.strftime("%H:%M:%S")
    if color:
        print(f"\033[{color}m[{timestamp}] {msg}\033[0m")
    else:
        print(f"[{timestamp}] {msg}")


def check_ram_and_offload(dataset_):
    usage = psutil.virtual_memory().used / psutil.virtual_memory().total
    if usage > RAM_USAGE_THRESHOLD:
        log(f"RAM > {RAM_USAGE_THRESHOLD * 100:.0f}%, offloading {len(dataset_)} entries...", color=33)
        offloaded = []
        if OFFLOAD_FILE.exists():
            with open(OFFLOAD_FILE, "r") as f:
                offloaded = json.load(f)
        offloaded.extend(dataset_)
        with open(OFFLOAD_FILE, "w") as f:
            json.dump(offloaded, f, indent=2)
        return []
    return dataset_


def load_dataset():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    return []


def save_dataset(dataset_):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset_, f, indent=2)


def plot_training(losses, cycle):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"Training Loss Cycle {cycle}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"training_plot_{cycle}.png")
    log(f"Saved training plot as training_plot_{cycle}.png", color=36)


# ------------------- MODEL LOAD -------------------
log("Loading generator model...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_gen = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE_GEN == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(DEVICE_GEN)
model_gen.eval()

log("Loading classifier model...")
model_emb = SentenceTransformer(CLS_MODEL_NAME, device=DEVICE_EMB)

dataset = load_dataset()
log(f"Dataset loaded with {len(dataset)} entries")

# ------------------- GENERATION -------------------
PROMPT_TEMPLATES = [
    "Write a Java code snippet that",
    "Create a secure function to",
    "Provide an example of a JDBC operation that",
    "Write a code segment handling exceptions that"
]


def generate_batch(batch_size=NUM_GENERATED):
    prompts = [random.choice(PROMPT_TEMPLATES) for _ in range(batch_size)]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE_GEN)
    with torch.inference_mode():
        outputs = model_gen.generate(
            **enc,
            max_new_tokens=MAX_LENGTH,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ------------------- SENSITIVE DETECTION -------------------
SENSITIVE_KEYWORDS = ["password", "ssn", "credit card", "secret", "private", "token"]


def detect_sensitive(texts):
    embeddings = model_emb.encode(texts, convert_to_tensor=True)
    flags = []
    for t, emb in zip(texts, embeddings):
        keyword_score = sum(kw in t.lower() for kw in SENSITIVE_KEYWORDS)
        flags.append(keyword_score > 0)
    return flags


# ------------------- TRAINING SIMULATION -------------------
def train_cycle():
    log("Starting training cycle...")
    losses = []
    for epoch in range(EPOCHS):
        batch_texts = generate_batch(BATCH_SIZE)
        sensitive_flags = detect_sensitive(batch_texts)
        loss = random.uniform(0.1, 0.5) / (epoch + 1)
        losses.append(loss)

        dataset.extend([{"text": t, "sensitive": f} for t, f in zip(batch_texts, sensitive_flags)])
        dataset[:] = check_ram_and_offload(dataset)

        log(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss:.4f} | Batch: {BATCH_SIZE} | Sensitive: {sum(sensitive_flags)}/{len(batch_texts)} | Dataset: {len(dataset)}")

        if epoch > 2 and abs(losses[-1] - losses[-2]) < 1e-3:
            log("Loss converged. Stopping early.", color=33)
            break
    save_dataset(dataset)
    return losses


# ------------------- MAIN -------------------
def main():
    log("Pipeline started.")
    total_cycles = 0
    while len(dataset) < MAX_DATASET_SIZE:
        print(f"=== Training Cycle {total_cycles + 1} ===")
        losses = train_cycle()
        plot_training(losses, total_cycles + 1)
        total_cycles += 1
    log("Pipeline completed.", color=32)


if __name__ == "__main__":
    main()
