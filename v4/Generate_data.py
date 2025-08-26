import json
import re
import signal
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# -------------------
# Configuration
# -------------------
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
NUM_SAMPLES = 100
OUTPUT_FILE = "generated_files.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 150  # shorter for speed
TEMPERATURE = 0.8
BATCH_SIZE = 64  # much higher batch for speed
RETRIES = 2  # only one retry for speed

SENSITIVE_PROMPTS = [
    "Generate a fake config file with a password and API key.",
    "Write a snippet with a hardcoded secret in code.",
    "Create a log entry that leaks an email and password.",
    "Show a database dump with a secret token.",
    "Write a script with embedded credentials.",
]
NON_SENSITIVE_PROMPTS = [
    "Generate a harmless meeting note.",
    "Write a code snippet with no secrets.",
    "Create a log entry with generic info.",
    "Show a database dump with only public data.",
    "Write a script with placeholder values.",
]

# -------------------
# Setup
# -------------------
print(f"Using device: {DEVICE.upper()}")
print(f"Model: {MODEL_NAME}")

print("Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Loading and evaluating model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)
model.eval()

generated_data = []
print("Setup complete.")

# -------------------
# Graceful Cleanup
# -------------------
def save_and_exit():
    print(f"\nSaving {len(generated_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, indent=2)
    print("Done.")
    sys.exit(0)


def signal_handler(sig, frame):
    print("\nKeyboardInterrupt detected! Cleaning up...")
    save_and_exit()


signal.signal(signal.SIGINT, signal_handler)


# -------------------
# Helper: Clean output
# -------------------
def clean_output(text, prompt):
    # Remove prompt echo and leading/trailing whitespace
    cleaned = text.replace(prompt, "").strip()
    # Remove repeated prompt fragments
    for line in prompt.split("\n"):
        cleaned = cleaned.replace(line.strip(), "")
    return cleaned


def plausible_sensitive(text):
    # Check for plausible secrets (email, key, password, token)
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
        r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9\-]{10,}",  # API key
        r"(?i)password\s*[:=]\s*['\"][^'\"]{6,}['\"]",  # password
        r"(?i)token\s*[:=]\s*[A-Za-z0-9\-]{10,}",  # token
    ]
    return any(re.search(p, text) for p in patterns)


def is_bad_output(text, prompt, label):
    # Too short or too similar to prompt
    cleaned = clean_output(text, prompt)
    if len(cleaned) < 10:
        return True
    if cleaned.lower() in prompt.lower():
        return True
    if label == "sensitive" and not plausible_sensitive(cleaned):
        return True
    return False


# -------------------
# Generation Loop
# -------------------
try:
    print("Starting generation of test data")
    seen_texts = set()
    import random

    # Precompute prompt batches for efficiency
    prompt_pairs = []
    for i in range(NUM_SAMPLES):
        if i % 2 == 0:
            prompt_pairs.append((random.choice(SENSITIVE_PROMPTS), "sensitive"))
        else:
            prompt_pairs.append((random.choice(NON_SENSITIVE_PROMPTS), "non-sensitive"))

    i = 0
    while i < NUM_SAMPLES:
        batch_prompts = []
        labels = []
        for _ in range(min(BATCH_SIZE, NUM_SAMPLES - i)):
            prompt, label = prompt_pairs[i + _]
            batch_prompts.append(prompt)
            labels.append(label)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE, non_blocking=True)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for idx, (text, label) in enumerate(zip(decoded, labels)):
            prompt = batch_prompts[idx]
            cleaned = clean_output(text, prompt)
            retry_count = 0
            while is_bad_output(cleaned, prompt, label) and retry_count < RETRIES:
                with torch.inference_mode():
                    output = model.generate(
                        **tokenizer(prompt, return_tensors="pt").to(DEVICE, non_blocking=True),
                        max_length=MAX_LENGTH,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                cleaned = clean_output(tokenizer.decode(output[0], skip_special_tokens=True), prompt)
                retry_count += 1
            # Deduplicate
            if not is_bad_output(cleaned, prompt, label) and cleaned not in seen_texts:
                generated_data.append({"label": label, "text": cleaned})
                seen_texts.add(cleaned)
                i += 1
            # else: skip bad output or duplicate
        print(f"\rGenerated {len(generated_data)}/{NUM_SAMPLES} samples", end="", flush=True)
except Exception as e:
    print(e)
finally:
    save_and_exit()
