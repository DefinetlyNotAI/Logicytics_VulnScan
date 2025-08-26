import json
import re
import signal
import sys
import time
import random

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


def signal_handler(*args, **kwargs):
    print(f"\nKeyboardInterrupt detected! Cleaning up... Provided arguments for handlers: {args}, {kwargs}")
    save_and_exit()


signal.signal(signal.SIGINT, signal_handler)


# -------------------
# Helper: Clean output
# -------------------
def clean_output(text_, prompt_):
    # Remove prompt echo and leading/trailing whitespace
    cleaned_ = text_.replace(prompt_, "").strip()
    # Remove repeated prompt fragments
    for line in prompt_.split("\n"):
        cleaned_ = cleaned_.replace(line.strip(), "")
    return cleaned_


def plausible_sensitive(text_):
    # Check for plausible secrets (email, key, password, token)
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
        r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9\-]{10,}",  # API key
        r"(?i)password\s*[:=]\s*['\"][^'\"]{6,}['\"]",  # password
        r"(?i)token\s*[:=]\s*[A-Za-z0-9\-]{10,}",  # token
    ]
    return any(re.search(p, text_) for p in patterns)


def is_bad_output(text_, prompt_, label_):
    # Too short or too similar to prompt
    cleaned_ = clean_output(text_, prompt_)
    if len(cleaned_) < 10:
        return True
    if cleaned_.lower() in prompt_.lower():
        return True
    if label_ == "sensitive" and not plausible_sensitive(cleaned_):
        return True
    return False


# -------------------
# Generation Loop
# -------------------
try:
    print("Starting generation of test data")
    seen_texts = set()

    # Precompute prompt batches for efficiency
    prompt_pairs = []
    for i in range(NUM_SAMPLES):
        if i % 2 == 0:
            prompt_pairs.append((random.choice(SENSITIVE_PROMPTS), "sensitive"))
        else:
            prompt_pairs.append((random.choice(NON_SENSITIVE_PROMPTS), "non-sensitive"))

    i = 0
    start_time = time.time()
    bar_len = 30  # Progress bar length
    while i < NUM_SAMPLES:
        done = len(generated_data)
        remaining = NUM_SAMPLES - done
        # Calculate values or use '?' if no samples yet
        if done == 0:
            avg_time = "?"
            expected_total = "?"
            expected_remaining = "?"
        else:
            elapsed = time.time() - start_time
            avg_time = f"{elapsed / done:.2f}"
            expected_total = f"{(elapsed / done) * NUM_SAMPLES:.1f}"
            expected_remaining = f"{(elapsed / done) * remaining:.1f}"

        filled = int(bar_len * done / NUM_SAMPLES)
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
        print(
            f"\r{bar} {done}/{NUM_SAMPLES} | Avg: {avg_time}s/sample | ETA: {expected_remaining}s | Total est: {expected_total}s",
            end="", flush=True
        )

        batch_start_time = time.time()  # optional: for per-batch timing
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
except Exception as e:
    print(e)
finally:
    save_and_exit()
