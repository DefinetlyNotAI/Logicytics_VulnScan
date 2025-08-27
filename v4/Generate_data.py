import json
import signal
import sys
import time
from itertools import cycle

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------- Configuration -------------------
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
NUM_SAMPLES = 5000
OUTPUT_FILE = "generated_files.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 100
TEMPERATURE = 1.0
BATCH_SIZE = 256
RETRIES = 3

SENSITIVE_PROMPTS = [
    "Generate a Python snippet with a password like 'Pa$$w0rd123' and an API key like 'ABCD-1234-EFGH-5678'.",
    "Write a config file containing a secret token of 16 alphanumeric characters.",
    "Create a log entry that exposes a user's email 'user@example.com' and password 'Secret!23'.",
]
NON_SENSITIVE_PROMPTS = [
    "Generate a harmless meeting note.",
    "Write a code snippet with no secrets.",
    "Create a log entry with generic info.",
    "Show a database dump with only public data.",
    "Write a script with placeholder values.",
]

# ------------------- Setup -------------------
print(f"Settings Used\n    Batch Size: {BATCH_SIZE} | Temperature: {TEMPERATURE}\n    Samples: {NUM_SAMPLES} | Max Length: {MAX_LENGTH}\n    Output: {OUTPUT_FILE}")
print(f"Using device: {DEVICE.upper()} | Model: {MODEL_NAME}\n")

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
if DEVICE == "cuda" and hasattr(torch, "compile"):
    model = torch.compile(model)

generated_data = []
print("Setup complete.\n")


# ------------------- Signal Handling -------------------
def save_and_exit():
    print(f"\nSaving {len(generated_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_:
        json.dump(generated_data, f_, indent=2)
    print("Done.")
    sys.exit(0)


signal.signal(signal.SIGINT, lambda *a: save_and_exit())


# ------------------- Helpers -------------------
def clean_output(text_, prompt_):
    cleaned_ = text_.replace(prompt_, "").strip()
    for line in prompt_.split("\n"):
        cleaned_ = cleaned_.replace(line.strip(), "")
    return cleaned_.strip()


def is_valid_output(text_, prompt_):
    cleaned_ = clean_output(text_, prompt_)
    if len(cleaned_) < 10:
        return False
    return True


# ------------------- Generation -------------------
# ------------------- Generation with Auto Flush -------------------
try:
    print("Starting generation...")
    start_time = time.time()
    seen_texts = set()
    prompts_cycle = cycle([(p, "sensitive") for p in SENSITIVE_PROMPTS] +
                          [(p, "non-sensitive") for p in NON_SENSITIVE_PROMPTS])

    FLUSH_THRESHOLD = int(NUM_SAMPLES * 0.9)  # 90% of total

    def print_progress_bar(done, total, start_time_):
        elapsed = time.time() - start_time_
        avg_time = elapsed / done if done else 0
        remaining = total - done
        eta = avg_time * remaining
        bar_len = 50
        filled = int(bar_len * done / total)
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
        sys.stdout.write(f"\r{bar} {done}/{total} | ETA: {eta:.1f}s | Avg: {avg_time:.2f}s/sample   ")
        sys.stdout.flush()


    print_progress_bar(len(generated_data), NUM_SAMPLES, start_time)

    while len(generated_data) < NUM_SAMPLES:
        batch_prompts, batch_labels = zip(
            *[next(prompts_cycle) for _ in range(min(BATCH_SIZE, NUM_SAMPLES - len(generated_data)))]
        )
        inputs = tokenizer(list(batch_prompts), return_tensors="pt", padding=True, truncation=True).to(DEVICE,
                                                                                                       non_blocking=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text, prompt, label in zip(decoded, batch_prompts, batch_labels):
            cleaned = clean_output(text, prompt)
            if cleaned not in seen_texts and is_valid_output(cleaned, prompt):
                generated_data.append({"label": label, "text": cleaned})
                seen_texts.add(cleaned)

        # Flush to file if we reach 90% of samples
        if len(generated_data) >= FLUSH_THRESHOLD:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, indent=2)

        print_progress_bar(len(generated_data), NUM_SAMPLES, start_time)

except Exception as e:
    print(f"\nError: {e}")
finally:
    print_progress_bar(len(generated_data), NUM_SAMPLES, start_time)
    print()
    save_and_exit()
