# high_quality_data_gen_tqdm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import signal
import sys

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # fully offline
NUM_SAMPLES = 20000
OUTPUT_FILE = "generated_files.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2000
TEMPERATURE = 0.7

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
print("Downloading/Loading GPT-Neo 1.3B model, this may take a few minutes...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
print("Model loaded!")

# -----------------------------
# Prompts
# -----------------------------
SENSITIVE_PROMPT = (
    "Generate a realistic sensitive file snippet (fake values). "
    "Examples of sensitive content:\n"
    "- Password: Abc12345\n"
    "- API Key: sk_test_4eC39HqLyjWDarjtT1zdp7dc\n"
    "- Credit Card: 4111-1111-1111-1111\n"
    "- Email: user@example.com\n"
    "Now generate a new sensitive snippet, try to embed it in normal text or values, "
    "like a short paragraph that explains the emails password, etc:\n"
)

NON_SENSITIVE_PROMPT = (
    "Generate a harmless, non-sensitive file snippet. Examples:\n"
    "- Meeting notes: Discuss project deadlines\n"
    "- Log: System started successfully\n"
    "- Code snippet: def add(a, b): return a + b\n"
    "- Note: Remember to buy groceries\n"
    "Now generate a new non-sensitive snippet, try make it look like sensitive content, but is not:\n"
)

# -----------------------------
# Helper Function
# -----------------------------
def generate_text(prompt, max_length=MAX_LENGTH):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

# -----------------------------
# Data and Cleanup Handling
# -----------------------------
data = []

def save_and_exit():
    print(f"\nSaving {len(data)} samples to {OUTPUT_FILE} before exit...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("JSON saved. Exiting.")
    sys.exit(0)

def handle_interrupt(sig, frame):
    save_and_exit()

signal.signal(signal.SIGINT, handle_interrupt)

# -----------------------------
# Generate Dataset
# -----------------------------
with tqdm(total=NUM_SAMPLES, desc="Generating files") as pbar:
    for i in range(NUM_SAMPLES):
        prompt = SENSITIVE_PROMPT if i % 2 == 0 else NON_SENSITIVE_PROMPT
        label = 1 if i % 2 == 0 else 0

        content = generate_text(prompt)

        if not content.strip():
            pbar.update(1)
            continue  # skip empty outputs

        if label == 1 and not any(char.isdigit() for char in content):
            pbar.update(1)
            continue  # filter bad sensitive examples

        data.append({
            "filename": f"file_{i}.txt",
            "content": content,
            "label": label
        })
        pbar.update(1)

# Final save
save_and_exit()
