import random

import psutil
import torch
from faker import Faker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedModel

from vulnscan.log import log
from vulnscan.config import cfg


# ---------------- SENSITIVE DATA ----------------
def generate_sensitive_text():
    field = random.choice(cfg.SENSITIVE_FIELDS)
    if field == "ssn":
        return f"SSN: {faker.ssn()}"
    elif field == "credit_card":
        return f"Credit Card: {faker.credit_card_number()}"
    elif field == "email":
        return f"Email: {faker.email()}"
    elif field == "phone_number":
        return f"Phone: {faker.phone_number()}"
    elif field == "address":
        return f"Address: {faker.address().replace(chr(10), ', ')}"
    elif field == "name":
        return f"Name: {faker.name()}"
    return "Sensitive info: [REDACTED]"


# ---------------- GPT TEXT GENERATION ----------------
def generate_gpt_text(gpt_tokenizer, gpt_model, lang: str, max_words: int = cfg.TEXT_MAX_LEN,
                      max_word_range: int = cfg.TEXT_MAX_LEN_JUMP_RANGE,
                      retry_limit: int = cfg.RETRY_LIMIT):
    max_words += random.randint(-max_word_range, max_word_range)
    for _ in range(retry_limit):
        prompt = f"Write one short, simple, natural sentence in {lang} about daily life:"
        input_enc = gpt_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = input_enc.input_ids.to(cfg.DEVICE)
        attention_mask = input_enc.attention_mask.to(cfg.DEVICE)
        with torch.no_grad():
            output_ids = gpt_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_words + len(input_ids[0]),
                do_sample=True,
                top_k=cfg.TOP_K,
                top_p=cfg.TOP_P,
                temperature=cfg.TEMPERATURE,
                pad_token_id=gpt_tokenizer.pad_token_id,
                eos_token_id=gpt_tokenizer.eos_token_id,
                repetition_penalty=cfg.REP_PENALTY
            )
        text = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        for p in ".!?":
            if p in text:
                text = text.split(p)[0].strip()
        if len(text.split()) > 1:
            return text
    return f"A short sentence in {lang}."


# ---------------- DATASET GENERATION ----------------
def generate_dataset(gpt_tokenizer: PreTrainedTokenizerFast, gpt_model: PreTrainedModel,
                     num_samples: int = cfg.DATASET_SIZE):
    dataset, labels = [], []
    log(f"Generating {num_samples} samples using GPT-Neo + Faker...")
    for _ in tqdm(range(num_samples)):
        try:
            sensitive = random.random() < cfg.SENSITIVE_PROB
            text = generate_sensitive_text() if sensitive else generate_gpt_text(gpt_tokenizer=gpt_tokenizer,
                                                                                 gpt_model=gpt_model,
                                                                                 lang=random.choice(cfg.MULTI_LANGUAGES))
            dataset.append(text)
            labels.append(int(sensitive))
        except KeyboardInterrupt:
            log(f"Dataset generation interrupted by user early. Premature dataset exit.")
            break
    return dataset, labels


# ---------------- EMBEDDINGS ----------------
def offload_embeddings(batch_embeddings: torch.Tensor, batch_labels: torch.Tensor, idx: int):
    path = f"{cfg.EMBED_CACHE_DIR}/batch_{idx}.pt"
    torch.save({'embeddings': batch_embeddings.cpu(), 'labels': batch_labels.cpu()}, path)


def generate_embeddings(embed_model: SentenceTransformer, texts: list[str], labels: list[int | float],
                        batch_size: int = cfg.BATCH_SIZE):
    batch_embeddings, batch_labels, batch_idx = [], [], 0
    for i in tqdm(range(0, len(texts), batch_size)):
        try:
            batch_texts = texts[i:i + batch_size]
            batch_lbls = labels[i:i + batch_size]

            # SentenceTransformer handles tokenization + encoding internally
            emb = embed_model.encode(sentences=batch_texts, convert_to_tensor=True, device=cfg.DEVICE)

            batch_embeddings.append(emb)
            batch_labels.append(torch.tensor(batch_lbls, dtype=torch.float32).unsqueeze(1))

            # Offload if memory usage too high
            if psutil.virtual_memory().percent / 100 > cfg.RAM_THRESHOLD:
                offload_embeddings(batch_embeddings=torch.cat(tensors=batch_embeddings, dim=0),
                                   batch_labels=torch.cat(tensors=batch_labels, dim=0), idx=batch_idx)
                batch_embeddings, batch_labels = [], []
                batch_idx += 1
        except KeyboardInterrupt:
            log(f"Embedding generation interrupted by user early. Premature generation exit, continuing.")
            break
    # Final offload
    if batch_embeddings:
        offload_embeddings(batch_embeddings=torch.cat(tensors=batch_embeddings, dim=0),
                           batch_labels=torch.cat(batch_labels, dim=0), idx=batch_idx)


faker = Faker()
