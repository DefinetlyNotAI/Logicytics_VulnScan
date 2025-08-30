import random
import sys

import psutil
import torch
from faker import Faker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedModel

from vulnscan.log import log
from vulnscan.config import TrainingConfig


class DataGen:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.faker = Faker()

    # ---------------- SENSITIVE DATA ----------------
    def sensitive_text(self):
        field = random.choice(self.cfg.SENSITIVE_FIELDS)
        if field == "ssn":
            return f"SSN: {self.faker.ssn()}"
        elif field == "credit_card":
            return f"Credit Card: {self.faker.credit_card_number()}"
        elif field == "email":
            return f"Email: {self.faker.email()}"
        elif field == "phone_number":
            return f"Phone: {self.faker.phone_number()}"
        elif field == "address":
            return f"Address: {self.faker.address().replace(chr(10), ', ')}"
        elif field == "name":
            return f"Name: {self.faker.name()}"
        return "Sensitive info: [REDACTED]"

    # ---------------- GPT TEXT GENERATION ----------------
    def gpt_text(self, gpt_tokenizer, gpt_model, lang: str):
        max_words: int = self.cfg.TEXT_MAX_LEN
        max_word_range: int = self.cfg.TEXT_MAX_LEN_JUMP_RANGE
        retry_limit: int = self.cfg.RETRY_LIMIT

        max_words += random.randint(-max_word_range, max_word_range)
        for _ in range(retry_limit):
            prompt = f"Write one short, simple, natural sentence in {lang} about daily life:"
            input_enc = gpt_tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = input_enc.input_ids.to(self.cfg.DEVICE)
            attention_mask = input_enc.attention_mask.to(self.cfg.DEVICE)
            with torch.no_grad():
                output_ids = gpt_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_words + len(input_ids[0]),
                    do_sample=True,
                    top_k=self.cfg.TOP_K,
                    top_p=self.cfg.TOP_P,
                    temperature=self.cfg.TEMPERATURE,
                    pad_token_id=gpt_tokenizer.pad_token_id,
                    eos_token_id=gpt_tokenizer.eos_token_id,
                    repetition_penalty=self.cfg.REP_PENALTY
                )
            text = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
            for p in ".!?":
                if p in text:
                    text = text.split(p)[0].strip()
            if len(text.split()) > 1:
                return text
        return f"A short sentence in {lang}."

    # ---------------- DATASET GENERATION ----------------
    def dataset(self, gpt_tokenizer: PreTrainedTokenizerFast, gpt_model: PreTrainedModel):
        num_samples = self.cfg.DATASET_SIZE
        dataset, labels = [], []
        log(f"Generating {num_samples} samples using GPT-Neo + self.faker...", cfg=self.cfg)
        for _ in tqdm(range(num_samples)):
            try:
                sensitive = random.random() < self.cfg.SENSITIVE_PROB
                if sensitive:
                    text = self.sensitive_text()
                else:
                    lang_choice = random.choice(self.cfg.MULTI_LANGUAGES)
                    text = self.gpt_text(gpt_tokenizer=gpt_tokenizer, gpt_model=gpt_model, lang=lang_choice)
                dataset.append(text)
                labels.append(int(sensitive))
            except KeyboardInterrupt:
                sys.exit(f"\nDataset generation interrupted by user early. Premature dataset exit.")
        torch.save({"texts": dataset, "labels": labels}, f"{self.cfg.DATASET_CACHE_DIR}/dataset_{self.cfg.DATASET_SIZE}.pt")
        return dataset, labels

    # ---------------- EMBEDDINGS ----------------
    def offload_embeddings(self, batch_embeddings: torch.Tensor, batch_labels: torch.Tensor, idx: int, split: str):
        torch.save({
            "embeddings": batch_embeddings.cpu(),
            "labels": batch_labels.cpu()
        }, f"{self.cfg.EMBED_CACHE_DIR}/{split}_{idx}.pt")

    def embeddings(self, embed_model: SentenceTransformer, texts: list[str], labels: list[int | float], split: str):
        batch_size = self.cfg.BATCH_SIZE
        batch_embeddings, batch_labels, batch_idx = [], [], 0
        for i in tqdm(range(0, len(texts), batch_size)):
            try:
                batch_texts = texts[i:i + batch_size]
                batch_lbls = labels[i:i + batch_size]

                # SentenceTransformer handles tokenization + encoding internally
                emb = embed_model.encode(sentences=batch_texts, convert_to_tensor=True, device=self.cfg.DEVICE)

                batch_embeddings.append(emb)
                batch_labels.append(torch.tensor(batch_lbls, dtype=torch.float32).unsqueeze(1))

                # Offload if memory usage too high
                if psutil.virtual_memory().percent > self.cfg.RAM_THRESHOLD:
                    self.offload_embeddings(
                        batch_embeddings=torch.cat(tensors=batch_embeddings, dim=0),
                        batch_labels=torch.cat(batch_labels, dim=0),
                        idx=batch_idx,
                        split=split
                    )
                    batch_embeddings, batch_labels = [], []
                    batch_idx += 1
            except KeyboardInterrupt:
                self.offload_embeddings(
                    batch_embeddings=torch.cat(tensors=batch_embeddings, dim=0),
                    batch_labels=torch.cat(batch_labels, dim=0),
                    idx=batch_idx,
                    split=split
                )
                sys.exit(f"Embedding generation interrupted by user early. Premature generation exit.")
        # Final offload
        if batch_embeddings:
            self.offload_embeddings(
                batch_embeddings=torch.cat(tensors=batch_embeddings, dim=0),
                batch_labels=torch.cat(batch_labels, dim=0),
                idx=batch_idx,
                split=split
            )
