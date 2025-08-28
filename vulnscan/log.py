from datetime import datetime

from vulnscan.config import TrainingConfig


def log(message: str, cfg: TrainingConfig, silent: bool = False):
    if not silent:
        print(message)
    with open(cfg.LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | {message}\n")
