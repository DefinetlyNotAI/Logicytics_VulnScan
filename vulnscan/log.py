from datetime import datetime

from vulnscan.config import cfg


def log(message: str):
    print(message)
    with open(cfg.LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | {message}\n")
