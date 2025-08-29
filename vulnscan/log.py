from datetime import datetime

from vulnscan.config import TrainingConfig


def log(message: str, cfg: TrainingConfig, silent: bool = False, only_console: bool = False):
    """Log a message to console and log file."""
    if only_console and silent:
        return  # nothing to do
    if not silent:
        print(message)
    if not only_console:
        with open(cfg.LOG_FILE, "a") as f:
            f.write(f"{datetime.now()} | {message}\n")
