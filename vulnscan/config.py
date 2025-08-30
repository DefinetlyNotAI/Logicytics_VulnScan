import os
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingConfig:
    def __init__(self):
        """
        Configuration class for training settings and hyperparameters.

        You must call the update method and set MODEL_NAME.
        """

        # Model / caching / logging
        self.MODEL_NAME = None
        self.writer = None
        self.LOG_FILE = None
        self.EMBED_CACHE_DIR = None

        self.CACHE_DIR = os.path.join(os.getcwd(), "cache")
        self.DATASET_CACHE_DIR = f"{self.CACHE_DIR}/dataset"

        existing_rounds = self.__get_existing_rounds(self.CACHE_DIR)  # Auto-increment round based on existing folders
        self.MODEL_ROUND = max(existing_rounds) + 1 if existing_rounds else 1

        # Training parameters
        self.BATCH_SIZE: int = 16
        self.MAX_EPOCHS: int = 35
        self.TRAIN_LOOPS: int = 3
        self.EARLY_STOPPING_PATIENCE: int = 5
        self.LR: float = 1e-3
        self.LR_JUMP: dict[str, int] = {"MAX": 5, "MIN": 0.1}
        self.COUNTER: dict[str, int] = {"PATIENCE": 0, "JUMP": 0}
        self.JUMP_PATIENCE: int = 3
        self.LR_DECAY: float = 0.9
        self.BEST_VAL_LOSS: float = float("inf")
        self.AUTO_CONTINUE: bool = False

        # Dataset / data generation
        self.DATASET_SIZE: int = 1000
        self.TEXT_MAX_LEN: int = 128
        self.TEXT_MAX_LEN_JUMP_RANGE: int = 10
        self.VAL_SPLIT: float = 0.85
        self.TRAIN_VAL_SPLIT: float = 0.7
        self.SENSITIVE_PROB: float = 0.3
        self.SENSITIVE_FIELDS: list[str] = ["ssn", "credit_card", "email", "phone_number", "address", "name"]

        # Language / generation
        self.MULTI_LANGUAGES: list[str] = ["english", "spanish", "french", "dutch", "arabic", "japanese"]
        self.TOP_K: int = 30
        self.TOP_P: float = 0.9
        self.TEMPERATURE: float = 0.9
        self.REP_PENALTY: float = 1.2
        self.RETRY_LIMIT: int = 3

        # Device / system
        self.DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.RAM_THRESHOLD: float = 0.85

    @staticmethod
    def __get_existing_rounds(cache_dir: str) -> list[int]:
        """
        Returns a list of round numbers based on existing folders in the cache directory.
        """
        rounds = []
        for f in os.listdir(os.path.join(os.path.dirname(__file__), cache_dir)):
            if f.startswith('round_'):
                round_str = f.split('_')[-1].split('-')[0]
                if round_str.isdigit():
                    rounds.append(int(round_str))
        return rounds

    def update(self, updates):
        """
        Update any configuration variable dynamically.
        Accepts a dict or a list of (key, value) pairs.
        Also updates dependent variables if their base values change.
        Example:
            cfg.update({'BATCH_SIZE': 32, 'LR': 5e-4})
            cfg.update([('BATCH_SIZE', 32), ('LR', 5e-4)])
        """
        if isinstance(updates, dict):
            items = updates.items()
        elif isinstance(updates, list):
            items = updates
        else:
            raise TypeError("updates must be a dict or a list of (key, value) pairs")
        for key, value in items:
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"TrainingConfig has no attribute '{key}'")
        # Update dependent variables if their base values changed
        if 'MODEL_NAME' in dict(items) or 'CACHE_DIR' in dict(items) or 'MODEL_ROUND' in dict(items):
            self.LOG_FILE = f"{self.CACHE_DIR}/{self.MODEL_NAME}/training.log"
            self.EMBED_CACHE_DIR = f"{self.CACHE_DIR}/{self.MODEL_NAME}/round_{self.MODEL_ROUND}/embeddings"
            self.writer = SummaryWriter(
                log_dir=f"{self.CACHE_DIR}/{self.MODEL_NAME}/round_{self.MODEL_ROUND}/tensorboard_logs")
            os.makedirs(self.EMBED_CACHE_DIR, exist_ok=True)
        if 'DATASET_CACHE_DIR' in dict(items):
            self.DATASET_CACHE_DIR = f"{self.CACHE_DIR}/dataset"
            os.makedirs(self.DATASET_CACHE_DIR, exist_ok=True)
