print("[INIT] Importing Libraries...")

import torch
from tabulate import tabulate

# ------------------- CONFIG -------------------
print("[INIT] Setting Config...")
GEN_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
CLS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE_GEN = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMB = "cpu"  # use CPU for embeddings to save VRAM

OUTPUT_FILE = "generated_files.json"
OFFLOAD_FILE = "offloaded_data.json"

NUM_GENERATED = 100
MAX_DATASET_SIZE = 2000
RELOAD_THRESHOLD = 500
MAX_LENGTH = 100
TEMPERATURE = 0.9
BATCH_SIZE = 8
EPOCHS = 30
MAX_TRAINS = 500

# Naming, check README.md for details
TypeOfModel: str = "Sense"
VersionNum: int = 4
RepeatNum: int = 2
ModelID: str = "n"


def naming_convention() -> str:
    Version = f"{VersionNum}{ModelID}{RepeatNum}"
    ModelName = f"Model_{TypeOfModel}.{Version}"

    AllowedModelTypes = ["SenseNano", "SenseMini", "Sense", "SenseMacro"]
    AllowedModelIDs = ["b", "dt", "et", "g", "l", "n", "nb", "r", "lr", "v", "x"]
    if TypeOfModel not in AllowedModelTypes:
        raise ValueError(f"TypeOfModel must be one of [{AllowedModelTypes}]")
    if ModelID not in AllowedModelIDs:
        raise ValueError(f"ModelID must be one of [{AllowedModelIDs}]")

    return ModelName


MODEL_NAME = naming_convention()

config_table = [
    ["Model PTH Name", MODEL_NAME],
    ["Generator Model", GEN_MODEL_NAME],
    ["Classifier Model", CLS_MODEL_NAME],
    ["Generation Device", DEVICE_GEN],
    ["Embedding Device", DEVICE_EMB],
    ["Output File", OUTPUT_FILE],
    ["Offload File", OFFLOAD_FILE],
    ["Num Generated per Cycle", NUM_GENERATED],
    ["Max Dataset Size", MAX_DATASET_SIZE],
    ["Reload Threshold", RELOAD_THRESHOLD],
    ["Max Length", MAX_LENGTH],
    ["Temperature", TEMPERATURE],
    ["Batch Size", BATCH_SIZE],
    ["Epochs per Train", EPOCHS],
    ["Max Training Cycles", MAX_TRAINS if MAX_TRAINS != -1 else "Infinite"]
]

print("[INIT] Configuration:")
print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))


def main():
    ...


if __name__ == "__main__":
    main()
