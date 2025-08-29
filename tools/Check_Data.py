import torch

val: int = 25000
data = torch.load(f"../cache/dataset/dataset_{val}.pt", map_location="cpu")  # use CPU to avoid GPU issues
print(type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    for k, v in data.items():
        print(f"{k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
if isinstance(data, dict):
    for k, v in data.items():
        print(f"\nPreview of {k}:")
        if isinstance(v, list):
            print(v[:5])  # first 5 items
        elif isinstance(v, torch.Tensor):
            print(v[:5])
