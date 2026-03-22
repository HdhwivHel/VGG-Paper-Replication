from pathlib import Path
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_pipeline import get_datasets
from models.vggnet import VGG_16
import yaml

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "VGG_16.pth"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

batch_size = config["testing"]["batch_size"]
workers = config["test_dataloader"]["num_workers"]


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataset = get_datasets()

    # Keep DataLoader stable on Windows with CUDA + multiple workers.
    use_pin_memory = device.type == "cuda" and not (os.name == "nt" and workers > 0)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Run train.py first to generate it."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_pin_memory,
        persistent_workers=workers > 0,
    )

    model = VGG_16().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    with torch.inference_mode():
        correct = 0
        total = 0

        for X, y in tqdm(test_loader):
            X = X.to(device, non_blocking=use_pin_memory)
            y = y.to(device, non_blocking=use_pin_memory)
            outputs = model(X)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    test()
