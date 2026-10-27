"""
Shared setup for the examples: data, model and command line.

The examples use Imagenette (Howard, 2019): ten easily told apart ImageNet
classes at 160 px, about 100 MB. Labels are mapped to their ImageNet indices,
so the logits of any ImageNet-pretrained timm model line up with them.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import (
    DataLoader,
    Subset
)
from torchvision.datasets import ImageFolder

from innards import load_model, style

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# ImageFolder sorts the WordNet ids, so dataset label i is ImageNet class IMAGENET_IDX[i]
IMAGENET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
NAMES = ["tench", "springer", "cassette player", "chain saw", "church", "French horn", "garbage truck",
         "gas pump", "golf ball", "parachute"]
NAME = dict(zip(IMAGENET_IDX, NAMES))


def parse(description: str, model: str, n: int = 500) -> argparse.Namespace:
    """Command-line arguments shared by all examples."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=model, help="timm model name")
    p.add_argument("--n", type=int, default=n, help="number of validation images (class-balanced)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data", default="data", help="where Imagenette is (or gets downloaded to)")
    p.add_argument("--out", default="figures", help="output directory for figures")
    p.add_argument("--format", default="pdf", help="figure format: pdf, svg or png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--random-weights", action="store_true", help="skip pretrained weights (for a quick smoke test)")
    args = p.parse_args()
    torch.manual_seed(0)
    style.use()
    return args


def imagenette(root: str | Path) -> Path:
    """Downloads and extracts Imagenette (160 px) once; returns its directory."""
    root = Path(root)
    folder = root / "imagenette2-160"
    if not folder.exists():
        archive = root / "imagenette2-160.tgz"
        if not archive.exists():
            root.mkdir(parents=True, exist_ok=True)
            print(f"downloading {URL} ...")
            urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
    return folder


def setup(args: argparse.Namespace):
    """
    The model (wrapped for nnsight), a class-balanced validation set and its loader.

    Returns:
        ``(model, dataset, loader)``; batches are ``(images, imagenet_labels)``.
    """
    model, preprocess = load_model(args.model, pretrained=not args.random_weights, device=args.device)
    data = ImageFolder(imagenette(args.data) / "val", preprocess, target_transform=lambda t: IMAGENET_IDX[t])
    targets = torch.tensor(data.targets)
    per_class = max(1, args.n // len(NAMES))
    idx = torch.cat([(targets == c).nonzero().squeeze(1)[:per_class] for c in range(len(NAMES))])
    subset = Subset(data, idx.tolist())
    return model, subset, DataLoader(subset, batch_size=args.batch_size, num_workers=2)


def normalisation(model) -> dict:
    """The model's input normalisation, to show its inputs as images (``plot.images(..., **norm)``)."""
    from innards import unwrap

    cfg = unwrap(model).pretrained_cfg
    return {"mean": cfg["mean"], "std": cfg["std"]}


def stack(dataset, idx) -> tuple[torch.Tensor, torch.Tensor]:
    """Images and labels of some dataset entries, as tensors."""
    items = [dataset[int(i)] for i in idx]
    return torch.stack([x for x, _ in items]), torch.tensor([y for _, y in items])


def dense(labels: torch.Tensor) -> torch.Tensor:
    """ImageNet labels -> 0..9."""
    return torch.tensor([IMAGENET_IDX.index(int(y)) for y in labels])


def figure_path(args: argparse.Namespace, name: str) -> Path:
    return Path(args.out) / f"{name}.{args.format}"
