"""Evaluate torchvision pretrained ImageNet weights on the local validation set."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in args.archs:
        weights_enum = torchvision.models.get_model_weights(arch)
        weights = weights_enum.DEFAULT if args.weights == "DEFAULT" else getattr(weights_enum, args.weights)
        model = torchvision.models.get_model(arch, weights=weights).to(device).eval()
        transform = weights.transforms()
        val_dir = find_val_dir(Path(args.imagenet_root).expanduser())
        val = datasets.ImageFolder(str(val_dir), transform=transform)
        if args.subsample_val > 0:
            val = Subset(val, list(range(min(args.subsample_val, len(val)))))
        loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        top1, top5, n = evaluate(model, loader, device=device)
        meta = getattr(weights, "meta", {})
        rows.append(
            {
                "arch": arch,
                "weights": str(weights).split(".")[-1],
                "local_top1": top1,
                "local_top5": top5,
                "n": n,
                "reference_top1": meta.get("acc@1", float("nan")),
                "reference_top5": meta.get("acc@5", float("nan")),
                "recipe": meta.get("recipe", ""),
            }
        )
        print(f"{arch} {weights}: local top1={top1:.3f} top5={top5:.3f} n={n}", flush=True)
    path = output_dir / "torchvision_weight_validation.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", default=os.environ.get("IMAGENET_ROOT", "data/imagenet"))
    parser.add_argument("--output-dir", default="results/imagenet_torchvision_weight_validation")
    parser.add_argument("--archs", nargs="+", default=["alexnet", "resnet18", "resnet50", "convnext_tiny"])
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--subsample-val", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def find_val_dir(root: Path) -> Path:
    for name in ["val", "validation", "Validation", "ILSVRC2012_img_val"]:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find ImageNet val split under {root}")


@torch.no_grad()
def evaluate(model, loader, *, device: torch.device) -> tuple[float, float, int]:
    correct1 = 0.0
    correct5 = 0.0
    n_seen = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        maxk = min(5, logits.shape[1])
        _, pred = logits.topk(maxk, dim=1)
        correct = pred.eq(target[:, None])
        correct1 += float(correct[:, :1].sum().item())
        correct5 += float(correct[:, :maxk].sum().item())
        n_seen += int(target.numel())
    return 100.0 * correct1 / max(n_seen, 1), 100.0 * correct5 / max(n_seen, 1), n_seen


if __name__ == "__main__":
    main()
