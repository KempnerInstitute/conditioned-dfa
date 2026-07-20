"""ImageNet-scale BP and block-DFA credit-assignment benchmarks.

This script is intentionally separate from the small manual-convnet DFA runner.
It uses standard torchvision ImageFolder data, 224 px ImageNet preprocessing,
top-1/top-5 metrics, mixed precision, and torchvision architectures. The DFA
variants replace downstream gradients at coarse feature blocks with fixed
random feedback from the output error, which is the ImageNet-scale comparison
that maps cleanly onto ResNet/AlexNet-style networks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main() -> None:
    args = parse_args()
    init_distributed(args)
    set_seed(args.seed + args.rank)
    output_dir = Path(args.output_dir)
    if args.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config(args, output_dir)

    train_loader, val_loader, n_classes = make_loaders(args)
    args.n_classes = n_classes
    model = make_model(args.arch, n_classes=n_classes, args=args).to(args.device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    controller = None
    aux_heads = None
    feedback_module_scales = parse_module_scales(args.feedback_module_scales)
    if args.method in {"block_dfa", "block_ndfa"}:
        controller = BlockDFAController(
            model,
            n_classes=n_classes,
            module_names=parse_module_names(args.feedback_modules),
            seed=args.feedback_seed,
            mode="ndfa" if args.method == "block_ndfa" else "dfa",
            feedback_scale=args.feedback_scale,
            blend_gamma=args.feedback_blend_gamma,
            module_scales=feedback_module_scales,
            spatial_mode=args.feedback_spatial_mode,
            norm_mode=args.dfa_norm,
            damping=args.natural_damping,
            whiten_mode=args.whiten_mode,
            device=args.device,
        )
    elif args.method == "local_aux":
        aux_heads = LocalAuxiliaryHeads(
            model,
            n_classes=n_classes,
            module_names=parse_module_names(args.feedback_modules),
            seed=args.feedback_seed,
            device=args.device,
        )
    elif args.method == "linear_probe":
        freeze_backbone(model)
    elif args.method != "bp":
        raise ValueError(f"Unknown method: {args.method}")

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank] if args.device.type == "cuda" else None)

    params = [p for p in model.parameters() if p.requires_grad]
    if aux_heads is not None:
        params.extend(aux_heads.parameters())
    optimizer = make_optimizer(params, args)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and args.device.type == "cuda",
        init_scale=args.amp_init_scale,
        growth_interval=args.amp_growth_interval,
    )
    criterion = MixedTargetCrossEntropy(label_smoothing=args.label_smoothing)
    ema = ModelEma(unwrap_model(model), decay=args.ema_decay, device=args.device) if args.ema_decay > 0 else None

    csv_path = output_dir / "imagenet_credit_assignment.csv"
    if args.rank == 0:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_fields())
            writer.writeheader()

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                args,
                controller=controller,
                aux_heads=aux_heads,
                ema=ema,
                epoch=epoch,
            )
        else:
            train_stats = {}
        raw_val_stats = evaluate(model, val_loader, criterion, args)
        ema_val_stats = evaluate(ema.module, val_loader, criterion, args) if ema is not None else {}
        val_stats = ema_val_stats if args.eval_ema and ema_val_stats else raw_val_stats
        controller_stats = controller.summary() if controller is not None else {}
        feedback_modules = active_feedback_modules(args, controller, aux_heads)
        row = {
            "dataset": args.dataset,
            "arch": args.arch,
            "model_source": args.model_source,
            "method": args.method,
            "epoch": epoch,
            "lr": current_lr(optimizer),
            "train_loss": train_stats.get("loss", float("nan")),
            "train_top1": train_stats.get("top1", float("nan")),
            "train_top5": train_stats.get("top5", float("nan")),
            "val_loss": val_stats["loss"],
            "val_top1": val_stats["top1"],
            "val_top5": val_stats["top5"],
            "raw_val_top1": raw_val_stats.get("top1", float("nan")),
            "raw_val_top5": raw_val_stats.get("top5", float("nan")),
            "ema_val_top1": ema_val_stats.get("top1", float("nan")),
            "ema_val_top5": ema_val_stats.get("top5", float("nan")),
            "feedback_modules": feedback_modules,
            "feedback_scale": args.feedback_scale,
            "feedback_blend_gamma": args.feedback_blend_gamma,
            "feedback_module_scales": format_module_scales(feedback_module_scales),
            "feedback_spatial_mode": args.feedback_spatial_mode,
            "natural_damping": args.natural_damping,
            "dfa_norm": args.dfa_norm,
            "aux_weight": args.aux_weight,
            "label_smoothing": args.label_smoothing,
            "randaugment": float(args.randaugment),
            "random_erasing": args.random_erasing,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "mixup_prob": args.mixup_prob,
            "ema_decay": args.ema_decay,
            "eval_ema": float(args.eval_ema),
            "repeated_aug": args.repeated_aug,
            "drop_path_rate": args.drop_path_rate,
            "seed": args.seed,
            "feedback_seed": args.feedback_seed,
            "world_size": args.world_size,
            "grad_clip_norm": args.grad_clip_norm,
            "grad_norm": train_stats.get("grad_norm", float("nan")),
            "nonfinite_steps": train_stats.get("nonfinite_steps", 0.0),
            "elapsed_sec": train_stats.get("elapsed_sec", 0.0),
            **controller_stats,
        }
        row = reduce_row(row, args)
        if args.rank == 0:
            append_row(csv_path, row)
            print(format_row(row), flush=True)
        if epoch == args.epochs:
            break

    if controller is not None:
        controller.close()
    if aux_heads is not None:
        aux_heads.close()
    cleanup_distributed(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="imagenet1k")
    parser.add_argument("--imagenet-root", default=os.environ.get("IMAGENET_ROOT", "data/imagenet"))
    parser.add_argument("--output-dir", default="results/imagenet_credit_assignment")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--model-source", choices=["auto", "torchvision", "timm"], default="auto")
    parser.add_argument("--method", choices=["bp", "block_dfa", "block_ndfa", "local_aux", "linear_probe"], default="bp")
    parser.add_argument("--feedback-modules", default="auto")
    parser.add_argument("--feedback-seed", type=int, default=0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-blend-gamma", type=float, default=1.0)
    parser.add_argument("--feedback-module-scales", default="")
    parser.add_argument(
        "--feedback-spatial-mode",
        choices=["broadcast", "activation", "bp_oracle", "bp_sign_oracle"],
        default="broadcast",
    )
    parser.add_argument("--dfa-norm", choices=["bp", "unit", "none"], default="bp")
    parser.add_argument("--natural-damping", type=float, default=0.1)
    parser.add_argument(
        "--whiten-mode",
        choices=["diag", "full"],
        default="diag",
        help="nDFA channel conditioning: 'diag' = per-channel variance (weak form); "
        "'full' = damped inverse-sqrt/ZCA of the C x C channel covariance.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize the backbone from torchvision/timm ImageNet-pretrained weights "
        "and replace the classifier head (matches the paper's fine-tuning design).",
    )
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--eval-ema", action="store_true")
    parser.add_argument("--repeated-aug", type=int, default=1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-resize", type=int, default=256)
    parser.add_argument("--subsample-train", type=int, default=0)
    parser.add_argument("--subsample-val", type=int, default=0)
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Fraction of training labels replaced by a random wrong class (deterministic given --seed).")
    parser.add_argument("--class-subset", type=int, default=0)
    parser.add_argument("--class-subset-seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-init-scale", type=float, default=65536.0)
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=100)
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> None:
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = args.world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device("cpu")
    if args.distributed:
        dist.init_process_group(backend="nccl" if args.device.type == "cuda" else "gloo")


def cleanup_distributed(args: argparse.Namespace) -> None:
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_imagenet_splits(root: Path) -> tuple[Path, Path]:
    train = next((root / n for n in ["train", "Train", "ILSVRC2012_img_train"] if (root / n).is_dir()), None)
    val = next((root / n for n in ["val", "validation", "Validation", "ILSVRC2012_img_val"] if (root / n).is_dir()), None)
    if train is None or val is None:
        raise FileNotFoundError(f"Could not find train/ and val/ ImageFolder splits under {root}")
    return train, val


def make_loaders(args: argparse.Namespace):
    root = Path(args.imagenet_root).expanduser()
    train_dir, val_dir = find_imagenet_splits(root)
    train_steps = [
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
    ]
    if args.randaugment:
        train_steps.append(transforms.RandAugment())
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if args.random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=args.random_erasing))
    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose(
        [
            transforms.Resize(args.val_resize),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val = datasets.ImageFolder(str(val_dir), transform=val_transform)
    n_classes = len(train.classes)
    if args.class_subset > 0 and args.class_subset < n_classes:
        train, val, n_classes = class_subset(train, val, args.class_subset, seed=args.class_subset_seed)
    if args.subsample_train > 0:
        train = Subset(train, balanced_indices(train, args.subsample_train, n_classes, seed=args.seed))
    if args.subsample_val > 0:
        val = Subset(val, balanced_indices(val, args.subsample_val, n_classes, seed=args.seed + 1))
    if args.label_noise > 0:
        train = NoisyLabelDataset(train, fraction=args.label_noise, n_classes=n_classes, seed=args.seed)
    if args.repeated_aug > 1:
        train = RepeatedAugDataset(train, repeats=args.repeated_aug)
    train_sampler = DistributedSampler(train, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val, shuffle=False) if args.distributed else None
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, n_classes


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base: datasets.ImageFolder, selected: Iterable[int]) -> None:
        self.base = base
        selected = list(selected)
        self.old_to_new = {old: new for new, old in enumerate(selected)}
        self.indices = [idx for idx, (_, y) in enumerate(base.samples) if int(y) in self.old_to_new]
        self.classes = [base.classes[i] for i in selected]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        x, y = self.base[self.indices[item]]
        return x, self.old_to_new[int(y)]


class RepeatedAugDataset(torch.utils.data.Dataset):
    """Repeat a dataset so each epoch sees multiple independent augmentations."""

    def __init__(self, base: torch.utils.data.Dataset, *, repeats: int) -> None:
        self.base = base
        self.repeats = max(int(repeats), 1)

    def __len__(self) -> int:
        return len(self.base) * self.repeats

    def __getitem__(self, item: int):
        return self.base[int(item) % len(self.base)]


class NoisyLabelDataset(torch.utils.data.Dataset):
    """Wrap a dataset and replace a fixed fraction of labels with random wrong labels.

    The corruption is deterministic given ``seed`` and is computed once at
    construction time so every epoch sees the same noisy labels.
    """

    def __init__(self, base: torch.utils.data.Dataset, *, fraction: float, n_classes: int, seed: int) -> None:
        self.base = base
        self.n_classes = int(n_classes)
        rng = np.random.default_rng(int(seed) + 7777)
        n = len(base)
        n_flip = int(round(float(fraction) * n))
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        self._noisy = {}
        for idx in flip_idx:
            true_label = target_at(base, int(idx))
            new = int(rng.integers(self.n_classes))
            while new == true_label and self.n_classes > 1:
                new = int(rng.integers(self.n_classes))
            self._noisy[int(idx)] = new

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        image, target = self.base[item]
        if int(item) in self._noisy:
            target = self._noisy[int(item)]
        return image, target


def class_subset(train, val, n_classes: int, *, seed: int):
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(np.arange(len(train.classes)), size=n_classes, replace=False)).tolist()
    return RemappedSubset(train, selected), RemappedSubset(val, selected), n_classes


def target_at(dataset, idx: int) -> int:
    if isinstance(dataset, Subset):
        return target_at(dataset.dataset, int(dataset.indices[idx]))
    if isinstance(dataset, RepeatedAugDataset):
        return target_at(dataset.base, int(idx) % len(dataset.base))
    if isinstance(dataset, RemappedSubset):
        old = int(dataset.base.samples[dataset.indices[idx]][1])
        return int(dataset.old_to_new[old])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[idx][1])
    _, y = dataset[idx]
    return int(y)


def balanced_indices(dataset, n_samples: int, n_classes: int, *, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    labels = np.array([target_at(dataset, i) for i in range(len(dataset))])
    per_class = max(int(math.ceil(n_samples / max(n_classes, 1))), 1)
    selected: list[int] = []
    for cls in range(n_classes):
        idx = np.flatnonzero(labels == cls)
        if idx.size == 0:
            continue
        take = min(per_class, idx.size)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(selected) > n_samples:
        selected = rng.choice(np.array(selected), size=n_samples, replace=False).tolist()
    rng.shuffle(selected)
    return [int(i) for i in selected]


def _replace_classifier(model: nn.Module, n_classes: int) -> nn.Module:
    """Swap the final classifier of a pretrained (1000-class) model for n_classes."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):  # ResNet family
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, n_classes)
        elif isinstance(clf, nn.Sequential) and isinstance(clf[-1], nn.Linear):
            clf[-1] = nn.Linear(clf[-1].in_features, n_classes)
        else:
            raise ValueError("Unsupported classifier structure for --pretrained head replacement")
    else:
        raise ValueError("Could not locate classifier head for --pretrained head replacement")
    return model


def make_model(arch: str, *, n_classes: int, args: argparse.Namespace) -> nn.Module:
    pretrained = bool(getattr(args, "pretrained", False))
    if args.model_source in {"auto", "timm"} and (args.model_source == "timm" or arch.startswith("timm:")):
        if timm is None:
            raise ImportError("timm is not installed; install timm or use --model-source torchvision")
        timm_arch = arch.removeprefix("timm:")
        # timm replaces the head for num_classes automatically, including when pretrained.
        return timm.create_model(timm_arch, pretrained=pretrained, num_classes=n_classes, drop_path_rate=args.drop_path_rate)
    if pretrained:
        # Load ImageNet-1k pretrained weights, then replace the head for n_classes.
        try:
            model = torchvision.models.get_model(arch, weights="DEFAULT")
        except Exception:
            builder = getattr(torchvision.models, arch)
            model = builder(weights="DEFAULT")
        return _replace_classifier(model, n_classes)
    try:
        return torchvision.models.get_model(arch, weights=None, num_classes=n_classes)
    except Exception:
        if args.model_source == "torchvision":
            builder = getattr(torchvision.models, arch)
            return builder(weights=None, num_classes=n_classes)
        if timm is None:
            raise
        return timm.create_model(arch, pretrained=False, num_classes=n_classes, drop_path_rate=args.drop_path_rate)


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features >= 10:
            for param in module.parameters():
                param.requires_grad = True


def parse_module_names(value: str) -> list[str]:
    if value == "auto":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_module_scales(value: str) -> dict[str, float]:
    if not value or value.strip().lower() in {"none", "auto"}:
        return {}
    scales: dict[str, float] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected module scale as name:scale, got {item!r}")
        name, raw_scale = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty module name in scale spec {item!r}")
        scales[name] = float(raw_scale)
    return scales


def format_module_scales(scales: dict[str, float]) -> str:
    if not scales:
        return ""
    return ",".join(f"{name}:{scales[name]:g}" for name in sorted(scales))


def active_feedback_modules(args: argparse.Namespace, controller, aux_heads) -> str:
    if controller is not None:
        return ",".join(controller.module_names)
    if aux_heads is not None:
        return ",".join(aux_heads.module_names)
    names = parse_module_names(args.feedback_modules)
    return ",".join(names) if names else args.feedback_modules


def named_module_dict(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def default_feedback_modules(model: nn.Module) -> list[str]:
    names = named_module_dict(model)
    if all(name in names for name in ["layer1", "layer2", "layer3", "layer4"]):
        return ["layer1", "layer2", "layer3", "layer4"]
    if "features" in names:
        return [name for name, module in names.items() if name.startswith("features.") and isinstance(module, nn.ReLU)]
    raise ValueError("Could not infer feedback modules. Pass --feedback-modules name1,name2,...")


def diagnostic_suffixes() -> list[str]:
    return [
        "cosine",
        "norm_ratio",
        "projection_ratio",
        "applied_cosine",
        "applied_norm_ratio",
        "applied_projection_ratio",
        "whiten_scale",
    ]


def dense_grad_stats(candidate_grad: torch.Tensor, bp_grad: torch.Tensor) -> tuple[float, float, float]:
    candidate = candidate_grad.float().reshape(-1)
    bp = bp_grad.float().reshape(-1)
    candidate_norm = candidate.norm().clamp_min(1e-12)
    bp_norm = bp.norm().clamp_min(1e-12)
    dot = candidate.dot(bp)
    cosine_value = float(dot.div(candidate_norm.mul(bp_norm).clamp_min(1e-12)).item())
    norm_ratio = float(candidate_norm.div(bp_norm).item())
    projection_ratio = float(dot.div(bp_norm.square().clamp_min(1e-12)).item())
    return cosine_value, norm_ratio, projection_ratio


def blended_grad_stats(
    *,
    cosine_value: float,
    norm_ratio: float,
    projection_ratio: float,
    gamma: float,
) -> tuple[float, float, float]:
    if not all(math.isfinite(v) for v in [cosine_value, norm_ratio, projection_ratio]):
        return float("nan"), float("nan"), float("nan")
    if gamma <= 0.0:
        return 1.0, 1.0, 1.0
    if gamma >= 1.0:
        return cosine_value, norm_ratio, projection_ratio
    applied_projection = (1.0 - gamma) + gamma * projection_ratio
    applied_norm_sq = (1.0 - gamma) ** 2 + (gamma * norm_ratio) ** 2 + 2.0 * gamma * (1.0 - gamma) * projection_ratio
    applied_norm = math.sqrt(max(applied_norm_sq, 1e-24))
    applied_cosine = applied_projection / max(applied_norm, 1e-12)
    return float(applied_cosine), float(applied_norm), float(applied_projection)


class BlockDFAController:
    def __init__(
        self,
        model: nn.Module,
        *,
        n_classes: int,
        module_names: list[str],
        seed: int,
        mode: str,
        feedback_scale: float,
        blend_gamma: float,
        module_scales: dict[str, float],
        spatial_mode: str,
        norm_mode: str,
        damping: float,
        device: torch.device,
        whiten_mode: str = "diag",
    ) -> None:
        self.model = model
        self.n_classes = int(n_classes)
        self.mode = mode
        self.whiten_mode = whiten_mode
        self.feedback_scale = float(feedback_scale)
        self.blend_gamma = float(blend_gamma)
        if not 0.0 <= self.blend_gamma <= 1.0:
            raise ValueError(f"--feedback-blend-gamma must be in [0, 1], got {self.blend_gamma}")
        self.module_scales = dict(module_scales)
        self.spatial_mode = spatial_mode
        self.norm_mode = norm_mode
        self.damping = float(damping)
        self.device = device
        self.output_error: torch.Tensor | None = None
        self.feedback: dict[str, torch.Tensor] = {}
        self.whiten_scales: dict[str, torch.Tensor] = {}
        self.whiten_mats: dict[str, torch.Tensor] = {}
        self.spatial_maps: dict[str, torch.Tensor] = {}
        self.handles = []
        self.stats: dict[str, list[float]] = defaultdict(list)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        modules = named_module_dict(model)
        if not module_names:
            module_names = default_feedback_modules(model)
        self.module_names = module_names
        unknown_scales = sorted(set(self.module_scales) - set(module_names))
        if unknown_scales:
            raise ValueError(f"Module scales were given for inactive modules: {unknown_scales}")
        for name in module_names:
            if name not in modules:
                raise ValueError(f"Feedback module {name!r} not found in model")
            self.handles.append(modules[name].register_forward_hook(self._make_forward_hook(name)))

    def _make_forward_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return output
            channels = int(output.shape[1]) if output.ndim == 4 else int(output.shape[-1])
            if name not in self.feedback:
                fb = torch.randn(self.n_classes, channels, generator=self.generator, device=self.device)
                fb = fb / fb.norm(dim=0, keepdim=True).clamp_min(1e-6)
                self.feedback[name] = fb
            if self.mode == "ndfa":
                if self.whiten_mode == "full":
                    self.whiten_mats[name] = self._full_whiten_mat(name, output.detach())
                else:
                    self.whiten_scales[name] = self._diag_whiten_scale(name, output.detach())
            if self.spatial_mode == "activation" and output.ndim == 4:
                self.spatial_maps[name] = self._activation_spatial_map(output.detach())
            if output.requires_grad:
                output.register_hook(lambda grad, module_name=name: self._backward_hook(module_name, grad))
            return output

        return hook

    def set_output_error(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)
            if target.ndim == 2:
                target_prob = target.detach().to(probs.device, dtype=probs.dtype)
            else:
                target_prob = torch.zeros_like(probs)
                target_prob.scatter_(1, target.detach().view(-1, 1), 1.0)
            self.output_error = (probs - target_prob) / max(int(logits.shape[0]), 1)

    def _backward_hook(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        if self.output_error is None:
            return grad
        err = self.output_error.to(device=grad.device, dtype=grad.dtype)
        fb = self.feedback[name].to(device=grad.device, dtype=grad.dtype)
        channel_grad = err @ fb
        if self.mode == "ndfa":
            if self.whiten_mode == "full":
                mat = self.whiten_mats.get(name)
                if mat is not None:
                    mat = mat.to(device=grad.device)
                    channel_grad = (channel_grad.float() @ mat).to(dtype=grad.dtype)
            else:
                scale = self.whiten_scales.get(name)
                if scale is not None:
                    channel_grad = channel_grad * scale.to(device=grad.device, dtype=grad.dtype)
        module_scale = self.feedback_scale * self.module_scales.get(name, 1.0)
        if grad.ndim == 4:
            dfa_grad, stat_cosine, stat_ratio, stat_projection = self._spatial_grad(
                name,
                channel_grad,
                grad,
                module_scale=module_scale,
            )
        else:
            dfa_grad = channel_grad.reshape_as(grad)
            dfa_grad = self._normalize(dfa_grad, grad)
            dfa_grad = dfa_grad * module_scale
            with torch.no_grad():
                stat_cosine, stat_ratio, stat_projection = dense_grad_stats(dfa_grad, grad)
        applied_cosine, applied_ratio, applied_projection = blended_grad_stats(
            cosine_value=stat_cosine,
            norm_ratio=stat_ratio,
            projection_ratio=stat_projection,
            gamma=self.blend_gamma,
        )
        self._append_stat(name, "cosine", stat_cosine)
        self._append_stat(name, "norm_ratio", stat_ratio)
        self._append_stat(name, "projection_ratio", stat_projection)
        self._append_stat(name, "applied_cosine", applied_cosine)
        self._append_stat(name, "applied_norm_ratio", applied_ratio)
        self._append_stat(name, "applied_projection_ratio", applied_projection)
        if self.blend_gamma <= 0.0:
            return grad
        if self.blend_gamma >= 1.0:
            return dfa_grad
        return grad.mul(1.0 - self.blend_gamma).add(dfa_grad, alpha=self.blend_gamma)

    def _spatial_grad(
        self,
        name: str,
        channel_grad: torch.Tensor,
        bp_grad: torch.Tensor,
        *,
        module_scale: float,
    ) -> tuple[torch.Tensor, float, float, float]:
        if self.spatial_mode == "broadcast":
            return self._spatial_broadcast_grad(channel_grad, bp_grad, module_scale=module_scale)
        if self.spatial_mode == "activation":
            spatial_map = self.spatial_maps.get(name)
            if spatial_map is None:
                spatial_map = torch.ones(
                    (bp_grad.shape[0], 1, bp_grad.shape[2], bp_grad.shape[3]),
                    device=bp_grad.device,
                    dtype=bp_grad.dtype,
                )
            spatial_map = spatial_map.to(device=bp_grad.device, dtype=bp_grad.dtype)
            base_grad = channel_grad[:, :, None, None].mul(spatial_map)
        elif self.spatial_mode == "bp_oracle":
            with torch.no_grad():
                channel_rms = bp_grad.float().square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-12)
                pattern = bp_grad.float().div(channel_rms)
            base_grad = channel_grad[:, :, None, None].mul(pattern.to(dtype=channel_grad.dtype))
        elif self.spatial_mode == "bp_sign_oracle":
            with torch.no_grad():
                channel_rms = bp_grad.float().square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-12)
                pattern = bp_grad.float().div(channel_rms)
                channel_sign = bp_grad.float().mean(dim=(2, 3), keepdim=True).sign()
                channel_sign = torch.where(channel_sign == 0, torch.ones_like(channel_sign), channel_sign)
            base_grad = channel_sign.to(dtype=channel_grad.dtype).mul(pattern.to(dtype=channel_grad.dtype))
        else:
            raise ValueError(f"Unknown feedback spatial mode: {self.spatial_mode}")
        dfa_grad = self._normalize(base_grad, bp_grad)
        dfa_grad = dfa_grad * module_scale
        with torch.no_grad():
            stat_cosine, stat_ratio, stat_projection = dense_grad_stats(dfa_grad, bp_grad)
        return dfa_grad, stat_cosine, stat_ratio, stat_projection

    def _spatial_broadcast_grad(
        self,
        channel_grad: torch.Tensor,
        bp_grad: torch.Tensor,
        *,
        module_scale: float,
    ) -> tuple[torch.Tensor, float, float, float]:
        spatial = max(int(bp_grad.shape[2] * bp_grad.shape[3]), 1)
        with torch.no_grad():
            bp_norm = bp_grad.norm().float().clamp_min(1e-12)
            base_norm = channel_grad.float().norm().div(math.sqrt(spatial)).clamp_min(1e-12)
            if self.norm_mode == "bp":
                norm_scale = bp_norm / base_norm
            elif self.norm_mode == "unit":
                norm_scale = 1.0 / base_norm
            elif self.norm_mode == "none":
                norm_scale = torch.ones((), device=channel_grad.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown DFA norm mode: {self.norm_mode}")
        scaled_channel = channel_grad.mul(norm_scale.to(channel_grad.dtype)).div(spatial)
        scaled_channel = scaled_channel * module_scale
        dfa_grad = scaled_channel[:, :, None, None].expand_as(bp_grad)
        with torch.no_grad():
            dfa_norm = scaled_channel.float().norm().mul(math.sqrt(spatial)).clamp_min(1e-12)
            bp_norm = bp_grad.norm().float().clamp_min(1e-12)
            dot = scaled_channel.float().mul(bp_grad.float().sum(dim=(2, 3))).sum()
            stat_ratio = float(dfa_norm.div(bp_norm).item())
            stat_cosine = float(dot.div(dfa_norm.mul(bp_norm).clamp_min(1e-12)).item())
            stat_projection = float(dot.div(bp_norm.square().clamp_min(1e-12)).item())
        return dfa_grad, stat_cosine, stat_ratio, stat_projection

    def _activation_spatial_map(self, activation: torch.Tensor) -> torch.Tensor:
        energy = activation.float().square().mean(dim=1, keepdim=True)
        mean_energy = energy.mean(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        return energy.div(mean_energy).detach()

    def _diag_whiten_scale(self, name: str, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim == 4:
            feat = activation.mean(dim=(2, 3))
        else:
            feat = activation
        var = feat.var(dim=0, unbiased=False).clamp_min(0.0)
        scale = torch.rsqrt(var + self.damping)
        self._append_stat(name, "whiten_scale", float(scale.mean().item()))
        return scale.detach()

    def _full_whiten_mat(self, name: str, activation: torch.Tensor) -> torch.Tensor:
        """Damped inverse-sqrt/ZCA of the C x C channel covariance.

        Reduces to the diagonal scale when the channel covariance is diagonal, but
        also removes cross-channel correlations. This is a power-1/2 diagnostic,
        not the full power-1 inverse-second-moment conditioner used by nDFA.
        """
        if activation.ndim == 4:
            feat = activation.mean(dim=(2, 3))  # (B, C) channel activity, pooled over space
        else:
            feat = activation
        feat = feat.float()
        feat = feat - feat.mean(dim=0, keepdim=True)
        n = max(int(feat.shape[0]), 1)
        cov = (feat.t() @ feat) / n  # (C, C)
        c = cov.shape[0]
        cov = cov + self.damping * torch.eye(c, device=cov.device, dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(1e-12)
        inv_sqrt = (evecs * evals.rsqrt()) @ evecs.t()  # V diag(1/sqrt(lambda)) V^T
        self._append_stat(name, "whiten_cond", float((evals.max() / evals.min()).item()))
        return inv_sqrt.detach()

    def _normalize(self, dfa_grad: torch.Tensor, bp_grad: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "none":
            return dfa_grad
        denom = dfa_grad.norm().clamp_min(1e-12)
        if self.norm_mode == "bp":
            return dfa_grad * (bp_grad.norm() / denom)
        if self.norm_mode == "unit":
            return dfa_grad / denom
        raise ValueError(f"Unknown DFA norm mode: {self.norm_mode}")

    def summary(self) -> dict[str, float]:
        values: dict[str, float] = {}
        aggregates: dict[str, list[float]] = defaultdict(list)
        for key, vals in self.stats.items():
            tail = [float(v) for v in vals[-200:] if math.isfinite(float(v))]
            if not tail:
                continue
            values[key] = float(np.mean(tail))
            for suffix in sorted(diagnostic_suffixes(), key=len, reverse=True):
                if key.endswith(f"_{suffix}"):
                    aggregates[suffix].extend(tail)
                    break
        for suffix, vals in aggregates.items():
            values[f"block_dfa_{suffix}_mean"] = float(np.mean(vals)) if vals else float("nan")
        values["feedback_module_count"] = float(len(self.module_names))
        return values

    def _append_stat(self, name: str, suffix: str, value: float) -> None:
        if math.isfinite(float(value)):
            self.stats[f"{name}_{suffix}"].append(float(value))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class LocalAuxiliaryHeads(nn.Module):
    def __init__(self, model: nn.Module, *, n_classes: int, module_names: list[str], seed: int, device: torch.device) -> None:
        super().__init__()
        self.features: dict[str, torch.Tensor] = {}
        self.handles = []
        modules = named_module_dict(model)
        if not module_names:
            module_names = default_feedback_modules(model)
        self.module_names = module_names
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        heads = {}
        for name in module_names:
            if name not in modules:
                raise ValueError(f"Auxiliary module {name!r} not found in model")
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))
            heads[name.replace(".", "_")] = nn.LazyLinear(n_classes, device=device)
        self.heads = nn.ModuleDict(heads).to(device)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.features[name] = output
            return output

        return hook

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for name, feature in self.features.items():
            key = name.replace(".", "_")
            pooled = feature.mean(dim=(2, 3)) if feature.ndim == 4 else feature
            losses.append(F.cross_entropy(self.heads[key](pooled), target))
        if not losses:
            return torch.zeros((), device=target.device)
        return torch.stack(losses).mean()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class MixedTargetCrossEntropy(nn.Module):
    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            return F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        target = target.to(logits.device, dtype=logits.dtype)
        if self.label_smoothing > 0:
            target = target * (1.0 - self.label_smoothing) + self.label_smoothing / target.shape[1]
        log_prob = F.log_softmax(logits, dim=1)
        return -(target * log_prob).sum(dim=1).mean()


class ModelEma:
    def __init__(self, model: nn.Module, *, decay: float, device: torch.device) -> None:
        self.module = copy.deepcopy(model).to(device).eval()
        self.decay = float(decay)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = unwrap_model(model).state_dict()
        target = self.module.state_dict()
        for key, value in target.items():
            src = source[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(src.to(value.device), alpha=1.0 - self.decay)
            else:
                value.copy_(src.to(value.device))


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def make_optimizer(params, args: argparse.Namespace):
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def adjust_lr(optimizer, epoch: int, step: int, n_steps: int, args: argparse.Namespace) -> None:
    progress = (epoch - 1) + step / max(n_steps, 1)
    if progress < args.warmup_epochs:
        lr = args.lr * (progress + 1.0 / max(n_steps, 1)) / max(args.warmup_epochs, 1)
    else:
        denom = max(args.epochs - args.warmup_epochs, 1)
        cosine_progress = min(max((progress - args.warmup_epochs) / denom, 0.0), 1.0)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * cosine_progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def all_ranks_true(value: bool, args: argparse.Namespace) -> bool:
    if not args.distributed:
        return bool(value)
    tensor = torch.tensor([1 if value else 0], device=args.device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def train_one_epoch(model, loader, criterion, optimizer, scaler, args, *, controller, aux_heads, ema, epoch: int):
    model.train()
    if aux_heads is not None:
        aux_heads.train()
    meters = defaultdict(float)
    n_seen = 0
    grad_norm_total = 0.0
    grad_norm_count = 0
    nonfinite_steps = 0
    clip_params = [p for p in model.parameters() if p.requires_grad]
    if aux_heads is not None:
        clip_params.extend([p for p in aux_heads.parameters() if p.requires_grad])
    start = time.time()
    for step, (images, target) in enumerate(loader):
        adjust_lr(optimizer, epoch, step, len(loader), args)
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        hard_target = target
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        images, target = apply_mixup_cutmix(images, target, args)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp and args.device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, target)
            if aux_heads is not None:
                loss = loss + args.aux_weight * aux_heads.loss(target)
        loss_is_finite = bool(torch.isfinite(loss.detach()).all().item())
        if not all_ranks_true(loss_is_finite, args):
            nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            message = f"non-finite loss at epoch={epoch} step={step}"
            if args.rank == 0:
                print(message, flush=True)
            if args.stop_on_nonfinite:
                raise FloatingPointError(message)
            continue
        if controller is not None:
            controller.set_output_error(logits, target)
        scaler.scale(loss).backward()
        grad_norm_value = float("nan")
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip_norm, error_if_nonfinite=False)
            grad_is_finite = bool(torch.isfinite(grad_norm.detach()).all().item())
            if not all_ranks_true(grad_is_finite, args):
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                message = f"non-finite gradient norm at epoch={epoch} step={step}"
                if args.rank == 0:
                    print(message, flush=True)
                if args.stop_on_nonfinite:
                    raise FloatingPointError(message)
                continue
            grad_norm_value = float(grad_norm.detach().float().item())
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        batch = int(hard_target.numel())
        top1, top5 = accuracy(logits.detach(), hard_target, topk=(1, min(5, args.n_classes)))
        meters["loss"] += float(loss.detach().item()) * batch
        meters["top1"] += top1 * batch
        meters["top5"] += top5 * batch
        if math.isfinite(grad_norm_value):
            grad_norm_total += grad_norm_value
            grad_norm_count += 1
        n_seen += batch
        if args.rank == 0 and args.print_freq > 0 and step % args.print_freq == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f} top1={top1:.2f}", flush=True)
    stats = {key: value / max(n_seen, 1) for key, value in meters.items()}
    stats["grad_norm"] = grad_norm_total / max(grad_norm_count, 1)
    stats["nonfinite_steps"] = float(nonfinite_steps)
    stats["elapsed_sec"] = time.time() - start
    return reduce_stats(stats, args)


def apply_mixup_cutmix(images: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if args.mixup_alpha <= 0 and args.cutmix_alpha <= 0:
        return images, target
    if torch.rand((), device=images.device).item() > args.mixup_prob:
        return images, target
    use_cutmix = args.cutmix_alpha > 0 and (args.mixup_alpha <= 0 or torch.rand((), device=images.device).item() < 0.5)
    alpha = args.cutmix_alpha if use_cutmix else args.mixup_alpha
    lam = sample_beta(alpha, device=images.device)
    perm = torch.randperm(images.shape[0], device=images.device)
    target_prob = one_hot(target, args.n_classes).to(images.dtype)
    if use_cutmix:
        x1, y1, x2, y2 = rand_bbox(images.shape[-2], images.shape[-1], lam, device=images.device)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        box_area = max((x2 - x1) * (y2 - y1), 0)
        lam = 1.0 - box_area / max(images.shape[-1] * images.shape[-2], 1)
    else:
        images = images * lam + images[perm] * (1.0 - lam)
    mixed = target_prob * lam + target_prob[perm] * (1.0 - lam)
    return images, mixed


def one_hot(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    out = torch.zeros((target.shape[0], n_classes), device=target.device)
    out.scatter_(1, target.view(-1, 1), 1.0)
    return out


def sample_beta(alpha: float, *, device: torch.device) -> float:
    dist_beta = torch.distributions.Beta(torch.tensor([alpha], device=device), torch.tensor([alpha], device=device))
    return float(dist_beta.sample().item())


def rand_bbox(height: int, width: int, lam: float, *, device: torch.device) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(1.0 - lam, 0.0))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = int(torch.randint(width, (1,), device=device).item())
    cy = int(torch.randint(height, (1,), device=device).item())
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


@torch.no_grad()
def evaluate(model, loader, criterion, args):
    model.eval()
    meters = defaultdict(float)
    n_seen = 0
    for images, target in loader:
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=args.amp and args.device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, target)
        batch = int(target.numel())
        top1, top5 = accuracy(logits, target, topk=(1, min(5, args.n_classes)))
        meters["loss"] += float(loss.item()) * batch
        meters["top1"] += top1 * batch
        meters["top5"] += top5 * batch
        n_seen += batch
    stats = {key: value / max(n_seen, 1) for key, value in meters.items()}
    return reduce_stats(stats, args)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out.append(float(correct_k.mul_(100.0 / target.numel()).item()))
    return out


def cosine(x: torch.Tensor, y: torch.Tensor, *, max_elements: int = 262_144) -> float:
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    if x_flat.numel() == 0 or y_flat.numel() == 0:
        return float("nan")
    if x_flat.numel() > max_elements:
        stride = math.ceil(x_flat.numel() / max_elements)
        x_flat = x_flat[::stride][:max_elements]
        y_flat = y_flat[::stride][:max_elements]
    x_f = x_flat.float()
    y_f = y_flat.float()
    denom = x_f.norm().mul(y_f.norm()).clamp_min(1e-12)
    return float(x_f.dot(y_f).div(denom).item())


def reduce_stats(stats: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    if not args.distributed:
        return stats
    keys = sorted(stats)
    tensor = torch.tensor([stats[k] for k in keys], device=args.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= args.world_size
    return {k: float(v.item()) for k, v in zip(keys, tensor)}


def diagnostic_result_fields() -> list[str]:
    fields = [f"block_dfa_{suffix}_mean" for suffix in diagnostic_suffixes()]
    for module in ["layer1", "layer2", "layer3", "layer4"]:
        fields.extend(f"{module}_{suffix}" for suffix in diagnostic_suffixes())
    return fields


def reduce_row(row: dict[str, float | str], args: argparse.Namespace) -> dict[str, float | str]:
    if not args.distributed:
        return row
    # Train/eval stats are already reduced with fixed keys inside train_one_epoch()
    # and evaluate(). Only reduce controller diagnostics here, and do it with a
    # fixed tensor shape so non-finite values on one rank cannot desynchronize NCCL.
    diagnostic_keys = diagnostic_result_fields()
    values = []
    finite = []
    for key in diagnostic_keys:
        value = row.get(key, float("nan"))
        is_finite = isinstance(value, (float, int)) and math.isfinite(float(value))
        values.append(float(value) if is_finite else 0.0)
        finite.append(1.0 if is_finite else 0.0)
    tensor = torch.tensor([values, finite], device=args.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    for index, key in enumerate(diagnostic_keys):
        count = float(tensor[1, index].item())
        row[key] = float(tensor[0, index].item() / count) if count == args.world_size else float("nan")
    return row


def result_fields() -> list[str]:
    return [
        "dataset",
        "arch",
        "model_source",
        "method",
        "epoch",
        "lr",
        "train_loss",
        "train_top1",
        "train_top5",
        "val_loss",
        "val_top1",
        "val_top5",
        "raw_val_top1",
        "raw_val_top5",
        "ema_val_top1",
        "ema_val_top5",
        "feedback_modules",
        "feedback_scale",
        "feedback_blend_gamma",
        "feedback_module_scales",
        "feedback_spatial_mode",
        "natural_damping",
        "dfa_norm",
        "aux_weight",
        "label_smoothing",
        "randaugment",
        "random_erasing",
        "mixup_alpha",
        "cutmix_alpha",
        "mixup_prob",
        "ema_decay",
        "eval_ema",
        "repeated_aug",
        "drop_path_rate",
        "seed",
        "feedback_seed",
        "world_size",
        "grad_clip_norm",
        "grad_norm",
        "nonfinite_steps",
        "elapsed_sec",
        "feedback_module_count",
        *diagnostic_result_fields(),
    ]


def append_row(path: Path, row: dict[str, float | str]) -> None:
    fields = result_fields()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writerow(row)


def format_row(row: dict[str, float | str]) -> str:
    return (
        f"epoch={int(row['epoch'])} method={row['method']} arch={row['arch']} "
        f"train_top1={float(row['train_top1']):.2f} val_top1={float(row['val_top1']):.2f} "
        f"val_top5={float(row['val_top5']):.2f} lr={float(row['lr']):.5g}"
    )


def write_config(args: argparse.Namespace, output_dir: Path) -> None:
    with (output_dir / "config.txt").open("w") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
