"""Exploratory, validation-only test of paired-view nuisance conditioning.

Every method receives the same two views and optimizes their mean loss. Only
the local activity metric changes. No test split is loaded or evaluated.
Synthetic views explicitly preserve latent task coordinates; CIFAR crops and
flips require the usual task-preservation assumption. This is not confirmation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.dfa import Gradients, ManualMLP, init_feedback
from infogeo.local_preconditioning import condition_local_update
from infogeo.paired_conditioning import (
    paired_activity_update, right_condition_from_samples, view_regularized_activity_update,
)


DEFAULT_METHODS = ("bp", "dfa", "ndfa", "centered_total", "paired", "wrong_pairs", "diagonal")
METHODS = DEFAULT_METHODS + ("view_regularized", "wrong_view_regularized")


@dataclass
class Data:
    train: torch.Tensor
    labels: torch.Tensor
    validation: torch.Tensor
    validation_labels: torch.Tensor
    input_dim: int
    classes: int
    task_indices: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    channel_mean: torch.Tensor | None = None
    channel_std: torch.Tensor | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "cifar10", "cifar100"], default="synthetic")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "torchvision"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--regime", choices=["task_high_variance", "task_low_variance"], default="task_high_variance")
    parser.add_argument("--task-view-noise", type=float, default=0.)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-seed", type=int, default=9173)
    parser.add_argument("--split-seed", type=int, default=80423)
    parser.add_argument("--feedback-seed", type=int, default=2901)
    parser.add_argument("--feedback-scale", type=float, default=1.)
    parser.add_argument("--standardize-inputs", action="store_true",
                        help="Vision only: per-channel training-pool mean/std, shared by train views and validation")
    parser.add_argument("--batchnorm", action="store_true", help="Apply BatchNorm to every method's hidden layers")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=1024, help="0 uses the full training pool")
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="Identities; each update uses twice this many views")
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--relative-damping", type=float, default=.3)
    parser.add_argument("--damping-floor", type=float, default=1e-6)
    parser.add_argument("--view-penalty", type=float, default=1.,
                        help="Additive view-regularization strength; zero is ordinary nDFA")
    parser.add_argument("--backend", choices=["auto", "feature", "sample"], default="auto")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    for name in ["batch_size", "validation_size", "steps", "eval_every", "threads"]:
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.batch_size < 2 or args.input_dim < 8 or args.n_train < 0:
        parser.error("require batch_size >= 2, input_dim >= 8, n_train >= 0")
    if args.dataset == "synthetic" and args.n_train == 0:
        parser.error("synthetic requires an explicit positive n_train")
    if args.dataset == "synthetic" and args.standardize_inputs:
        parser.error("standardize-inputs is defined only for vision datasets")
    if not math.isfinite(args.feedback_scale) or args.feedback_scale <= 0:
        parser.error("feedback-scale must be positive and finite")
    if not math.isfinite(args.view_penalty) or args.view_penalty < 0:
        parser.error("view-penalty must be finite and nonnegative")
    scales = [args.lr, args.relative_damping, args.damping_floor, args.task_view_noise]
    if not all(math.isfinite(v) for v in scales) or min(scales[:3]) <= 0 or args.task_view_noise < 0:
        parser.error("lr and damping must be positive; task-view-noise nonnegative")
    return args


def load_data(args) -> Data:
    generator = torch.Generator().manual_seed(args.data_seed)
    if args.dataset == "synthetic":
        dim = args.input_dim
        task = torch.arange(4) if args.regime == "task_high_variance" else torch.arange(dim - 4, dim)
        scales = torch.ones(dim)
        scales[:dim // 2] = 3.  # Same complete population spectrum in both regimes.
        rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
        train = torch.randn(args.n_train, dim, generator=generator)
        validation_latent = torch.randn(args.validation_size, dim, generator=generator)
        data = Data(train, train[:, task].argmax(1), validation_latent,
                    validation_latent[:, task].argmax(1), dim, 4, task, scales, rotation)
        data.validation = synthetic_view(validation_latent, data, generator, args.task_view_noise)
        return data
    from torchvision import datasets
    classes = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
    dataset = classes[args.dataset](root=args.data_dir, train=True, download=args.download)
    images = torch.as_tensor(dataset.data)
    if args.dataset == "mnist":
        images = images[:, None, :, :]
    else:
        images = images.permute(0, 3, 1, 2)
    labels = torch.as_tensor(dataset.targets, dtype=torch.long)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(args.split_seed))
    if args.validation_size >= len(order):
        raise ValueError("validation must leave at least one training example")
    val_idx, train_idx = order[:args.validation_size], order[args.validation_size:]
    if args.n_train:
        train_idx = train_idx[:args.n_train]
    data = Data(images[train_idx], labels[train_idx], images[val_idx], labels[val_idx],
                images[0].numel(), 100 if args.dataset == "cifar100" else 10)
    if args.standardize_inputs:
        # Compute population moments only after both the validation split and
        # optional training subset selection. Chunking bounds the float64 copy.
        channel_sum = torch.zeros(data.train.shape[1], dtype=torch.float64)
        channel_square_sum = torch.zeros_like(channel_sum)
        for chunk in data.train.split(1024):
            values = chunk.double().div_(255.)
            channel_sum += values.sum(dim=(0, 2, 3))
            channel_square_sum += values.square().sum(dim=(0, 2, 3))
        count = len(data.train) * data.train.shape[2] * data.train.shape[3]
        mean = channel_sum / count
        variance = (channel_square_sum / count - mean.square()).clamp_min(0.)
        data.channel_mean = mean.float()
        data.channel_std = variance.sqrt().clamp_min(1e-6).float()
    return data


def synthetic_view(latent, data, generator, task_noise):
    view = torch.randn(latent.shape, generator=generator)
    view[:, data.task_indices] = latent[:, data.task_indices]
    if task_noise:
        view[:, data.task_indices] += task_noise * torch.randn(
            len(latent), len(data.task_indices), generator=generator)
    return (view * data.scales) @ data.rotation


def standardize_images(values, data):
    """Apply recorded channel statistics to floating-point [0, 1] images."""
    if data is not None and data.channel_mean is not None:
        values = (values - data.channel_mean[None, :, None, None]) / data.channel_std[None, :, None, None]
    return values


def image_view(images, generator, dataset, data=None):
    values = images.float() / 255.
    # Reflect-padded random translation for MNIST; crop+flip for CIFAR.
    padding = 2 if dataset == "mnist" else 4
    padded = F.pad(values, (padding,) * 4, mode="reflect")
    offsets = torch.randint(0, 2 * padding + 1, (len(values), 2), generator=generator)
    height, width = values.shape[-2:]
    result = torch.stack([padded[i, :, u:u+height, v:v+width]
                          for i, (u, v) in enumerate(offsets.tolist())])
    if dataset != "mnist":
        flip = torch.rand(len(values), generator=generator) < .5
        result[flip] = result[flip].flip(-1)
    return standardize_images(result, data).flatten(1)


def paired_batch(data, indices, args, generator):
    values = data.train[indices]
    if args.dataset == "synthetic":
        first = synthetic_view(values, data, generator, args.task_view_noise)
        second = synthetic_view(values, data, generator, args.task_view_noise)
    else:
        first = image_view(values, generator, args.dataset, data)
        second = image_view(values, generator, args.dataset, data)
    return torch.cat([first, second]).to(args.device), data.labels[indices].repeat(2).to(args.device)


@contextmanager
def preserve_batchnorm_state(model):
    """Restore ManualMLP's replaced BN tensors/cache after a diagnostic pass."""
    if not model.batchnorm:
        yield
        return
    means, variances, cache = list(model.bn_running_mean), list(model.bn_running_var), list(model._bn_cache)
    try:
        yield
    finally:
        model.bn_running_mean[:] = means
        model.bn_running_var[:] = variances
        model._bn_cache[:] = cache


@torch.no_grad()
def condition(model, raw, x, method, args):
    if method in {"bp", "dfa"}:
        return raw
    # Use the same training-mode batch moments as the gradient pass, but do
    # not update running statistics twice or replace its backward cache.
    with preserve_batchnorm_state(model):
        _, activities, _ = model.forward(x)
    weights = list(raw.weights)
    n = x.shape[0] // 2
    for layer in range(model.n_hidden_layers):
        activity = activities[layer]
        damping = max(args.relative_damping * activity.square().mean().item(), args.damping_floor)
        if method == "ndfa":
            update = condition_local_update(activity, raw.deltas[layer] * len(x),
                                            activity_damping=damping, mode="activity", backend=args.backend)
        elif method in {"view_regularized", "wrong_view_regularized"}:
            errors = raw.deltas[layer] * len(x)
            if method == "wrong_view_regularized":
                # Reorder matching activity/error rows together: G and C_A stay
                # fixed, while only the within-pair penalty changes.
                activity = torch.cat([activity[:n], activity[n:].roll(1, dims=0)])
                errors = torch.cat([errors[:n], errors[n:].roll(1, dims=0)])
            update = view_regularized_activity_update(
                activity, errors, penalty=args.view_penalty, damping=damping, backend=args.backend)
        elif method == "centered_total":
            # Keep the raw update and uncentered trace-based damping fixed so
            # this control isolates removal of the activity mean from pairing.
            update = right_condition_from_samples(
                raw.weights[layer], activity - activity.mean(0),
                damping=damping, backend=args.backend)
        elif method == "diagonal":
            update = raw.weights[layer] / (activity.square().mean(0) + damping)
        else:
            first, second = activity[:n], activity[n:]
            if method == "wrong_pairs":
                second = second.roll(1, dims=0)
            update = paired_activity_update(raw.weights[layer], first, second,
                                            damping=damping, backend=args.backend)
        original_norm, new_norm = raw.weights[layer].norm(), update.norm()
        if new_norm > 0:
            update = update * (original_norm / new_norm)
        weights[layer] = update
    return Gradients(weights, raw.biases, raw.deltas, raw.loss, raw.bn_gammas, raw.bn_betas)


@torch.no_grad()
def evaluate(model, data, args):
    total_loss, correct = 0., 0
    was_training = model.training
    try:
        model.training = False
        with preserve_batchnorm_state(model):
            for start in range(0, len(data.validation), 256):
                x = data.validation[start:start+256]
                if args.dataset != "synthetic":
                    x = standardize_images(x.float().div(255.), data).flatten(1)
                y = data.validation_labels[start:start+256].to(args.device)
                logits = model.forward(x.to(args.device))[0]
                total_loss += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
    finally:
        model.training = was_training
    return total_loss / len(data.validation), correct / len(data.validation)


def make_model_and_feedback(data, args, seed):
    model = ManualMLP(data.input_dim, args.hidden_dims, data.classes,
                      seed=10000+seed, device=args.device, batchnorm=args.batchnorm)
    feedback = init_feedback(model, seed=args.feedback_seed + 100 * seed, scale=args.feedback_scale)
    return model, feedback


def parameter_count(model):
    return (sum(w.numel() + b.numel() for w, b in zip(model.weights, model.biases))
            + sum(g.numel() + b.numel() for g, b in zip(model.bn_gamma, model.bn_beta)))


def synchronize(args):
    if args.device == "cuda":
        torch.cuda.synchronize()


def train_one(args, data, seed, method):
    """Train one fixed configuration; return its final model and validation rows.

    This shared routine never loads or evaluates a test split. Initialization,
    data/view streams, diagnostics, and BatchNorm behavior match development.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    rows = []
    model, feedback = make_model_and_feedback(data, args, seed)
    sampler = torch.Generator().manual_seed(30000 + seed)
    views = torch.Generator().manual_seed(40000 + seed)
    training_seconds = 0.
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps + 1):
        if step:
            synchronize(args)
            start = time.perf_counter()
            idx = torch.randint(len(data.train), (args.batch_size,), generator=sampler)
            x, y = paired_batch(data, idx, args, views)
            model.training = True
            raw = model.bp_gradients(x, y) if method == "bp" else model.dfa_gradients(x, y, feedback)
            update = condition(model, raw, x, method, args)
            model.apply_gradients(update, lr=args.lr)
            if not all(torch.isfinite(w).all() for w in model.weights):
                raise FloatingPointError(f"nonfinite weights: seed={seed}, method={method}, step={step}")
            synchronize(args)
            training_seconds += time.perf_counter() - start
        if step == 0 or step == args.steps or step % args.eval_every == 0:
            loss, accuracy = evaluate(model, data, args)
            rows.append({"seed": seed, "method": method, "step": step,
                         "validation_loss": loss, "validation_accuracy": accuracy,
                         "training_seconds": training_seconds,
                         "parameter_count": parameter_count(model),
                         "batchnorm_affine_parameter_count": sum(g.numel() + b.numel() for g, b in zip(model.bn_gamma, model.bn_beta)),
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None})
    return model, rows


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing a silent CPU training fallback")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = load_data(args)
    source_paths = [Path(__file__), ROOT / "infogeo/paired_conditioning.py", ROOT / "infogeo/local_preconditioning.py", ROOT / "infogeo/dfa.py"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
                "stage": "exploratory development; validation only", "test_evaluated": False,
                "torch": torch.__version__, "python": sys.version,
                "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "training_examples": len(data.train), "validation_examples": len(data.validation),
                "input_standardization": None if data.channel_mean is None else {
                    "source": "selected training pool before augmentation; per channel over images and pixels",
                    "mean": data.channel_mean.tolist(), "std": data.channel_std.tolist(),
                    "std_definition": "population standard deviation", "std_floor": 1e-6,
                    "input_scale": "uint8 divided by 255", "validation_used": False},
                "batchnorm_affine_parameter_count": 2 * sum(args.hidden_dims) if args.batchnorm else 0,
                "peak_memory_scope": "total allocated CUDA bytes, including model, training and validation; reset independently after constructing each model",
                "comparability": "same two views, labels, initial weights and minibatch order across methods; conditioned hidden weights norm-matched to own raw DFA; all biases and classifier updates unchanged"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for seed in args.seeds:
        for method in args.methods:
            model, run_rows = train_one(args, data, seed, method)
            rows.extend(run_rows)
            print(json.dumps(rows[-1]), flush=True)
            (output / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            # Release the preceding run's GPU tensors before measuring the next
            # method. Otherwise old gradients can inflate later memory peaks.
            del model
    endpoints = [r for r in rows if r["step"] == args.steps]
    (output / "endpoints.json").write_text(json.dumps(endpoints, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
