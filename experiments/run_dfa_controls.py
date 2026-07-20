"""Control experiments for the conditioned-DFA paper.

This script runs matched comparisons that address two alternative explanations
for the conditioning gains:

  (a) Is conditioned DFA winning over BP because conditioning is genuinely
      different, or because the BP recipe is under-regularized? We sweep
      BP + L2, BP + label-smoothing, and BP + early-stop on val loss.

  (b) Is conditioning winning over raw DFA because of whitening, or merely
      because of per-layer gradient-norm matching? We add a DFA + norm-matching
      variant that rescales the DFA gradient at each layer to match the BP
      gradient norm on a held-out evaluation batch.

Two data sources are supported by ``--task``:

  - ``synthetic_nuisance`` : one of the existing 128-cell stress conditions
    (default: ``nuisance_dominant`` with the canonical config).
  - ``fashion_mnist_noisy``: torchvision Fashion-MNIST with optional
    train-label corruption.

Outputs land in ``results/dfa_controls_<task>_<timestamp>/`` as
``dfa_controls_results.csv`` plus per-method learning-curve CSVs and a
markdown summary, ready to be picked up by ``write_infodfa_paper_tables.py``
(see ``analysis/write_infodfa_controls_table.py``).
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_multioutput_synthetic import (  # noqa: E402
    MultiOutputDataset,
    make_multioutput_dataset,
)
from experiments.run_dfa_synthetic import (  # noqa: E402
    feedback_mode_from_method,
    natural_mode_from_method,
    natural_precondition_gradients,
)
from experiments.run_dfa_vision_baselines import (  # noqa: E402
    dataset_to_tensors,
    minibatches,
    normalize_train_test,
)
from infogeo.dfa import Gradients, ManualMLP, init_feedback  # noqa: E402


SUPPORTED_TASKS = ("synthetic_nuisance", "fashion_mnist_noisy")

# Method names follow the pattern <base>[+<modifier>] where <base> selects the
# learning rule and <modifier> is a control we apply on top of it.
ALL_METHODS = (
    "bp",
    "bp+l2",
    "bp+label_smoothing",
    "bp+early_stop",
    "dfa_random",
    "dfa_random+norm_match",
    "ndfa_random",
    "ndfa_random_kronecker",
)


@dataclass
class TrainArtifacts:
    test_acc: float
    train_acc: float
    val_loss_best: float
    epoch_best: int
    history: list[dict[str, float]] = field(default_factory=list)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "controls_config.txt"
    config_path.write_text("\n".join(f"{k} = {v}" for k, v in vars(args).items()))

    rows = []
    for seed in range(args.n_seeds):
        train_x, train_y, val_x, val_y, test_x, test_y, n_classes = prepare_task(args, seed=seed)
        for method in args.methods:
            torch.manual_seed(seed)
            start = perf_counter()
            artifact = train_one(
                method=method,
                seed=seed,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                n_classes=n_classes,
                args=args,
            )
            elapsed = perf_counter() - start
            row = {
                "task": args.task,
                "method": method,
                "seed": seed,
                "test_acc": artifact.test_acc,
                "train_acc": artifact.train_acc,
                "val_loss_best": artifact.val_loss_best,
                "epoch_best": artifact.epoch_best,
                "weight_decay": args.weight_decay if "+l2" in method else 0.0,
                "label_smoothing": args.label_smoothing if "+label_smoothing" in method else 0.0,
                "early_stop": int("+early_stop" in method),
                "norm_match": int("+norm_match" in method),
                "elapsed_sec": elapsed,
                "n_classes": n_classes,
            }
            print(
                f"{args.task:22s} {method:32s} seed={seed} test={artifact.test_acc:.3f} "
                f"train={artifact.train_acc:.3f} val_best@{artifact.epoch_best} ({elapsed:.1f}s)",
                flush=True,
            )
            rows.append(row)
            # Per-method history CSVs
            history = pd.DataFrame(artifact.history)
            history.insert(0, "method", method)
            history.insert(0, "seed", seed)
            history.to_csv(output_dir / f"history_{method.replace('+','_')}_seed{seed}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dfa_controls_results.csv", index=False)
    write_summary(df, output_dir, args)
    print(f"\nSaved results to {output_dir / 'dfa_controls_results.csv'}")


# -----------------------------------------------------------------------------
# Task setup
# -----------------------------------------------------------------------------


def prepare_task(args: argparse.Namespace, *, seed: int):
    if args.task == "synthetic_nuisance":
        dataset = make_multioutput_dataset(
            condition=args.synthetic_condition,
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            n_classes=args.n_classes_synthetic,
            nuisance_dim=args.nuisance_dim,
            input_noise=args.input_noise,
            train_label_noise=args.label_noise,
            test_label_noise=0.0,
            task_scale_override=None,
            nuisance_scale_override=None,
            seed=seed,
        )
        train_x = torch.tensor(dataset.x_train, dtype=torch.float32)
        train_y = torch.tensor(dataset.y_train, dtype=torch.long)
        test_x = torch.tensor(dataset.x_test, dtype=torch.float32)
        test_y = torch.tensor(dataset.y_test, dtype=torch.long)
    elif args.task == "fashion_mnist_noisy":
        try:
            from torchvision import datasets, transforms
        except Exception as exc:
            raise ImportError("Install torchvision to run fashion_mnist_noisy") from exc
        transform = transforms.Compose([transforms.ToTensor()])
        root = Path(args.data_dir)
        train_full = datasets.FashionMNIST(root=str(root), train=True, download=args.download, transform=transform)
        test_full = datasets.FashionMNIST(root=str(root), train=False, download=args.download, transform=transform)
        train_x, train_y = dataset_to_tensors(train_full, args.n_train, seed=seed)
        test_x, test_y = dataset_to_tensors(test_full, args.n_test, seed=seed + 1)
        train_x, test_x = normalize_train_test(train_x, test_x)
        if args.label_noise > 0:
            train_y = corrupt_labels(train_y, fraction=args.label_noise, n_classes=10, seed=seed)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Carve a held-out validation split from training for early stopping AND
    # for norm-match calibration. We use a small slice so the train budget is
    # essentially preserved.
    n_total = train_x.shape[0]
    n_val = max(min(int(args.val_fraction * n_total), 2048), 256)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_x = train_x[val_idx]
    val_y = train_y[val_idx]
    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    n_classes = int(train_y.max().item() + 1) if args.task == "fashion_mnist_noisy" else args.n_classes_synthetic
    return train_x, train_y, val_x, val_y, test_x, test_y, n_classes


def corrupt_labels(labels: torch.Tensor, *, fraction: float, n_classes: int, seed: int) -> torch.Tensor:
    if fraction <= 0:
        return labels
    rng = np.random.default_rng(seed + 12345)
    labels_np = labels.cpu().numpy().copy()
    n = labels_np.shape[0]
    n_flip = int(round(fraction * n))
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    for idx in flip_idx:
        new = rng.integers(n_classes)
        while new == labels_np[idx]:
            new = rng.integers(n_classes)
        labels_np[idx] = new
    return torch.tensor(labels_np, dtype=labels.dtype)


# -----------------------------------------------------------------------------
# Trainer with control wrappers
# -----------------------------------------------------------------------------


def train_one(
    *,
    method: str,
    seed: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_classes: int,
    args: argparse.Namespace,
) -> TrainArtifacts:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = ManualMLP(
        input_dim=train_x.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=n_classes,
        seed=10_000 + seed,
        device=device,
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    base_method = method.split("+", 1)[0]
    feedback = None
    if base_method != "bp":
        feedback = init_feedback(
            model,
            mode=feedback_mode_from_method(base_method),
            seed=20_000 + seed,
            scale=args.feedback_scale,
            rank=None if args.feedback_rank <= 0 else args.feedback_rank,
        )

    apply_l2 = "+l2" in method
    apply_label_smoothing = "+label_smoothing" in method
    apply_early_stop = "+early_stop" in method
    apply_norm_match = "+norm_match" in method

    rng = np.random.default_rng(30_000 + seed)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
    best_epoch = 0

    for epoch in range(args.epochs + 1):
        train_acc = model.accuracy(train_x, train_y)
        val_logits, _, _ = model.forward(val_x)
        val_loss = float(F.cross_entropy(val_logits, val_y).item())
        test_acc = model.accuracy(test_x, test_y)
        history.append(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "test_acc": test_acc,
            }
        )
        if apply_early_stop and val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = (
                [w.detach().clone() for w in model.weights],
                [b.detach().clone() for b in model.biases],
            )
        elif not apply_early_stop:
            best_val_loss = val_loss
            best_epoch = epoch
        if epoch == args.epochs:
            break
        for batch in minibatches(train_x.shape[0], args.batch_size, rng):
            xb = train_x[batch]
            yb = train_y[batch]
            grads = compute_gradients_with_controls(
                model,
                xb,
                yb,
                method=method,
                base_method=base_method,
                feedback=feedback,
                apply_label_smoothing=apply_label_smoothing,
                label_smoothing=args.label_smoothing if apply_label_smoothing else 0.0,
                apply_norm_match=apply_norm_match,
                norm_match_x=val_x,
                norm_match_y=val_y,
                args=args,
            )
            if apply_l2 and args.weight_decay > 0:
                grads = add_weight_decay(grads, model, weight_decay=args.weight_decay)
            model.apply_gradients(grads, lr=args.lr)

    if apply_early_stop and best_state is not None:
        weights, biases = best_state
        for i, (w, b) in enumerate(zip(weights, biases)):
            model.weights[i] = w
            model.biases[i] = b
    final_test_acc = model.accuracy(test_x, test_y)
    final_train_acc = model.accuracy(train_x, train_y)
    return TrainArtifacts(
        test_acc=final_test_acc,
        train_acc=final_train_acc,
        val_loss_best=best_val_loss,
        epoch_best=best_epoch,
        history=history,
    )


def compute_gradients_with_controls(
    model: ManualMLP,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    method: str,
    base_method: str,
    feedback,
    apply_label_smoothing: bool,
    label_smoothing: float,
    apply_norm_match: bool,
    norm_match_x: torch.Tensor,
    norm_match_y: torch.Tensor,
    args: argparse.Namespace,
) -> Gradients:
    if apply_label_smoothing and base_method == "bp":
        return _bp_gradients_label_smoothed(model, xb, yb, label_smoothing=label_smoothing)
    if base_method == "bp":
        return model.bp_gradients(xb, yb)
    if base_method.startswith("dfa") or base_method.startswith("ndfa"):
        gradients = model.dfa_gradients(xb, yb, feedback)
        if base_method.startswith("ndfa"):
            gradients = natural_precondition_gradients(
                model,
                gradients,
                xb,
                damping=args.natural_damping,
                mode=natural_mode_from_method(base_method),
            )
        if apply_norm_match:
            bp_ref = model.bp_gradients(norm_match_x, norm_match_y)
            gradients = _rescale_to_match(gradients, reference=bp_ref)
        return gradients
    raise ValueError(f"Unsupported base method: {base_method}")


def _bp_gradients_label_smoothed(
    model: ManualMLP,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    label_smoothing: float,
) -> Gradients:
    logits, activations, preactivations = model.forward(xb)
    n_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(yb, num_classes=n_classes).float()
    targets = (1.0 - label_smoothing) * one_hot + label_smoothing / n_classes
    delta = (probs - targets) / logits.shape[0]
    grad_w = [torch.empty_like(w) for w in model.weights]
    grad_b = [torch.empty_like(b) for b in model.biases]
    deltas = [torch.empty(0, device=model.device) for _ in model.weights]
    last = len(model.weights) - 1
    deltas[last] = delta
    grad_w[last] = delta.T @ activations[last]
    grad_b[last] = delta.sum(dim=0)
    for layer_idx in range(last - 1, -1, -1):
        delta = (delta @ model.weights[layer_idx + 1]) * (preactivations[layer_idx] > 0).float()
        deltas[layer_idx] = delta
        grad_w[layer_idx] = delta.T @ activations[layer_idx]
        grad_b[layer_idx] = delta.sum(dim=0)
    loss = float(F.cross_entropy(logits, yb).item())
    return Gradients(grad_w, grad_b, deltas, loss)


def _rescale_to_match(local: Gradients, *, reference: Gradients) -> Gradients:
    weights = []
    biases = []
    for w_local, w_ref in zip(local.weights, reference.weights):
        ref_norm = w_ref.norm()
        loc_norm = w_local.norm().clamp_min(1e-12)
        weights.append(w_local * (ref_norm / loc_norm))
    for b_local, b_ref in zip(local.biases, reference.biases):
        ref_norm = b_ref.norm()
        loc_norm = b_local.norm().clamp_min(1e-12)
        biases.append(b_local * (ref_norm / loc_norm))
    return Gradients(weights, biases, local.deltas, local.loss)


def add_weight_decay(grads: Gradients, model: ManualMLP, *, weight_decay: float) -> Gradients:
    weights = [g + weight_decay * w for g, w in zip(grads.weights, model.weights)]
    return Gradients(weights, list(grads.biases), grads.deltas, grads.loss)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_summary(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    summary = df.groupby("method").agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_sem=("test_acc", "sem"),
        train_acc_mean=("train_acc", "mean"),
        epoch_best_mean=("epoch_best", "mean"),
        n=("seed", "count"),
    ).reset_index()
    summary.to_csv(output_dir / "dfa_controls_summary.csv", index=False)

    if "bp" in summary["method"].values:
        bp_acc = float(summary.loc[summary["method"] == "bp", "test_acc_mean"].iloc[0])
    else:
        bp_acc = float("nan")
    if "dfa_random" in summary["method"].values:
        dfa_acc = float(summary.loc[summary["method"] == "dfa_random", "test_acc_mean"].iloc[0])
    else:
        dfa_acc = float("nan")

    lines = [
        "# DFA controls comparison",
        "",
        f"Task: `{args.task}`",
        f"Seeds: {args.n_seeds}",
        f"Hidden dims: {args.hidden_dims}",
        f"Train/val/test sizes: {args.n_train} / val from train / {args.n_test}",
        f"Label noise: {args.label_noise}",
        "",
        "## Final test accuracy by method",
        "",
        "| method | test acc (mean +/- SEM) | train acc | best epoch | n | gap vs BP | gap vs DFA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        gap_bp = f"{100.0 * (row['test_acc_mean'] - bp_acc):+.2f}" if np.isfinite(bp_acc) else "n/a"
        gap_dfa = f"{100.0 * (row['test_acc_mean'] - dfa_acc):+.2f}" if np.isfinite(dfa_acc) else "n/a"
        lines.append(
            f"| {row['method']} | {100.0*row['test_acc_mean']:.2f} +/- {100.0*row['test_acc_sem']:.2f} "
            f"| {100.0*row['train_acc_mean']:.2f} | {row['epoch_best_mean']:.1f} | {int(row['n'])} | "
            f"{gap_bp} | {gap_dfa} |"
        )
    lines.extend(
        [
            "",
            "## Reading guide",
            "",
            "- Compare `bp+l2`, `bp+label_smoothing`, `bp+early_stop` against `bp` to test ",
            "  whether stronger BP regularization closes the gap to conditioned DFA. ",
            "- Compare `dfa_random+norm_match` against `dfa_random` and against `ndfa_random` ",
            "  to isolate whether the conditioning win is from whitening or just per-layer ",
            "  norm matching. ",
            "- `ndfa_random` and `ndfa_random_kronecker` rows give the reference conditioned ",
            "  rules at matched seeds.",
        ]
    )
    (output_dir / "dfa_controls_summary.md").write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="synthetic_nuisance")
    parser.add_argument("--output-dir", default="results/dfa_controls_synthetic_nuisance_v1")
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    parser.add_argument("--n-seeds", type=int, default=5)

    # Synthetic task config
    parser.add_argument("--synthetic-condition", default="nuisance_dominant")
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--n-classes-synthetic", type=int, default=8)
    parser.add_argument("--nuisance-dim", type=int, default=24)
    parser.add_argument("--input-noise", type=float, default=0.05)
    parser.add_argument("--label-noise", type=float, default=0.1)

    # Vision task config
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--download", action="store_true")

    # Trainer config
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-rank", type=int, default=0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    # Controls
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
