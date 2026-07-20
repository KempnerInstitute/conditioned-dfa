"""ColoredMNIST benchmark for conditioned DFA.

ColoredMNIST is the canonical spurious-correlation benchmark
(\\citealp{arjovsky2019irm}). Each binary class (digits 0--4 vs 5--9) is
correlated with a color (red/green) in training, with a small environment-
dependent flip. At test time the color--label correlation is *reversed*, so a
classifier that latches onto color suffers and a classifier that ignores color
generalizes. BP MLPs are notoriously prone to the color shortcut; the question
is whether conditioned DFA, which whitens the activity covariance, resists it.

We run a single-environment ColoredMNIST configuration with adjustable color
strength. Methods compared: BP, BP+L2, DFA, DFA+norm-match, nDFA, K-nDFA.

Outputs: ``results/dfa_coloredmnist_v1/dfa_coloredmnist_results.csv`` plus
per-method history CSVs and a markdown summary.
"""

from __future__ import annotations

import argparse
import sys
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

from experiments.run_dfa_controls import (  # noqa: E402
    add_weight_decay,
    compute_gradients_with_controls,
)
from experiments.run_dfa_synthetic import (  # noqa: E402
    feedback_mode_from_method,
)
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.dfa import ManualMLP, init_feedback  # noqa: E402


METHODS = (
    "bp",
    "bp+l2",
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
    grayscale_acc: float
    history: list[dict[str, float]] = field(default_factory=list)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in vars(args).items()))

    rows = []
    for seed in range(args.n_seeds):
        train_x, train_y, val_x, val_y, test_x, test_y, gray_x, gray_y = make_coloredmnist(args, seed=seed)
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
                gray_x=gray_x,
                gray_y=gray_y,
                args=args,
            )
            elapsed = perf_counter() - start
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_acc": artifact.test_acc,
                    "train_acc": artifact.train_acc,
                    "grayscale_acc": artifact.grayscale_acc,
                    "val_loss_best": artifact.val_loss_best,
                    "epoch_best": artifact.epoch_best,
                    "elapsed_sec": elapsed,
                }
            )
            print(
                f"{method:32s} seed={seed} test={artifact.test_acc:.3f} "
                f"gray={artifact.grayscale_acc:.3f} train={artifact.train_acc:.3f} ({elapsed:.1f}s)",
                flush=True,
            )
            history = pd.DataFrame(artifact.history)
            history.insert(0, "method", method)
            history.insert(0, "seed", seed)
            history.to_csv(output_dir / f"history_{method.replace('+', '_')}_seed{seed}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dfa_coloredmnist_results.csv", index=False)
    write_summary(df, output_dir)
    print(f"\nSaved results to {output_dir / 'dfa_coloredmnist_results.csv'}")


# -----------------------------------------------------------------------------
# ColoredMNIST dataset
# -----------------------------------------------------------------------------


def make_coloredmnist(args: argparse.Namespace, *, seed: int) -> tuple:
    """Generate train/val/test ColoredMNIST tensors.

    Training environment: digit-label correlated with color with probability
    ``args.train_color_correlation``. Test environment: correlation is
    \emph{reversed} (i.e., the color is anti-correlated with label by the same
    amount). A held-out grayscale split removes color entirely and measures the
    color-independent accuracy.
    """

    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise ImportError("Install torchvision to run ColoredMNIST") from exc

    root = Path(args.data_dir)
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root=str(root), train=True, download=args.download, transform=transform)
    full_test = datasets.MNIST(root=str(root), train=False, download=args.download, transform=transform)

    rng = np.random.default_rng(seed + 999)
    train_x, train_y = _build_split(full_train, rng=rng, color_correlation=args.train_color_correlation, n_per_class=args.n_train // 2)
    val_x, val_y = _build_split(full_train, rng=rng, color_correlation=args.train_color_correlation, n_per_class=args.n_val // 2, offset=args.n_train)
    test_x, test_y = _build_split(full_test, rng=rng, color_correlation=1.0 - args.train_color_correlation, n_per_class=args.n_test // 2)
    gray_x, gray_y = _build_split(full_test, rng=rng, color_correlation=0.5, n_per_class=args.n_test // 2, grayscale=True, offset=args.n_test)

    return train_x, train_y, val_x, val_y, test_x, test_y, gray_x, gray_y


def _build_split(
    base,
    *,
    rng: np.random.Generator,
    color_correlation: float,
    n_per_class: int,
    grayscale: bool = False,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n_per_class`` images per binary class and apply color correlation.

    Binary label rule: digits 0..4 -> class 0, 5..9 -> class 1.
    """

    images_per_class = {0: [], 1: []}
    seen = 0
    for image, digit in base:
        seen += 1
        if seen <= offset:
            continue
        cls = 0 if int(digit) < 5 else 1
        if len(images_per_class[cls]) < n_per_class:
            images_per_class[cls].append(image.squeeze(0).numpy())
        if all(len(v) >= n_per_class for v in images_per_class.values()):
            break

    xs, ys = [], []
    for cls in (0, 1):
        for img in images_per_class[cls]:
            # Choose color according to label-color correlation.
            if grayscale:
                color = 0.5  # explicit gray: no color signal
            elif rng.random() < color_correlation:
                color = 1.0 if cls == 0 else 0.0
            else:
                color = 0.0 if cls == 0 else 1.0
            colored = _apply_color(img, color, grayscale=grayscale)
            xs.append(colored)
            ys.append(cls)
    perm = rng.permutation(len(xs))
    x = torch.tensor(np.stack([xs[i] for i in perm]), dtype=torch.float32)
    y = torch.tensor([ys[i] for i in perm], dtype=torch.long)
    return x, y


def _apply_color(image: np.ndarray, color: float, *, grayscale: bool) -> np.ndarray:
    """Turn a grayscale MNIST digit into a (3, 28, 28) colored image.

    ``color`` in [0, 1] selects between red (0) and green (1). ``grayscale=True``
    forces equal red/green channels.
    """

    img = image.astype(np.float32)
    if grayscale:
        return np.stack([img, img, np.zeros_like(img)], axis=0)
    red = img * (1.0 - color)
    green = img * color
    blue = np.zeros_like(img)
    return np.stack([red, green, blue], axis=0)


# -----------------------------------------------------------------------------
# Trainer (wraps run_dfa_controls.compute_gradients_with_controls)
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
    gray_x: torch.Tensor,
    gray_y: torch.Tensor,
    args: argparse.Namespace,
) -> TrainArtifacts:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    input_dim = int(train_x.shape[1] * train_x.shape[2] * train_x.shape[3])
    n_classes = int(train_y.max().item()) + 1
    model = ManualMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=n_classes,
        seed=10_000 + seed,
        device=device,
    )
    train_x = train_x.reshape(train_x.shape[0], -1).to(device)
    train_y = train_y.to(device)
    val_x = val_x.reshape(val_x.shape[0], -1).to(device)
    val_y = val_y.to(device)
    test_x = test_x.reshape(test_x.shape[0], -1).to(device)
    test_y = test_y.to(device)
    gray_x = gray_x.reshape(gray_x.shape[0], -1).to(device)
    gray_y = gray_y.to(device)

    base_method = method.split("+", 1)[0]
    feedback = None
    if base_method != "bp":
        feedback = init_feedback(
            model,
            mode=feedback_mode_from_method(base_method),
            seed=20_000 + seed,
            scale=args.feedback_scale,
            rank=None,
        )

    apply_l2 = "+l2" in method
    apply_norm_match = "+norm_match" in method
    rng = np.random.default_rng(30_000 + seed)
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs + 1):
        train_acc = model.accuracy(train_x, train_y)
        val_logits, _, _ = model.forward(val_x)
        val_loss = float(F.cross_entropy(val_logits, val_y).item())
        test_acc = model.accuracy(test_x, test_y)
        gray_acc = model.accuracy(gray_x, gray_y)
        history.append({"epoch": epoch, "train_acc": train_acc, "val_loss": val_loss, "test_acc": test_acc, "gray_acc": gray_acc})
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
                apply_label_smoothing=False,
                label_smoothing=0.0,
                apply_norm_match=apply_norm_match,
                norm_match_x=val_x,
                norm_match_y=val_y,
                args=args,
            )
            if apply_l2 and args.weight_decay > 0:
                grads = add_weight_decay(grads, model, weight_decay=args.weight_decay)
            model.apply_gradients(grads, lr=args.lr)

    return TrainArtifacts(
        test_acc=model.accuracy(test_x, test_y),
        train_acc=model.accuracy(train_x, train_y),
        val_loss_best=history[-1]["val_loss"],
        epoch_best=int(history[-1]["epoch"]),
        grayscale_acc=model.accuracy(gray_x, gray_y),
        history=history,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    summary = df.groupby("method").agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_sem=("test_acc", "sem"),
        gray_acc_mean=("grayscale_acc", "mean"),
        gray_acc_sem=("grayscale_acc", "sem"),
        train_acc_mean=("train_acc", "mean"),
        n=("seed", "count"),
    ).reset_index()
    summary.to_csv(output_dir / "dfa_coloredmnist_summary.csv", index=False)

    bp_row = summary[summary["method"] == "bp"]
    bp_test = float(bp_row["test_acc_mean"].iloc[0]) if len(bp_row) else float("nan")
    bp_gray = float(bp_row["gray_acc_mean"].iloc[0]) if len(bp_row) else float("nan")

    lines = [
        "# ColoredMNIST conditioned-DFA comparison",
        "",
        "Spurious-correlation benchmark: training has positive color-label correlation; test has the correlation reversed.",
        "Grayscale accuracy probes color-independent learning.",
        "",
        "| method | test acc (color reversed) | grayscale acc | train acc | n | gap vs BP (test) | gap vs BP (gray) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        gap_test = f"{100.0 * (row['test_acc_mean'] - bp_test):+.2f}" if np.isfinite(bp_test) else "n/a"
        gap_gray = f"{100.0 * (row['gray_acc_mean'] - bp_gray):+.2f}" if np.isfinite(bp_gray) else "n/a"
        lines.append(
            f"| {row['method']} | "
            f"{100.0 * row['test_acc_mean']:.2f} $\\pm$ {100.0 * row['test_acc_sem']:.2f} | "
            f"{100.0 * row['gray_acc_mean']:.2f} $\\pm$ {100.0 * row['gray_acc_sem']:.2f} | "
            f"{100.0 * row['train_acc_mean']:.2f} | {int(row['n'])} | {gap_test} | {gap_gray} |"
        )
    lines.append("")
    lines.append("Higher is better. A method that ignores color should generalize to the color-reversed test set "
                 "and to the grayscale probe; a method that latches onto color shows degraded test accuracy.")
    (output_dir / "dfa_coloredmnist_summary.md").write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dfa_coloredmnist_v1")
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--train-color-correlation", type=float, default=0.85,
                        help="Probability that color matches the canonical color of the label class.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-rank", type=int, default=0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
