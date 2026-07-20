"""MLP-Mixer DFA/nDFA comparison on noisy-label CIFAR-10 (pre-registered P5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_convnet_baselines import load_image_dataset  # noqa: E402
from experiments.run_dfa_nmnc_comparison import corrupt_labels  # noqa: E402
from experiments.run_dfa_synthetic import natural_mode_from_method  # noqa: E402
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.analysis import dataframe_to_markdown  # noqa: E402
from infogeo.mixer_dfa import (  # noqa: E402
    ManualMixer,
    init_mixer_feedback,
    mixer_gradient_cosines,
    natural_precondition_mixer_gradients,
)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 256
        args.n_test = 256
        args.epochs = 1
        args.depth = 2
        args.dim = 32
        args.eval_size = 128
        args.n_seeds = 1
        args.n_feedback_seeds = 1
        args.methods = ["bp", "dfa_random", "ndfa_random"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y, input_shape, n_classes = load_image_dataset(args)
    args.n_classes = n_classes
    train_y = corrupt_labels(train_y, n_classes=args.n_classes, noise=args.label_noise, seed=args.seed + 101)
    test_y = corrupt_labels(test_y, n_classes=args.n_classes, noise=args.test_label_noise, seed=args.seed + 202)
    rows = []
    for seed in range(args.n_seeds):
        for method in args.methods:
            feedback_seeds = [0] if method == "bp" else range(args.n_feedback_seeds)
            for feedback_seed in feedback_seeds:
                rows.extend(
                    run_one(
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        input_shape,
                        method=method,
                        seed=seed,
                        feedback_seed=int(feedback_seed),
                        args=args,
                    )
                )
    df = pd.DataFrame(rows)
    csv_path = output_dir / "dfa_mixer_results.csv"
    df.to_csv(csv_path, index=False)
    write_report(df, output_dir)
    plot_results(df, output_dir)
    print(final_summary(df).to_string(index=False))
    print(f"\nSaved results to {csv_path}")


def run_one(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    input_shape: tuple[int, ...],
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = ManualMixer(
        input_shape,
        args.n_classes,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        expansion=args.expansion,
        layernorm=args.layernorm,
        seed=10_000 + seed,
        device=device,
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    feedback = None
    if method != "bp":
        feedback = init_mixer_feedback(
            model,
            seed=20_000 + 100 * seed + feedback_seed,
            scale=args.feedback_scale,
            rank=None if args.feedback_rank <= 0 else args.feedback_rank,
        )
    precond_cache = {} if args.cov_refresh_interval > 1 else None
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    rows = []
    eval_n = min(args.eval_size, train_x.shape[0])
    step = 0
    for epoch in range(args.epochs + 1):
        rows.append(
            evaluate(
                model,
                train_x[:eval_n],
                train_y[:eval_n],
                test_x,
                test_y,
                method,
                feedback,
                seed,
                feedback_seed,
                epoch,
                args,
            )
        )
        if epoch == args.epochs:
            break
        for batch in minibatches(train_x.shape[0], args.batch_size, rng):
            xb = train_x[batch]
            yb = train_y[batch]
            gradients = compute_gradients(
                model,
                xb,
                yb,
                method=method,
                feedback=feedback,
                args=args,
                precond_cache=precond_cache,
                refresh_precond=step % max(args.cov_refresh_interval, 1) == 0,
            )
            model.apply_gradients(gradients, lr=args.lr)
            step += 1
    return rows


def compute_gradients(
    model: ManualMixer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    method: str,
    feedback,
    args: argparse.Namespace,
    precond_cache: dict | None = None,
    refresh_precond: bool = True,
):
    if method == "bp":
        return model.bp_gradients(x, y)
    if feedback is None:
        raise ValueError(f"{method} requires feedback")
    gradients = model.dfa_gradients(x, y, feedback)
    if method.startswith("ndfa_"):
        gradients = natural_precondition_mixer_gradients(
            model,
            gradients,
            x,
            damping=args.natural_damping,
            mode=natural_mode_from_method(method),
            cache=precond_cache,
            refresh=refresh_precond,
        )
    return gradients


def evaluate(
    model: ManualMixer,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    method: str,
    feedback,
    seed: int,
    feedback_seed: int,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    diagnostics = {"param_cosine": np.nan}
    if feedback is not None:
        local = compute_gradients(model, x_eval, y_eval, method=method, feedback=feedback, args=args)
        diagnostics.update(mixer_gradient_cosines(bp, local))
    return {
        "dataset": args.dataset,
        "method": method,
        "seed": float(seed),
        "feedback_seed": float(feedback_seed),
        "feedback_rank": float(args.feedback_rank),
        "natural_damping": float(args.natural_damping),
        "n_train": float(args.n_train),
        "n_test": float(args.n_test),
        "label_noise": float(args.label_noise),
        "test_label_noise": float(args.test_label_noise),
        "layernorm": float(args.layernorm),
        "patch_size": float(args.patch_size),
        "dim": float(args.dim),
        "depth": float(args.depth),
        "expansion": float(args.expansion),
        "epoch": float(epoch),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval),
        "test_acc": model.accuracy(x_test, y_test, batch_size=args.eval_batch_size),
        **diagnostics,
    }


def final_summary(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed"]).tail(1)
    return (
        final.groupby(["dataset", "method"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            train_mean=("train_eval_acc", "mean"),
            param_cosine=("param_cosine", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["dataset", "test_mean"], ascending=[True, False])
    )


def write_report(df: pd.DataFrame, output_dir: Path) -> None:
    summary = final_summary(df)
    lines = [
        "# MLP-Mixer DFA Comparison",
        "",
        "## Design",
        "",
        "Small ReLU MLP-Mixer (patch embedding, token-mixing + channel-mixing blocks, average-pool linear head). DFA sends the output error to every mixer sublayer through fixed random feedback broadcast over the non-mixing axis; nDFA right-multiplies each sublayer update by the damped inverse presynaptic second moment (token profiles for token-mixing weights, channel vectors for channel-mixing weights); K-nDFA adds the error-side factor. The classifier head trains on the exact output error.",
        "",
        "## Final Summary",
        "",
        dataframe_to_markdown(summary, float_format=".4f"),
        "",
    ]
    (output_dir / "dfa_mixer_report.md").write_text("\n".join(lines))


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, group in df.groupby("method"):
        curve = group.groupby("epoch")["test_acc"].mean()
        axes[0].plot(curve.index, curve.values, marker="o", label=method)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    final = df.sort_values("epoch").groupby(["method", "seed", "feedback_seed"]).tail(1)
    by_method = final.groupby("method")["test_acc"].mean().sort_values()
    axes[1].barh(by_method.index, by_method.values)
    axes[1].set_xlabel("Final test accuracy")
    fig.savefig(output_dir / "dfa_mixer_summary.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/infodfa_mixer_v1")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker"],
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of training labels replaced by an incorrect class.")
    parser.add_argument("--test-label-noise", type=float, default=0.0, help="Optional test-label corruption; defaults to clean evaluation.")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--layernorm", dest="layernorm", action="store_true", default=True, help="Pre-LN mixer blocks (standard Mixer; the default).")
    parser.add_argument("--no-layernorm", dest="layernorm", action="store_false", help="Ablate both LayerNorms in every mixer block.")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-rank", type=int, default=0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cov-refresh-interval", type=int, default=1, help="Recompute the damped covariance inverse for ndfa preconditioning every k training batches (1 = every batch, the default behavior).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
