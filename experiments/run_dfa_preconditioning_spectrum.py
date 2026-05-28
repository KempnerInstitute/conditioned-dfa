"""Continuous spectrum from DFA to damped Kronecker natural-DFA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_multioutput_synthetic import make_multioutput_dataset  # noqa: E402
from experiments.run_dfa_synthetic import natural_precondition_gradients  # noqa: E402
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.analysis import dataframe_to_markdown, predictor_scores, write_markdown_report  # noqa: E402
from infogeo.dfa import Gradients, ManualMLP, gradient_cosines, init_feedback  # noqa: E402


@dataclass(frozen=True)
class SpectrumSpec:
    condition: str
    seed: int
    feedback_seed: int
    feedback_rank: int
    gamma: float
    damping: float


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 1024
        args.n_test = 512
        args.epochs = 3
        args.hidden_dims = [128]
        args.n_seeds = 1
        args.n_feedback_seeds = 1
        args.conditions = [args.condition]
        args.feedback_ranks = [0, 2]
        args.gammas = [0.0, 0.5, 1.0]
        args.dampings = [0.3]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    if args.n_shards is not None:
        if args.shard_index is None:
            raise ValueError("--shard-index is required when --n-shards is set")
        specs = [spec for idx, spec in enumerate(specs) if idx % args.n_shards == args.shard_index]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    print(f"Running {len(specs)} DFA-to-K-nDFA spectrum specs", flush=True)
    rows = []
    for spec_idx, spec in enumerate(specs, start=1):
        rows.extend(run_one(spec, args=args))
        if args.checkpoint_every > 0 and spec_idx % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(output_dir / "dfa_preconditioning_spectrum.partial.csv", index=False)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "dfa_preconditioning_spectrum.csv"
    df.to_csv(csv_path, index=False)
    if not args.skip_analysis:
        plot_results(df, output_dir)
        write_report(df, output_dir)
    print(f"Saved {csv_path}")


def build_specs(args: argparse.Namespace) -> list[SpectrumSpec]:
    return [
        SpectrumSpec(condition=condition, seed=seed, feedback_seed=feedback_seed, feedback_rank=rank, gamma=float(gamma), damping=float(damping))
        for condition in args.conditions
        for seed in range(args.n_seeds)
        for feedback_seed in range(args.n_feedback_seeds)
        for rank in args.feedback_ranks
        for damping in args.dampings
        for gamma in args.gammas
    ]


def run_one(spec: SpectrumSpec, *, args: argparse.Namespace) -> list[dict[str, float | str]]:
    dataset = make_multioutput_dataset(
        condition=spec.condition,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        nuisance_dim=args.nuisance_dim,
        input_noise=args.input_noise,
        seed=spec.seed,
    )
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = ManualMLP(
        input_dim=dataset.x_train.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=dataset.n_classes,
        seed=10_000 + spec.seed,
        device=device,
    )
    feedback = init_feedback(
        model,
        mode="random",
        seed=20_000 + 100 * spec.seed + spec.feedback_seed,
        scale=args.feedback_scale,
        rank=None if spec.feedback_rank <= 0 else spec.feedback_rank,
    )
    x_train = torch.tensor(dataset.x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long, device=device)
    x_test = torch.tensor(dataset.x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long, device=device)
    rng = np.random.default_rng(30_000 + 100 * spec.seed + spec.feedback_seed)
    rows = []
    eval_n = min(args.eval_size, len(x_train))
    for epoch in range(args.epochs + 1):
        rows.append(evaluate(model, x_train[:eval_n], y_train[:eval_n], x_test, y_test, feedback, spec, epoch, args))
        if epoch == args.epochs:
            break
        for batch in minibatches(len(x_train), args.batch_size, rng):
            xb = x_train[batch]
            yb = y_train[batch]
            raw = model.dfa_gradients(xb, yb, feedback)
            kndfa = natural_precondition_gradients(model, raw, xb, damping=spec.damping, mode="kronecker")
            gradients = blend_gradients(raw, kndfa, gamma=spec.gamma)
            model.apply_gradients(gradients, lr=args.lr)
    print(
        f"{spec.condition:18s} gamma={spec.gamma:.2f} damping={spec.damping:.3g} "
        f"rank={spec.feedback_rank} seed={spec.seed}/{spec.feedback_seed} test={model.accuracy(x_test, y_test):.3f}",
        flush=True,
    )
    return rows


def blend_gradients(raw: Gradients, kndfa: Gradients, *, gamma: float) -> Gradients:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    weights = [(1.0 - gamma) * a + gamma * b for a, b in zip(raw.weights, kndfa.weights)]
    biases = [(1.0 - gamma) * a + gamma * b for a, b in zip(raw.biases, kndfa.biases)]
    return Gradients(weights=weights, biases=biases, deltas=raw.deltas, loss=raw.loss)


def evaluate(
    model: ManualMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feedback,
    spec: SpectrumSpec,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    raw = model.dfa_gradients(x_eval, y_eval, feedback)
    kndfa = natural_precondition_gradients(model, raw, x_eval, damping=spec.damping, mode="kronecker")
    blended = blend_gradients(raw, kndfa, gamma=spec.gamma)
    raw_scores = gradient_cosines(bp, raw)
    blended_scores = gradient_cosines(bp, blended)
    return {
        "condition": spec.condition,
        "method": "dfa_to_kndfa",
        "seed": float(spec.seed),
        "feedback_seed": float(spec.feedback_seed),
        "feedback_rank": float(spec.feedback_rank),
        "gamma": float(spec.gamma),
        "damping": float(spec.damping),
        "epoch": float(epoch),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval),
        "test_acc": model.accuracy(x_test, y_test),
        "raw_param_cosine": float(raw_scores.get("param_cosine", np.nan)),
        "blended_param_cosine": float(blended_scores.get("param_cosine", np.nan)),
        "raw_weight_step": weight_projected_step(bp, raw),
        "kndfa_weight_step": weight_projected_step(bp, kndfa),
        "blended_weight_step": weight_projected_step(bp, blended),
        "kndfa_weight_norm_ratio": weight_norm_ratio(kndfa, raw),
    }


def weight_projected_step(reference: Gradients, estimate: Gradients) -> float:
    numerator = 0.0
    denominator = 0.0
    for ref, est in zip(reference.weights, estimate.weights):
        numerator += float(torch.sum(ref * est).item())
        denominator += float(torch.sum(ref * ref).item())
    return numerator / max(denominator, 1e-12)


def weight_norm_ratio(numerator_grad: Gradients, denominator_grad: Gradients) -> float:
    num = sum(float(torch.sum(w * w).item()) for w in numerator_grad.weights)
    den = sum(float(torch.sum(w * w).item()) for w in denominator_grad.weights)
    return float(np.sqrt(num / max(den, 1e-12)))


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    final = final_rows(df)
    for condition, sub in final.groupby("condition"):
        curve = sub.groupby("gamma", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
        axes[0].errorbar(curve["gamma"], curve["test"], yerr=curve["sem"], marker="o", label=condition)
    axes[0].set_xlabel("Preconditioning blend gamma")
    axes[0].set_ylabel("Final test accuracy")
    axes[0].legend(frameon=False, fontsize=7)

    for condition, sub in df.groupby("condition"):
        selected = sub[np.isclose(sub["gamma"], 1.0)]
        curve = selected.groupby("epoch", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
        axes[1].plot(curve["epoch"], curve["test"], marker="o", label=condition)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("K-nDFA test accuracy")

    for condition, sub in final.groupby("condition"):
        curve = sub.groupby("gamma", as_index=False).agg(step=("blended_weight_step", "mean"))
        axes[2].plot(curve["gamma"], curve["step"], marker="o", label=condition)
    axes[2].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2].set_xlabel("Preconditioning blend gamma")
    axes[2].set_ylabel("Projected BP weight-step")

    fig.savefig(output_dir / "dfa_preconditioning_spectrum.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, output_dir: Path) -> None:
    final = final_rows(df)
    summary = (
        final.groupby(["condition", "feedback_rank", "damping", "gamma"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            blended_step=("blended_weight_step", "mean"),
            blended_cosine=("blended_param_cosine", "mean"),
            norm_ratio=("kndfa_weight_norm_ratio", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["condition", "feedback_rank", "damping", "gamma"])
    )
    scores = predictor_scores(
        final,
        target="test_acc",
        predictors=["gamma", "damping", "feedback_rank", "blended_weight_step", "blended_param_cosine", "kndfa_weight_norm_ratio"],
    )
    write_markdown_report(
        output_dir / "dfa_preconditioning_spectrum.md",
        title="DFA to K-nDFA Preconditioning Spectrum",
        sections=[
            (
                "Design",
                "Gamma=0 is raw DFA and gamma=1 is damped K-nDFA. Intermediate values blend the two weight updates, testing whether geometry-aware covariance whitening improves learning continuously rather than only as a method label.",
            ),
            ("Summary", dataframe_to_markdown(summary, float_format=".4f")),
            ("Predictors", dataframe_to_markdown(scores, float_format=".4f")),
        ],
    )


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("epoch").groupby(
        ["condition", "seed", "feedback_seed", "feedback_rank", "gamma", "damping"],
        as_index=False,
    ).tail(1)


def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.shape[0] <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.shape[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/dfa_preconditioning_spectrum_v1")
    parser.add_argument("--condition", default="task_aligned")
    parser.add_argument("--conditions", nargs="+", default=["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--feedback-ranks", type=int, nargs="+", default=[0, 2, 8])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--dampings", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--nuisance-dim", type=int, default=24)
    parser.add_argument("--input-noise", type=float, default=0.05)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
