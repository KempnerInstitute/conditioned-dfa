"""Factorial test of activity- and error-side conditioning in a ReLU DFA MLP.

The experiment crosses two independently controlled interventions:

* activity anisotropy: task-irrelevant nuisance coordinates are rescaled while
  the latent task, labels, and additive input noise remain unchanged;
* error anisotropy: columns of each fixed DFA feedback matrix are rescaled and
  RMS-renormalized, producing unequal local credit-coordinate variance without
  changing the input distribution.

Raw DFA, activity nDFA, error nDFA, and K-nDFA share the same initialization,
minibatches, and fixed feedback in each cell.  Conditioned hidden-layer weight
gradients are norm-matched to raw DFA, so comparisons isolate direction rather
than scalar step size.  Unlike the DFA-Stall confirmations, this test uses a
two-hidden-layer ReLU MLP and multiclass softmax cross-entropy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_multioutput_synthetic import make_multioutput_dataset  # noqa: E402
from experiments.run_dfa_synthetic import natural_precondition_gradients  # noqa: E402
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.dfa import Gradients, ManualMLP, error_second_moment, init_feedback  # noqa: E402


METHODS = ("dfa", "ndfa", "endfa", "kndfa")
METHOD_LABEL = {
    "dfa": "DFA",
    "ndfa": "activity nDFA",
    "endfa": "error nDFA",
    "kndfa": "K-nDFA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dfa_factorial_synthetic_v1")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--feedback-seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--cells",
        nargs="+",
        choices=["baseline", "activity", "error", "both"],
        default=["baseline", "activity", "error", "both"],
    )
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--nuisance-dim", type=int, default=24)
    parser.add_argument("--task-scale", type=float, default=1.0)
    parser.add_argument("--baseline-nuisance-scale", type=float, default=0.25)
    parser.add_argument("--activity-nuisance-scale", type=float, default=3.0)
    parser.add_argument(
        "--feedback-anisotropy",
        type=float,
        default=30.0,
        help="Ratio of largest to smallest feedback-coordinate scale in error-anisotropic cells.",
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--activity-damping", type=float, default=0.3)
    parser.add_argument("--error-damping", type=float, default=10.0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True

    rows: list[dict[str, float | int | str | bool]] = []
    run_index = 0
    for seed in args.seeds:
        for feedback_seed in args.feedback_seeds:
            for cell in args.cells:
                for method in args.methods:
                    rows.extend(run_one(args, seed, feedback_seed, cell, method, device))
                    run_index += 1
                    if args.checkpoint_every > 0 and run_index % args.checkpoint_every == 0:
                        pd.DataFrame(rows).to_csv(output_dir / "factorial_results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "factorial_results.csv", index=False)
    endpoints = df.sort_values("epoch").groupby(
        ["cell", "activity_anisotropy", "error_anisotropy", "method", "seed", "feedback_seed"],
        as_index=False,
    ).tail(1)
    endpoints.to_csv(output_dir / "factorial_endpoints.csv", index=False)
    write_report(endpoints, output_dir, args)
    print(f"wrote {output_dir}")


def cell_flags(cell: str) -> tuple[bool, bool]:
    return cell in {"activity", "both"}, cell in {"error", "both"}


def run_one(
    args: argparse.Namespace,
    seed: int,
    feedback_seed: int,
    cell: str,
    method: str,
    device: str,
) -> list[dict[str, float | int | str | bool]]:
    activity_anisotropy, error_anisotropy = cell_flags(cell)
    nuisance_scale = args.activity_nuisance_scale if activity_anisotropy else args.baseline_nuisance_scale
    dataset = make_multioutput_dataset(
        condition="task_aligned",
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        nuisance_dim=args.nuisance_dim,
        input_noise=0.0,
        train_label_noise=0.0,
        test_label_noise=0.0,
        task_scale_override=args.task_scale,
        nuisance_scale_override=nuisance_scale,
        seed=seed,
    )
    model = ManualMLP(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.n_classes,
        seed=10_000 + seed,
        device=device,
    )
    feedback = init_feedback(
        model,
        mode="random",
        seed=20_000 + 100 * seed + feedback_seed,
        scale=args.feedback_scale,
    )
    if error_anisotropy:
        feedback = anisotropize_feedback(
            feedback,
            ratio=args.feedback_anisotropy,
            seed=40_000 + 100 * seed + feedback_seed,
        )

    x_train = torch.as_tensor(dataset.x_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(dataset.y_train, dtype=torch.long, device=device)
    x_test = torch.as_tensor(dataset.x_test, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(dataset.y_test, dtype=torch.long, device=device)
    eval_n = min(int(args.eval_size), int(x_train.shape[0]))
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)

    rows: list[dict[str, float | int | str | bool]] = []
    for epoch in range(args.epochs + 1):
        rows.append(
            evaluate(
                model,
                x_train[:eval_n],
                y_train[:eval_n],
                x_test,
                y_test,
                feedback,
                args=args,
                method=method,
                seed=seed,
                feedback_seed=feedback_seed,
                cell=cell,
                activity_anisotropy=activity_anisotropy,
                error_anisotropy=error_anisotropy,
                epoch=epoch,
            )
        )
        if epoch == args.epochs:
            break
        for batch in minibatches(len(x_train), args.batch_size, rng):
            xb, yb = x_train[batch], y_train[batch]
            raw = model.dfa_gradients(xb, yb, feedback)
            gradients = condition_gradients(model, raw, xb, method=method, args=args)
            model.apply_gradients(gradients, lr=args.lr)

    print(
        f"cell={cell:8s} method={method:5s} seed={seed:3d} fb={feedback_seed} "
        f"test={rows[-1]['test_acc']:.3f}",
        flush=True,
    )
    return rows


def anisotropize_feedback(
    feedback: list[torch.Tensor],
    *,
    ratio: float,
    seed: int,
) -> list[torch.Tensor]:
    """Rescale hidden credit coordinates while preserving feedback RMS norm."""

    if ratio < 1.0:
        raise ValueError("feedback anisotropy ratio must be at least one")
    out: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    half_log = 0.5 * math.log(max(float(ratio), 1.0))
    for matrix in feedback:
        scales = torch.exp(
            torch.linspace(-half_log, half_log, matrix.shape[1], dtype=matrix.dtype, device="cpu")
        )
        scales = scales[torch.randperm(matrix.shape[1], generator=generator)]
        scales = scales / scales.square().mean().sqrt().clamp_min(1e-12)
        out.append(matrix * scales.to(matrix.device).unsqueeze(0))
    return out


def condition_gradients(
    model: ManualMLP,
    raw: Gradients,
    x: torch.Tensor,
    *,
    method: str,
    args: argparse.Namespace,
) -> Gradients:
    if method == "dfa":
        return raw
    mode = {"ndfa": "activity", "endfa": "error", "kndfa": "kronecker"}[method]
    conditioned = natural_precondition_gradients(
        model,
        raw,
        x,
        damping=args.activity_damping,
        error_damping=args.error_damping,
        mode=mode,
    )
    weights = [weight.clone() for weight in conditioned.weights]
    for layer_idx in range(model.n_hidden_layers):
        raw_norm = raw.weights[layer_idx].norm().clamp_min(1e-12)
        conditioned_norm = weights[layer_idx].norm().clamp_min(1e-12)
        weights[layer_idx] = weights[layer_idx] * (raw_norm / conditioned_norm)
    return Gradients(
        weights,
        [bias.clone() for bias in conditioned.biases],
        conditioned.deltas,
        conditioned.loss,
        conditioned.bn_gammas,
        conditioned.bn_betas,
    )


@torch.no_grad()
def evaluate(
    model: ManualMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feedback: list[torch.Tensor],
    *,
    args: argparse.Namespace,
    method: str,
    seed: int,
    feedback_seed: int,
    cell: str,
    activity_anisotropy: bool,
    error_anisotropy: bool,
    epoch: int,
) -> dict[str, float | int | str | bool]:
    train_logits, activations, _ = model.forward(x_train)
    test_logits, _, _ = model.forward(x_test)
    local = model.dfa_gradients(x_train, y_train, feedback)
    activity_stats = []
    error_stats = []
    for layer_idx in range(model.n_hidden_layers):
        activity = activations[layer_idx]
        activity_cov = activity.T @ activity / max(int(activity.shape[0]), 1)
        error_cov = error_second_moment(local.deltas[layer_idx], normalization_count=x_train.shape[0])
        activity_stats.append(spectrum_stats(activity_cov, damping=args.activity_damping))
        error_stats.append(spectrum_stats(error_cov, damping=args.error_damping))
    return {
        "cell": cell,
        "activity_anisotropy": activity_anisotropy,
        "error_anisotropy": error_anisotropy,
        "method": method,
        "method_label": METHOD_LABEL[method],
        "seed": seed,
        "feedback_seed": feedback_seed,
        "epoch": epoch,
        "activity_damping": float(args.activity_damping),
        "error_damping": float(args.error_damping),
        "feedback_anisotropy_ratio": float(args.feedback_anisotropy),
        "nuisance_scale": float(args.activity_nuisance_scale if activity_anisotropy else args.baseline_nuisance_scale),
        "train_loss": float(F.cross_entropy(train_logits, y_train).item()),
        "train_acc": float((train_logits.argmax(1) == y_train).float().mean().item()),
        "test_loss": float(F.cross_entropy(test_logits, y_test).item()),
        "test_acc": float((test_logits.argmax(1) == y_test).float().mean().item()),
        "activity_log_condition": float(np.mean([item["log_condition"] for item in activity_stats])),
        "error_log_condition": float(np.mean([item["log_condition"] for item in error_stats])),
        "activity_effective_rank": float(np.mean([item["effective_rank"] for item in activity_stats])),
        "error_effective_rank": float(np.mean([item["effective_rank"] for item in error_stats])),
        "activity_top_fraction": float(np.mean([item["top_fraction"] for item in activity_stats])),
        "error_top_fraction": float(np.mean([item["top_fraction"] for item in error_stats])),
    }


def spectrum_stats(cov: torch.Tensor, *, damping: float) -> dict[str, float]:
    eig = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0.0)
    trace = eig.sum().clamp_min(1e-12)
    probs = eig / trace
    effective_rank = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    damped_condition = (eig[-1] + damping) / (eig[0] + damping)
    return {
        "log_condition": float(torch.log10(damped_condition.clamp_min(1.0)).item()),
        "effective_rank": float(effective_rank.item()),
        "top_fraction": float((eig[-1] / trace).item()),
    }


def write_report(endpoints: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    summary = endpoints.groupby(["cell", "method", "method_label"], as_index=False).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_sem=("test_acc", sem),
        test_loss_mean=("test_loss", "mean"),
        n=("test_acc", "count"),
    )
    lines = [
        "# Factorial activity/error conditioning experiment",
        "",
        f"Two-hidden-layer ReLU MLP ({'/'.join(map(str, args.hidden_dims))}) with softmax cross-entropy; "
        f"{args.epochs} epochs; activity/error damping {args.activity_damping:g}/{args.error_damping:g}.",
        f"Activity intervention changes only nuisance scale ({args.baseline_nuisance_scale:g} to "
        f"{args.activity_nuisance_scale:g}); additive input and label noise are zero. Error intervention "
        f"uses an RMS-matched feedback-coordinate scale ratio of {args.feedback_anisotropy:g}.",
        "Conditioned hidden gradients are norm-matched to raw DFA.",
        "",
        summary.to_markdown(index=False, floatfmt=".5f"),
    ]
    (output_dir / "factorial_summary.md").write_text("\n".join(lines) + "\n")


def sem(values: pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


if __name__ == "__main__":
    main()
