"""Audit the frozen Fashion-MNIST BP-error source-swap comparator.

The registered source swap reused the local error damping for a BP-error
covariance whose spectrum can have a very different scale.  This script follows
the exact frozen trajectory and records, layer by layer, (i) local and BP-error
second-moment scales and (ii) the cosine of each two-sided update with the
activity-only update after the paper's layerwise norm matching.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_dfa_stall_comparison as comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/dfa_stall_fashion_bpsource_scale_audit_v1",
    )
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--feedback-seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--activity-damping", type=float, default=0.03)
    parser.add_argument("--error-damping", type=float, default=30.0)
    parser.add_argument("--spectra-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stall = comparison.load_dfa_stall_module()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    x_train, y_train, _, _ = comparison.load_dataset(
        stall, "fashion_mnist", Path(args.data_dir)
    )
    split_gen = torch.Generator(device="cpu").manual_seed(24_680)
    order = torch.randperm(int(x_train.shape[0]), generator=split_gen)
    x_train = x_train[order[5_000:]]
    y_train = y_train[order[5_000:]]

    torch.manual_seed(args.seed)
    model = stall.TanhMLP(784, args.hidden, 10, args.seed).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    feedback_args = SimpleNamespace(hidden=args.hidden, feedback_scale=1.0)
    feedback = comparison.init_feedback(
        feedback_args,
        10,
        device,
        seed=args.seed,
        feedback_seed=args.feedback_seed,
    )
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    rows: list[dict[str, float | int]] = []

    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, len(x_train), (args.batch_size,), generator=rng)
        xb = x_train[idx].to(device).float()
        targets = stall.to_one_hot(y_train[idx], 10).to(device)

        model(xb)
        rows.extend(source_metrics(model, xb, targets, feedback, args, step))
        grads, _, _, _ = comparison.dfa_family_gradients(
            stall,
            model,
            xb,
            targets,
            feedback,
            method="kndfa_bp",
            activity_damping=args.activity_damping,
            error_damping=args.error_damping,
            norm_match_hidden=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for layer, (weight_grad, bias_grad) in zip(model.layers, grads):
            layer.weight.grad = weight_grad.detach().clone()
            layer.bias.grad = bias_grad.detach().clone()
        optimizer.step()

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "source_scale_audit.csv", index=False)
    write_summary(frame, output_dir, args)
    print((output_dir / "source_scale_audit_summary.md").read_text())


@torch.no_grad()
def source_metrics(model, xb, targets, feedback, args, step: int) -> list[dict[str, float | int]]:
    predictions = model(xb)
    output_error = predictions - targets
    bp_deltas = comparison.exact_bp_hidden_deltas(model, output_error)
    record_spectra = step == 1 or step == args.total_steps or step % args.spectra_every == 0
    rows: list[dict[str, float | int]] = []

    for layer_index, _ in enumerate(model.layers[:-1]):
        activity = xb if layer_index == 0 else model.acts[layer_index - 1]
        gate = 1.0 - model.preacts[layer_index].tanh().pow(2)
        local_delta = (output_error @ feedback[layer_index].t()) * gate
        raw = local_delta.t() @ activity / xb.shape[0]
        raw_norm = raw.norm().clamp_min(1e-12)

        activity_cov = activity.t() @ activity / activity.shape[0]
        activity_update = comparison.damped_solve(
            activity_cov, raw.t(), damping=args.activity_damping
        ).t()
        activity_update = activity_update * (
            raw_norm / activity_update.norm().clamp_min(1e-12)
        )

        local_cov = local_delta.t() @ local_delta / local_delta.shape[0]
        bp_delta = bp_deltas[layer_index]
        bp_cov = bp_delta.t() @ bp_delta / bp_delta.shape[0]
        local_update = comparison.damped_solve(
            local_cov, activity_update, damping=args.error_damping
        )
        bp_update = comparison.damped_solve(
            bp_cov, activity_update, damping=args.error_damping
        )
        local_update = local_update * (raw_norm / local_update.norm().clamp_min(1e-12))
        bp_update = bp_update * (raw_norm / bp_update.norm().clamp_min(1e-12))

        row: dict[str, float | int] = {
            "step": step,
            "layer": layer_index + 1,
            "bp_update_cosine_with_activity": cosine(bp_update, activity_update),
            "local_update_cosine_with_activity": cosine(local_update, activity_update),
        }
        if record_spectra:
            local_eigs = torch.linalg.eigvalsh(0.5 * (local_cov + local_cov.t()))
            bp_eigs = torch.linalg.eigvalsh(0.5 * (bp_cov + bp_cov.t()))
            row.update(
                {
                    "local_lambda_max": float(local_eigs[-1].item()),
                    "local_trace": float(local_eigs.sum().item()),
                    "bp_lambda_max": float(bp_eigs[-1].item()),
                    "bp_trace": float(bp_eigs.sum().item()),
                    "local_damped_spectral_ratio": float(
                        ((local_eigs[-1] + args.error_damping) /
                         (local_eigs[0] + args.error_damping)).item()
                    ),
                    "bp_damped_spectral_ratio": float(
                        ((bp_eigs[-1] + args.error_damping) /
                         (bp_eigs[0] + args.error_damping)).item()
                    ),
                }
            )
        rows.append(row)
    return rows


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.reshape(-1) @ right.reshape(-1) /
         (left.norm() * right.norm()).clamp_min(1e-12)).item()
    )


def write_summary(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    spectra = frame.dropna(subset=["bp_lambda_max"])
    lines = [
        "# Fashion-MNIST BP-source scale audit",
        "",
        (
            f"Frozen activity/error dampings: {args.activity_damping:g}/"
            f"{args.error_damping:g}; seed {args.seed}, feedback seed "
            f"{args.feedback_seed}; {args.total_steps} steps."
        ),
        "",
        "| quantity | minimum | maximum |",
        "|---|---:|---:|",
    ]
    for column in [
        "bp_update_cosine_with_activity",
        "local_update_cosine_with_activity",
        "bp_lambda_max",
        "bp_trace",
        "local_lambda_max",
        "local_trace",
        "bp_damped_spectral_ratio",
        "local_damped_spectral_ratio",
    ]:
        source = frame if "cosine" in column else spectra
        lines.append(
            f"| {column} | {source[column].min():.8g} | {source[column].max():.8g} |"
        )
    lines.append("")
    (output_dir / "source_scale_audit_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
