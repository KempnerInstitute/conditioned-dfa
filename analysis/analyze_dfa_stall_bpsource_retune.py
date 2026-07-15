"""Aggregate the post-hoc Fashion-MNIST BP-source damping audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-root",
        default="results/dfa_stall_fashion_bpsource_retune_dev_v1",
    )
    parser.add_argument(
        "--confirmation-root",
        default="results/dfa_stall_fashion_bpsource_retune_confirmation_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="results/dfa_stall_fashion_bpsource_retune_analysis_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = load_development(Path(args.dev_root))
    selected = select_damping(dev)
    confirmation = load_confirmation(Path(args.confirmation_root))
    seed_level = (
        confirmation.groupby(["method", "seed"], as_index=False)
        .agg(test_acc=("final_test_acc", "mean"), test_loss=("final_test_loss", "mean"))
    )
    method_summary = summarize_methods(seed_level)
    contrasts = summarize_contrasts(seed_level)

    dev.to_csv(output_dir / "development_summary.csv", index=False)
    seed_level.to_csv(output_dir / "confirmation_seed_level.csv", index=False)
    method_summary.to_csv(output_dir / "confirmation_method_summary.csv", index=False)
    contrasts.to_csv(output_dir / "confirmation_contrasts.csv", index=False)
    write_report(dev, selected, method_summary, contrasts, output_dir)
    print((output_dir / "bpsource_retune_summary.md").read_text())


def load_development(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("error_d*/dfa_stall_comparison_summary.csv")):
        match = re.fullmatch(r"error_d(.+)", path.parent.name)
        if match is None:
            continue
        damping = float(match.group(1).replace("p", "."))
        frame = pd.read_csv(path)
        rows.append(
            {
                "error_damping": damping,
                "validation_acc_mean": frame["final_probe_acc"].mean(),
                "validation_acc_sem": sem(frame["final_probe_acc"]),
                "validation_loss_mean": frame["final_probe_loss"].mean(),
                "validation_loss_sem": sem(frame["final_probe_loss"]),
                "n_seeds": len(frame),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No development summaries below {root}")
    return pd.DataFrame(rows).sort_values("error_damping").reset_index(drop=True)


def select_damping(dev: pd.DataFrame) -> pd.Series:
    best_accuracy = dev["validation_acc_mean"].max()
    tied = dev[np.isclose(dev["validation_acc_mean"], best_accuracy, atol=1e-12)]
    return tied.sort_values(["validation_loss_mean", "error_damping"]).iloc[0]


def load_confirmation(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/fb*/dfa_stall_comparison_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No confirmation summaries below {root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def summarize_methods(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in seed_level.groupby("method"):
        rows.append(
            {
                "method": method,
                "test_acc_mean": frame["test_acc"].mean(),
                "test_acc_sem": sem(frame["test_acc"]),
                "test_loss_mean": frame["test_loss"].mean(),
                "test_loss_sem": sem(frame["test_loss"]),
                "n_seeds": len(frame),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def summarize_contrasts(seed_level: pd.DataFrame) -> pd.DataFrame:
    accuracy = seed_level.pivot(index="seed", columns="method", values="test_acc")
    loss = seed_level.pivot(index="seed", columns="method", values="test_loss")
    pairs = [
        ("kndfa", "ndfa"),
        ("kndfa_bp", "ndfa"),
        ("kndfa_bp", "kndfa"),
    ]
    rows = []
    for left, right in pairs:
        acc_delta = 100.0 * (accuracy[left] - accuracy[right])
        loss_delta = loss[left] - loss[right]
        rows.append(
            {
                "contrast": f"{left} - {right}",
                "accuracy_delta_pp": acc_delta.mean(),
                "accuracy_delta_sem_pp": sem(acc_delta),
                "accuracy_positive": int((acc_delta > 0).sum()),
                "loss_delta": loss_delta.mean(),
                "loss_delta_sem": sem(loss_delta),
                "loss_lower": int((loss_delta < 0).sum()),
                "n_seeds": len(acc_delta),
            }
        )
    return pd.DataFrame(rows)


def sem(values: pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def write_report(dev, selected, methods, contrasts, output_dir: Path) -> None:
    lines = [
        "# Post-hoc scale-matched BP-source audit",
        "",
        (
            f"Validation selected BP-source error damping "
            f"**{selected['error_damping']:g}**: mean accuracy "
            f"{100 * selected['validation_acc_mean']:.2f}% and loss "
            f"{selected['validation_loss_mean']:.4f}. Accuracy ties are broken "
            "by lower validation loss."
        ),
        "",
        "## Development sweep",
        "",
        "| damping | validation accuracy (%) | validation loss |",
        "|---:|---:|---:|",
    ]
    for row in dev.itertuples(index=False):
        lines.append(
            f"| {row.error_damping:g} | {100 * row.validation_acc_mean:.2f} ± "
            f"{100 * row.validation_acc_sem:.2f} | {row.validation_loss_mean:.4f} "
            f"± {row.validation_loss_sem:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fresh-seed confirmation",
            "",
            "| method | test accuracy (%) | test loss |",
            "|---|---:|---:|",
        ]
    )
    for row in methods.itertuples(index=False):
        lines.append(
            f"| {row.method} | {100 * row.test_acc_mean:.2f} ± "
            f"{100 * row.test_acc_sem:.2f} | {row.test_loss_mean:.4f} ± "
            f"{row.test_loss_sem:.4f} |"
        )
    lines.extend(["", "## Paired contrasts", ""])
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"- {row.contrast}: {row.accuracy_delta_pp:+.3f} ± "
            f"{row.accuracy_delta_sem_pp:.3f} pp "
            f"({row.accuracy_positive}/{row.n_seeds} positive); loss "
            f"{row.loss_delta:+.4f} ± {row.loss_delta_sem:.4f} "
            f"({row.loss_lower}/{row.n_seeds} lower)."
        )
    lines.append("")
    (output_dir / "bpsource_retune_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
