"""Aggregate ImageNet credit-assignment benchmark shards."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"No input CSVs matched: {args.inputs}")
    df = pd.concat([pd.read_csv(path).assign(source=path) for path in paths], ignore_index=True)
    df.to_csv(output_dir / "imagenet_credit_assignment_all.csv", index=False)
    final = df.sort_values("epoch").groupby(["dataset", "arch", "method", "seed", "feedback_seed", "source"], dropna=False).tail(1)
    summary_keys = [
        col
        for col in [
            "dataset",
            "arch",
            "method",
            "model_source",
            "feedback_modules",
            "feedback_scale",
            "feedback_blend_gamma",
            "feedback_module_scales",
            "feedback_spatial_mode",
            "natural_damping",
            "dfa_norm",
        ]
        if col in final.columns
    ]
    summary = (
        final.groupby(summary_keys, dropna=False)
        .agg(
            val_top1_mean=("val_top1", "mean"),
            val_top1_sem=("val_top1", sem),
            val_top5_mean=("val_top5", "mean"),
            train_top1_mean=("train_top1", "mean"),
            n=("val_top1", "size"),
        )
        .reset_index()
        .sort_values(["dataset", "arch", "val_top1_mean"], ascending=[True, True, False])
    )
    summary.to_csv(output_dir / "imagenet_credit_assignment_summary.csv", index=False)
    best = summary.sort_values("val_top1_mean", ascending=False).groupby(["dataset", "arch", "method"], dropna=False).head(1)
    best.to_csv(output_dir / "imagenet_credit_assignment_best_by_method.csv", index=False)
    write_report(output_dir, paths, summary)
    plot_learning_curves(df, output_dir, args.figure_name)
    plot_final_bars(summary, output_dir, args.figure_name)
    print(summary.to_string(index=False))


def sem(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / (len(series) ** 0.5))


def write_report(output_dir: Path, paths: list[str], summary: pd.DataFrame) -> None:
    with (output_dir / "imagenet_credit_assignment_aggregate.md").open("w") as f:
        f.write("# ImageNet Credit-Assignment Aggregate\n\n")
        f.write("## Inputs\n\n")
        for path in paths:
            f.write(f"- `{path}`\n")
        f.write("\n## Final Summary\n\n")
        f.write(csv_table(summary))
        f.write("\n")


def csv_table(df: pd.DataFrame) -> str:
    lines = [", ".join(df.columns)]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append(", ".join(values))
    return "\n".join(lines)


def plot_learning_curves(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for (arch, method), group in df.groupby(["arch", "method"]):
        curve = group.groupby("epoch", as_index=False)["val_top1"].mean()
        ax.plot(curve["epoch"], curve["val_top1"], marker="o", markersize=2.5, linewidth=1.8, label=f"{arch}/{method}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 (%)")
    ax.set_title("ImageNet learning curves")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{name}_learning_curves.{ext}", dpi=220)
    plt.close(fig)


def plot_final_bars(summary: pd.DataFrame, output_dir: Path, name: str) -> None:
    labels = [f"{row.arch}\n{row.method}" for row in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(labels)), 4.0))
    ax.bar(range(len(summary)), summary["val_top1_mean"], yerr=summary["val_top1_sem"], color="#4477AA", alpha=0.9)
    ax.set_xticks(range(len(summary)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final validation top-1 (%)")
    ax.set_title("ImageNet final accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{name}_final_bars.{ext}", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/imagenet_credit_assignment_aggregate")
    parser.add_argument("--figure-name", default="imagenet_credit_assignment")
    parser.add_argument("--inputs", nargs="+", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
