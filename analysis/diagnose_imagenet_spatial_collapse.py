"""Diagnose instability in ImageNet spatial-feedback credit-assignment runs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_COLUMNS = [
    "method",
    "feedback_modules",
    "feedback_spatial_mode",
    "feedback_scale",
    "feedback_blend_gamma",
    "feedback_module_scales",
    "seed",
    "feedback_seed",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"No input CSVs matched: {args.inputs}")

    df = pd.concat([pd.read_csv(path).assign(source=path) for path in paths], ignore_index=True)
    df.to_csv(output_dir / "spatial_collapse_all_epochs.csv", index=False)

    summary = summarize_runs(df)
    summary.to_csv(output_dir / "spatial_collapse_run_summary.csv", index=False)
    epoch_diag = epoch_diagnostics(df)
    epoch_diag.to_csv(output_dir / "spatial_collapse_epoch_diagnostics.csv", index=False)

    write_report(output_dir, paths, summary)
    plot_learning_curves(df, output_dir, args.figure_name)
    plot_worst_runs(df, summary, output_dir, args.figure_name, n=args.n_worst)
    print(summary.sort_values("max_drop", ascending=False).head(args.n_worst).to_string(index=False))


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in df.sort_values("epoch").groupby("source", dropna=False):
        final = group.tail(1).iloc[0]
        val = group["val_top1"].astype(float)
        running_peak = val.cummax()
        drop = running_peak - val
        drop_idx = int(drop.idxmax())
        drop_row = group.loc[drop_idx]
        peak_before_drop = float(running_peak.loc[drop_idx])
        row = {
            "source": source,
            "final_epoch": int(final["epoch"]),
            "final_val_top1": float(final["val_top1"]),
            "final_val_top5": float(final["val_top5"]),
            "max_val_top1": float(val.max()),
            "min_val_top1": float(val.min()),
            "max_drop": float(drop.max()),
            "drop_epoch": int(drop_row["epoch"]),
            "peak_before_drop": peak_before_drop,
            "drop_val_top1": float(drop_row["val_top1"]),
            "nonfinite_total": float(group.get("nonfinite_steps", pd.Series(0.0, index=group.index)).fillna(0).sum()),
            "max_grad_norm": finite_max(group.get("grad_norm")),
            "min_block_cosine": finite_min(group.get("block_dfa_cosine_mean")),
            "final_block_cosine": finite_last(group.get("block_dfa_cosine_mean")),
        }
        for col in CONDITION_COLUMNS:
            if col in final.index:
                row[col] = final[col]
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        col
        for col in [
            "val_top1",
            "train_top1",
            "grad_norm",
            "nonfinite_steps",
            "block_dfa_cosine_mean",
            "block_dfa_projection_ratio_mean",
            "layer1_cosine",
            "layer2_cosine",
            "layer3_cosine",
            "layer4_cosine",
        ]
        if col in df.columns
    ]
    for source, group in df.sort_values("epoch").groupby("source", dropna=False):
        final = group.tail(1).iloc[0]
        for _, row in group.iterrows():
            out = {"source": source, "epoch": int(row["epoch"])}
            for col in CONDITION_COLUMNS:
                if col in final.index:
                    out[col] = final[col]
            for col in metric_cols:
                out[col] = row[col]
            rows.append(out)
    return pd.DataFrame(rows)


def finite_max(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def finite_min(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else float("nan")


def finite_last(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def condition_label(row: pd.Series) -> str:
    parts = []
    for col in ["method", "feedback_modules", "feedback_spatial_mode"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    if "seed" in row.index:
        parts.append(f"seed={row['seed']}")
    return " | ".join(parts)


def write_report(output_dir: Path, paths: list[str], summary: pd.DataFrame) -> None:
    with (output_dir / "spatial_collapse_report.md").open("w") as f:
        f.write("# ImageNet Spatial-Feedback Collapse Diagnosis\n\n")
        f.write("## Inputs\n\n")
        for path in paths:
            f.write(f"- `{path}`\n")
        f.write("\n## Worst Drops\n\n")
        cols = [
            col
            for col in [
                "method",
                "feedback_modules",
                "feedback_spatial_mode",
                "final_val_top1",
                "max_val_top1",
                "max_drop",
                "drop_epoch",
                "nonfinite_total",
                "max_grad_norm",
                "min_block_cosine",
            ]
            if col in summary.columns
        ]
        f.write(csv_table(summary.sort_values("max_drop", ascending=False)[cols].head(12)))
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
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for _, group in df.sort_values("epoch").groupby("source", dropna=False):
        final = group.tail(1).iloc[0]
        ax.plot(group["epoch"], group["val_top1"], linewidth=1.8, label=condition_label(final))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 (%)")
    ax.set_title("Spatial-feedback learning curves")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{name}_learning_curves.{ext}", dpi=220)
    plt.close(fig)


def plot_worst_runs(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, name: str, *, n: int) -> None:
    worst = summary.sort_values("max_drop", ascending=False).head(n)
    if worst.empty:
        return
    metric_cols = [
        col
        for col in [
            "val_top1",
            "grad_norm",
            "nonfinite_steps",
            "block_dfa_cosine_mean",
            "layer1_cosine",
            "layer2_cosine",
            "layer3_cosine",
            "layer4_cosine",
        ]
        if col in df.columns
    ]
    fig, axes = plt.subplots(len(metric_cols), 1, figsize=(8.0, max(3.0, 1.8 * len(metric_cols))), sharex=True)
    if len(metric_cols) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_cols):
        for source in worst["source"]:
            group = df[df["source"] == source].sort_values("epoch")
            label = condition_label(group.tail(1).iloc[0])
            ax.plot(group["epoch"], group[metric], linewidth=1.5, label=label)
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(fontsize=6, ncol=1)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{name}_worst_runs.{ext}", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/imagenet_spatial_collapse_diagnosis")
    parser.add_argument("--figure-name", default="imagenet_spatial_collapse")
    parser.add_argument("--n-worst", type=int, default=4)
    parser.add_argument("--inputs", nargs="+", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
