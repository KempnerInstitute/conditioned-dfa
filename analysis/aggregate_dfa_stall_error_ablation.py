"""Aggregate DFA-Stall error-side damping ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["dfa", "ndfa", "endfa", "kndfa"]
METHOD_LABEL = {
    "dfa": "DFA",
    "ndfa": "activity nDFA",
    "endfa": "error nDFA",
    "kndfa": "K-nDFA",
}
METHOD_COLOR = {
    "dfa": "#0072B2",
    "ndfa": "#009E73",
    "endfa": "#D55E00",
    "kndfa": "#6A3D9A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/dfa_stall_error_ablation_damping_v1")
    parser.add_argument("--output-dir", default="results/dfa_stall_error_ablation_damping_v1/aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run = load_runs(input_dir)
    per_run.to_csv(output_dir / "dfa_stall_error_ablation_runs.csv", index=False)
    summary = summarize(per_run)
    summary.to_csv(output_dir / "dfa_stall_error_ablation_summary.csv", index=False)
    write_report(summary, output_dir)
    make_figure(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"\nwrote {output_dir}")


def load_runs(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*_d*")):
        if not path.is_dir():
            continue
        summary_path = path / "dfa_stall_comparison_summary.csv"
        if not summary_path.exists():
            continue
        match = re.match(r"(faithful|normmatch)_d(.+)$", path.name)
        if match is None:
            continue
        condition_raw, damping_raw = match.groups()
        condition = "faithful" if condition_raw == "faithful" else "norm-matched"
        damping = float(damping_raw.replace("p", "."))
        df = pd.read_csv(summary_path)
        df.insert(0, "condition", condition)
        df.insert(1, "damping", damping)
        df.insert(2, "source_dir", str(path))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No run summaries found under {input_dir}")
    return pd.concat(rows, ignore_index=True)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    out = (
        per_run.groupby(["condition", "damping", "method"], as_index=False)
        .agg(
            final_probe_acc_mean=("final_probe_acc", "mean"),
            final_probe_acc_sem=("final_probe_acc", sem),
            final_loss_mean=("final_loss", "mean"),
            final_loss_sem=("final_loss", sem),
            loss_drop_mean=("loss_drop", "mean"),
            final_grad_alignment_mean=("final_grad_alignment", "mean"),
            final_grad_alignment_sem=("final_grad_alignment", sem),
            mean_grad_alignment=("mean_grad_alignment", "mean"),
            n=("seed", "count"),
        )
        .sort_values(["condition", "damping", "method"])
    )
    out["method_label"] = out["method"].map(METHOD_LABEL).fillna(out["method"])
    return out


def sem(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    best = (
        summary.sort_values(["condition", "final_probe_acc_mean"])
        .groupby("condition", as_index=False)
        .tail(4)
        .sort_values(["condition", "final_probe_acc_mean"], ascending=[True, False])
    )
    lines = [
        "# DFA-Stall error-side ablation",
        "",
        "Methods:",
        "- `dfa`: raw direct feedback alignment.",
        "- `ndfa`: activity/input-side second-moment preconditioning.",
        "- `endfa`: error/local-delta-side second-moment preconditioning only.",
        "- `kndfa`: both activity and error-side preconditioning.",
        "",
        "The faithful condition uses the raw preconditioned step. The norm-matched",
        "condition rescales each hidden preconditioned gradient back to the raw-DFA",
        "layerwise norm.",
        "",
        "## Best cells by final probe accuracy",
        "",
        best.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Full summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Takeaway: activity-side nDFA is the robust component at low damping. The",
        "error-side factor is harmful when under-damped, especially in K-nDFA. With",
        "norm matching and large damping, the error-side factor becomes safe and can",
        "help, consistent with the error covariance being noisy rather than useless.",
    ]
    (output_dir / "dfa_stall_error_ablation_summary.md").write_text("\n".join(lines) + "\n")


def make_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.6), constrained_layout=True)
    conditions = ["faithful", "norm-matched"]
    metrics = [
        ("final_probe_acc_mean", "final_probe_acc_sem", "final probe acc. (%)", 100.0),
        ("final_loss_mean", "final_loss_sem", "final train loss", 1.0),
        ("final_grad_alignment_mean", "final_grad_alignment_sem", r"final $\cos(g,g_{\rm BP})$", 1.0),
    ]
    for row_idx, condition in enumerate(conditions):
        cond = summary[summary["condition"] == condition]
        for col_idx, (metric, err_metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for method in METHOD_ORDER:
                sub = cond[cond["method"] == method].sort_values("damping")
                if sub.empty:
                    continue
                x = sub["damping"].to_numpy(dtype=float)
                y = scale * sub[metric].to_numpy(dtype=float)
                yerr = scale * sub[err_metric].to_numpy(dtype=float)
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    ms=3.0,
                    lw=1.4,
                    capsize=2,
                    color=METHOD_COLOR[method],
                    label=METHOD_LABEL[method],
                )
            ax.set_xscale("log")
            ax.set_xlabel("damping")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{condition}: {ylabel}")
            if col_idx == 2:
                ax.axhline(0.0, color="#555555", lw=0.7)
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"dfa_stall_error_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )


if __name__ == "__main__":
    main()
