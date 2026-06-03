"""Aggregate DFA-Stall comparison pilots.

Reads outputs from ``experiments/run_dfa_stall_comparison.py`` and writes a
compact cross-condition summary plus an overview figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABEL = {"dfa": "DFA", "ndfa": "nDFA", "kndfa": "K-nDFA"}
METHOD_COLOR = {"dfa": "#0072B2", "ndfa": "#009E73", "kndfa": "#6A3D9A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faithful-dir", default="results/dfa_stall_comparison_3seed_v1")
    parser.add_argument("--normmatch-dir", default="results/dfa_stall_comparison_normmatch_3seed_v1")
    parser.add_argument("--output-dir", default="results/dfa_stall_comparison_overview_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        ("faithful", Path(args.faithful_dir)),
        ("norm-matched", Path(args.normmatch_dir)),
    ]
    curves = []
    summaries = []
    for condition, root in runs:
        result_path = root / "dfa_stall_comparison_results.csv"
        summary_path = root / "dfa_stall_comparison_summary.csv"
        if not result_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing comparison outputs under {root}")
        c = pd.read_csv(result_path)
        s = pd.read_csv(summary_path)
        c.insert(0, "condition", condition)
        s.insert(0, "condition", condition)
        curves.append(c)
        summaries.append(s)
    curves_df = pd.concat(curves, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)
    method_summary = summarize(summary_df)
    method_summary.to_csv(output_dir / "dfa_stall_comparison_overview.csv", index=False)
    write_report(method_summary, output_dir)
    make_figure(curves_df, method_summary, output_dir)
    print(method_summary.to_string(index=False))
    print(f"\nwrote {output_dir}")


def summarize(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["condition", "method"], as_index=False)
        .agg(
            final_probe_acc_mean=("final_probe_acc", "mean"),
            final_probe_acc_sem=("final_probe_acc", sem),
            final_loss_mean=("final_loss", "mean"),
            final_loss_sem=("final_loss", sem),
            loss_drop_mean=("loss_drop", "mean"),
            mean_grad_alignment=("mean_grad_alignment", "mean"),
            final_grad_alignment=("final_grad_alignment", "mean"),
            stall_duration_mean=("stall_duration", "mean"),
            n=("seed", "count"),
        )
        .sort_values(["condition", "method"])
    )
    grouped["method_label"] = grouped["method"].map(METHOD_LABEL).fillna(grouped["method"])
    return grouped


def sem(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def write_report(method_summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# DFA-Stall comparison overview",
        "",
        "Faithful preconditioning uses the raw nDFA/K-nDFA step returned by the local second-moment solve.",
        "Norm-matched preconditioning rescales each hidden preconditioned gradient back to the raw-DFA",
        "layerwise norm, isolating the direction of the conditioned update from its step size.",
        "",
        method_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Takeaway: on this clean MNIST/tanh stall setup, the unscaled conditioned steps are not",
        "a direct rescue for the DFA stall. The nDFA direction becomes strongly beneficial once",
        "the hidden-step norm is controlled, whereas K-nDFA remains poorly aligned.",
    ]
    (output_dir / "dfa_stall_comparison_overview.md").write_text("\n".join(lines) + "\n")


def make_figure(curves: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    for col, condition in enumerate(["faithful", "norm-matched"]):
        subset = curves[curves["condition"] == condition]
        for method, method_df in subset.groupby("method", sort=False):
            curve = method_df.groupby("step", as_index=False).agg(
                train_loss=("train_loss", "mean"),
                probe_acc=("probe_acc", "mean"),
                grad_alignment=("grad_alignment", "mean"),
            )
            color = METHOD_COLOR.get(method, "#777777")
            label = METHOD_LABEL.get(method, method)
            axes[0, col].plot(curve["step"], smooth(curve["train_loss"]), color=color, label=label)
            eval_curve = curve.dropna(subset=["probe_acc"])
            axes[1, col].plot(eval_curve["step"], 100.0 * eval_curve["probe_acc"], color=color, marker="o", ms=2.4, label=label)
        axes[0, col].set_title(f"{condition}: loss")
        axes[0, col].set_yscale("log")
        axes[1, col].set_title(f"{condition}: probe accuracy")
        axes[0, col].set_xlabel("step")
        axes[1, col].set_xlabel("step")
    axes[0, 0].set_ylabel("train loss")
    axes[1, 0].set_ylabel("probe accuracy (%)")
    axes[0, 0].legend(frameon=False, ncol=1)

    ax = axes[0, 2]
    x_base = np.arange(2)
    width = 0.23
    for i, method in enumerate(["dfa", "ndfa", "kndfa"]):
        sub = summary[summary["method"] == method].set_index("condition").loc[["faithful", "norm-matched"]]
        ax.bar(
            x_base + (i - 1) * width,
            100.0 * sub["final_probe_acc_mean"].to_numpy(),
            yerr=100.0 * sub["final_probe_acc_sem"].to_numpy(),
            width=width,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
            error_kw={"lw": 0.7, "capsize": 2},
        )
    ax.set_xticks(x_base, ["faithful", "norm-matched"], rotation=15, ha="right")
    ax.set_ylabel("final probe acc. (%)")
    ax.set_title("Final accuracy")

    ax = axes[1, 2]
    for i, method in enumerate(["dfa", "ndfa", "kndfa"]):
        sub = summary[summary["method"] == method].set_index("condition").loc[["faithful", "norm-matched"]]
        ax.bar(
            x_base + (i - 1) * width,
            sub["final_grad_alignment"].to_numpy(),
            width=width,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
        )
    ax.axhline(0.0, color="#555555", lw=0.7)
    ax.set_xticks(x_base, ["faithful", "norm-matched"], rotation=15, ha="right")
    ax.set_ylabel(r"final $\cos(g,g_{\rm BP})$")
    ax.set_title("Alignment")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"dfa_stall_comparison_overview.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
        }
    )


def smooth(values, alpha: float = 0.06) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for idx in range(1, arr.size):
        out[idx] = alpha * arr[idx] + (1.0 - alpha) * out[idx - 1]
    return out


if __name__ == "__main__":
    main()
