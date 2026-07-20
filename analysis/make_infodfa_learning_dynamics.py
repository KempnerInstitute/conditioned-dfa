"""Make Conditioned DFA learning-dynamics figures from completed aggregate CSVs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHODS = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc", "drtp_random"]
COLORS = {
    "bp": "#4C78A8",
    "dfa_random": "#59A14F",
    "ndfa_random": "#B07AA1",
    "ndfa_random_kronecker": "#9D755D",
    "vnc": "#E15759",
    "nmnc": "#F28E2B",
    "drtp_random": "#9D9D9D",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    multi = load_csv(args.multioutput_csv)
    nmnc = load_csv(args.nmnc_csv)
    make_learning_figure(multi, nmnc, output_dir)
    mirror_figure(output_dir)
    print(f"Saved learning dynamics to {output_dir}")


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {"condition", "dataset", "method"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def make_learning_figure(multi: pd.DataFrame, nmnc: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.8, 6.2), constrained_layout=True)
    axes = axes.ravel()

    conditions = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        sub = multi[multi["condition"] == condition]
        plot_curves(ax, sub, group_cols=["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval"], y_col="test_acc")
        ax.set_title(f"{chr(65 + idx)}  {pretty_condition(condition)}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)

    datasets = ["mnist", "fashion_mnist", "cifar10"]
    for jdx, dataset in enumerate(datasets):
        ax = axes[4 + jdx]
        sub = nmnc[nmnc["dataset"] == dataset]
        plot_curves(ax, sub, group_cols=["dataset", "method", "seed", "feedback_seed", "nc_update_interval", "nc_manifold_rank"], y_col="test_acc")
        ax.set_title(f"{chr(69 + jdx)}  {dataset.replace('_', '-').upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)

    ax = axes[7]
    for method in ["dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc"]:
        sub = multi[multi["method"] == method]
        if sub.empty:
            continue
        run = run_means(sub, ["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval"], "projected_step_ratio_mean")
        curve = run.groupby("epoch", as_index=False).agg(mean=("value", "mean"), sem=("value", sem))
        ax.plot(curve["epoch"], curve["mean"], color=COLORS.get(method, "0.4"), linewidth=1.5, label=pretty(method))
        ax.fill_between(
            curve["epoch"].to_numpy(dtype=float),
            (curve["mean"] - curve["sem"]).to_numpy(dtype=float),
            (curve["mean"] + curve["sem"]).to_numpy(dtype=float),
            color=COLORS.get(method, "0.4"),
            alpha=0.16,
            linewidth=0,
        )
    ax.axhline(0.1, color="0.55", linestyle="--", linewidth=0.8)
    ax.set_title("H  Useful projected step")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Projected BP-step ratio")
    ax.legend(frameon=False, fontsize=5.9, ncol=2)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Conditioned DFA learning dynamics: speed, stability, and useful local descent", fontsize=10.5, y=1.05)
    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"infodfa_learning_dynamics.{suffix}", **kwargs)
    plt.close(fig)


def plot_curves(ax, df: pd.DataFrame, *, group_cols: list[str], y_col: str) -> None:
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        run = run_means(sub, group_cols, y_col)
        curve = run.groupby("epoch", as_index=False).agg(mean=("value", "mean"), sem=("value", sem))
        ax.plot(curve["epoch"], curve["mean"], color=COLORS.get(method, "0.4"), linewidth=1.55, label=pretty(method))
        ax.fill_between(
            curve["epoch"].to_numpy(dtype=float),
            (curve["mean"] - curve["sem"]).to_numpy(dtype=float),
            (curve["mean"] + curve["sem"]).to_numpy(dtype=float),
            color=COLORS.get(method, "0.4"),
            alpha=0.13,
            linewidth=0,
        )


def run_means(df: pd.DataFrame, group_cols: list[str], y_col: str) -> pd.DataFrame:
    cols = [col for col in group_cols if col in df.columns]
    out = df[cols + ["epoch", y_col]].replace([np.inf, -np.inf], np.nan).dropna(subset=["epoch", y_col])
    out = out.groupby(cols + ["epoch"], as_index=False).agg(value=(y_col, "mean"))
    return out


def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.shape[0] <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.shape[0]))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def mirror_figure(output_dir: Path) -> None:
    dest = ROOT / "drafts" / "Info-DFA" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png", "svg"]:
        src = output_dir / f"infodfa_learning_dynamics.{suffix}"
        if src.exists():
            shutil.copy2(src, dest / src.name)


def pretty(method: str) -> str:
    return {
        "bp": "BP",
        "dfa_random": "DFA",
        "ndfa_random": "nDFA",
        "ndfa_random_kronecker": "K-nDFA",
        "drtp_random": "DRTP",
        "vnc": "VNC",
        "nmnc": "NMNC",
    }.get(method, method)


def pretty_condition(condition: str) -> str:
    return {
        "task_aligned": "Task-aligned",
        "nuisance_dominant": "Nuisance-dominant",
        "mixed_context": "Mixed context",
        "low_sample_noisy": "Low-sample/noisy",
    }.get(condition, condition.replace("_", " ").title())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multioutput-csv", default="results/dfa_multioutput_synthetic_aggregate_v1/dfa_multioutput_all.csv")
    parser.add_argument("--nmnc-csv", default="results/dfa_nmnc_aggregate_v1/dfa_nmnc_all.csv")
    parser.add_argument("--output-dir", default="results/infodfa_learning_dynamics_v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
