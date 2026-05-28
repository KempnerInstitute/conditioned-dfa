"""Aggregate DFA/NMNC comparison shards."""

from __future__ import annotations

import argparse
import glob
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

from infogeo.analysis import dataframe_to_markdown, write_markdown_report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs)
    df.to_csv(output_dir / "dfa_nmnc_all.csv", index=False)
    if df.empty:
        write_markdown_report(output_dir / "dfa_nmnc_aggregate.md", title="DFA/NMNC Aggregate", sections=[("Status", "_No rows._")])
        return
    final = df.sort_values("epoch").groupby(
        optional_group_cols(
            df,
            [
                "dataset",
                "n_train",
                "n_test",
                "label_noise",
                "test_label_noise",
                "method",
                "seed",
                "feedback_seed",
                "feedback_rank",
                "natural_damping",
                "nc_update_interval",
                "nc_manifold_rank",
            ],
        ),
        as_index=False,
    ).tail(1)
    summary = summarize(final)
    summary.to_csv(output_dir / "dfa_nmnc_summary.csv", index=False)
    best = best_by_method(summary)
    best.to_csv(output_dir / "dfa_nmnc_best_by_method.csv", index=False)
    make_figure(summary, final, output_dir)
    mirror_figure(output_dir)
    write_report(df, summary, best, output_dir)
    print(f"Aggregated {len(df)} rows into {output_dir}")


def load_inputs(patterns: list[str]) -> pd.DataFrame:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    frames = []
    for path in sorted(set(paths)):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if col not in {"dataset", "method"}:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                pass
    return df.replace([np.inf, -np.inf], np.nan)


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    group_cols = optional_group_cols(
        final,
        [
            "dataset",
            "n_train",
            "n_test",
            "label_noise",
            "test_label_noise",
            "method",
            "feedback_rank",
            "natural_damping",
            "nc_update_interval",
            "nc_manifold_rank",
        ],
    )
    agg_cols = {
        "test_mean": ("test_acc", "mean"),
        "test_sem": ("test_acc", "sem"),
        "train_mean": ("train_eval_acc", "mean"),
        "activity_angle": ("activity_angle_deg_mean", "mean"),
        "projected_step": ("projected_step_ratio_mean", "mean"),
        "alpha": ("manifold_gradient_alpha_mean", "mean"),
        "dim_fraction": ("manifold_dim_fraction_mean", "mean"),
        "margin": ("manifold_condition_margin_mean", "mean"),
        "feedback_norm": ("feedback_norm", "mean"),
        "n": ("test_acc", "size"),
    }
    for col in [
        "pre_activity_condition_mean",
        "pre_activity_effective_rank_mean",
        "pre_activity_top1_fraction_mean",
        "local_error_condition_mean",
        "local_error_effective_rank_mean",
        "local_error_top1_fraction_mean",
        "bp_error_condition_mean",
        "bp_error_effective_rank_mean",
        "bp_error_top1_fraction_mean",
    ]:
        if col in final.columns:
            agg_cols[col] = (col, "mean")
    return (
        final.groupby(group_cols, as_index=False)
        .agg(**agg_cols)
        .sort_values(["dataset", "method", "nc_manifold_rank", "nc_update_interval"])
    )


def optional_group_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def best_by_method(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = optional_group_cols(summary, ["dataset", "n_train", "n_test", "label_noise", "test_label_noise", "method"])
    for _, sub in summary.groupby(group_cols, dropna=False):
        idx = sub["test_mean"].idxmax()
        rows.append(summary.loc[idx].to_dict())
    return pd.DataFrame(rows).sort_values(["dataset", "test_mean"], ascending=[True, False])


def make_figure(summary: pd.DataFrame, final: pd.DataFrame, output_dir: Path) -> None:
    setup_style()
    datasets = [name for name in ["mnist", "fashion_mnist", "cifar10", "cifar100"] if name in set(summary["dataset"])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    axes = axes.ravel()
    colors = {
        "bp": "#4C78A8",
        "dfa_random": "#59A14F",
        "fa_random": "#76B7B2",
        "ndfa_random": "#B07AA1",
        "ndfa_random_kronecker": "#9D755D",
        "drtp_random": "#9D9D9D",
        "vnc": "#E15759",
        "nmnc": "#F28E2B",
    }

    ax = axes[0]
    best = best_by_method(summary)
    for dataset in datasets:
        sub = best[best["dataset"] == dataset].sort_values("test_mean")
        y = [f"{dataset}:{pretty(m)}" for m in sub["method"]]
        ax.barh(y, sub["test_mean"], color=[colors.get(m, "0.4") for m in sub["method"]])
    ax.set_xlabel("Best final test accuracy")
    ax.set_title("A  Best method/settings")

    for ax, dataset in zip(axes[1:4], datasets):
        sub = summary[(summary["dataset"] == dataset) & (summary["method"].isin(["vnc", "nmnc"]))]
        for method in ["vnc", "nmnc"]:
            for rank, rank_sub in sub[sub["method"] == method].groupby("nc_manifold_rank"):
                rank_sub = rank_sub.sort_values("nc_update_interval")
                label = f"{pretty(method)} d={int(rank)}"
                ax.errorbar(
                    rank_sub["nc_update_interval"],
                    rank_sub["test_mean"],
                    yerr=rank_sub["test_sem"],
                    marker="o",
                    linewidth=1.2,
                    color=colors.get(method, "0.4"),
                    alpha=0.45 + 0.15 * np.log2(max(rank, 1)) / 6.0,
                    label=label,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Feedback update interval b")
        ax.set_ylabel("Final test accuracy")
        ax.set_title(f"{chr(66 + datasets.index(dataset))}  {dataset.replace('_', '-').upper()}")
    if datasets:
        axes[1].legend(frameon=False, fontsize=5.6, ncol=2)

    ax = axes[4]
    nc = final[final["method"].isin(["vnc", "nmnc"])].copy()
    if not nc.empty:
        for method in ["vnc", "nmnc"]:
            sub = nc[nc["method"] == method]
            ax.scatter(sub["manifold_condition_margin_mean"], sub["test_acc"], s=18, alpha=0.45, color=colors[method], label=pretty(method))
    ax.axvline(0, color="0.65", linewidth=0.8)
    ax.set_xlabel(r"Manifold-gradient margin $\alpha-d/n$")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("E  Manifold condition")
    ax.legend(frameon=False, fontsize=6.2)

    ax = axes[5]
    if not nc.empty:
        for method in ["vnc", "nmnc"]:
            sub = nc[nc["method"] == method]
            ax.scatter(sub["projected_step_ratio_mean"], sub["test_acc"], s=18, alpha=0.45, color=colors[method], label=pretty(method))
    ax.axvline(0, color="0.65", linewidth=0.8)
    ax.set_xlabel("Projected BP-step ratio")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("F  Useful step size")

    fig.suptitle("Info-DFA external benchmark: vanilla vs neural-manifold noise correlation", fontsize=10.5)
    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"infodfa_nmnc_comparison.{suffix}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> None:
    sections = [
        (
            "Design",
            "Aggregates BP, DFA/nDFA/DRTP, vanilla noise correlation (VNC), and neural-manifold noise correlation (NMNC). Rows are grouped by dataset, method, feedback update interval, and PCA manifold rank.",
        ),
        ("Best By Dataset/Method", dataframe_to_markdown(best, float_format=".4f")),
        ("Full Summary", dataframe_to_markdown(summary, float_format=".4f")),
    ]
    write_markdown_report(output_dir / "dfa_nmnc_aggregate.md", title="DFA/NMNC Aggregate", sections=sections)


def mirror_figure(output_dir: Path) -> None:
    dest = ROOT / "drafts" / "Info-DFA" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png", "svg"]:
        src = output_dir / f"infodfa_nmnc_comparison.{suffix}"
        if src.exists():
            shutil.copy2(src, dest / src.name)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def pretty(method: str) -> str:
    return {
        "bp": "BP",
        "dfa_random": "DFA",
        "fa_random": "FA",
        "ndfa_random": "nDFA",
        "ndfa_random_kronecker": "K-nDFA",
        "drtp_random": "DRTP",
        "vnc": "VNC",
        "nmnc": "NMNC",
    }.get(method, method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/dfa_nmnc_aggregate_v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
