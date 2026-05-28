"""Aggregate DFA-to-K-nDFA preconditioning spectrum shards."""

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

from infogeo.analysis import dataframe_to_markdown, predictor_scores, write_markdown_report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs)
    df.to_csv(output_dir / "dfa_preconditioning_spectrum_all.csv", index=False)
    if df.empty:
        write_markdown_report(output_dir / "dfa_preconditioning_spectrum_aggregate.md", title="DFA to K-nDFA Spectrum", sections=[("Status", "_No rows._")])
        return
    final = final_rows(df)
    summary = summarize(final)
    summary.to_csv(output_dir / "dfa_preconditioning_spectrum_summary.csv", index=False)
    scores = predictor_scores(
        final,
        target="test_acc",
        predictors=["gamma", "damping", "feedback_rank", "blended_weight_step", "blended_param_cosine", "kndfa_weight_norm_ratio"],
    )
    scores.to_csv(output_dir / "dfa_preconditioning_spectrum_predictors.csv", index=False)
    make_figure(df, final, output_dir)
    mirror_figure(output_dir)
    write_markdown_report(
        output_dir / "dfa_preconditioning_spectrum_aggregate.md",
        title="DFA to K-nDFA Preconditioning Spectrum",
        sections=[
            ("Design", "Gamma=0 is raw DFA. Gamma=1 is damped K-nDFA. Intermediate values blend the two weight updates."),
            ("Summary", dataframe_to_markdown(summary, float_format=".4f")),
            ("Predictors", dataframe_to_markdown(scores, float_format=".4f")),
        ],
    )
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
        if col not in {"condition", "method"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("epoch").groupby(
        ["condition", "seed", "feedback_seed", "feedback_rank", "gamma", "damping"],
        as_index=False,
    ).tail(1)


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    return (
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


def make_figure(df: pd.DataFrame, final: pd.DataFrame, output_dir: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.2), constrained_layout=True)
    axes = axes.ravel()
    conditions = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
    colors = {
        "task_aligned": "#4C78A8",
        "nuisance_dominant": "#E15759",
        "mixed_context": "#F28E2B",
        "low_sample_noisy": "#59A14F",
    }

    for condition in conditions:
        sub = final[(final["condition"] == condition) & (final["feedback_rank"] == 0)]
        if sub.empty:
            continue
        curve = sub.groupby("gamma", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
        axes[0].errorbar(curve["gamma"], curve["test"], yerr=curve["sem"], marker="o", color=colors[condition], label=pretty_condition(condition))
    axes[0].set_title("A  Turning on K-nDFA")
    axes[0].set_xlabel(r"Preconditioning blend $\gamma$")
    axes[0].set_ylabel("Final test accuracy")
    axes[0].legend(frameon=False, fontsize=6.0)

    for ax, condition in zip(axes[1:5], conditions):
        sub = df[(df["condition"] == condition) & (df["feedback_rank"] == 0)]
        for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gsub = sub[np.isclose(sub["gamma"], gamma)]
            if gsub.empty:
                continue
            curve = gsub.groupby("epoch", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
            ax.plot(curve["epoch"], curve["test"], marker="o", linewidth=1.2, label=fr"$\gamma={gamma:g}$")
        ax.set_title(f"{chr(66 + conditions.index(condition))}  {pretty_condition(condition)}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False, fontsize=5.6, ncol=2)

    ax = axes[5]
    for condition in conditions:
        sub = final[(final["condition"] == condition) & (final["feedback_rank"] == 0)]
        if sub.empty:
            continue
        curve = sub.groupby("gamma", as_index=False).agg(step=("blended_weight_step", "mean"))
        ax.plot(curve["gamma"], curve["step"], marker="o", color=colors[condition], label=pretty_condition(condition))
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_title("F  Projected weight-step")
    ax.set_xlabel(r"Preconditioning blend $\gamma$")
    ax.set_ylabel("Projection onto BP update")

    fig.suptitle("A continuous spectrum from DFA to K-nDFA", fontsize=10.5)
    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"infodfa_preconditioning_spectrum.{suffix}", **kwargs)
    plt.close(fig)


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
        src = output_dir / f"infodfa_preconditioning_spectrum.{suffix}"
        if src.exists():
            shutil.copy2(src, dest / src.name)


def pretty_condition(condition: str) -> str:
    return {
        "task_aligned": "Task-aligned",
        "nuisance_dominant": "Nuisance-dominant",
        "mixed_context": "Mixed context",
        "low_sample_noisy": "Low-sample/noisy",
    }.get(condition, condition.replace("_", " ").title())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/dfa_preconditioning_spectrum_aggregate_v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
