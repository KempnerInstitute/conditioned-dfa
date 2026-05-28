"""Aggregate convolutional DFA/K-nDFA benchmark shards."""

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


COLORS = {
    "bp": "#4C78A8",
    "dfa_random": "#59A14F",
    "ndfa_random": "#B07AA1",
    "ndfa_random_kronecker": "#9D755D",
    "drtp_random": "#9D9D9D",
    "local_loss": "#F28E2B",
}
METHODS = ["bp", "dfa_random", "drtp_random", "ndfa_random", "ndfa_random_kronecker", "local_loss"]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs)
    df = ensure_diagnostic_columns(df)
    df.to_csv(output_dir / "dfa_convnet_all.csv", index=False)
    if df.empty:
        write_markdown_report(output_dir / "dfa_convnet_aggregate.md", title="Convnet DFA/K-nDFA Aggregate", sections=[("Status", "_No rows._")])
        return
    final = final_rows(df)
    summary = summarize(final)
    best = best_by_method(summary)
    scores = predictor_scores(
        final,
        target="test_acc",
        predictors=available_columns(
            final,
            [
                "feedback_rank",
                "natural_damping",
                "projected_weight_step",
                "hidden_projected_weight_step",
                "param_cosine",
                "hidden_param_cosine",
                "activity_cosine_mean",
                "kndfa_weight_norm_ratio",
                "kndfa_hidden_weight_norm_ratio",
            ],
        ),
    )
    summary.to_csv(output_dir / "dfa_convnet_summary.csv", index=False)
    best.to_csv(output_dir / "dfa_convnet_best_by_method.csv", index=False)
    scores.to_csv(output_dir / "dfa_convnet_predictors.csv", index=False)
    make_figure(df, final, output_dir, figure_name=args.figure_name)
    if not args.no_mirror:
        mirror_figure(output_dir, figure_name=args.figure_name)
    write_markdown_report(
        output_dir / "dfa_convnet_aggregate.md",
        title="Convnet DFA/K-nDFA Aggregate",
        sections=[
            (
                "Design",
                "Manual stride-conv networks on harder vision tasks. DFA projects output error directly to each convolutional activation. nDFA preconditions conv filters by input-channel covariance; K-nDFA additionally preconditions by local feedback-error channel covariance.",
            ),
            ("Best By Dataset/Method", dataframe_to_markdown(best, float_format=".4f")),
            ("Full Summary", dataframe_to_markdown(summary, float_format=".4f")),
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
    text_cols = {"dataset", "architecture", "method", "feedback_mode", "feedback_normalization", "optimizer"}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def ensure_diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "projected_weight_step",
        "raw_projected_weight_step",
        "hidden_projected_weight_step",
        "raw_hidden_projected_weight_step",
        "param_cosine",
        "hidden_param_cosine",
        "activity_cosine_mean",
        "kndfa_weight_norm_ratio",
        "kndfa_hidden_weight_norm_ratio",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = optional_group_cols(
        df,
        [
            "dataset",
            "method",
            "feedback_mode",
            "feedback_normalization",
            "optimizer",
            "label_noise",
            "test_label_noise",
            "n_train",
            "n_test",
            "lr",
            "weight_decay",
            "gradient_clip",
            "feedback_scale",
            "seed",
            "feedback_seed",
            "feedback_rank",
            "natural_damping",
        ],
    )
    return df.sort_values("epoch").groupby(
        group_cols,
        as_index=False,
    ).tail(1)


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    group_cols = optional_group_cols(
        final,
        [
            "dataset",
            "method",
            "feedback_mode",
            "feedback_normalization",
            "optimizer",
            "label_noise",
            "test_label_noise",
            "n_train",
            "n_test",
            "lr",
            "weight_decay",
            "gradient_clip",
            "feedback_scale",
            "feedback_rank",
            "natural_damping",
        ],
    )
    return (
        final.groupby(group_cols, as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            train_mean=("train_eval_acc", "mean"),
            projected_step=("projected_weight_step", "mean"),
            raw_projected_step=("raw_projected_weight_step", "mean"),
            hidden_projected_step=("hidden_projected_weight_step", "mean"),
            raw_hidden_projected_step=("raw_hidden_projected_weight_step", "mean"),
            param_cosine=("param_cosine", "mean"),
            hidden_param_cosine=("hidden_param_cosine", "mean"),
            activity_cosine=("activity_cosine_mean", "mean"),
            norm_ratio=("kndfa_weight_norm_ratio", "mean"),
            hidden_norm_ratio=("kndfa_hidden_weight_norm_ratio", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["dataset", "method", "feedback_rank", "natural_damping"])
    )


def optional_group_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()]


def best_by_method(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sub in summary.groupby(["dataset", "method"], dropna=False):
        rows.append(sub.loc[sub["test_mean"].idxmax()].to_dict())
    return pd.DataFrame(rows).sort_values(["dataset", "test_mean"], ascending=[True, False])


def make_figure(df: pd.DataFrame, final: pd.DataFrame, output_dir: Path, *, figure_name: str) -> None:
    setup_style()
    datasets = [d for d in ["cifar10", "cifar100", "fashion_mnist", "mnist"] if d in set(df["dataset"])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    best = best_by_method(summarize(final))
    for dataset in datasets:
        sub = best[best["dataset"] == dataset].sort_values("test_mean")
        ax.barh([f"{dataset}:{pretty(m)}" for m in sub["method"]], sub["test_mean"], color=[COLORS.get(m, "0.45") for m in sub["method"]])
    ax.set_xlabel("Best final test accuracy")
    ax.set_title("A  Convnet final performance")

    for ax, dataset in zip(axes[1:3], datasets[:2]):
        sub = df[df["dataset"] == dataset]
        for method in METHODS:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            curve = msub.groupby("epoch", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
            ax.plot(curve["epoch"], curve["test"], color=COLORS.get(method, "0.45"), marker="o", linewidth=1.3, label=pretty(method))
            ax.fill_between(
                curve["epoch"].to_numpy(dtype=float),
                (curve["test"] - curve["sem"]).to_numpy(dtype=float),
                (curve["test"] + curve["sem"]).to_numpy(dtype=float),
                color=COLORS.get(method, "0.45"),
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(f"{chr(66 + datasets.index(dataset))}  {dataset.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
    axes[1].legend(frameon=False, fontsize=6.0)

    ax = axes[3]
    local = final[final["method"] != "bp"]
    step_col = "hidden_projected_weight_step" if "hidden_projected_weight_step" in local.columns else "projected_weight_step"
    ax.scatter(local[step_col], local["test_acc"], s=18, alpha=0.45, c=[COLORS.get(m, "0.4") for m in local["method"]])
    ax.axvline(0.0, color="0.65", linewidth=0.8)
    ax.set_xlabel("Hidden projected BP step")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("D  Useful conv update")

    ax = axes[4]
    for method in ["dfa_random", "ndfa_random", "ndfa_random_kronecker"]:
        sub = final[final["method"] == method]
        if sub.empty:
            continue
        curve = sub.groupby("feedback_rank", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
        ax.errorbar(curve["feedback_rank"], curve["test"], yerr=curve["sem"], marker="o", color=COLORS.get(method, "0.4"), label=pretty(method))
    ax.set_xlabel("Feedback rank")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("E  Rank sweep")
    ax.legend(frameon=False, fontsize=6.0)

    ax = axes[5]
    for method in ["ndfa_random", "ndfa_random_kronecker"]:
        sub = final[final["method"] == method]
        if sub.empty:
            continue
        curve = sub.groupby("natural_damping", as_index=False).agg(test=("test_acc", "mean"), sem=("test_acc", sem))
        ax.errorbar(curve["natural_damping"], curve["test"], yerr=curve["sem"], marker="o", color=COLORS.get(method, "0.4"), label=pretty(method))
    ax.set_xscale("log")
    ax.set_xlabel("Damping")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("F  Damping")
    ax.legend(frameon=False, fontsize=6.0)

    fig.suptitle("Conv-layer DFA and K-nDFA on harder vision tasks", fontsize=10.5)
    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"{figure_name}.{suffix}", **kwargs)
    plt.close(fig)


def mirror_figure(output_dir: Path, *, figure_name: str) -> None:
    dest = ROOT / "drafts" / "Info-DFA" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png", "svg"]:
        src = output_dir / f"{figure_name}.{suffix}"
        if src.exists():
            shutil.copy2(src, dest / src.name)


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


def pretty(method: str) -> str:
    return {
        "bp": "BP",
        "dfa_random": "DFA",
        "drtp_random": "DRTP",
        "ndfa_random": "nDFA",
        "ndfa_random_kronecker": "K-nDFA",
        "local_loss": "Local loss",
    }.get(method, method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/dfa_convnet_aggregate_v1")
    parser.add_argument("--figure-name", default="infodfa_convnet_baselines")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
