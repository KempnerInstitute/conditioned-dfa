"""Aggregate multi-output DFA synthetic stress-test shards."""

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


METHOD_ORDER = ["bp", "dfa_random", "fa_random", "ndfa_random", "ndfa_random_kronecker", "drtp_random", "vnc", "nmnc"]
COLORS = {
    "bp": "#4C78A8",
    "dfa_random": "#59A14F",
    "fa_random": "#76B7B2",
    "ndfa_random": "#B07AA1",
    "ndfa_random_kronecker": "#9D755D",
    "drtp_random": "#9D9D9D",
    "vnc": "#E15759",
    "nmnc": "#F28E2B",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs)
    df.to_csv(output_dir / "dfa_multioutput_all.csv", index=False)
    if df.empty:
        write_markdown_report(
            output_dir / "dfa_multioutput_aggregate.md",
            title="DFA Multi-Output Synthetic Aggregate",
            sections=[("Status", "_No rows._")],
        )
        return

    final = add_diagnostic_columns(final_rows(df))
    summary = summarize(final)
    best = best_by_condition_method(summary)
    elbows = rank_elbows(summary)
    diagnostics = diagnostic_tests(final)
    scores = predictor_scores(
        final,
        target="test_acc",
        predictors=[
            "feedback_rank",
            "effective_feedback_rank",
            "rank_fraction_of_error_space",
            "task_energy_fraction",
            "task_weighted_projected_step",
            "task_weighted_manifold_margin",
            "task_weighted_margin_step",
            "activity_angle_deg_mean",
            "projected_step_ratio_mean",
            "manifold_gradient_alpha_mean",
            "manifold_condition_margin_mean",
            "feedback_update_norm",
        ],
    )
    final.to_csv(output_dir / "dfa_multioutput_final_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "dfa_multioutput_summary.csv", index=False)
    best.to_csv(output_dir / "dfa_multioutput_best_by_method.csv", index=False)
    elbows.to_csv(output_dir / "dfa_multioutput_rank_elbows.csv", index=False)
    diagnostics.to_csv(output_dir / "dfa_multioutput_diagnostic_tests.csv", index=False)
    scores.to_csv(output_dir / "dfa_multioutput_predictors.csv", index=False)
    make_figure(summary, final, output_dir)
    make_diagnostic_figure(summary, final, elbows, output_dir)
    mirror_figure(output_dir)
    write_report(summary, best, elbows, diagnostics, scores, output_dir)
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
    text_cols = {"condition", "method"}
    for col in df.columns:
        if col not in text_cols:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                pass
    return df.replace([np.inf, -np.inf], np.nan)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = optional_group_cols(
        df,
        [
            "condition",
            "input_noise",
            "n_train",
            "n_test",
            "train_label_noise",
            "test_label_noise",
            "task_scale",
            "nuisance_scale",
            "nuisance_dim",
            "method",
            "seed",
            "feedback_seed",
            "feedback_rank",
            "nc_update_interval",
            "nc_manifold_rank",
        ],
    )
    return df.sort_values("epoch").groupby(group_cols, as_index=False).tail(1).copy()


def optional_group_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_diagnostic_columns(final: pd.DataFrame) -> pd.DataFrame:
    final = final.copy()
    n_classes = final.get("n_classes", pd.Series(np.nan, index=final.index)).astype(float)
    max_error_rank = np.maximum(n_classes - 1.0, 1.0)
    rank = final.get("feedback_rank", pd.Series(np.nan, index=final.index)).astype(float)
    effective_rank = np.where(rank <= 0, max_error_rank, np.minimum(rank, max_error_rank))
    final["effective_feedback_rank"] = effective_rank
    final["max_error_rank"] = max_error_rank
    final["rank_fraction_of_error_space"] = effective_rank / max_error_rank

    task_scale = final.get("task_scale", pd.Series(np.nan, index=final.index)).astype(float)
    nuisance_scale = final.get("nuisance_scale", pd.Series(np.nan, index=final.index)).astype(float)
    nuisance_dim = final.get("nuisance_dim", pd.Series(np.nan, index=final.index)).astype(float)
    task_energy = 4.0 * task_scale * task_scale
    nuisance_energy = nuisance_dim * nuisance_scale * nuisance_scale
    final["task_energy_fraction"] = task_energy / np.maximum(task_energy + nuisance_energy, 1e-12)
    final["nuisance_energy_ratio"] = nuisance_energy / np.maximum(task_energy, 1e-12)

    final["rank_sufficient_full_error"] = final["rank_fraction_of_error_space"] >= 0.999
    final["positive_manifold_margin"] = final["manifold_condition_margin_mean"] > 0.0
    final["useful_projected_step"] = final["projected_step_ratio_mean"] > 0.1
    final["nuisance_dominant_design"] = final["task_energy_fraction"] < 0.1
    final["positive_projected_step"] = final["projected_step_ratio_mean"].clip(lower=0.0)
    final["positive_manifold_margin_value"] = final["manifold_condition_margin_mean"].clip(lower=0.0)
    final["task_weighted_projected_step"] = final["task_energy_fraction"] * final["positive_projected_step"]
    final["task_weighted_manifold_margin"] = final["task_energy_fraction"] * final["positive_manifold_margin_value"]
    final["task_weighted_margin_step"] = (
        final["task_energy_fraction"] * final["positive_manifold_margin_value"] * final["positive_projected_step"]
    )
    return final


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    group_cols = optional_group_cols(
        final,
        [
            "condition",
            "input_noise",
            "n_train",
            "n_test",
            "train_label_noise",
            "test_label_noise",
            "task_scale",
            "nuisance_scale",
            "nuisance_dim",
            "method",
            "feedback_rank",
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
        "update_norm": ("feedback_update_norm", "mean"),
        "effective_feedback_rank": ("effective_feedback_rank", "mean"),
        "rank_fraction": ("rank_fraction_of_error_space", "mean"),
        "task_energy_fraction": ("task_energy_fraction", "mean"),
        "nuisance_energy_ratio": ("nuisance_energy_ratio", "mean"),
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
        .sort_values(["condition", "method", "feedback_rank", "nc_update_interval"])
    )


def best_by_condition_method(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = optional_group_cols(
        summary,
        [
            "condition",
            "input_noise",
            "n_train",
            "n_test",
            "train_label_noise",
            "test_label_noise",
            "task_scale",
            "nuisance_scale",
            "nuisance_dim",
            "method",
        ],
    )
    for _, sub in summary.groupby(group_cols, dropna=False):
        rows.append(sub.loc[sub["test_mean"].idxmax()].to_dict())
    return pd.DataFrame(rows).sort_values(["condition", "test_mean"], ascending=[True, False])


def rank_elbows(summary: pd.DataFrame, *, fraction_of_best: float = 0.95) -> pd.DataFrame:
    rows = []
    local = summary[summary["method"] != "bp"].copy()
    if local.empty:
        return pd.DataFrame()
    group_cols = optional_group_cols(
        local,
        [
            "condition",
            "input_noise",
            "n_train",
            "n_test",
            "train_label_noise",
            "test_label_noise",
            "task_scale",
            "nuisance_scale",
            "nuisance_dim",
            "method",
        ],
    )
    for key, sub in local.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        context = dict(zip(group_cols, key))
        by_rank = (
            sub.groupby("effective_feedback_rank", as_index=False)
            .agg(test_mean=("test_mean", "mean"), n=("n", "sum"))
            .sort_values("effective_feedback_rank")
        )
        best = float(by_rank["test_mean"].max())
        threshold = fraction_of_best * best
        above = by_rank[by_rank["test_mean"] >= threshold]
        elbow = float(above["effective_feedback_rank"].iloc[0]) if not above.empty else float("nan")
        rows.append(
            {
                **context,
                "best_test": best,
                "threshold": threshold,
                "rank_elbow_95pct_best": elbow,
                "min_rank_test": float(by_rank["test_mean"].iloc[0]),
                "full_rank_test": float(by_rank["test_mean"].iloc[-1]),
                "rank_gain": float(by_rank["test_mean"].iloc[-1] - by_rank["test_mean"].iloc[0]),
                "n": int(by_rank["n"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "rank_elbow_95pct_best", "method"])


def diagnostic_tests(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    local = final[final["method"] != "bp"].copy()
    if local.empty:
        return pd.DataFrame()

    rows.extend(binary_effect(local, "rank_sufficient_full_error", "rank sufficient for full output-error space"))
    rows.extend(binary_effect(local, "positive_manifold_margin", "manifold-gradient margin > 0"))
    rows.extend(binary_effect(local, "useful_projected_step", "projected BP-step ratio > 0.1"))
    rows.extend(binary_effect(local, "nuisance_dominant_design", "nuisance energy dominates design", positive_is_good=False))

    for predictor in [
        "rank_fraction_of_error_space",
        "task_energy_fraction",
        "task_weighted_projected_step",
        "task_weighted_manifold_margin",
        "task_weighted_margin_step",
        "projected_step_ratio_mean",
        "manifold_condition_margin_mean",
        "activity_angle_deg_mean",
    ]:
        subset = local[["test_acc", predictor]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset.shape[0] < 3:
            continue
        x = subset[predictor].to_numpy(dtype=float)
        y = subset["test_acc"].to_numpy(dtype=float)
        rows.append(
            {
                "test": f"continuous predictor: {predictor}",
                "n_low": np.nan,
                "n_high": int(subset.shape[0]),
                "mean_low": np.nan,
                "mean_high": np.nan,
                "delta_high_minus_low": np.nan,
                "pearson_r": safe_corr(x, y),
                "interpretation": "positive means larger diagnostic predicts better accuracy",
            }
        )
        residual = residualized_corr(local, predictor, group_cols=["condition", "method"])
        if np.isfinite(residual):
            rows.append(
                {
                    "test": f"within condition/method predictor: {predictor}",
                    "n_low": np.nan,
                    "n_high": int(subset.shape[0]),
                    "mean_low": np.nan,
                    "mean_high": np.nan,
                    "delta_high_minus_low": np.nan,
                    "pearson_r": residual,
                    "interpretation": "correlation after removing condition and method means",
                }
            )
    return pd.DataFrame(rows)


def binary_effect(df: pd.DataFrame, column: str, label: str, *, positive_is_good: bool = True) -> list[dict[str, float | str]]:
    subset = df[["test_acc", column]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty or subset[column].nunique() < 2:
        return []
    high = subset[subset[column].astype(bool)]["test_acc"].astype(float)
    low = subset[~subset[column].astype(bool)]["test_acc"].astype(float)
    delta = float(high.mean() - low.mean())
    expected = "positive" if positive_is_good else "negative"
    return [
        {
            "test": label,
            "n_low": int(low.shape[0]),
            "n_high": int(high.shape[0]),
            "mean_low": float(low.mean()),
            "mean_high": float(high.mean()),
            "delta_high_minus_low": delta,
            "pearson_r": np.nan,
            "interpretation": f"expected {expected} delta",
        }
    ]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residualized_corr(df: pd.DataFrame, predictor: str, *, group_cols: list[str]) -> float:
    subset = df[group_cols + ["test_acc", predictor]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if subset.shape[0] < 3:
        return float("nan")
    subset["target_residual"] = subset["test_acc"] - subset.groupby(group_cols)["test_acc"].transform("mean")
    subset["predictor_residual"] = subset[predictor] - subset.groupby(group_cols)[predictor].transform("mean")
    return safe_corr(subset["predictor_residual"].to_numpy(dtype=float), subset["target_residual"].to_numpy(dtype=float))


def make_figure(summary: pd.DataFrame, final: pd.DataFrame, output_dir: Path) -> None:
    setup_style()
    conditions = [c for c in ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"] if c in set(summary["condition"])]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.7), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    best = best_by_condition_method(summary)
    for condition in conditions:
        sub = best[best["condition"] == condition].copy()
        sub["method_order"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
        sub = sub.sort_values(["test_mean", "method_order"])
        labels = [f"{short_condition(condition)}:{pretty(m)}" for m in sub["method"]]
        ax.barh(labels, sub["test_mean"], color=[COLORS.get(m, "0.45") for m in sub["method"]], alpha=0.92)
    ax.set_xlabel("Best final test accuracy")
    ax.set_title("A  Best method by stress condition")

    rank_methods = ["dfa_random", "ndfa_random", "ndfa_random_kronecker", "drtp_random", "vnc", "nmnc"]
    for ax, condition in zip(axes[1:5], conditions):
        sub = summary[(summary["condition"] == condition) & (summary["method"].isin(rank_methods))]
        for method in rank_methods:
            method_sub = sub[sub["method"] == method]
            if method_sub.empty:
                continue
            curve = method_sub.groupby("feedback_rank", as_index=False).agg(
                test_mean=("test_mean", "mean"),
                test_sem=("test_sem", "mean"),
            )
            ax.errorbar(
                curve["feedback_rank"],
                curve["test_mean"],
                yerr=curve["test_sem"],
                marker="o",
                linewidth=1.1,
                color=COLORS.get(method, "0.45"),
                label=pretty(method),
            )
        bp = summary[(summary["condition"] == condition) & (summary["method"] == "bp")]
        if not bp.empty:
            ax.axhline(float(bp["test_mean"].max()), color=COLORS["bp"], linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_xlabel("Feedback rank")
        ax.set_ylabel("Final test accuracy")
        ax.set_title(f"{chr(66 + conditions.index(condition))}  {pretty_condition(condition)}")
        ax.set_ylim(0.0, 1.02)
    if len(conditions) >= 1:
        axes[1].legend(frameon=False, fontsize=5.7, ncol=2)

    ax = axes[5]
    nc = final[final["method"].isin(["vnc", "nmnc"])]
    if not nc.empty:
        for method in ["vnc", "nmnc"]:
            sub = nc[nc["method"] == method]
            ax.scatter(
                sub["manifold_condition_margin_mean"],
                sub["test_acc"],
                s=18,
                alpha=0.45,
                color=COLORS[method],
                label=pretty(method),
            )
    ax.axvline(0.0, color="0.65", linewidth=0.8)
    ax.set_xlabel(r"Manifold-gradient margin $\alpha-d/n$")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("F  When manifold restriction should help")
    ax.legend(frameon=False, fontsize=6.2)

    fig.suptitle("Info-DFA stress test: multi-output tasks with nuisance manifolds", fontsize=10.5)
    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"infodfa_multioutput_synthetic.{suffix}", **kwargs)
    plt.close(fig)


def make_diagnostic_figure(summary: pd.DataFrame, final: pd.DataFrame, elbows: pd.DataFrame, output_dir: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.3), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    if not elbows.empty:
        methods = [m for m in METHOD_ORDER if m in set(elbows["method"])]
        x_positions = {m: i for i, m in enumerate(methods)}
        for condition, sub in elbows.groupby("condition"):
            xs = [x_positions[m] + condition_offset(condition) for m in sub["method"]]
            ax.scatter(xs, sub["rank_elbow_95pct_best"], s=30, label=pretty_condition(condition), alpha=0.75)
        ax.set_xticks(list(x_positions.values()), [pretty(m) for m in methods], rotation=35, ha="right")
    ax.set_ylabel("Smallest rank reaching 95% of best")
    ax.set_title("A  Rank sufficiency is measurable")
    ax.legend(frameon=False, fontsize=5.7, ncol=2)

    ax = axes[1]
    local = final[final["method"] != "bp"]
    for method in [m for m in METHOD_ORDER if m in set(local["method"])]:
        sub = local[local["method"] == method]
        ax.scatter(
            sub["projected_step_ratio_mean"],
            sub["test_acc"],
            s=16,
            color=COLORS.get(method, "0.45"),
            alpha=0.45,
            label=pretty(method),
        )
    ax.axvline(0.1, color="0.65", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Projected BP-step ratio")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("B  Cosine is not enough: useful step")

    ax = axes[2]
    for method in [m for m in METHOD_ORDER if m in set(local["method"])]:
        sub = local[local["method"] == method]
        ax.scatter(
            sub["task_energy_fraction"],
            sub["test_acc"],
            s=16,
            color=COLORS.get(method, "0.45"),
            alpha=0.45,
            label=pretty(method),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Task energy / total latent energy")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("C  Nuisance-dominant geometry is a failure mode")

    ax = axes[3]
    if "manifold_condition_margin_mean" in local:
        for method in ["vnc", "nmnc", "ndfa_random", "ndfa_random_kronecker"]:
            sub = local[local["method"] == method]
            if sub.empty:
                continue
            ax.scatter(
                sub["manifold_condition_margin_mean"],
                sub["test_acc"],
                s=16,
                color=COLORS.get(method, "0.45"),
                alpha=0.45,
                label=pretty(method),
            )
    ax.axvline(0.0, color="0.65", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"Manifold-gradient margin $\alpha-d/n$")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("D  Manifold overlap predicts when geometry helps")
    handles, labels = axes[3].get_legend_handles_labels()
    if handles:
        axes[3].legend(frameon=False, fontsize=5.7, ncol=2)

    fig.suptitle("Quantitative success and failure conditions for Info-DFA", fontsize=10.5)
    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"infodfa_multioutput_diagnostics.{suffix}", **kwargs)
    plt.close(fig)


def condition_offset(condition: str) -> float:
    order = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
    if condition not in order:
        return 0.0
    return (order.index(condition) - 1.5) * 0.12


def write_report(
    summary: pd.DataFrame,
    best: pd.DataFrame,
    elbows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    output_dir: Path,
) -> None:
    sections = [
        (
            "Design",
            "Multi-output synthetic tasks use eight classes on a latent circle plus high-dimensional nuisance factors. The benchmark tests cases where the task manifold is aligned, where nuisance variance dominates unsupervised geometry, where context changes the label mapping, and where sampling/noise make covariance estimates harder.",
        ),
        (
            "Expected Win Conditions",
            "Geometry-aware local rules should help when the task-relevant error is low-rank, feedback rank is at least the task rank, hidden covariance is anisotropic but estimable, and the neural manifold contains a large projected BP-gradient component. They should not help when PCA is dominated by nuisance variation, covariance estimates are noisy, or the required error is effectively full-rank.",
        ),
        (
            "Quantified Conditions",
            "The aggregate now reports rank elbows, task-energy fraction, manifold-gradient margin, and projected BP-step ratio. These are intended as falsifiable success/failure diagnostics rather than post-hoc labels.",
        ),
        ("Best By Condition/Method", dataframe_to_markdown(best, float_format=".4f")),
        ("Rank Elbows", dataframe_to_markdown(elbows, float_format=".4f")),
        ("Diagnostic Tests", dataframe_to_markdown(diagnostics, float_format=".4f")),
        ("Full Summary", dataframe_to_markdown(summary, float_format=".4f")),
        ("Predictors", dataframe_to_markdown(scores, float_format=".4f")),
    ]
    write_markdown_report(output_dir / "dfa_multioutput_aggregate.md", title="DFA Multi-Output Synthetic Aggregate", sections=sections)


def mirror_figure(output_dir: Path) -> None:
    dest = ROOT / "drafts" / "Info-DFA" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png", "svg"]:
        src = output_dir / f"infodfa_multioutput_synthetic.{suffix}"
        if src.exists():
            shutil.copy2(src, dest / src.name)
        diagnostic_src = output_dir / f"infodfa_multioutput_diagnostics.{suffix}"
        if diagnostic_src.exists():
            shutil.copy2(diagnostic_src, dest / diagnostic_src.name)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
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


def pretty_condition(condition: str) -> str:
    return {
        "task_aligned": "Task-aligned",
        "nuisance_dominant": "Nuisance-dominant",
        "mixed_context": "Mixed context",
        "low_sample_noisy": "Low-sample/noisy",
    }.get(condition, condition.replace("_", " ").title())


def short_condition(condition: str) -> str:
    return {
        "task_aligned": "aligned",
        "nuisance_dominant": "nuis",
        "mixed_context": "context",
        "low_sample_noisy": "noisy",
    }.get(condition, condition)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/dfa_multioutput_synthetic_aggregate_v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
