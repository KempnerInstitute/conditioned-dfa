"""Make the main-text figure from the original confirmation cohorts.

The appendix analyses may contain later extension seeds. Those seeds must not
silently change this figure's original five/five/eight-seed comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "error_kndfa_replication_figure_v1"
PAPER_FIGURES = ROOT / "drafts" / "Info-DFA" / "figures"
STEM = "iclr_fig_error_kndfa_replication"
COLORS = {"dfa": "#7F7F7F", "ndfa": "#009E73", "endfa": "#D55E00", "kndfa": "#6A3D9A"}
LABELS = {"dfa": "DFA", "ndfa": "activity nDFA", "endfa": "error nDFA", "kndfa": "K-nDFA"}
COHORTS = {
    "tanh MNIST": tuple(range(50, 55)),
    "tanh Fashion": tuple(range(70, 75)),
    "ReLU MNIST": tuple(range(100, 108)),
}
ANALYSIS_DIRS = {
    "tanh MNIST": "dfa_stall_threefactor_analysis_v1",
    "tanh Fashion": "dfa_stall_fashion_threefactor_analysis_v1",
    "ReLU MNIST": "dfa_relu_mnist_threefactor_analysis_v1",
}
FEEDBACK_SEEDS = (0, 1, 2)


def original_cohort(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Select explicit seed IDs and reject missing or duplicated method pairs."""
    selected = frame[
        frame["seed"].isin(COHORTS[label]) & frame["method"].isin(COLORS)
    ].copy()
    expected = pd.MultiIndex.from_product(
        [COHORTS[label], tuple(COLORS)], names=["seed", "method"]
    )
    actual = pd.MultiIndex.from_frame(selected[["seed", "method"]])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError(f"{label}: incomplete or duplicated original seed/method pairs")
    if not selected["n_feedback_seeds"].eq(len(FEEDBACK_SEEDS)).all():
        raise ValueError(f"{label}: every original pair must average three feedback seeds")
    if not np.isfinite(selected[["test_acc", "test_loss"]].to_numpy()).all():
        raise ValueError(f"{label}: nonfinite original test endpoints")
    return selected.sort_values(["seed", "method"]).reset_index(drop=True)


def load_seed_means() -> dict[str, pd.DataFrame]:
    frames = {}
    for label, directory in ANALYSIS_DIRS.items():
        path = RESULTS / directory / "confirmation_seed_means.csv"
        frame = pd.read_csv(path)
        frames[label] = original_cohort(frame, label)
        runs = pd.read_csv(RESULTS / directory / "confirmation_runs.csv")
        runs = runs[runs["seed"].isin(COHORTS[label]) & runs["method"].isin(COLORS)]
        keys = ["seed", "method", "feedback_seed"]
        actual = pd.MultiIndex.from_frame(runs[keys])
        expected = pd.MultiIndex.from_product(
            [COHORTS[label], tuple(COLORS), FEEDBACK_SEEDS], names=keys
        )
        if actual.has_duplicates or set(actual) != set(expected):
            raise ValueError(f"{label}: original runs must cover feedback seeds 0, 1, 2 exactly")
        columns = ["test_acc", "test_loss"] if label == "ReLU MNIST" else ["final_test_acc", "final_test_loss"]
        recomputed = runs.groupby(["seed", "method"])[columns].mean()
        saved = frames[label].set_index(["seed", "method"])[["test_acc", "test_loss"]]
        if not np.allclose(recomputed.to_numpy(), saved.to_numpy(), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{label}: saved seed means disagree with the original runs")
    return frames


def load_original_curves() -> pd.DataFrame:
    curves = pd.read_csv(RESULTS / ANALYSIS_DIRS["ReLU MNIST"] / "confirmation_curves.csv")
    curves = curves[
        curves["seed"].isin(COHORTS["ReLU MNIST"]) & curves["method"].isin(COLORS)
    ].copy()
    keys = ["seed", "method", "step", "feedback_seed"]
    if curves.empty or curves.duplicated(keys).any():
        raise ValueError("ReLU MNIST: missing or duplicated original validation curves")
    expected = pd.MultiIndex.from_product(
        [COHORTS["ReLU MNIST"], tuple(COLORS), sorted(curves["step"].unique()), FEEDBACK_SEEDS],
        names=keys,
    )
    actual = pd.MultiIndex.from_frame(curves[keys])
    if set(actual) != set(expected) or not np.isfinite(curves["val_acc"]).all():
        raise ValueError("ReLU MNIST: incomplete original validation curves or nonfinite values")
    return curves.groupby(["seed", "method", "step"], as_index=False)["val_acc"].mean()


def paired_deltas(frames: dict[str, pd.DataFrame], method: str, reference: str) -> dict[str, pd.Series]:
    out = {}
    for label, frame in frames.items():
        wide = frame.pivot(index="seed", columns="method", values="test_acc")
        out[label] = 100 * (wide[method] - wide[reference])
    return out


def plot_paired(ax, deltas: dict[str, pd.Series], *, color: str, title: str, ylabel: str) -> None:
    rng = np.random.default_rng(20260715)
    labels = list(deltas)
    for index, label in enumerate(labels):
        values = deltas[label].to_numpy(float)
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        ax.scatter(index + jitter, values, s=18, color=color, alpha=0.52, edgecolor="none", zorder=2)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        ax.errorbar(index, mean, yerr=sem, fmt="o", ms=5.5, color=color, mec="white", mew=0.7, capsize=2.5, zorder=4)
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)), labels, rotation=16, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, lw=0.5)


def make_figure(*, output_dir: Path = OUT, copy_to_paper: bool = True) -> None:
    frames = load_seed_means()
    error_delta = paired_deltas(frames, "endfa", "dfa")
    k_delta = paired_deltas(frames, "kndfa", "ndfa")
    curves = load_original_curves()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.titlesize": 8.7,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    plot_paired(
        axes[0],
        error_delta,
        color=COLORS["endfa"],
        title="A  Error conditioning over DFA",
        ylabel=r"$\Delta$ test accuracy (pp)",
    )
    plot_paired(
        axes[1],
        k_delta,
        color=COLORS["kndfa"],
        title="B  Error factor after activity",
        ylabel=r"$\Delta$ test accuracy (pp)",
    )
    for method in ("dfa", "endfa", "ndfa", "kndfa"):
        sub = curves[curves["method"].eq(method)]
        summary = sub.groupby("step")["val_acc"].agg(["mean", "sem"]).reset_index()
        x = summary["step"].to_numpy(float)
        mean = 100 * summary["mean"].to_numpy(float)
        error = 100 * summary["sem"].fillna(0).to_numpy(float)
        axes[2].plot(x, mean, color=COLORS[method], label=LABELS[method], lw=1.5)
        axes[2].fill_between(x, mean - error, mean + error, color=COLORS[method], alpha=0.12, lw=0)
    axes[2].set_xlabel("training update")
    axes[2].set_ylabel("validation accuracy (%)")
    axes[2].set_title("C  ReLU/softmax confirmation", loc="left", fontweight="bold")
    axes[2].grid(alpha=0.18, lw=0.5)
    axes[2].legend(frameon=False, loc="lower right")

    directories = [output_dir] + ([PAPER_FIGURES] if copy_to_paper else [])
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png", "svg"):
            fig.savefig(directory / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for contrast, values_by_setting in (("error-DFA", error_delta), ("K-activity", k_delta)):
        for setting, values in values_by_setting.items():
            rows.append(
                {
                    "contrast": contrast,
                    "setting": setting,
                    "n_seeds": values.size,
                    "mean_delta_pp": values.mean(),
                    "sem_delta_pp": values.sem(),
                    "wins": int((values > 0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "replication_contrasts.csv", index=False)
    input_paths = [
        RESULTS / directory / filename
        for directory in ANALYSIS_DIRS.values()
        for filename in ("confirmation_seed_means.csv", "confirmation_runs.csv")
    ] + [RESULTS / ANALYSIS_DIRS["ReLU MNIST"] / "confirmation_curves.csv"]
    manifest = {
        "figure": STEM,
        "cohort": "original confirmation; post-hoc extension seeds excluded",
        "model_seed_ids": COHORTS,
        "feedback_seed_ids": FEEDBACK_SEEDS,
        "aggregation": "average feedback seeds within model seed; mean and SEM across model seeds",
        "caption_counts": {label: len(seeds) for label, seeds in COHORTS.items()},
        "sources": [
            {"path": str(path.relative_to(RESULTS)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in input_paths
        ],
    }
    (output_dir / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--no-copy-to-paper", action="store_true", help="write only to the output directory")
    args = parser.parse_args()
    make_figure(output_dir=args.output_dir, copy_to_paper=not args.no_copy_to_paper)
