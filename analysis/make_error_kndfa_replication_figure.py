"""Make the main-text cross-setting error/K-nDFA replication figure."""

from __future__ import annotations

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


def load_seed_means() -> dict[str, pd.DataFrame]:
    paths = {
        "tanh MNIST": RESULTS / "dfa_stall_threefactor_analysis_v1" / "confirmation_seed_means.csv",
        "tanh Fashion": RESULTS / "dfa_stall_fashion_threefactor_analysis_v1" / "confirmation_seed_means.csv",
        "ReLU MNIST": RESULTS / "dfa_relu_mnist_threefactor_analysis_v1" / "confirmation_seed_means.csv",
    }
    frames = {}
    for label, path in paths.items():
        frame = pd.read_csv(path)
        frame = frame[frame["method"].isin(COLORS)].copy()
        frames[label] = frame
    return frames


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


def make_figure() -> None:
    frames = load_seed_means()
    error_delta = paired_deltas(frames, "endfa", "dfa")
    k_delta = paired_deltas(frames, "kndfa", "ndfa")
    curves = pd.read_csv(RESULTS / "dfa_relu_mnist_threefactor_analysis_v1" / "confirmation_curves.csv")
    curves = curves.groupby(["seed", "method", "step"], as_index=False)["val_acc"].mean()

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

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUT, PAPER_FIGURES):
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
    pd.DataFrame(rows).to_csv(OUT / "replication_contrasts.csv", index=False)


if __name__ == "__main__":
    make_figure()
