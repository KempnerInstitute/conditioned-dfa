"""Make paper-facing figures for Info-DFA noise/sample-size follow-ups.

The default inputs are the completed v2 aggregates used in the current paper
draft. The script remains tolerant of missing inputs so it can still be reused
while future sweeps are landing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DRAFT_FIGS = ROOT / "drafts" / "Info-DFA" / "figures"
PAPER_FIGS = RESULTS / "paper_figures" / "Info-DFA"

COLORS = {
    "bp": "#1F2937",
    "dfa_random": "#2E7D32",
    "fa_random": "#6B7280",
    "ndfa_random": "#2563EB",
    "ndfa_random_kronecker": "#D55E00",
    "local_loss": "#8B5CF6",
    "vnc": "#009E73",
    "nmnc": "#CC79A7",
    "drtp_random": "#A3A3A3",
    "nuisance_dominant": "#D55E00",
    "mixed_context": "#CC79A7",
    "low_sample_noisy": "#009E73",
    "task_aligned": "#2563EB",
}

MARKERS = {
    "bp": "o",
    "dfa_random": "o",
    "ndfa_random": "D",
    "ndfa_random_kronecker": "^",
    "local_loss": "P",
    "nmnc": "X",
    "vnc": "v",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    DRAFT_FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    setup_style()

    data = {
        "multi_summary": read_csv(first_existing(args.multi_summary, RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_summary.csv")),
        "multi_final": read_csv(first_existing(args.multi_final, RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_final_diagnostics.csv")),
        "nmnc_summary": read_csv(first_existing(args.nmnc_summary, RESULTS / "dfa_nmnc_aggregate_v1" / "dfa_nmnc_summary.csv")),
        "nmnc_all": read_csv(first_existing(args.nmnc_all, RESULTS / "dfa_nmnc_aggregate_v1" / "dfa_nmnc_all.csv")),
        "conv_summary": read_csv(first_existing(args.conv_summary, RESULTS / "dfa_convnet_cifar100_harder_matched_bp_aggregate_v1" / "dfa_convnet_summary.csv")),
        "conv_all": read_csv(first_existing(args.conv_all, RESULTS / "dfa_convnet_cifar100_harder_matched_bp_aggregate_v1" / "dfa_convnet_all.csv")),
    }
    make_figure(data, output_dir)
    write_report(data, output_dir)
    print(f"Wrote noise-sweep figures to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-summary", default="results/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_summary.csv")
    parser.add_argument("--multi-final", default="results/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_final_diagnostics.csv")
    parser.add_argument("--nmnc-summary", default="results/infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_summary.csv")
    parser.add_argument("--nmnc-all", default="results/infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_all.csv")
    parser.add_argument("--conv-summary", default="results/infodfa_hard_cifar100_confirm_aggregate_v2/dfa_convnet_summary.csv")
    parser.add_argument("--conv-all", default="results/infodfa_hard_cifar100_confirm_aggregate_v2/dfa_convnet_all.csv")
    parser.add_argument("--output-dir", default="results/infodfa_noise_sweep_figures_v2")
    return parser.parse_args()


def first_existing(*paths: str | Path) -> Path:
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    return Path(paths[0])


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 420,
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_figure(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.85), constrained_layout=True)
    axes = axes.ravel()
    plot_synthetic_label_noise(axes[0], data["multi_summary"])
    plot_synthetic_sample_size(axes[1], data["multi_summary"])
    plot_vision_noise(axes[2], data["nmnc_summary"])
    plot_projected_step(axes[3], data["multi_final"])
    plot_covariance_mechanism(axes[4], data["multi_final"])
    plot_conv_confirm(axes[5], data["conv_summary"])
    label_panels(axes)
    save(fig, output_dir, "infodfa_noise_sample_mechanism")


def plot_synthetic_label_noise(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_panel(ax, "Synthetic noise sweep")
        return
    best = best_rows(summary, context_cols(summary, ["condition", "train_label_noise", "input_noise", "n_train", "method"]))
    if "train_label_noise" in best and best["train_label_noise"].nunique(dropna=True) > 1:
        sub = canonical_slice(best, prefer_min=["input_noise"], prefer_max=["n_train"])
        for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy"]:
            for method, ls in [("dfa_random", "--"), ("ndfa_random_kronecker", "-")]:
                m = sub[(sub["condition"] == condition) & (sub["method"] == method)].sort_values("train_label_noise")
                if m.empty:
                    continue
                ax.errorbar(
                    m["train_label_noise"],
                    100.0 * m["test_mean"],
                    yerr=100.0 * m["test_sem"].fillna(0.0),
                    color=COLORS[condition],
                    ls=ls,
                    marker=MARKERS.get(method, "o"),
                    lw=1.5,
                    ms=3.8,
                    capsize=1.8,
                    label=f"{pretty_condition(condition)} {pretty_method(method)}",
                )
        ax.set_xlabel("Training label noise")
        ax.set_ylabel("Best test accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Synthetic: robustness to noisy supervision")
        ax.legend(frameon=False, fontsize=5.7, ncol=1, loc="lower left")
    else:
        plot_gain_bars(ax, best, "condition", ["nuisance_dominant", "mixed_context", "low_sample_noisy"], "Synthetic gains over raw DFA")
    clean_axes(ax)


def plot_synthetic_sample_size(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_panel(ax, "Synthetic sample-size sweep")
        return
    best = best_rows(summary, context_cols(summary, ["condition", "n_train", "train_label_noise", "input_noise", "method"]))
    if "n_train" in best and best["n_train"].nunique(dropna=True) > 1:
        sub = canonical_slice(best, prefer_min=["input_noise", "train_label_noise"])
        for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy"]:
            dfa = sub[(sub["condition"] == condition) & (sub["method"] == "dfa_random")].set_index("n_train")["test_mean"]
            kndfa = sub[(sub["condition"] == condition) & (sub["method"] == "ndfa_random_kronecker")].set_index("n_train")["test_mean"]
            x = sorted(set(dfa.dropna().index).intersection(kndfa.dropna().index))
            if not x:
                continue
            y = [100.0 * (float(kndfa.loc[v]) - float(dfa.loc[v])) for v in x]
            ax.plot(x, y, color=COLORS[condition], marker="o", lw=1.7, ms=4.0, label=pretty_condition(condition))
        ax.set_xscale("log")
        ax.axhline(0.0, color="#111827", lw=0.8)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("K-nDFA gain vs DFA (pp)")
        ax.set_title("Synthetic: benefit at low sample size")
        ax.legend(frameon=False, loc="best")
    else:
        plot_gain_bars(ax, best, "condition", ["nuisance_dominant", "mixed_context", "low_sample_noisy"], "Synthetic low-sample gains")
    clean_axes(ax)


def plot_vision_noise(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_panel(ax, "Vision MLP noise sweep")
        return
    best = best_rows(summary, context_cols(summary, ["dataset", "label_noise", "n_train", "method"]))
    if "label_noise" in best and best["label_noise"].nunique(dropna=True) > 1:
        sub = canonical_slice(best, prefer_max=["n_train"])
        for dataset, color in [("fashion_mnist", "#2563EB"), ("cifar10", "#009E73")]:
            for method, ls in [("dfa_random", "--"), ("ndfa_random_kronecker", "-"), ("bp", ":")]:
                m = sub[(sub["dataset"] == dataset) & (sub["method"] == method)].sort_values("label_noise")
                if m.empty:
                    continue
                ax.errorbar(
                    m["label_noise"],
                    100.0 * m["test_mean"],
                    yerr=100.0 * m["test_sem"].fillna(0.0),
                    color=color,
                    ls=ls,
                    marker=MARKERS.get(method, "o"),
                    lw=1.5,
                    ms=3.8,
                    capsize=1.8,
                    label=f"{pretty_dataset(dataset)} {pretty_method(method)}",
                )
        ax.set_xlabel("Training label noise")
        ax.set_ylabel("Best test accuracy (%)")
        ax.set_title("Vision MLP: noisy-label robustness")
        ax.legend(frameon=False, fontsize=5.7, ncol=1, loc="best")
    else:
        plot_gain_bars(ax, best, "dataset", ["fashion_mnist", "cifar10", "mnist"], "Vision MLP gains over DFA")
    clean_axes(ax)


def plot_projected_step(ax: plt.Axes, final: pd.DataFrame) -> None:
    if final.empty or "projected_step_ratio_mean" not in final:
        empty_panel(ax, "Projected-step mechanism")
        return
    local = final[final["method"] != "bp"].copy()
    for method, sub in local.groupby("method"):
        ax.scatter(
            sub["projected_step_ratio_mean"],
            100.0 * sub["test_acc"],
            s=16,
            color=COLORS.get(method, "0.4"),
            alpha=0.35,
            edgecolor="none",
            label=pretty_method(method),
        )
    ax.axvline(0.1, color="#6B7280", ls="--", lw=0.9)
    ax.set_xlabel("Projected BP-step ratio")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Mechanism: useful local descent")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower right")
    clean_axes(ax)


def plot_covariance_mechanism(ax: plt.Axes, final: pd.DataFrame) -> None:
    if final.empty:
        empty_panel(ax, "Covariance mechanism")
        return
    xcol = next((c for c in ["local_error_top1_fraction_mean", "local_error_condition_mean", "pre_activity_condition_mean"] if c in final.columns), None)
    if xcol is None:
        plot_projected_step(ax, final)
        ax.set_title("Covariance diagnostics pending")
        return
    local = final[final["method"].isin(["dfa_random", "ndfa_random", "ndfa_random_kronecker"])].copy()
    for method, sub in local.groupby("method"):
        ax.scatter(
            sub[xcol],
            100.0 * sub["test_acc"],
            s=18,
            color=COLORS.get(method, "0.4"),
            alpha=0.42,
            edgecolor="none",
            label=pretty_method(method),
        )
    xlabel = {
        "local_error_top1_fraction_mean": "Local-error top eigenvalue fraction",
        "local_error_condition_mean": "Local-error covariance condition",
        "pre_activity_condition_mean": "Presynaptic covariance condition",
    }[xcol]
    if "condition_mean" in xcol:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Mechanism: covariance structure")
    ax.legend(frameon=False, fontsize=5.8, loc="best")
    clean_axes(ax)


def plot_conv_confirm(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_panel(ax, "Hard CIFAR-100 confirm")
        return
    best = best_rows(summary, context_cols(summary, ["dataset", "method"]))
    dataset = "cifar100" if "cifar100" in set(best.get("dataset", [])) else (best["dataset"].iloc[0] if "dataset" in best and not best.empty else None)
    sub = best[best["dataset"] == dataset] if dataset is not None and "dataset" in best else best
    order = ["drtp_random", "dfa_random", "ndfa_random_kronecker", "bp", "local_loss", "ndfa_random"]
    sub = sub[sub["method"].isin(order)].copy()
    sub["order"] = sub["method"].map({m: i for i, m in enumerate(order)})
    sub = sub.sort_values("order")
    ax.barh(
        [pretty_method(m) for m in sub["method"]],
        100.0 * sub["test_mean"],
        xerr=100.0 * sub["test_sem"].fillna(0.0),
        color=[COLORS.get(m, "0.5") for m in sub["method"]],
        error_kw={"lw": 0.75, "capsize": 2.0},
        zorder=3,
    )
    ax.set_xlabel("Best test accuracy (%)")
    ax.set_title("Hard CIFAR-100 confirmation")
    clean_axes(ax)


def best_rows(df: pd.DataFrame, group_cols: list[str], value_col: str = "test_mean") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, sub in df.groupby(group_cols, dropna=False):
        rows.append(sub.loc[pd.to_numeric(sub[value_col], errors="coerce").idxmax()].to_dict())
    return pd.DataFrame(rows)


def context_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def canonical_slice(df: pd.DataFrame, *, prefer_min: list[str] | None = None, prefer_max: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in prefer_min or []:
        if col in out and out[col].notna().any():
            out = out[np.isclose(out[col], out[col].min())]
    for col in prefer_max or []:
        if col in out and out[col].notna().any():
            out = out[np.isclose(out[col], out[col].max())]
    return out


def plot_gain_bars(ax: plt.Axes, best: pd.DataFrame, group_col: str, groups: list[str], title: str) -> None:
    rows = []
    for group in groups:
        base = value_for(best, group_col, group, "dfa_random")
        ndfa = value_for(best, group_col, group, "ndfa_random")
        kndfa = value_for(best, group_col, group, "ndfa_random_kronecker")
        gain = max(ndfa, kndfa) - base
        if np.isfinite(gain):
            rows.append((pretty_group(group_col, group), gain, group))
    rows = sorted(rows, key=lambda item: item[1])
    ax.barh([r[0] for r in rows], [100.0 * r[1] for r in rows], color=[COLORS.get(r[2], "#2563EB") for r in rows], zorder=3)
    ax.axvline(0.0, color="#111827", lw=0.8)
    ax.set_xlabel("Best nDFA/K-nDFA gain vs DFA (pp)")
    ax.set_title(title)


def value_for(best: pd.DataFrame, group_col: str, group: str, method: str) -> float:
    if best.empty or group_col not in best:
        return np.nan
    sub = best[(best[group_col] == group) & (best["method"] == method)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["test_mean"])


def pretty_group(group_col: str, value: str) -> str:
    return pretty_condition(value) if group_col == "condition" else pretty_dataset(value)


def pretty_method(method: str) -> str:
    return {
        "bp": "BP",
        "dfa_random": "DFA",
        "fa_random": "FA",
        "ndfa_random": "nDFA",
        "ndfa_random_kronecker": "K-nDFA",
        "local_loss": "Local aux",
        "drtp_random": "DRTP",
        "vnc": "VNC",
        "nmnc": "NMNC",
    }.get(str(method), str(method).replace("_", " "))


def pretty_condition(condition: str) -> str:
    return {
        "task_aligned": "Task-aligned",
        "nuisance_dominant": "Nuisance-dominant",
        "mixed_context": "Mixed-context",
        "low_sample_noisy": "Low-sample/noisy",
    }.get(str(condition), str(condition).replace("_", " ").title())


def pretty_dataset(dataset: str) -> str:
    return {
        "mnist": "MNIST",
        "fashion_mnist": "Fashion-MNIST",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
    }.get(str(dataset), str(dataset).replace("_", "-").upper())


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", zorder=0)
    ax.tick_params(pad=1.5)


def empty_panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "waiting for aggregate", transform=ax.transAxes, ha="center", va="center", color="#6B7280")
    ax.set_xticks([])
    ax.set_yticks([])


def label_panels(axes: np.ndarray) -> None:
    for ax, label in zip(np.ravel(axes), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", color="#111827")


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if ext == "png":
            kwargs["dpi"] = 420
        fig.savefig(path, **kwargs)
        shutil.copy2(path, DRAFT_FIGS / path.name)
        shutil.copy2(path, PAPER_FIGS / path.name)
    plt.close(fig)


def write_report(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    lines = [
        "# Info-DFA Noise/Sample Follow-Up Figure Report",
        "",
        "This report is generated from the newest available noise/sample aggregates, falling back to completed v1 aggregates when follow-up jobs have not landed yet.",
        "",
        "## Inputs",
        "",
    ]
    for name, df in data.items():
        lines.append(f"- {name}: {len(df)} rows")
    lines.extend(
        [
            "",
            "## Interpretation Targets",
            "",
            "- Label-noise curves should show whether nDFA/K-nDFA degrade more gracefully than raw DFA.",
            "- Sample-size curves should show whether the gain is largest when data are scarce.",
            "- Projected-step and covariance panels should show whether the performance gain tracks useful local descent and better-conditioned local error/activity spectra.",
            "- Hard CIFAR-100 confirmation should decide whether the current positive convnet result survives more seeds and a cleaner matched baseline.",
        ]
    )
    (output_dir / "infodfa_noise_sweep_figure_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
