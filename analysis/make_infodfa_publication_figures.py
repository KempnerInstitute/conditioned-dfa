"""Build publication-style Info-DFA figures with a shared visual language.

This script intentionally supersedes the older exploratory plotting scripts for
the figures used in the Info-DFA draft.  It reads the existing aggregate CSVs,
uses one palette/typographic system, and mirrors outputs into the draft figure
folder so Overleaf/local LaTeX compiles with the polished versions.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "infodfa_publication_figures_v2"
DRAFT_FIGS = ROOT / "drafts" / "Info-DFA" / "figures"
PAPER_FIGS = RESULTS / "paper_figures" / "Info-DFA"

METHOD_ORDER = [
    "bp",
    "dfa_random",
    "fa_random",
    "ndfa_random",
    "ndfa_random_kronecker",
    "local_loss",
    "vnc",
    "nmnc",
    "drtp_random",
]

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
    "dfa_tangent_biased": "#059669",
    "dfa_tangent_orthogonal": "#EF4444",
    "dfa_bp_aligned_dynamic": "#16A34A",
    "dfa_bp_orthogonal_dynamic": "#DC2626",
    "task_aligned": "#2563EB",
    "nuisance_dominant": "#D55E00",
    "mixed_context": "#CC79A7",
    "low_sample_noisy": "#009E73",
}

MARKERS = {
    "bp": "o",
    "dfa_random": "o",
    "fa_random": "s",
    "ndfa_random": "D",
    "ndfa_random_kronecker": "^",
    "local_loss": "P",
    "vnc": "v",
    "nmnc": "X",
    "drtp_random": "x",
    "dfa_tangent_biased": "o",
    "dfa_tangent_orthogonal": "o",
    "dfa_bp_aligned_dynamic": "s",
    "dfa_bp_orthogonal_dynamic": "s",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DRAFT_FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    setup_style()

    make_schematic()
    make_main_method_result()
    make_legacy_synthetic_results()
    make_rank_conditioning()
    make_legacy_vision_bridge()
    make_multioutput()
    make_multioutput_diagnostics()
    make_preconditioning_spectrum()
    make_nmnc_comparison()
    make_learning_dynamics()
    make_convnet_baselines()
    make_convnet_conditioning()
    make_convnet_extra_baselines()
    make_convnet_harder()

    print(f"Wrote publication Info-DFA figures to {OUT}")
    print(f"Mirrored figures to {DRAFT_FIGS}")


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
            "legend.fontsize": 6.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


def read_first_existing(*paths: str | Path) -> pd.DataFrame:
    for path in paths:
        df = read_csv(RESULTS / path if not Path(path).is_absolute() else path)
        if not df.empty:
            return df
    return pd.DataFrame()


def read_vision_bridge() -> pd.DataFrame:
    frames = []
    for rel in [
        "dfa_vision_full_slurm_v4/mnist/dfa_vision_results.csv",
        "dfa_vision_full_slurm_v4/fashion_mnist/dfa_vision_results.csv",
        "dfa_vision_full_slurm_v4/cifar10/dfa_vision_results.csv",
        "dfa_vision/dfa_vision_results.csv",
    ]:
        df = read_csv(RESULTS / rel)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if ext == "png":
            kwargs["dpi"] = 420
        fig.savefig(path, **kwargs)
        shutil.copy2(path, DRAFT_FIGS / path.name)
        shutil.copy2(path, PAPER_FIGS / path.name)
    plt.close(fig)


def clean_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="y", zorder=0)
    ax.tick_params(pad=1.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        color="#111827",
    )


def label_panels(axes: list[plt.Axes] | np.ndarray, labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> None:
    for ax, label in zip(np.ravel(axes), labels):
        panel_label(ax, label)


def legend_for_methods(fig: plt.Figure, methods: list[str], *, ncol: int = 6, y: float = 0.99) -> None:
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=COLORS.get(method, "0.35"),
            marker=MARKERS.get(method, "o"),
            lw=1.8,
            ms=4.0,
            label=pretty_method(method),
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.35,
    )


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
        "dfa_tangent_biased": "Tangent DFA",
        "dfa_tangent_orthogonal": "Tangent-orthogonal DFA",
        "dfa_bp_aligned_dynamic": "BP-aligned DFA",
        "dfa_bp_orthogonal_dynamic": "BP-orthogonal DFA",
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


def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.shape[0] <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.shape[0]))


def best_rows(df: pd.DataFrame, group_cols: list[str], value_col: str = "test_mean") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, sub in df.groupby(group_cols, dropna=False):
        rows.append(sub.loc[sub[value_col].idxmax()].to_dict())
    return pd.DataFrame(rows)


def final_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "epoch" not in df:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    return df.sort_values("epoch").groupby(cols, as_index=False).tail(1).copy()


def line_with_sem(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    yerr: pd.Series | None,
    *,
    method: str | None = None,
    color: str | None = None,
    label: str | None = None,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    c = color or COLORS.get(method or "", "0.3")
    marker = MARKERS.get(method or "", "o")
    ax.errorbar(
        pd.to_numeric(x, errors="coerce"),
        pd.to_numeric(y, errors="coerce"),
        yerr=None if yerr is None else pd.to_numeric(yerr, errors="coerce").fillna(0.0),
        color=c,
        marker=marker,
        lw=1.7,
        ms=4.1,
        elinewidth=0.8,
        capsize=1.8,
        label=label or (pretty_method(method) if method else None),
        linestyle=linestyle,
        alpha=alpha,
        zorder=3,
    )


def make_schematic() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()

    def box(ax: plt.Axes, xy, wh, text, *, fc="#F9FAFB", ec="#374151", color="#111827", size=8.0):
        patch = FancyBboxPatch(
            xy,
            wh[0],
            wh[1],
            boxstyle="round,pad=0.025,rounding_size=0.025",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=size, color=color)
        return patch

    def arrow(ax: plt.Axes, start, end, *, color="#4B5563", lw=1.1, style="-|>"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, lw=lw, color=color))

    ax = axes[0]
    ax.set_title("Error transport", loc="left", pad=2)
    box(ax, (0.05, 0.56), (0.24, 0.22), "input\nactivity", fc="#EFF6FF", ec=COLORS["ndfa_random"])
    box(ax, (0.40, 0.56), (0.24, 0.22), "hidden\nlayer", fc="#ECFDF5", ec=COLORS["dfa_random"])
    box(ax, (0.74, 0.56), (0.20, 0.22), "output\nerror", fc="#FEF3C7", ec="#B45309")
    arrow(ax, (0.29, 0.67), (0.40, 0.67), color="#111827")
    arrow(ax, (0.74, 0.60), (0.64, 0.60), color=COLORS["bp"])
    arrow(ax, (0.84, 0.56), (0.54, 0.36), color=COLORS["dfa_random"])
    box(ax, (0.37, 0.22), (0.30, 0.17), "DFA:\n$B_\\ell e$", fc="#F0FDF4", ec=COLORS["dfa_random"])
    box(
        ax,
        (0.04, 0.025),
        (0.92, 0.125),
        "BP transports exact downstream weights;\nDFA broadcasts a fixed/random error projection.",
        fc="#FFFFFF",
        ec="#D1D5DB",
        size=6.55,
    )
    panel_label(ax, "A")

    ax = axes[1]
    ax.set_title("Representation information", loc="left", pad=2)
    t = np.linspace(-1.0, 1.0, 100)
    ax.plot(0.22 + 0.24 * t, 0.55 + 0.16 * np.sin(2.6 * t), color=COLORS["bp"], lw=2.0)
    ax.plot(0.62 + 0.22 * t, 0.55 + 0.08 * np.sin(2.6 * t), color=COLORS["ndfa_random"], lw=2.0)
    ax.text(0.16, 0.72, "raw manifold", fontsize=7.2)
    ax.text(0.56, 0.72, "whitened\nby uncertainty", fontsize=7.2)
    arrow(ax, (0.43, 0.56), (0.54, 0.56), color="#111827")
    ax.text(0.43, 0.63, "$\\Sigma^{-1/2}$", fontsize=8)
    box(ax, (0.14, 0.18), (0.72, 0.18), "$d'^2 = \\Delta\\mu^\\top\\Sigma^{-1}\\Delta\\mu$\n$G=J^\\top\\Sigma^{-1}J$", fc="#F8FAFC", ec="#475569", size=8.0)
    ax.text(0.09, 0.07, "Task information is measured as discriminability\nor Fisher length in noise-whitened coordinates.", fontsize=7.1, color="#374151")
    panel_label(ax, "B")

    ax = axes[2]
    ax.set_title("From DFA to nDFA/K-nDFA", loc="left", pad=2)
    box(ax, (0.04, 0.58), (0.25, 0.18), "DFA\n$\\delta h^\\top$", fc="#F0FDF4", ec=COLORS["dfa_random"])
    box(ax, (0.37, 0.58), (0.25, 0.18), "nDFA\n$\\delta h^\\top C^{-1}$", fc="#EFF6FF", ec=COLORS["ndfa_random"])
    box(ax, (0.70, 0.58), (0.25, 0.18), "K-nDFA\n$D^{-1}\\delta\\,h_C^\\top$", fc="#FFF7ED", ec=COLORS["ndfa_random_kronecker"], size=7.0)
    arrow(ax, (0.29, 0.67), (0.37, 0.67), color="#111827")
    arrow(ax, (0.62, 0.67), (0.70, 0.67), color="#111827")
    box(ax, (0.17, 0.23), (0.67, 0.20), "$F_W \\approx D \\otimes C$\nlocal Fisher/Kronecker block", fc="#F9FAFB", ec="#4B5563")
    ax.text(0.07, 0.08, "nDFA whitens presynaptic activity; K-nDFA also whitens\nfeedback-error coordinates. Both are local approximations.", fontsize=7.1, color="#374151")
    panel_label(ax, "C")

    ax = axes[3]
    ax.set_title("Falsifiable success conditions", loc="left", pad=2)
    items = [
        ("rank$(B)$ reaches task-error rank", COLORS["ndfa_random"]),
        ("projected BP-step is nonzero", COLORS["dfa_random"]),
        ("task geometry beats nuisance geometry", COLORS["nmnc"]),
        ("covariance damping is stable", COLORS["ndfa_random_kronecker"]),
    ]
    for i, (text, color) in enumerate(items):
        y = 0.75 - i * 0.16
        ax.scatter([0.12], [y], s=55, color=color)
        ax.text(0.20, y, text, va="center", fontsize=7.5)
    box(
        ax,
        (0.14, 0.035),
        (0.72, 0.18),
        "Prediction: local learning works when\nrank, projected step, and conditioning\nhold together",
        fc="#F8FAFC",
        ec="#475569",
        size=6.6,
    )
    panel_label(ax, "D")

    save(fig, "infodfa_fig1_schematic")


def make_main_method_result() -> None:
    """Make the main result figure with a deliberately small comparison set.

    The exploratory figures compare every method in every setting.  This
    figure is different: it answers the reader's first question about the
    proposed rule, namely where covariance-preconditioned DFA helps over raw
    DFA and which diagnostic explains that improvement.
    """
    summary = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_summary.csv")
    final = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_final_diagnostics.csv")
    spectrum = read_csv(RESULTS / "dfa_preconditioning_spectrum_aggregate_v1" / "dfa_preconditioning_spectrum_summary.csv")
    if summary.empty:
        return

    conditions = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
    headline_methods = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "nmnc"]
    best = best_rows(summary[summary["method"].isin(headline_methods)], ["condition", "method"])

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    x0 = np.arange(len(conditions))
    width = 0.145
    for i, method in enumerate(headline_methods):
        vals = best[best["method"] == method].set_index("condition")["test_mean"].reindex(conditions)
        errs = best[best["method"] == method].set_index("condition")["test_sem"].reindex(conditions).fillna(0.0)
        offset = (i - (len(headline_methods) - 1) / 2) * width
        ax.bar(
            x0 + offset,
            vals,
            width=width,
            color=COLORS[method],
            yerr=errs,
            error_kw={"lw": 0.7, "capsize": 1.5, "capthick": 0.7},
            label=pretty_method(method),
            zorder=3,
        )
    ax.set_xticks(x0, [pretty_condition(c) for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Best test accuracy")
    ax.set_ylim(0.0, 1.04)
    ax.set_title("Calibrated stress-test comparison")
    clean_axes(ax)

    ax = axes[1]
    baseline = best[best["method"] == "dfa_random"].set_index("condition")["test_mean"]
    gain_methods = ["ndfa_random", "ndfa_random_kronecker", "nmnc"]
    gain_width = 0.22
    for i, method in enumerate(gain_methods):
        vals = best[best["method"] == method].set_index("condition")["test_mean"].reindex(conditions) - baseline.reindex(conditions)
        ax.bar(
            x0 + (i - 1) * gain_width,
            vals,
            width=gain_width,
            color=COLORS[method],
            label=pretty_method(method),
            zorder=3,
        )
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks(x0, [pretty_condition(c) for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Accuracy gain vs DFA")
    ax.set_title("The method helps most in hard regimes")
    clean_axes(ax)

    ax = axes[2]
    if not final.empty and "projected_step_ratio_mean" in final:
        local = final[final["method"].isin(["dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc", "drtp_random", "fa_random"])].copy()
        local["useful_step"] = local["projected_step_ratio_mean"] >= 0.1
        rng = np.random.default_rng(8)
        groups = [(False, "low projected step", "#9CA3AF"), (True, "useful projected step", COLORS["ndfa_random"])]
        means = []
        ses = []
        labels = []
        colors = []
        for flag, label, color in groups:
            vals = local.loc[local["useful_step"] == flag, "test_acc"].dropna()
            means.append(vals.mean())
            ses.append(sem(vals))
            labels.append(label)
            colors.append(color)
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), len(labels) - 1) + jitter, vals, s=8, color=color, alpha=0.16, edgecolor="none", zorder=2)
        ax.bar(np.arange(2), means, yerr=ses, color=colors, width=0.55, error_kw={"lw": 0.8, "capsize": 2.0}, zorder=3)
        ax.set_xticks(np.arange(2), labels, rotation=0)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Mechanistic diagnostic")
    ax.set_ylim(0.0, 1.02)
    clean_axes(ax)

    ax = axes[3]
    if not spectrum.empty:
        hard_conditions = ["nuisance_dominant", "mixed_context", "low_sample_noisy"]
        for condition in hard_conditions:
            sub = spectrum[
                (spectrum["condition"] == condition)
                & np.isclose(spectrum["feedback_rank"], 0.0)
                & np.isclose(spectrum["damping"], 0.1)
            ].sort_values("gamma")
            if sub.empty:
                continue
            line_with_sem(
                ax,
                sub["gamma"],
                sub["test_mean"],
                sub["test_sem"],
                color=COLORS[condition],
                label=pretty_condition(condition),
            )
    ax.set_xlabel("Blend from DFA to K-nDFA, $\\gamma$")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("K-nDFA changes learning continuously")
    ax.set_ylim(0.0, 1.02)
    clean_axes(ax)

    label_panels(axes)
    handles = [
        mpl.lines.Line2D([0], [0], color=COLORS[m], marker=MARKERS.get(m, "o"), lw=1.8, ms=4.0, label=pretty_method(m))
        for m in headline_methods
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=5, frameon=False)
    axes[3].legend(frameon=False, loc="lower right", fontsize=6.4)
    save(fig, "infodfa_main_method_result")


def predictor_label(name: str) -> str:
    return {
        "early_param_cosine": "Parameter cosine",
        "early_activity_cosine": "Activity cosine",
        "early_tangent_cosine": "Tangent cosine",
        "early_task_tangent_cosine": "Task-tangent cosine",
        "early_fisher_trace": "Whitened Fisher",
        "early_weight_rayleigh": "Rayleigh quotient",
        "delta_class_dprime2_early": "Class $d'$ growth",
        "early_class_dprime2": "Class $d'$",
    }.get(str(name), str(name).replace("_", " "))


def linestyle_for_method(method: str) -> str:
    return {
        "dfa_tangent_biased": "--",
        "dfa_tangent_orthogonal": "--",
        "dfa_bp_aligned_dynamic": "-.",
        "dfa_bp_orthogonal_dynamic": "-.",
        "drtp_random": "-",
    }.get(method, "-")


def make_legacy_synthetic_results() -> None:
    """Regenerate the original synthetic Figure 2 in the new publication style."""
    df = read_first_existing(
        "dfa_synthetic_full_slurm_v5/dfa_synthetic_results.csv",
        "dfa_synthetic_multimanifold/dfa_synthetic_results.csv",
        "dfa_synthetic/dfa_synthetic_results.csv",
    )
    summary = read_first_existing(
        "dfa_synthetic_full_slurm_v5/dfa_run_summary.csv",
        "dfa_synthetic_multimanifold/dfa_run_summary.csv",
        "dfa_synthetic/dfa_run_summary.csv",
    )
    scores = read_first_existing(
        "dfa_synthetic_full_slurm_v5/dfa_predictor_scores.csv",
        "dfa_synthetic_multimanifold/dfa_predictor_scores.csv",
        "dfa_synthetic/dfa_predictor_scores.csv",
    )
    if df.empty and summary.empty:
        return

    methods = [
        "bp",
        "dfa_random",
        "dfa_bp_aligned_dynamic",
        "dfa_bp_orthogonal_dynamic",
        "dfa_tangent_biased",
        "dfa_tangent_orthogonal",
        "ndfa_random_kronecker",
        "drtp_random",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    if not df.empty:
        acc = df[["method", "epoch", "test_acc", "seed", "feedback_seed", "feedback_scale", "manifold"]].drop_duplicates()
        for method in methods:
            sub = acc[acc["method"] == method]
            if sub.empty:
                continue
            run_cols = [c for c in ["method", "seed", "feedback_seed", "feedback_scale", "manifold", "epoch"] if c in sub.columns]
            run = sub.groupby(run_cols, as_index=False).agg(value=("test_acc", "mean"))
            curve = run.groupby("epoch", as_index=False).agg(mean=("value", "mean"), se=("value", sem)).sort_values("epoch")
            line_with_sem(
                ax,
                curve["epoch"],
                curve["mean"],
                curve["se"],
                method=method,
                label=pretty_method(method),
                linestyle=linestyle_for_method(method),
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Learning curves")
    clean_axes(ax)

    ax = axes[1]
    if not df.empty:
        layer1 = df[(df["layer"] == 1) & (df["method"].isin(methods[1:]))]
        for method in methods[1:]:
            sub = layer1[layer1["method"] == method]
            if sub.empty:
                continue
            run_cols = [c for c in ["method", "seed", "feedback_seed", "feedback_scale", "manifold", "epoch"] if c in sub.columns]
            run = sub.groupby(run_cols, as_index=False).agg(value=("task_tangent_cosine", "mean"))
            curve = run.groupby("epoch", as_index=False).agg(mean=("value", "mean"), se=("value", sem)).sort_values("epoch")
            line_with_sem(
                ax,
                curve["epoch"],
                curve["mean"],
                curve["se"],
                method=method,
                label=pretty_method(method),
                linestyle=linestyle_for_method(method),
            )
    ax.axhline(0.0, color="#111827", lw=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Task-tangent cosine")
    ax.set_ylim(-0.25, 0.9)
    ax.set_title("Alignment in task geometry")
    clean_axes(ax)

    ax = axes[2]
    if not summary.empty:
        dfa_runs = summary[summary["method"] != "bp"].copy()
        metric_colors = {
            "early_task_tangent_cosine": COLORS["dfa_random"],
            "early_fisher_trace": COLORS["ndfa_random"],
            "early_weight_rayleigh": COLORS["ndfa_random_kronecker"],
        }
        for metric, color in metric_colors.items():
            if metric not in dfa_runs:
                continue
            values = pd.to_numeric(dfa_runs[metric], errors="coerce")
            z = (values - values.mean()) / (values.std(ddof=0) + 1e-12)
            ax.scatter(z, dfa_runs["final_test_acc"], s=18, color=color, alpha=0.42, edgecolor="none", label=predictor_label(metric))
    ax.set_xlabel("Early predictor (z score)")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Early diagnostics predict learning")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    clean_axes(ax)

    ax = axes[3]
    if not scores.empty:
        keep = [
            "early_task_tangent_cosine",
            "early_tangent_cosine",
            "early_fisher_trace",
            "early_param_cosine",
            "delta_class_dprime2_early",
            "early_weight_rayleigh",
        ]
        sub = scores[scores["predictor"].isin(keep)].copy()
        sub["label"] = sub["predictor"].map(predictor_label)
        sub = sub.sort_values("r2", ascending=True)
        ax.barh(sub["label"], sub["r2"], color="#4B5563", zorder=3)
    ax.set_xlabel("$R^2$ predicting final accuracy")
    ax.set_title("Predictor comparison")
    clean_axes(ax)

    label_panels(axes)
    legend_for_methods(fig, methods, ncol=4, y=1.08)
    save(fig, "infodfa_fig2_synthetic_results")


def make_legacy_vision_bridge() -> None:
    """Regenerate the original vision bridge in the same style as the new figures."""
    df = read_vision_bridge()
    if df.empty:
        return
    methods = [
        "bp",
        "dfa_bp_aligned_dynamic",
        "dfa_random",
        "ndfa_random",
        "ndfa_random_kronecker",
        "dfa_bp_orthogonal_dynamic",
        "drtp_random",
    ]
    final = final_rows(df[df["method"].isin(methods)], ["dataset", "method", "seed", "feedback_seed", "feedback_rank"])
    datasets = [d for d in ["mnist", "fashion_mnist", "cifar10"] if d in set(df["dataset"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.15), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    x = np.arange(len(datasets))
    width = min(0.10, 0.78 / max(len(methods), 1))
    for i, method in enumerate(methods):
        sub = final[final["method"] == method].groupby("dataset", as_index=False).agg(mean=("test_acc", "mean"), se=("test_acc", sem))
        vals = sub.set_index("dataset")["mean"].reindex(datasets)
        err = sub.set_index("dataset")["se"].reindex(datasets).fillna(0)
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            vals,
            yerr=err,
            width=width,
            color=COLORS.get(method, "0.5"),
            zorder=3,
            error_kw={"lw": 0.65, "capsize": 1.6},
        )
    ax.set_xticks(x, [pretty_dataset(d) for d in datasets], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Dataset comparison")
    clean_axes(ax)

    ax = axes[1]
    if "local_pca_tangent_cosine" in final:
        sub = final[(final["method"] != "bp") & final["local_pca_tangent_cosine"].notna()]
        for method, m in sub.groupby("method"):
            ax.scatter(m["local_pca_tangent_cosine"], m["test_acc"], s=28, color=COLORS.get(method, "0.4"), alpha=0.65, edgecolor="white", linewidth=0.35)
        rho = sub["local_pca_tangent_cosine"].corr(sub["test_acc"], method="spearman")
        ax.text(0.04, 0.07, f"Spearman $\\rho$={rho:.2f}", transform=ax.transAxes, fontsize=6.6, color="#374151")
    ax.set_xlabel("Local-PCA tangent cosine")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Data-estimated tangent is not sufficient")
    clean_axes(ax)

    for ax, dataset in zip(axes[2:], ["mnist", "cifar10"]):
        sub = df[(df["dataset"] == dataset) & (df["method"].isin(methods))]
        plot_epoch_curves(
            ax,
            sub,
            methods,
            y_col="test_acc",
            group_cols=["dataset", "method", "seed", "feedback_seed", "feedback_rank"],
            fill=True,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(pretty_dataset(dataset))
        clean_axes(ax)

    label_panels(axes)
    legend_for_methods(fig, methods, ncol=4, y=1.08)
    save(fig, "infodfa_fig3_vision_bridge")


def make_rank_conditioning() -> None:
    synth = read_csv(RESULTS / "infodfa_rank_conditioning_v1" / "infodfa_synthetic_rank_summary.csv")
    vision = read_csv(RESULTS / "infodfa_rank_conditioning_v1" / "infodfa_vision_rank_summary.csv")
    damping = read_csv(RESULTS / "infodfa_rank_conditioning_v1" / "infodfa_kronecker_damping_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 5.55), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    if not synth.empty:
        order = ["low_rank", "swiss_roll", "torus"]
        labels = ["Low-rank", "Swiss roll", "Torus"]
        for method in ["bp", "dfa_random", "dfa_tangent_biased", "drtp_random"]:
            sub = synth[synth["method"] == method]
            if sub.empty:
                continue
            vals = sub.groupby("manifold")["test_mean"].mean().reindex(order)
            line_with_sem(ax, pd.Series(np.arange(len(order))), vals.reset_index(drop=True), None, method=method, label=pretty_method(method), linestyle=linestyle_for_method(method))
        ax.set_xticks(range(len(order)), labels, rotation=18, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Controlled manifolds")
    clean_axes(ax)

    for ax, xcol, title, xlabel in [
        (axes[1], "task_tangent", "Task tangent alignment", "Task-tangent cosine"),
        (axes[2], "fisher", "Whitened Fisher", "$\\log_{10}$ Fisher proxy"),
    ]:
        if not synth.empty:
            sub = synth[synth["method"].isin(["dfa_random", "dfa_tangent_biased", "drtp_random"])].copy()
            x = np.log10(sub[xcol].clip(lower=1e-8)) if xcol == "fisher" else sub[xcol]
            for method, msub in sub.assign(_x=x).groupby("method"):
                ax.scatter(msub["_x"], msub["test_mean"], s=28, color=COLORS.get(method, COLORS["dfa_random"]), alpha=0.75, edgecolor="white", linewidth=0.35)
            rho = pd.Series(x).corr(sub["test_mean"], method="spearman")
            ax.text(0.04, 0.06, f"Spearman $\\rho$={rho:.2f}", transform=ax.transAxes, fontsize=7, color="#374151")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final test accuracy")
        ax.set_title(title)
        clean_axes(ax)

    for ax, dataset in zip(axes[3:5], ["mnist", "fashion_mnist"]):
        sub = vision[vision["dataset"] == dataset] if not vision.empty else pd.DataFrame()
        for method in ["bp", "dfa_random", "dfa_bp_aligned_dynamic", "ndfa_random", "drtp_random"]:
            m = sub[sub["method"] == method].sort_values("feedback_rank")
            if m.empty:
                continue
            line_with_sem(ax, m["feedback_rank"], m["test_mean"], m["test_sem"], method=method, label=pretty_method(method), linestyle=linestyle_for_method(method))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 2, 4, 8, 16])
        ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
        ax.set_ylim(0.05, 1.0)
        ax.set_xlabel("Feedback rank")
        ax.set_ylabel("Final test accuracy")
        ax.set_title(pretty_dataset(dataset))
        clean_axes(ax)

    ax = axes[5]
    if not damping.empty:
        for dataset, color, ls in [("mnist", COLORS["ndfa_random_kronecker"], "-"), ("fashion_mnist", "#B45309", "--")]:
            sub = damping[(damping["dataset"] == dataset) & (damping["method"] == "ndfa_random_kronecker")].sort_values("damping")
            if not sub.empty:
                line_with_sem(ax, sub["damping"], sub["test_mean"], sub["test_sem"], color=color, label=pretty_dataset(dataset), linestyle=ls)
            bp = damping[(damping["dataset"] == dataset) & (damping["method"] == "bp")]
            if not bp.empty:
                ax.axhline(bp["test_mean"].mean(), color=color, lw=0.9, ls=":", alpha=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0.05, 1.0)
    ax.set_xlabel("K-nDFA damping")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Conditioning transition")
    clean_axes(ax)

    label_panels(axes)
    legend_for_methods(fig, ["bp", "dfa_random", "dfa_tangent_biased", "dfa_bp_aligned_dynamic", "ndfa_random", "ndfa_random_kronecker", "drtp_random"], ncol=4, y=1.08)
    save(fig, "infodfa_rank_conditioning")


def make_multioutput() -> None:
    summary = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_summary.csv")
    if summary.empty:
        return
    best = best_rows(summary, ["condition", "method"])
    conditions = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
    methods = ["bp", "dfa_random", "fa_random", "ndfa_random", "ndfa_random_kronecker", "drtp_random", "vnc", "nmnc"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.45), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    width = 0.10
    x0 = np.arange(len(conditions))
    for i, method in enumerate(methods):
        vals = best[best["method"] == method].set_index("condition")["test_mean"].reindex(conditions)
        ax.bar(x0 + (i - (len(methods) - 1) / 2) * width, vals, width=width, color=COLORS[method], label=pretty_method(method), zorder=3)
    ax.set_xticks(x0, [pretty_condition(c) for c in conditions], rotation=18, ha="right")
    ax.set_ylabel("Best final test accuracy")
    ax.set_ylim(0.0, 1.04)
    ax.set_title("Stress-suite summary")
    clean_axes(ax)

    for ax, condition in zip(axes[1:5], conditions):
        sub = summary[(summary["condition"] == condition) & (summary["method"].isin(methods[1:]))]
        for method in methods[1:]:
            m = sub[sub["method"] == method].groupby("effective_feedback_rank", as_index=False).agg(test_mean=("test_mean", "mean"), test_sem=("test_sem", "mean")).sort_values("effective_feedback_rank")
            if m.empty:
                continue
            line_with_sem(ax, m["effective_feedback_rank"], m["test_mean"], m["test_sem"], method=method, label=pretty_method(method))
        bp = summary[(summary["condition"] == condition) & (summary["method"] == "bp")]
        if not bp.empty:
            ax.axhline(bp["test_mean"].max(), color=COLORS["bp"], lw=1.0, ls=":", alpha=0.8)
        ax.set_xlabel("Effective feedback rank")
        ax.set_ylabel("Final test accuracy")
        ax.set_ylim(0.0, 1.04)
        ax.set_title(pretty_condition(condition))
        clean_axes(ax)

    ax = axes[5]
    final = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_final_diagnostics.csv")
    if not final.empty:
        local = final[final["method"].isin(["dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc", "drtp_random"])]
        for method, sub in local.groupby("method"):
            ax.scatter(sub["projected_step_ratio_mean"], sub["test_acc"], s=18, color=COLORS.get(method, "0.4"), alpha=0.45, edgecolor="none")
        ax.axvline(0.1, color="#6B7280", lw=0.9, ls="--")
    ax.set_xlabel("Projected BP-step ratio")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Useful projected descent")
    clean_axes(ax)

    label_panels(axes)
    legend_for_methods(fig, methods, ncol=4, y=1.09)
    save(fig, "infodfa_multioutput_synthetic")


def make_multioutput_diagnostics() -> None:
    final = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_final_diagnostics.csv")
    elbows = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_rank_elbows.csv")
    if final.empty:
        return
    local = final[final["method"] != "bp"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.15), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    if not elbows.empty:
        methods = [m for m in ["dfa_random", "ndfa_random", "ndfa_random_kronecker", "nmnc", "drtp_random"] if m in set(elbows["method"])]
        conditions = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
        mat = np.full((len(conditions), len(methods)), np.nan)
        for i, condition in enumerate(conditions):
            vals = elbows[elbows["condition"] == condition].set_index("method")["rank_elbow_95pct_best"].reindex(methods)
            mat[i, :] = vals.to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=1, vmax=7)
        ax.set_xticks(np.arange(len(methods)), [pretty_method(m) for m in methods], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(conditions)), [pretty_condition(c) for c in conditions])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.8, color="#111827")
        cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.012)
        cb.set_label("rank")
    ax.set_title("Feedback-rank sufficiency")
    clean_axes(ax, grid=False)

    ax = axes[1]
    if "projected_step_ratio_mean" in local:
        local["useful_step"] = local["projected_step_ratio_mean"] >= 0.1
        groups = [(False, "low\nprojected step", "#9CA3AF"), (True, "useful\nprojected step", COLORS["ndfa_random"])]
        for i, (flag, label, color) in enumerate(groups):
            vals = local.loc[local["useful_step"] == flag, "test_acc"].dropna()
            ax.bar(i, vals.mean(), yerr=sem(vals), width=0.54, color=color, error_kw={"lw": 0.8, "capsize": 2.0}, zorder=3)
            ax.text(i, vals.mean() + 0.055, f"n={len(vals)}", ha="center", fontsize=6.5, color="#374151")
        ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Projected descent is the strongest diagnostic")
    clean_axes(ax)

    ax = axes[2]
    if "nuisance_dominant_design" in local:
        groups = [(False, "task-bearing\ngeometry", COLORS["ndfa_random"]), (True, "nuisance-\ndominant", COLORS["ndfa_random_kronecker"])]
        for i, (flag, label, color) in enumerate(groups):
            vals = local.loc[local["nuisance_dominant_design"] == flag, "test_acc"].dropna()
            ax.bar(i, vals.mean(), yerr=sem(vals), width=0.54, color=color, error_kw={"lw": 0.8, "capsize": 2.0}, zorder=3)
            ax.text(i, vals.mean() + 0.055, f"n={len(vals)}", ha="center", fontsize=6.5, color="#374151")
        ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Nuisance geometry is a failure mode")
    clean_axes(ax)

    ax = axes[3]
    xcol = "manifold_condition_margin_mean"
    corr_labels = ["global", "after condition\nand method means"]
    corrs = [np.nan, np.nan]
    if xcol in local:
        xy = local[[xcol, "test_acc", "condition", "method"]].dropna().copy()
        if len(xy) > 2:
            corrs[0] = float(np.corrcoef(xy[xcol], xy["test_acc"])[0, 1])
            xy["x_resid"] = xy[xcol] - xy.groupby(["condition", "method"])[xcol].transform("mean")
            xy["y_resid"] = xy["test_acc"] - xy.groupby(["condition", "method"])["test_acc"].transform("mean")
            ok = xy[["x_resid", "y_resid"]].dropna()
            if len(ok) > 2 and ok["x_resid"].std() > 0 and ok["y_resid"].std() > 0:
                corrs[1] = float(np.corrcoef(ok["x_resid"], ok["y_resid"])[0, 1])
    colors = ["#9CA3AF", COLORS["dfa_random"]]
    ax.bar([0, 1], corrs, color=colors, width=0.56, zorder=3)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks([0, 1], corr_labels)
    ax.set_ylabel("Correlation with accuracy")
    ax.set_ylim(-0.45, 0.45)
    ax.set_title("Generic manifold overlap is not enough")
    clean_axes(ax)

    label_panels(axes)
    save(fig, "infodfa_multioutput_diagnostics")


def make_preconditioning_spectrum() -> None:
    summary = read_csv(RESULTS / "dfa_preconditioning_spectrum_aggregate_v1" / "dfa_preconditioning_spectrum_summary.csv")
    all_df = read_csv(RESULTS / "dfa_preconditioning_spectrum_aggregate_v1" / "dfa_preconditioning_spectrum_all.csv")
    if summary.empty:
        return
    conditions = ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.2), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    for condition in conditions:
        sub = summary[(summary["condition"] == condition) & (summary["feedback_rank"] == 0)].groupby("gamma", as_index=False).agg(test_mean=("test_mean", "mean"), test_sem=("test_sem", "mean")).sort_values("gamma")
        if not sub.empty:
            line_with_sem(ax, sub["gamma"], sub["test_mean"], sub["test_sem"], color=COLORS[condition], label=pretty_condition(condition))
    ax.set_xlabel("K-nDFA blend $\\gamma$")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Turning on local whitening")
    clean_axes(ax)

    for ax, condition in zip(axes[1:3], ["nuisance_dominant", "mixed_context"]):
        sub = all_df[(all_df["condition"] == condition) & (all_df["feedback_rank"] == 0)] if not all_df.empty else pd.DataFrame()
        for gamma in [0.0, 0.25, 0.5, 1.0]:
            g = sub[np.isclose(sub["gamma"], gamma)]
            if g.empty:
                continue
            curve = g.groupby("epoch", as_index=False).agg(mean=("test_acc", "mean"), se=("test_acc", sem))
            line_with_sem(ax, curve["epoch"], curve["mean"], curve["se"], color=mpl.cm.viridis(float(gamma)), label=f"$\\gamma$={gamma:g}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(pretty_condition(condition))
        clean_axes(ax)

    ax = axes[3]
    for condition in conditions:
        sub = summary[(summary["condition"] == condition) & (summary["feedback_rank"] == 0)].groupby("gamma", as_index=False).agg(step=("blended_step", "mean")).sort_values("gamma")
        if not sub.empty:
            ax.plot(sub["gamma"], sub["step"], color=COLORS[condition], marker="o", lw=1.7, ms=4.0, label=pretty_condition(condition))
    ax.axhline(0.0, color="#6B7280", lw=0.9)
    ax.set_xlabel("K-nDFA blend $\\gamma$")
    ax.set_ylabel("Projection onto BP update")
    ax.set_title("Useful step changes continuously")
    clean_axes(ax)

    label_panels(axes)
    axes[0].legend(frameon=False, ncol=1, loc="lower right")
    axes[1].legend(frameon=False, ncol=2, loc="lower right")
    save(fig, "infodfa_preconditioning_spectrum")


def make_nmnc_comparison() -> None:
    summary = read_csv(RESULTS / "dfa_nmnc_aggregate_v1" / "dfa_nmnc_summary.csv")
    all_df = read_csv(RESULTS / "dfa_nmnc_aggregate_v1" / "dfa_nmnc_all.csv")
    if summary.empty:
        return
    best = best_rows(summary, ["dataset", "method"])
    datasets = [d for d in ["mnist", "fashion_mnist", "cifar10"] if d in set(best["dataset"])]
    methods = ["bp", "dfa_random", "fa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc", "drtp_random"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.2), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.095
    for i, method in enumerate(methods):
        vals = best[best["method"] == method].set_index("dataset")["test_mean"].reindex(datasets)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, color=COLORS[method], zorder=3)
    ax.set_xticks(x, [pretty_dataset(d) for d in datasets], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Best final test accuracy")
    ax.set_title("Standard MLP benchmark")
    clean_axes(ax)

    ax = axes[1]
    for dataset, ls in [("mnist", "-"), ("fashion_mnist", "--"), ("cifar10", ":")]:
        sub = summary[(summary["dataset"] == dataset) & (summary["method"].isin(["vnc", "nmnc"]))]
        for method in ["vnc", "nmnc"]:
            m = sub[sub["method"] == method].groupby("nc_update_interval", as_index=False).agg(test_mean=("test_mean", "mean"), test_sem=("test_sem", "mean")).sort_values("nc_update_interval")
            if not m.empty:
                line_with_sem(ax, m["nc_update_interval"], m["test_mean"], m["test_sem"], method=method, linestyle=ls, label=f"{pretty_method(method)} {pretty_dataset(dataset)}")
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation update interval")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Noise-correlation hyperparameters")
    clean_axes(ax)

    final = final_rows(all_df, ["dataset", "method", "seed", "feedback_seed", "nc_update_interval", "nc_manifold_rank"])
    for ax, xcol, xlabel, title in [
        (axes[2], "manifold_condition_margin_mean", "Manifold-gradient margin", "Manifold diagnostic"),
        (axes[3], "projected_step_ratio_mean", "Projected BP-step ratio", "Useful-step diagnostic"),
    ]:
        nc = final[final["method"].isin(["vnc", "nmnc"])]
        for method, sub in nc.groupby("method"):
            ax.scatter(sub[xcol], sub["test_acc"], color=COLORS[method], s=18, alpha=0.42, edgecolor="none", label=pretty_method(method))
        ax.axvline(0, color="#6B7280", lw=0.8, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final test accuracy")
        ax.set_title(title)
        clean_axes(ax)

    label_panels(axes)
    legend_for_methods(fig, methods, ncol=4, y=1.08)
    save(fig, "infodfa_nmnc_comparison")


def make_learning_dynamics() -> None:
    multi = read_csv(RESULTS / "dfa_multioutput_synthetic_aggregate_v1" / "dfa_multioutput_all.csv")
    nmnc = read_csv(RESULTS / "dfa_nmnc_aggregate_v1" / "dfa_nmnc_all.csv")
    if multi.empty or nmnc.empty:
        return
    methods = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc", "drtp_random"]
    fig, axes = plt.subplots(2, 4, figsize=(8.9, 5.05), constrained_layout=True)
    axes = axes.ravel()

    for ax, condition in zip(axes[:4], ["task_aligned", "nuisance_dominant", "mixed_context", "low_sample_noisy"]):
        sub = multi[multi["condition"] == condition]
        plot_epoch_curves(ax, sub, methods, y_col="test_acc", group_cols=["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval"])
        ax.set_title(pretty_condition(condition))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)
        clean_axes(ax)

    for ax, dataset in zip(axes[4:7], ["mnist", "fashion_mnist", "cifar10"]):
        sub = nmnc[nmnc["dataset"] == dataset]
        plot_epoch_curves(ax, sub, methods, y_col="test_acc", group_cols=["dataset", "method", "seed", "feedback_seed", "nc_update_interval", "nc_manifold_rank"])
        ax.set_title(pretty_dataset(dataset))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.02)
        clean_axes(ax)

    ax = axes[7]
    plot_epoch_curves(ax, multi[multi["method"] != "bp"], ["dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc"], y_col="projected_step_ratio_mean", group_cols=["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval"], fill=False)
    ax.axhline(0.1, color="#6B7280", lw=0.9, ls="--")
    ax.set_title("Useful projected step")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Projected BP-step ratio")
    clean_axes(ax)

    label_panels(axes)
    legend_for_methods(fig, methods, ncol=7, y=1.055)
    save(fig, "infodfa_learning_dynamics")


def plot_epoch_curves(ax: plt.Axes, df: pd.DataFrame, methods: list[str], *, y_col: str, group_cols: list[str], fill: bool = True) -> None:
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty or y_col not in sub:
            continue
        cols = [c for c in group_cols if c in sub.columns]
        run = sub[cols + ["epoch", y_col]].dropna(subset=["epoch", y_col]).groupby(cols + ["epoch"], as_index=False).agg(value=(y_col, "mean"))
        curve = run.groupby("epoch", as_index=False).agg(mean=("value", "mean"), se=("value", sem)).sort_values("epoch")
        color = COLORS.get(method, "0.4")
        ax.plot(curve["epoch"], curve["mean"], color=color, lw=1.5, label=pretty_method(method), zorder=3)
        if fill:
            ax.fill_between(curve["epoch"].to_numpy(float), (curve["mean"] - curve["se"]).to_numpy(float), (curve["mean"] + curve["se"]).to_numpy(float), color=color, alpha=0.10, linewidth=0)


def make_convnet_baselines() -> None:
    summary = read_csv(RESULTS / "dfa_convnet_calibration_aggregate_v1" / "dfa_convnet_summary.csv")
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75), constrained_layout=True)
    best = best_rows(summary, ["method", "feedback_mode", "feedback_normalization"])
    ax = axes[0]
    show = best.sort_values("test_mean", ascending=False).head(10).sort_values("test_mean")
    labels = [f"{pretty_method(r.method)}\n{r.feedback_mode}/{r.feedback_normalization}" for r in show.itertuples()]
    ax.barh(labels, show["test_mean"], color=[COLORS.get(m, "0.5") for m in show["method"]], zorder=3)
    ax.set_xlabel("Final test accuracy")
    ax.set_title("Calibrated CIFAR-10 convnet")
    clean_axes(ax)

    for ax, xcol, xlabel in [(axes[1], "projected_step", "All-parameter projected step"), (axes[2], "param_cosine", "All-parameter cosine")]:
        local = summary[summary["method"] != "bp"]
        for method, sub in local.groupby("method"):
            ax.scatter(sub[xcol], sub["test_mean"], s=25, color=COLORS.get(method, "0.4"), alpha=0.7, edgecolor="white", linewidth=0.35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final test accuracy")
        ax.set_title("Diagnostic, not just accuracy")
        clean_axes(ax)
    label_panels(axes)
    save(fig, "infodfa_convnet_baselines")


def make_convnet_conditioning() -> None:
    summary = read_csv(RESULTS / "dfa_convnet_conditioning_aggregate_v1" / "dfa_convnet_summary.csv")
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75), constrained_layout=True)
    for ax, method in zip(axes[:2], ["dfa_random", "ndfa_random"]):
        sub = summary[summary["method"] == method]
        for damping, dsub in sub.groupby("natural_damping"):
            curve = dsub.groupby("feedback_rank", as_index=False).agg(test_mean=("test_mean", "mean"), test_sem=("test_sem", "mean")).sort_values("feedback_rank")
            line_with_sem(ax, curve["feedback_rank"], curve["test_mean"], curve["test_sem"], method=method, label=f"$\\lambda$={damping:g}")
        bp = summary[summary["method"] == "bp"]
        if not bp.empty:
            ax.axhline(bp["test_mean"].max(), color=COLORS["bp"], lw=1.0, ls=":")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Feedback rank")
        ax.set_ylabel("Final test accuracy")
        ax.set_title(pretty_method(method))
        clean_axes(ax)
    ax = axes[2]
    local = summary[summary["method"] != "bp"]
    ax.scatter(local["projected_step"], local["test_mean"], c=[COLORS.get(m, "0.4") for m in local["method"]], s=24, alpha=0.7, edgecolor="white", linewidth=0.35)
    ax.set_xlabel("Projected step")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Conditioning works through useful step")
    clean_axes(ax)
    label_panels(axes)
    axes[0].legend(frameon=False, fontsize=6.0, loc="lower right")
    save(fig, "infodfa_convnet_conditioning")


def make_convnet_extra_baselines() -> None:
    summary = read_csv(RESULTS / "dfa_convnet_extra_baselines_aggregate_v1" / "dfa_convnet_summary.csv")
    if summary.empty:
        return
    methods = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "local_loss", "drtp_random"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.15), constrained_layout=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes[:2], ["cifar10", "cifar100"]):
        sub = summary[(summary["dataset"] == dataset) & (summary["method"].isin(methods))]
        best = best_rows(sub, ["method"]).set_index("method").reindex(methods).reset_index()
        ax.bar(np.arange(len(methods)), best["test_mean"], yerr=best["test_sem"].fillna(0), color=[COLORS[m] for m in methods], zorder=3, error_kw={"lw": 0.8, "capsize": 2})
        ax.set_xticks(np.arange(len(methods)), [pretty_method(m) for m in methods], rotation=25, ha="right")
        ax.set_ylim(0.0, max(0.75, float(best["test_mean"].max()) + 0.08))
        ax.set_ylabel("Final test accuracy")
        ax.set_title(pretty_dataset(dataset))
        clean_axes(ax)
    local = summary[summary["method"].isin(["dfa_random", "ndfa_random", "ndfa_random_kronecker"])]
    for ax, xcol, title in [(axes[2], "hidden_projected_step", "Hidden projected step"), (axes[3], "hidden_param_cosine", "Hidden gradient cosine")]:
        if xcol in local:
            for method, sub in local.groupby("method"):
                ax.scatter(sub[xcol], sub["test_mean"], color=COLORS[method], s=28, alpha=0.72, edgecolor="white", linewidth=0.35)
        ax.set_xlabel(title)
        ax.set_ylabel("Final test accuracy")
        ax.set_title("Hidden-credit diagnostic")
        clean_axes(ax)
    label_panels(axes)
    legend_for_methods(fig, methods, ncol=6, y=1.07)
    save(fig, "infodfa_convnet_extra_baselines")


def make_convnet_harder() -> None:
    summary = read_csv(RESULTS / "dfa_convnet_cifar100_harder_aggregate_v1" / "dfa_convnet_summary.csv")
    matched = read_csv(RESULTS / "dfa_convnet_cifar100_harder_matched_bp_aggregate_v1" / "dfa_convnet_summary.csv")
    if summary.empty and matched.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
    combined = pd.concat([summary, matched], ignore_index=True)
    methods = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "local_loss", "drtp_random"]
    best = best_rows(combined[combined["method"].isin(methods)], ["method"]).set_index("method").reindex(methods).reset_index()
    ax = axes[0]
    ax.bar(np.arange(len(methods)), best["test_mean"], yerr=best["test_sem"].fillna(0), color=[COLORS[m] for m in methods], zorder=3, error_kw={"lw": 0.8, "capsize": 2})
    ax.set_xticks(np.arange(len(methods)), [pretty_method(m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Larger CIFAR-100 convnet")
    clean_axes(ax)
    ax = axes[1]
    local = combined[combined["method"].isin(["dfa_random", "ndfa_random", "ndfa_random_kronecker"])]
    if "hidden_projected_step" in local:
        for method, sub in local.groupby("method"):
            ax.scatter(sub["hidden_projected_step"], sub["test_mean"], color=COLORS[method], s=30, alpha=0.78, edgecolor="white", linewidth=0.35)
    ax.set_xlabel("Hidden projected step")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Mechanism check")
    clean_axes(ax)
    label_panels(axes)
    save(fig, "infodfa_convnet_cifar100_harder")


if __name__ == "__main__":
    main()
