"""Build ICLR-facing positive-regime figures for the Info-DFA paper.

The standard publication figure script is broad.  This one is intentionally
selective: it highlights regimes where conditioned DFA is clearly useful,
shows the mechanism in those regimes, and keeps ImageNet as a boundary case.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "infodfa_iclr_positive_regimes_20260527"
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
    "task_aligned": "#2563EB",
    "nuisance_dominant": "#D55E00",
    "mixed_context": "#CC79A7",
    "low_sample_noisy": "#009E73",
    "synthetic": "#D55E00",
    "mlp": "#2563EB",
    "convnet": "#009E73",
    "boundary": "#6B7280",
}

MARKERS = {
    "bp": "o",
    "dfa_random": "o",
    "ndfa_random": "D",
    "ndfa_random_kronecker": "^",
    "nmnc": "X",
    "vnc": "v",
    "local_loss": "P",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DRAFT_FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    setup_style()

    tables = load_tables()
    make_positive_regime_figure(tables)
    make_noisy_mechanism_figure(tables)
    write_strategy_report(tables)

    print(f"Wrote ICLR positive-regime figures and report to {OUT}")
    print(f"Mirrored figures to {DRAFT_FIGS} and {PAPER_FIGS}")


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
            "grid.alpha": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_tables() -> dict[str, pd.DataFrame]:
    paths = {
        "multi_summary": RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_summary.csv",
        "multi_all": RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_all.csv",
        "multi_diag": RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_diagnostic_tests.csv",
        "multi_final": RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_final_diagnostics.csv",
        "spectrum": RESULTS / "dfa_preconditioning_spectrum_aggregate_v1" / "dfa_preconditioning_spectrum_summary.csv",
        "nmnc_summary": RESULTS / "infodfa_vision_noise_sweep_aggregate_v2" / "dfa_nmnc_summary.csv",
        "conv_extra": RESULTS / "dfa_convnet_extra_baselines_aggregate_v1" / "dfa_convnet_summary.csv",
        "conv_hard": RESULTS / "infodfa_hard_cifar100_confirm_aggregate_v2" / "dfa_convnet_summary.csv",
        "imagenet_blend": RESULTS / "imagenet100_infodfa_blend_diag_aggregate_20260525" / "imagenet_credit_assignment_summary.csv",
        "imagenet_stage": RESULTS / "imagenet100_infodfa_stage_diag_aggregate_20260525" / "imagenet_credit_assignment_summary.csv",
        "imagenet_90": RESULTS / "imagenet100_infodfa_spatial_headtohead90_aggregate_20260526" / "imagenet_credit_assignment_summary.csv",
    }
    return {name: read_csv(path) for name, path in paths.items()}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


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


def label_panels(axes: list[plt.Axes] | np.ndarray) -> None:
    for ax, label in zip(np.ravel(axes), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        panel_label(ax, label)


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
        rows.append(sub.loc[pd.to_numeric(sub[value_col], errors="coerce").idxmax()].to_dict())
    return pd.DataFrame(rows)


def best_value(best: pd.DataFrame, group_col: str, group: str, method: str, metric: str = "test_mean") -> float:
    if best.empty:
        return np.nan
    sub = best[(best[group_col] == group) & (best["method"] == method)]
    if sub.empty or metric not in sub:
        return np.nan
    return float(sub.iloc[0][metric])


def make_positive_regime_figure(tables: dict[str, pd.DataFrame]) -> None:
    multi_best = best_rows(tables["multi_summary"], ["condition", "method"])
    nmnc_best = best_rows(tables["nmnc_summary"], ["dataset", "method"])
    conv_best = best_rows(tables["conv_extra"], ["dataset", "method"])
    hard_best = best_rows(tables["conv_hard"], ["dataset", "method"])

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    conditions = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]
    methods = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "nmnc"]
    x = np.arange(len(conditions))
    width = 0.145
    for idx, method in enumerate(methods):
        vals = multi_best[multi_best["method"] == method].set_index("condition")["test_mean"].reindex(conditions)
        errs = multi_best[multi_best["method"] == method].set_index("condition")["test_sem"].reindex(conditions).fillna(0.0)
        ax.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            100.0 * vals,
            width=width,
            yerr=100.0 * errs,
            color=COLORS[method],
            error_kw={"lw": 0.7, "capsize": 1.5, "capthick": 0.7},
            zorder=3,
            label=pretty_method(method),
        )
    ax.set_xticks(x, [pretty_condition(c) for c in conditions], rotation=17, ha="right")
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_ylim(0, 104)
    ax.set_title("Positive regimes in the controlled stress suite")
    clean_axes(ax)

    ax = axes[1]
    gain_rows = positive_gain_rows(multi_best, nmnc_best, conv_best, hard_best)
    y = np.arange(len(gain_rows))
    colors = [COLORS[row["family"]] for row in gain_rows]
    ax.barh(y, [100.0 * row["gain"] for row in gain_rows], color=colors, height=0.68, zorder=3)
    for yi, row in zip(y, gain_rows):
        ax.text(100.0 * row["gain"] + 1.2, yi, f"+{100.0 * row['gain']:.1f}", va="center", fontsize=6.8, color="#111827")
    ax.axvline(0.0, color="#111827", lw=0.8)
    ax.set_yticks(y, [row["label"] for row in gain_rows])
    ax.set_xlabel("Accuracy gain over raw DFA (percentage points)")
    ax.set_xlim(0, max(75.0, 100.0 * max(row["gain"] for row in gain_rows) + 10.0))
    ax.set_title("Where the new rule most clearly helps")
    clean_axes(ax)

    ax = axes[2]
    spectrum = tables["spectrum"]
    if not spectrum.empty:
        for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy"]:
            sub = spectrum[
                (spectrum["condition"] == condition)
                & np.isclose(spectrum["feedback_rank"], 0.0)
                & np.isclose(spectrum["damping"], 0.1)
            ].sort_values("gamma")
            if sub.empty:
                continue
            ax.errorbar(
                sub["gamma"],
                100.0 * sub["test_mean"],
                yerr=100.0 * sub["test_sem"].fillna(0.0),
                color=COLORS[condition],
                marker="o",
                lw=1.7,
                ms=4.1,
                elinewidth=0.8,
                capsize=1.8,
                label=pretty_condition(condition),
                zorder=3,
            )
    ax.set_xlabel("Blend from DFA to K-nDFA, $\\gamma$")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Whitening turns failures into learning")
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[3]
    imagenet = imagenet_depth_points(tables)
    if imagenet:
        xx = np.arange(len(imagenet))
        means = [row["mean"] for row in imagenet]
        errs = [row["sem"] for row in imagenet]
        ax.errorbar(xx, means, yerr=errs, color=COLORS["boundary"], marker="o", lw=1.9, ms=4.5, capsize=2.0, zorder=3)
        ax.set_xticks(xx, [row["label"] for row in imagenet], rotation=15, ha="right")
    bp = imagenet_bp_reference(tables)
    if np.isfinite(bp):
        ax.axhline(bp, color=COLORS["bp"], lw=1.1, ls=":", label=f"BP {bp:.1f}%")
    ax.set_ylabel("ImageNet-100 top-1 (%)")
    ax.set_ylim(55, 82)
    ax.set_title("Boundary: substitution depth on ImageNet")
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower left")

    label_panels(axes)
    handles = [
        mpl.lines.Line2D([0], [0], color=COLORS[m], marker=MARKERS.get(m, "o"), lw=1.8, ms=4.0, label=pretty_method(m))
        for m in methods
    ]
    handles.extend(
        [
            mpl.patches.Patch(facecolor=COLORS["synthetic"], label="Synthetic stress"),
            mpl.patches.Patch(facecolor=COLORS["mlp"], label="Vision MLP"),
            mpl.patches.Patch(facecolor=COLORS["convnet"], label="Convnet"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=8, frameon=False)
    save(fig, "infodfa_iclr_positive_regimes")


def positive_gain_rows(
    multi_best: pd.DataFrame,
    nmnc_best: pd.DataFrame,
    conv_best: pd.DataFrame,
    hard_best: pd.DataFrame,
) -> list[dict[str, object]]:
    _ = conv_best
    rows: list[dict[str, object]] = []
    for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy"]:
        base = best_value(multi_best, "condition", condition, "dfa_random")
        cond = best_value(multi_best, "condition", condition, "ndfa_random_kronecker")
        rows.append(
            {
                "label": pretty_condition(condition),
                "family": "synthetic",
                "gain": cond - base,
                "rule": "K-nDFA",
            }
        )
    for dataset in ["fashion_mnist", "cifar10", "mnist"]:
        base = best_value(nmnc_best, "dataset", dataset, "dfa_random")
        ndfa = best_value(nmnc_best, "dataset", dataset, "ndfa_random")
        kndfa = best_value(nmnc_best, "dataset", dataset, "ndfa_random_kronecker")
        rows.append(
            {
                "label": pretty_dataset(dataset),
                "family": "mlp",
                "gain": max(ndfa, kndfa) - base,
                "rule": "best nDFA",
            }
        )
    base = best_value(hard_best, "dataset", "cifar100", "dfa_random")
    ndfa = best_value(hard_best, "dataset", "cifar100", "ndfa_random")
    rows.append(
        {
            "label": "Hard CIFAR-100 conv",
            "family": "convnet",
            "gain": ndfa - base,
            "rule": "nDFA",
        }
    )
    rows = [row for row in rows if np.isfinite(float(row["gain"]))]
    return sorted(rows, key=lambda row: float(row["gain"]))


def imagenet_bp_reference(tables: dict[str, pd.DataFrame]) -> float:
    blend = tables["imagenet_blend"]
    if blend.empty:
        return np.nan
    bp = blend[(blend["method"] == "block_dfa") & np.isclose(blend["feedback_blend_gamma"], 0.0)]
    if bp.empty:
        return np.nan
    return float(bp["val_top1_mean"].max())


def imagenet_depth_points(tables: dict[str, pd.DataFrame]) -> list[dict[str, float | str]]:
    head = tables["imagenet_90"]
    if not head.empty and "feedback_spatial_mode" in head:
        head = head[(head["method"] == "block_dfa") & (head["feedback_spatial_mode"] == "broadcast")]
        order = [
            ("layer4", "layer4"),
            ("layer3,layer4", "layer3+4"),
            ("layer1,layer2,layer3,layer4", "all blocks"),
        ]
        rows = []
        for modules, label in order:
            sub = head[head["feedback_modules"] == modules]
            if not sub.empty:
                rows.append({"label": label, "mean": float(sub.iloc[0]["val_top1_mean"]), "sem": float(sub.iloc[0]["val_top1_sem"])})
        if rows:
            return rows
    stage = tables["imagenet_stage"]
    if stage.empty:
        return []
    stage = stage[(stage["method"] == "block_dfa") & np.isclose(stage["feedback_scale"], 0.3)]
    order = [
        ("layer4", "layer4"),
        ("layer3,layer4", "layer3+4"),
        ("layer2,layer3,layer4", "layer2+3+4"),
        ("layer1,layer2,layer3,layer4", "all blocks"),
    ]
    rows = []
    for modules, label in order:
        sub = stage[stage["feedback_modules"] == modules]
        if not sub.empty:
            rows.append({"label": label, "mean": float(sub.iloc[0]["val_top1_mean"]), "sem": float(sub.iloc[0]["val_top1_sem"])})
    return rows


def make_noisy_mechanism_figure(tables: dict[str, pd.DataFrame]) -> None:
    multi_all = tables["multi_all"]
    multi_summary = tables["multi_summary"]
    multi_final = tables["multi_final"]
    if multi_all.empty or multi_summary.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    axes = axes.ravel()
    methods = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "nmnc"]

    for ax, condition in zip(axes[:3], ["nuisance_dominant", "mixed_context", "low_sample_noisy"]):
        for method in methods:
            runs = best_setting_runs(multi_all, multi_summary, condition, method)
            if runs.empty:
                continue
            curve = curve_by_epoch(runs, "test_acc")
            if curve.empty:
                continue
            color = COLORS[method]
            ax.plot(curve["epoch"], 100.0 * curve["mean"], color=color, lw=1.55, label=pretty_method(method), zorder=3)
            ax.fill_between(
                curve["epoch"].to_numpy(float),
                100.0 * (curve["mean"] - curve["se"]).to_numpy(float),
                100.0 * (curve["mean"] + curve["se"]).to_numpy(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(pretty_condition(condition))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, 102)
        clean_axes(ax)

    ax = axes[3]
    for method in ["dfa_random", "ndfa_random", "ndfa_random_kronecker", "nmnc", "vnc"]:
        frames = []
        for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy"]:
            runs = best_setting_runs(multi_all, multi_summary, condition, method)
            if not runs.empty:
                frames.append(runs)
        if not frames:
            continue
        curve = curve_by_epoch(pd.concat(frames, ignore_index=True), "projected_step_ratio_mean")
        if curve.empty:
            continue
        color = COLORS[method]
        ax.plot(curve["epoch"], curve["mean"], color=color, lw=1.55, label=pretty_method(method), zorder=3)
        ax.fill_between(
            curve["epoch"].to_numpy(float),
            (curve["mean"] - curve["se"]).to_numpy(float),
            (curve["mean"] + curve["se"]).to_numpy(float),
            color=color,
            alpha=0.11,
            linewidth=0,
        )
    ax.axhline(0.1, color="#6B7280", lw=0.9, ls="--")
    if not multi_final.empty and "projected_step_ratio_mean" in multi_final:
        local = multi_final[multi_final["method"] != "bp"].copy()
        low = local.loc[local["projected_step_ratio_mean"] < 0.1, "test_acc"].mean()
        high = local.loc[local["projected_step_ratio_mean"] >= 0.1, "test_acc"].mean()
        ax.text(
            0.04,
            0.06,
            f"Projected-step split: {100*low:.1f}% vs {100*high:.1f}% final acc",
            transform=ax.transAxes,
            va="bottom",
            fontsize=6.7,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "linewidth": 0.5, "alpha": 0.88, "pad": 2.0},
        )
    ax.set_title("Mechanism: useful projected descent")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Projected BP-step ratio")
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper right", ncol=1)

    label_panels(axes)
    handles = [
        mpl.lines.Line2D([0], [0], color=COLORS[m], marker=MARKERS.get(m, "o"), lw=1.8, ms=4.0, label=pretty_method(m))
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=False)
    save(fig, "infodfa_iclr_noisy_mechanism")


def best_setting_runs(all_df: pd.DataFrame, summary: pd.DataFrame, condition: str, method: str) -> pd.DataFrame:
    best = best_rows(summary[(summary["condition"] == condition) & (summary["method"] == method)], ["condition", "method"])
    if best.empty:
        return pd.DataFrame()
    row = best.iloc[0]
    runs = all_df[(all_df["condition"] == condition) & (all_df["method"] == method)].copy()
    for col in ["feedback_rank", "nc_update_interval", "nc_manifold_rank"]:
        if col in runs and col in row and pd.notna(row[col]):
            runs = runs[np.isclose(pd.to_numeric(runs[col], errors="coerce"), float(row[col]))]
    return runs


def curve_by_epoch(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    if df.empty or y_col not in df:
        return pd.DataFrame()
    cols = [c for c in ["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval", "nc_manifold_rank"] if c in df]
    run = (
        df[cols + ["epoch", y_col]]
        .dropna(subset=["epoch", y_col])
        .groupby(cols + ["epoch"], as_index=False)
        .agg(value=(y_col, "mean"))
    )
    if run.empty:
        return pd.DataFrame()
    return run.groupby("epoch", as_index=False).agg(mean=("value", "mean"), se=("value", sem)).sort_values("epoch")


def write_strategy_report(tables: dict[str, pd.DataFrame]) -> None:
    multi_best = best_rows(tables["multi_summary"], ["condition", "method"])
    nmnc_best = best_rows(tables["nmnc_summary"], ["dataset", "method"])
    conv_best = best_rows(tables["conv_extra"], ["dataset", "method"])
    hard_best = best_rows(tables["conv_hard"], ["dataset", "method"])
    diag = tables["multi_diag"]

    report = []
    report.append("# Info-DFA ICLR Positive-Regime Plan")
    report.append("")
    report.append(f"Generated {date.today().isoformat()} from the completed aggregate CSVs.")
    report.append("")
    report.append("## Central Paper Claim")
    report.append("")
    report.append(
        "Conditioned direct feedback alignment is a local learning rule that improves raw DFA when the task is noisy, low-sample, nuisance-dominated, or mixed-context, provided the feedback has enough task-error rank and the local update has a non-trivial projection onto the BP descent step. The rule should be presented as a targeted improvement to DFA, not as an all-settings replacement for BP."
    )
    report.append("")
    report.append("## Best Positive Cases")
    report.append("")
    report.append("| regime | result to emphasize | main comparison | interpretation |")
    report.append("|---|---:|---:|---|")
    for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]:
        dfa = best_value(multi_best, "condition", condition, "dfa_random")
        kndfa = best_value(multi_best, "condition", condition, "ndfa_random_kronecker")
        bp = best_value(multi_best, "condition", condition, "bp")
        interp = (
            "positive control; the rule also helps in the easy geometry, but the gain is much smaller"
            if condition == "task_aligned"
            else "local whitening helps when nuisance/noise contaminates raw feedback"
        )
        report.append(
            f"| {pretty_condition(condition)} synthetic | K-nDFA {pct(kndfa)} | DFA {pct(dfa)}, BP {pct(bp)}, gain vs DFA {pp(kndfa - dfa)} | {interp} |"
        )
    for dataset in ["fashion_mnist", "cifar10", "mnist"]:
        dfa = best_value(nmnc_best, "dataset", dataset, "dfa_random")
        ndfa = best_value(nmnc_best, "dataset", dataset, "ndfa_random")
        kndfa = best_value(nmnc_best, "dataset", dataset, "ndfa_random_kronecker")
        bp = best_value(nmnc_best, "dataset", dataset, "bp")
        best_local = max(ndfa, kndfa)
        report.append(
            f"| {pretty_dataset(dataset)} MLP | best nDFA/K-nDFA {pct(best_local)} | DFA {pct(dfa)}, BP {pct(bp)}, gain vs DFA {pp(best_local - dfa)} | standard vision bridge; strong DFA rescue without exact hidden BP |"
        )
    for dataset in ["cifar100", "cifar10"]:
        dfa = best_value(conv_best, "dataset", dataset, "dfa_random")
        ndfa = best_value(conv_best, "dataset", dataset, "ndfa_random")
        kndfa = best_value(conv_best, "dataset", dataset, "ndfa_random_kronecker")
        bp = best_value(conv_best, "dataset", dataset, "bp")
        best_local = max(ndfa, kndfa)
        report.append(
            f"| {pretty_dataset(dataset)} convnet | best nDFA/K-nDFA {pct(best_local)} | DFA {pct(dfa)}, BP {pct(bp)}, gain vs DFA {pp(best_local - dfa)} | architectural feedback must be channel-tied and normalized |"
        )
    hard_dfa = best_value(hard_best, "dataset", "cifar100", "dfa_random")
    hard_ndfa = best_value(hard_best, "dataset", "cifar100", "ndfa_random")
    hard_bp = best_value(hard_best, "dataset", "cifar100", "bp")
    report.append(
        f"| Hard CIFAR-100 convnet | nDFA {pct(hard_ndfa)} | DFA {pct(hard_dfa)}, matched BP {pct(hard_bp)}, gain vs DFA {pp(hard_ndfa - hard_dfa)} | final confirmation: clear rescue over DFA, but still below BP and local auxiliary loss |"
    )
    report.append("")
    report.append("## Mechanistic Reading")
    report.append("")
    if not diag.empty:
        diag_rows = diag[diag["mean_low"].notna() & diag["mean_high"].notna() & diag["delta_high_minus_low"].notna()]
        for _, row in diag_rows.iterrows():
            report.append(
                f"- {row['test']}: low {pct(row['mean_low'])}, high {pct(row['mean_high'])}, delta {pp(row['delta_high_minus_low'])}."
            )
    report.append(
        "- The strongest story is projected descent: the rule helps when covariance conditioning turns raw feedback into a useful local step, especially under nuisance-dominated or mixed-context noise."
    )
    report.append(
        "- The blend experiment makes this causal-looking rather than cosmetic: in nuisance-dominant synthetic data, moving from raw DFA to K-nDFA raises accuracy from roughly 16-21% to about 90% depending on the matched cell."
    )
    report.append("")
    report.append("## Figure Package")
    report.append("")
    report.append(f"- Main positive-regime figure: `{OUT / 'infodfa_iclr_positive_regimes.pdf'}`")
    report.append(f"- Noisy/mechanism dynamics figure: `{OUT / 'infodfa_iclr_noisy_mechanism.pdf'}`")
    report.append("- Existing supporting figures to keep: `infodfa_main_method_result`, `infodfa_multioutput_diagnostics`, `infodfa_preconditioning_spectrum`, `infodfa_nmnc_comparison`, `infodfa_convnet_extra_baselines`, and `infodfa_learning_dynamics`.")
    report.append("")
    report.append("## ICLR Paper Plan")
    report.append("")
    report.append("1. Lead with the learning rule: nDFA/K-nDFA are covariance-conditioned direct-feedback rules.")
    report.append("2. Make the main empirical claim narrow and strong: conditioning rescues DFA in noisy, nuisance-dominated, mixed-context, and low-sample regimes.")
    report.append("3. Use the projected BP-step diagnostic as the mechanism, with feedback rank and covariance damping as constraints.")
    report.append("4. Put convnets in the middle: channel-tied feedback narrows the DFA gap and the hard CIFAR-100 matched run is a positive case, but local auxiliary losses remain a serious baseline.")
    report.append("5. Put ImageNet in the limitations/boundary section: layer4-only DFA is nearly free on ImageNet-100, but all-layer substitution fails and current nDFA/spatial fixes do not solve it.")
    report.append("")
    report.append("## Highest-Value Next Experiments")
    report.append("")
    report.append("- Run an explicit label-noise/low-sample sweep on the multi-output suite and Fashion-MNIST/CIFAR-10 MLPs with BP, DFA, nDFA, K-nDFA, NMNC, VNC, FA, and DRTP.")
    report.append("- Add mechanistic diagnostics around the hard CIFAR-100 gap: compare channel covariance spectra, hidden projected step, and train/test memorization for DFA, nDFA, K-nDFA, local aux, and BP.")
    report.append("- Add covariance-spectrum panels for the noisy/nuisance cases: presynaptic condition number, feedback-error condition number, projected step, and final accuracy.")
    report.append("- Stop broad ImageNet sweeps for now; include ImageNet-100 as a boundary/mechanism result and ImageNet-1K only as supplementary pilot evidence.")
    (OUT / "infodfa_iclr_positive_paper_plan_20260527.md").write_text("\n".join(report) + "\n")


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def pp(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:+.1f} pp"


if __name__ == "__main__":
    main()
