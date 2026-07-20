"""Test whether conditioning decouples the learning outcome from the feedback draw.

Hypothesis: raw DFA's final accuracy depends on how the random feedback B lands
on the dominant activity eigendirections, so across feedback seeds (data seed
fixed) it has high outcome variance; nDFA / K-nDFA equalize eigendirection
weighting, so the outcome becomes nearly independent of B ("variance collapse",
first observed on ColoredMNIST: DFA 74.2+-4.6 -> nDFA 82.5+-0.3).

For every cell x method we decompose the final test accuracy into a
feedback-seed component (pooled within-data-seed variance over feedback seeds)
and a data-seed component (moment estimator of the between-data-seed variance,
i.e. a one-way nested ANOVA), then compare sd_fb across methods at matched
full-rank feedback.  Ceiling controls: (i) the same decomposition on
logit-transformed accuracy, (ii) OLS of log10 var_fb on mean accuracy with
method dummies, (iii) the ratio restricted to cells where DFA and nDFA mean
accuracies agree to within 2pp.

Inputs (5 data seeds x 3 feedback seeds unless noted):
  - <legacy>/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv
    (128-cell synthetic suite, feedback ranks {0,1,2,4,8}; main analysis at
    full-rank feedback_rank==0, rank-restricted ratios reported as robustness)
  - <legacy>/infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_all.csv
    (24 noisy-label vision MLP cells: mnist/fashion/cifar10 x n_train x noise)
  - results/infodfa_mixer_v1/*/dfa_mixer_results.csv (4 CIFAR-10 mixer cells)
  - <legacy>/dfa_coloredmnist_v1/dfa_coloredmnist_results.csv (context only:
    a single seed drives both data and feedback, so no decomposition there)

Outputs (results/infodfa_feedback_variance_v1):
  - feedback_variance_cells.csv     per cell x method: mean acc, sd_fb, sd_data,
                                    logit-scale sd_fb, run counts
  - feedback_variance_ratios.csv    per cell: sd_fb(DFA)/sd_fb(nDFA|K-nDFA)
  - feedback_variance_regime_summary.csv  per-regime medians + Wilcoxon tests
  - feedback_variance_report.md     markdown report incl. ceiling controls
  - infodfa_feedback_variance.{pdf,png,svg}  two-panel figure

CPU-only; the big synthetic CSV is streamed in chunks (about 1-2 min).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.analysis import dataframe_to_markdown, write_markdown_report

LEGACY = Path(os.environ.get("INFODFA_LEGACY_RESULTS", ROOT / ".." / "Info-Man" / "results")).resolve()
OUT = ROOT / "results" / "infodfa_feedback_variance_v1"

SYN_ALL = LEGACY / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_all.csv"
VIS_ALL = LEGACY / "infodfa_vision_noise_sweep_aggregate_v2" / "dfa_nmnc_all.csv"
MIXER_GLOB = str(ROOT / "results" / "infodfa_mixer_v1" / "*" / "dfa_mixer_results.csv")
COLORED = LEGACY / "dfa_coloredmnist_v1" / "dfa_coloredmnist_results.csv"

SYN_CELL = ["condition", "n_train", "input_noise", "train_label_noise"]
VIS_CELL = ["dataset", "n_train", "label_noise"]
MIX_CELL = ["dataset", "label_noise", "layernorm"]
METHODS = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker"]
COND_ORDER = ["nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned"]
REGIME_ORDER = COND_ORDER + ["vision_mnist", "vision_fashion_mnist", "vision_cifar10", "mixer_cifar10"]
MATCH_TOLS = [0.02, 0.05]  # "matched accuracy" = |mean(DFA) - mean(conditioned)| <= tol
LOGIT_EPS = 1e-3
CHANCE = {"synthetic": 1.0 / 8.0, "vision": 1.0 / 10.0, "mixer": 1.0 / 10.0}
FLOOR_TOL = 0.01  # a method is "at the chance floor" if mean_acc <= chance + 1pp

# Methods follow the house scheme (DFA grey-dashed, nDFA green, K-nDFA purple);
# regimes use a distinct palette deliberately disjoint from the method colours
# so a regime marker can never be misread as a training rule.
COLORS = {
    "dfa_random": "#999999",             # house grey
    "ndfa_random": "#009E73",            # house green
    "ndfa_random_kronecker": "#6A3D9A",  # house purple
    "task_aligned": "#332288",           # indigo
    "nuisance_dominant": "#882255",      # wine
    "mixed_context": "#CC6677",          # rose
    "low_sample_noisy": "#DDCC77",       # sand
    "vision_mnist": "#88CCEE",           # cyan
    "vision_fashion_mnist": "#661100",   # dark brown-red
    "vision_cifar10": "#000000",         # black
    "mixer_cifar10": "#999933",          # olive
}
# House line styles for the panel-B method fits (grayscale-legible identity).
METHOD_LS = {"dfa_random": (0, (4, 2)), "ndfa_random": "-", "ndfa_random_kronecker": (0, (5, 1.5))}
MARKERS = {"synthetic": "o", "vision": "D", "mixer": "^"}
PRETTY = {
    "nuisance_dominant": "nuisance-dominant",
    "low_sample_noisy": "low-sample noisy",
    "mixed_context": "mixed-context",
    "task_aligned": "task-aligned",
    "vision_mnist": "MNIST (noisy labels)",
    "vision_fashion_mnist": "Fashion-MNIST (noisy labels)",
    "vision_cifar10": "CIFAR-10 (noisy labels)",
    "mixer_cifar10": "CIFAR-10 mixer",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()

    syn = load_synthetic_final()
    vis = load_vision_final()
    mix = load_mixer_final()

    frames = []
    for df, cell_keys, family in [
        (syn[syn["feedback_rank"] == 0.0], SYN_CELL, "synthetic"),
        (vis, VIS_CELL, "vision"),
        (mix, MIX_CELL, "mixer"),
    ]:
        stats_df = decompose(df, cell_keys)
        stats_df["family"] = family
        stats_df["regime"] = regime_labels(stats_df, family)
        stats_df["cell"] = cell_ids(stats_df, cell_keys)
        frames.append(stats_df)
    cells = pd.concat(frames, ignore_index=True)
    cells["at_floor"] = cells["mean_acc"] <= cells["family"].map(CHANCE) + FLOOR_TOL
    cells.to_csv(OUT / "feedback_variance_cells.csv", index=False)

    ratios = build_ratios(cells)
    ratios.to_csv(OUT / "feedback_variance_ratios.csv", index=False)

    regime = regime_summary(ratios)
    regime.to_csv(OUT / "feedback_variance_regime_summary.csv", index=False)

    ceiling = ceiling_controls(cells, ratios)
    rank_rob = rank_robustness(syn)
    colored = coloredmnist_context()

    make_figure(cells, ratios, ceiling)
    write_report(cells, ratios, regime, ceiling, rank_rob, colored)
    print(f"Wrote feedback-variance collapse analysis to {OUT}")


# ---------------------------------------------------------------------------
# loading


def load_synthetic_final() -> pd.DataFrame:
    """Stream the 544MB synthetic aggregate; keep final-epoch rows only."""
    usecols = SYN_CELL + ["method", "seed", "feedback_seed", "feedback_rank", "epoch", "test_acc"]
    epoch_max = pd.read_csv(SYN_ALL, usecols=["epoch"])["epoch"].max()
    chunks = []
    for chunk in pd.read_csv(SYN_ALL, usecols=usecols, chunksize=500_000):
        keep = chunk[(chunk["epoch"] == epoch_max) & chunk["method"].isin(METHODS)]
        if not keep.empty:
            chunks.append(keep.drop(columns=["epoch"]))
    return pd.concat(chunks, ignore_index=True)


def load_vision_final() -> pd.DataFrame:
    usecols = VIS_CELL + ["method", "seed", "feedback_seed", "feedback_rank", "epoch", "test_acc"]
    df = pd.read_csv(VIS_ALL, usecols=usecols)
    df = df[df["method"].isin(METHODS) & (df["feedback_rank"] == 0.0)]
    run_keys = VIS_CELL + ["method", "seed", "feedback_seed"]
    return df.sort_values("epoch").groupby(run_keys, as_index=False).tail(1)


def load_mixer_final() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(glob.glob(MIXER_GLOB))]
    if not frames:
        return pd.DataFrame(columns=MIX_CELL + ["method", "seed", "feedback_seed", "test_acc"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(METHODS)]
    run_keys = MIX_CELL + ["method", "seed", "feedback_seed"]
    return df.sort_values("epoch").groupby(run_keys, as_index=False).tail(1)


# ---------------------------------------------------------------------------
# variance decomposition


def decompose(df: pd.DataFrame, cell_keys: list[str]) -> pd.DataFrame:
    """Nested decomposition of final test accuracy per cell x method.

    var_fb   pooled within-data-seed variance over feedback seeds
    var_data moment estimator: max(0, var(seed means) - var_fb / n_fb)
    """
    df = df.copy()
    df["acc_logit"] = logit(df["test_acc"])
    per_seed = df.groupby(cell_keys + ["method", "seed"]).agg(
        seed_mean=("test_acc", "mean"),
        fb_var=("test_acc", lambda x: x.var(ddof=1)),
        fb_var_logit=("acc_logit", lambda x: x.var(ddof=1)),
        n_fb=("test_acc", "size"),
    )
    per_seed = per_seed[per_seed["n_fb"] >= 2]
    rows = []
    for key, grp in per_seed.groupby(level=list(range(len(cell_keys) + 1))):
        var_fb = grp["fb_var"].mean()
        n_fb = grp["n_fb"].mean()
        var_seed_means = grp["seed_mean"].var(ddof=1) if len(grp) >= 2 else np.nan
        var_data = max(0.0, var_seed_means - var_fb / n_fb) if np.isfinite(var_seed_means) else np.nan
        rows.append(
            dict(
                zip(cell_keys + ["method"], key),
                mean_acc=grp["seed_mean"].mean(),
                sd_fb=np.sqrt(var_fb),
                sd_data=np.sqrt(var_data) if np.isfinite(var_data) else np.nan,
                sd_fb_logit=np.sqrt(grp["fb_var_logit"].mean()),
                n_data_seeds=len(grp),
                n_fb_seeds=n_fb,
            )
        )
    return pd.DataFrame(rows)


def logit(acc: pd.Series) -> pd.Series:
    clipped = acc.clip(LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(clipped / (1.0 - clipped))


def regime_labels(stats_df: pd.DataFrame, family: str) -> pd.Series:
    if family == "synthetic":
        return stats_df["condition"].astype(str)
    if family == "vision":
        return "vision_" + stats_df["dataset"].astype(str)
    return "mixer_" + stats_df["dataset"].astype(str)


def cell_ids(stats_df: pd.DataFrame, cell_keys: list[str]) -> pd.Series:
    return stats_df[cell_keys].astype(str).agg("|".join, axis=1)


def build_ratios(cells: pd.DataFrame) -> pd.DataFrame:
    idx = ["family", "regime", "cell"]
    wide = cells.pivot_table(index=idx, columns="method", values=["sd_fb", "sd_fb_logit", "mean_acc"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for cond, tag in [("ndfa_random", "ndfa"), ("ndfa_random_kronecker", "kndfa")]:
        wide[f"ratio_dfa_{tag}"] = safe_ratio(wide["sd_fb_dfa_random"], wide[f"sd_fb_{cond}"])
        wide[f"ratio_dfa_{tag}_logit"] = safe_ratio(wide["sd_fb_logit_dfa_random"], wide[f"sd_fb_logit_{cond}"])
        wide[f"acc_gap_{tag}"] = wide[f"mean_acc_{cond}"] - wide["mean_acc_dfa_random"]
    wide["floor_dfa"] = wide["mean_acc_dfa_random"] <= wide["family"].map(CHANCE) + FLOOR_TOL
    return wide


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# statistics


def regime_summary(ratios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = [r for r in REGIME_ORDER if r in set(ratios["regime"])]
    nonfloor = ratios[~ratios["floor_dfa"]]
    scopes = [("all", ratios), ("all_dfa_learns", nonfloor)]
    scopes += [(r, ratios[ratios["regime"] == r]) for r in order]
    for scope, grp in scopes:
        row = {
            "regime": scope,
            "n_cells": len(grp),
            "n_dfa_at_floor": int(grp["floor_dfa"].sum()),
        }
        for tag in ["ndfa", "kndfa"]:
            r = grp[f"ratio_dfa_{tag}"].dropna()
            r_nf = grp.loc[~grp["floor_dfa"], f"ratio_dfa_{tag}"].dropna()
            row[f"median_ratio_{tag}"] = r.median()
            row[f"iqr_lo_{tag}"], row[f"iqr_hi_{tag}"] = (r.quantile(0.25), r.quantile(0.75)) if len(r) else (np.nan, np.nan)
            row[f"median_ratio_{tag}_nonfloor"] = r_nf.median()
            row[f"frac_gt1_{tag}"] = (r > 1.0).mean() if len(r) else np.nan
            row[f"wilcoxon_p_{tag}"] = wilcoxon_log(r)
            row[f"median_ratio_{tag}_logit"] = grp[f"ratio_dfa_{tag}_logit"].dropna().median()
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_log(ratio: pd.Series) -> float:
    vals = np.log(ratio[ratio > 0].dropna())
    if len(vals) < 5:
        return np.nan
    return float(stats.wilcoxon(vals, alternative="greater").pvalue)


def ceiling_controls(cells: pd.DataFrame, ratios: pd.DataFrame) -> dict:
    """OLS of log10 var_fb on mean accuracy + method dummies, and matched-acc subset."""
    sub = cells[cells["method"].isin(["dfa_random", "ndfa_random", "ndfa_random_kronecker"])]
    sub = sub[(sub["sd_fb"] > 0) & ~sub["at_floor"]].copy()
    y = np.log10(sub["sd_fb"] ** 2)
    x_cols = {"mean_acc": sub["mean_acc"].to_numpy()}
    x_cols["is_ndfa"] = (sub["method"] == "ndfa_random").astype(float).to_numpy()
    x_cols["is_kndfa"] = (sub["method"] == "ndfa_random_kronecker").astype(float).to_numpy()
    regimes = [r for r in REGIME_ORDER if r in set(sub["regime"])][1:]
    for r in regimes:
        x_cols[f"regime_{r}"] = (sub["regime"] == r).astype(float).to_numpy()
    ols = fit_ols(y.to_numpy(), x_cols)

    matched = []
    for tag in ["ndfa", "kndfa"]:
        for tol in MATCH_TOLS:
            m = ratios[~ratios["floor_dfa"] & (ratios[f"acc_gap_{tag}"].abs() <= tol)]
            r = m[f"ratio_dfa_{tag}"].dropna()
            matched.append(
                {
                    "comparison": f"DFA vs {tag}",
                    "tol_pp": 100.0 * tol,
                    "n_cells": len(m),
                    "median_ratio": r.median() if len(r) else np.nan,
                    "wilcoxon_p": wilcoxon_log(r),
                }
            )
    return {"ols": ols, "matched": pd.DataFrame(matched), "n_ols_rows": len(sub)}


def fit_ols(y: np.ndarray, x_cols: dict[str, np.ndarray]) -> pd.DataFrame:
    names = ["intercept"] + list(x_cols)
    X = np.column_stack([np.ones_like(y)] + list(x_cols.values()))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "t": t, "p": p})


def rank_robustness(syn: pd.DataFrame) -> pd.DataFrame:
    """Median sd_fb ratios per feedback rank on the synthetic suite."""
    rows = []
    for rank, grp in syn.groupby("feedback_rank"):
        stats_df = decompose(grp, SYN_CELL)
        stats_df["family"] = "synthetic"
        stats_df["regime"] = regime_labels(stats_df, "synthetic")
        stats_df["cell"] = cell_ids(stats_df, SYN_CELL)
        r = build_ratios(stats_df)
        rows.append(
            {
                "feedback_rank": rank,
                "n_cells": len(r),
                "median_ratio_ndfa": r["ratio_dfa_ndfa"].median(),
                "median_ratio_kndfa": r["ratio_dfa_kndfa"].median(),
                "wilcoxon_p_ndfa": wilcoxon_log(r["ratio_dfa_ndfa"].dropna()),
            }
        )
    return pd.DataFrame(rows).sort_values("feedback_rank")


def coloredmnist_context() -> pd.DataFrame:
    if not COLORED.exists():
        return pd.DataFrame()
    df = pd.read_csv(COLORED)
    return df.groupby("method", as_index=False).agg(
        mean_acc=("test_acc", "mean"), sd_total=("test_acc", lambda x: x.std(ddof=1)), n=("test_acc", "size")
    )


# ---------------------------------------------------------------------------
# figure


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
            "legend.fontsize": 6.4,
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


def make_figure(cells: pd.DataFrame, ratios: pd.DataFrame, ceiling: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), constrained_layout=True)

    # A: per-cell feedback-seed std, DFA vs nDFA (log-log, diagonal)
    ax = axes[0]
    floor = 0.02  # pp; keeps zero-variance cells visible on log axes
    x = np.maximum(100.0 * ratios["sd_fb_dfa_random"], floor)
    y = np.maximum(100.0 * ratios["sd_fb_ndfa_random"], floor)
    lims = (floor * 0.8, max(x.max(), y.max()) * 1.6)
    ax.plot(lims, lims, color="#111827", lw=0.8, zorder=2)
    for factor, label in [(1 / 3, r"$\times$1/3"), (1 / 10, r"$\times$1/10")]:
        ax.plot(lims, (lims[0] * factor, lims[1] * factor), color="#9CA3AF", lw=0.7, ls="--", zorder=2)
        ax.text(lims[1] * 0.55, lims[1] * factor * 0.42, label, fontsize=6.0, color="#6B7280")
    for regime in [r for r in REGIME_ORDER if r in set(ratios["regime"])]:
        grp = ratios[ratios["regime"] == regime]
        marker = MARKERS[grp["family"].iloc[0]]
        for is_floor, sub in grp.groupby("floor_dfa"):
            ax.scatter(
                np.maximum(100.0 * sub["sd_fb_dfa_random"], floor),
                np.maximum(100.0 * sub["sd_fb_ndfa_random"], floor),
                s=13,
                marker=marker,
                facecolor="none" if is_floor else COLORS[regime],
                edgecolor=COLORS[regime] if is_floor else "white",
                linewidth=0.6 if is_floor else 0.35,
                alpha=0.55 if is_floor else 0.9,
                label=None if is_floor else PRETTY[regime],
                zorder=3,
            )
    ax.scatter([], [], s=13, marker="o", facecolor="none", edgecolor="#6B7280",
               linewidth=0.6, label="DFA at chance (open)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("feedback-seed std, raw DFA (pp)")
    ax.set_ylabel("feedback-seed std, nDFA (pp)")
    ax.set_title("Feedback-seed variance per cell")
    ax.legend(frameon=False, handletextpad=0.25, borderaxespad=0.1, loc="upper left")
    ax.grid(True, which="major", zorder=0)
    ax.tick_params(pad=1.5)

    # B: ceiling control — sd_fb vs mean accuracy with per-method fits
    ax = axes[1]
    method_labels = {"dfa_random": "DFA", "ndfa_random": "nDFA", "ndfa_random_kronecker": "K-nDFA"}
    for method in ["dfa_random", "ndfa_random", "ndfa_random_kronecker"]:
        grp = cells[(cells["method"] == method) & (cells["sd_fb"] > 0) & ~cells["at_floor"]]
        acc = 100.0 * grp["mean_acc"]
        sd = 100.0 * grp["sd_fb"]
        ax.scatter(acc, sd, s=10, marker="o", facecolor=COLORS[method], edgecolor="white",
                   linewidth=0.3, alpha=0.55, zorder=3)
        slope, intercept = np.polyfit(acc, np.log10(sd), 1)
        xs = np.linspace(acc.min(), acc.max(), 50)
        ax.plot(xs, 10 ** (intercept + slope * xs), color=COLORS[method], lw=1.6,
                ls=METHOD_LS[method], label=method_labels[method], zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("mean final test accuracy (%)")
    ax.set_ylabel("feedback-seed std (pp)")
    ax.set_title("Ceiling control: std vs accuracy (above-chance cells)")
    ax.legend(frameon=False, handletextpad=0.4, loc="lower left")
    ax.grid(True, which="major", zorder=0)
    ax.tick_params(pad=1.5)

    for ax, label in zip(axes, "AB"):
        ax.text(-0.14, 1.07, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, weight="bold", color="#111827")

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"infodfa_feedback_variance.{ext}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report


def write_report(
    cells: pd.DataFrame,
    ratios: pd.DataFrame,
    regime: pd.DataFrame,
    ceiling: dict,
    rank_rob: pd.DataFrame,
    colored: pd.DataFrame,
) -> None:
    n_syn = (ratios["family"] == "synthetic").sum()
    n_vis = (ratios["family"] == "vision").sum()
    n_mix = (ratios["family"] == "mixer").sum()
    n_floor = int(ratios["floor_dfa"].sum())
    data_note = (
        f"Cells: {n_syn} synthetic (full-rank feedback), {n_vis} vision MLP, {n_mix} mixer. "
        "Per cell x method: 5 data seeds x 3 feedback seeds (final epoch). "
        "sd_fb = pooled within-data-seed std over feedback seeds; sd_data = nested-ANOVA "
        "between-data-seed std. Ratios are sd_fb(DFA)/sd_fb(conditioned) per cell. "
        f"In {n_floor} cells raw DFA sits at the chance floor (mean acc within "
        f"{100 * FLOOR_TOL:.0f}pp of chance), where its variance is degenerately small "
        "because the run fails identically for every draw; the `all_dfa_learns` scope and "
        "the *_nonfloor columns exclude those cells."
    )

    floor_note = (
        "`feedback_seed` seeds the feedback matrices B and the minibatch-order stream "
        "(experiments/run_dfa_multioutput_synthetic.py: feedback seed 20000+100*seed+fs, "
        "training rng 30000+100*seed+fs); model init depends only on the data seed, and BP "
        "was run at a single feedback seed. sd_fb therefore includes a small shared "
        "batch-order component; the conditioned methods' sd_fb is an upper bound on that "
        "common stochastic floor, so the DFA/conditioned ratio is a lower bound on the "
        "pure feedback-draw effect."
    )

    med = (
        cells[~cells["at_floor"]]
        .groupby("method")[["sd_fb", "sd_data"]]
        .median()
        * 100.0
    )
    decomp_note = (
        dataframe_to_markdown(
            med.reset_index().rename(columns={"sd_fb": "median sd_fb (pp)", "sd_data": "median sd_data (pp)"})
        )
        + "\n\nAbove-chance cells only. For raw DFA the feedback-seed component dominates the "
        "data-seed component; conditioning drops the feedback component below the data component."
    )

    ols_md = dataframe_to_markdown(ceiling["ols"], float_format=".4g")
    matched_note = (
        dataframe_to_markdown(ceiling["matched"], float_format=".3g")
        + "\n\nCells (DFA above chance) where DFA and the conditioned method reach the same mean "
        "accuracy to within the tolerance, so the ratio cannot be a ceiling artifact there."
    )

    colored_note = (
        dataframe_to_markdown(colored)
        + "\n\nColoredMNIST runs seed data and feedback jointly, so only the total seed std is "
        "available there; it motivated this analysis but is not part of the decomposition."
        if not colored.empty
        else "_ColoredMNIST results not found._"
    )

    write_markdown_report(
        OUT / "feedback_variance_report.md",
        title="Feedback-seed variance collapse under conditioning",
        sections=[
            ("Data", data_note),
            ("Per-regime ratios sd_fb(DFA)/sd_fb(conditioned)", dataframe_to_markdown(regime, float_format=".3g")),
            ("Variance decomposition medians", decomp_note),
            ("What feedback_seed randomizes", floor_note),
            (
                "Ceiling control: OLS log10 var_fb ~ mean_acc + method + regime",
                f"{ceiling['n_ols_rows']} cell-method rows (above-chance cells).\n\n{ols_md}",
            ),
            ("Ceiling control: matched-accuracy cells", matched_note),
            (
                "Rank-restricted feedback (synthetic)",
                dataframe_to_markdown(rank_rob, float_format=".3g")
                + "\n\nWith 8 output classes, rank 8 equals full rank, hence duplicates rank 0.",
            ),
            ("ColoredMNIST context (total seed variance)", colored_note),
        ],
    )


if __name__ == "__main__":
    main()
