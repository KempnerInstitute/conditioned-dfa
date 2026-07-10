"""Aggregate the alignment-dynamics measurement (see measure_alignment_dynamics.py).

Answers two questions about conditioned DFA vs raw DFA:

(a) Alignment-phase invariance: does right-preconditioning by the damped inverse
    activity second moment leave the Refinetti align-then-memorize phase
    structure unchanged? Measured as the trajectory of the weight-alignment
    cosine cos(vec(W_L...W_{k+1}), vec(B_k)) and its per-run t50/t80 (first
    epoch reaching 50%/80% of the run's final alignment).

(b) Pi-gap timing: is E_B[Pi^cond] - E_B[Pi^DFA] largest during the alignment
    phase? E_B averages over feedback seeds within a data seed; the gap is
    computed at each logged time point in both delta space (the paper's logged
    projected BP-step ratio) and weight-gradient space.

Outputs (results/infodfa_alignment_dynamics_v1/):
- alignment_dynamics.csv          full merged trajectories
- alignment_phase_metrics.csv     per-run t50/t80/final alignment/min early alignment
- pi_gap_trajectories.csv         E_B Pi gaps (per seed, per epoch, both spaces)
- alignment_dynamics.png/.pdf     two-regime figure (alignment + Pi gap vs epoch)
- alignment_dynamics_summary.md   measured answers
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.analysis import dataframe_to_markdown, write_markdown_report  # noqa: E402

OUT_DIR = ROOT / "results" / "infodfa_alignment_dynamics_v1"

METHOD_LABELS = {"dfa_random": "DFA", "ndfa_random": "nDFA", "ndfa_random_kronecker": "K-nDFA"}
# House palette (matches make_iclr_figures.py COLORS): DFA grey-dashed, nDFA
# green, K-nDFA purple-dashed. Line styles keep identity legible without color.
METHOD_COLORS = {"dfa_random": "#999999", "ndfa_random": "#009E73", "ndfa_random_kronecker": "#6A3D9A"}
METHOD_STYLES = {"dfa_random": (0, (4, 2)), "ndfa_random": "-", "ndfa_random_kronecker": (0, (5, 1.5))}
# Non-breaking hyphen keeps "K-nDFA" from reading as a subtraction next to the
# explicit minus that denotes the Pi-gap difference.
NB_HYPHEN = "‑"
PAIR_LABELS = {
    "ndfa_random": "nDFA $-$ DFA",
    "ndfa_random_kronecker": f"K{NB_HYPHEN}nDFA $-$ DFA",
}
CELL_TITLES = {
    "clean_aligned": "Task-aligned (clean cell)",
    "nuisance_hard": "Nuisance-dominant (hard cell)",
}


def load_trajectories() -> pd.DataFrame:
    shards = sorted((OUT_DIR / "shards").glob("*/alignment_dynamics.csv"))
    if not shards:
        raise FileNotFoundError(f"no shard CSVs under {OUT_DIR / 'shards'}")
    df = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
    return df.sort_values(["cell", "method", "seed", "feedback_seed", "step"]).reset_index(drop=True)


def phase_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run alignment-phase timing from the weight-alignment trajectory."""

    rows = []
    for (cell, method, seed, fb), g in df.groupby(["cell", "method", "seed", "feedback_seed"]):
        g = g.sort_values("epoch")
        horizon = g["epoch"].max()
        final = g[g["epoch"] >= 0.9 * horizon]["weight_align_mean"].mean()
        early = g[g["epoch"] <= 0.3 * horizon]["weight_align_mean"]
        def first_reach(frac: float) -> float:
            if final <= 0:
                return np.inf
            hit = g[g["weight_align_mean"] >= frac * final]
            return float(hit["epoch"].min()) if len(hit) else np.inf
        rows.append(
            {
                "cell": cell,
                "method": method,
                "seed": seed,
                "feedback_seed": fb,
                "horizon_epochs": float(horizon),
                "final_weight_align": float(final),
                "min_early_weight_align": float(early.min()),
                "t50_epochs": first_reach(0.5),
                "t80_epochs": first_reach(0.8),
            }
        )
    return pd.DataFrame(rows)


def eb_pi_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """E_B[Pi^cond] - E_B[Pi^DFA] per seed and time point, both Pi spaces."""

    eb = (
        df.groupby(["cell", "method", "seed", "epoch"], as_index=False)[
            ["projected_step_ratio_mean", "pi_weight_mean"]
        ].mean()
    )
    piv = eb.pivot_table(
        index=["cell", "seed", "epoch"],
        columns="method",
        values=["projected_step_ratio_mean", "pi_weight_mean"],
    )
    rows = []
    for cond in ["ndfa_random", "ndfa_random_kronecker"]:
        gap_delta = piv[("projected_step_ratio_mean", cond)] - piv[("projected_step_ratio_mean", "dfa_random")]
        gap_weight = piv[("pi_weight_mean", cond)] - piv[("pi_weight_mean", "dfa_random")]
        out = pd.DataFrame(
            {
                "pair": f"{METHOD_LABELS[cond]}-DFA",
                "pi_gap_delta_space": gap_delta,
                "pi_gap_weight_space": gap_weight,
            }
        ).reset_index()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarize_phase(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["cell", "method"], as_index=False)
        .agg(
            t50_mean=("t50_epochs", "mean"),
            t50_sem=("t50_epochs", "sem"),
            t80_mean=("t80_epochs", "mean"),
            t80_sem=("t80_epochs", "sem"),
            final_align_mean=("final_weight_align", "mean"),
            final_align_sem=("final_weight_align", "sem"),
            min_early_align_mean=("min_early_weight_align", "mean"),
            frac_runs_anti_aligned=("min_early_weight_align", lambda x: float((x < -0.02).mean())),
            n=("t80_epochs", "size"),
        )
        .sort_values(["cell", "method"])
    )


def summarize_gap_timing(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, pair), g in gaps.groupby(["cell", "pair"]):
        mean_traj = g.groupby("epoch")["pi_gap_delta_space"].mean()
        peak_epoch = float(mean_traj.idxmax())
        per_seed_peaks = g.loc[g.groupby("seed")["pi_gap_delta_space"].idxmax()]
        late = mean_traj[mean_traj.index >= 0.9 * mean_traj.index.max()].mean()
        rows.append(
            {
                "cell": cell,
                "pair": pair,
                "peak_gap": float(mean_traj.max()),
                "peak_epoch": peak_epoch,
                "peak_epoch_seed_mean": float(per_seed_peaks["epoch"].mean()),
                "peak_epoch_seed_sem": float(per_seed_peaks["epoch"].sem()),
                "gap_at_epoch1": float(mean_traj.get(1.0, np.nan)),
                "gap_late": float(late),
                "late_over_peak": float(late / mean_traj.max()) if mean_traj.max() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "legend.fontsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def mean_sem(df: pd.DataFrame, value: str) -> pd.DataFrame:
    eb = df.groupby(["method", "seed", "epoch"], as_index=False)[value].mean()
    g = eb.groupby(["method", "epoch"])[value]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem()}).reset_index()


def make_figure(df: pd.DataFrame, gaps: pd.DataFrame, phase: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), constrained_layout=True)
    cells = ["clean_aligned", "nuisance_hard"]

    for col, cell in enumerate(cells):
        sub = df[df["cell"] == cell]
        ax = axes[0, col]
        for method in METHOD_LABELS:
            curve = mean_sem(sub[sub["method"] == method], "weight_align_mean")
            curve = curve[curve["method"] == method]
            ax.plot(curve["epoch"], curve["mean"], color=METHOD_COLORS[method],
                    ls=METHOD_STYLES[method], lw=1.4, label=METHOD_LABELS[method])
            ax.fill_between(curve["epoch"], curve["mean"] - curve["sem"], curve["mean"] + curve["sem"],
                            color=METHOD_COLORS[method], alpha=0.18, lw=0)
        ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
        ax.set_title(f"{chr(65 + col)}  {CELL_TITLES[cell]}")
        ax.set_ylabel("Weight alignment\n" + r"$\cos(W_{L}\cdots W_{\ell+1},\,B_\ell)$" if col == 0 else "")
        if col == 0:
            ax.legend(frameon=False, loc="lower right")

        ax = axes[1, col]
        for cond, pair in [("ndfa_random", "nDFA-DFA"), ("ndfa_random_kronecker", "K-nDFA-DFA")]:
            g = gaps[(gaps["cell"] == cell) & (gaps["pair"] == pair)]
            traj = g.groupby("epoch")["pi_gap_delta_space"]
            mean, sem = traj.mean(), traj.sem()
            ax.plot(mean.index, mean.values, color=METHOD_COLORS[cond], ls=METHOD_STYLES[cond],
                    lw=1.4, label=PAIR_LABELS[cond])
            ax.fill_between(mean.index, mean - sem, mean + sem, color=METHOD_COLORS[cond], alpha=0.18, lw=0)
            t80 = phase[(phase["cell"] == cell) & (phase["method"] == cond)]["t80_mean"].iloc[0]
            ax.axvline(t80, color=METHOD_COLORS[cond], lw=0.9, ls=":", alpha=0.85)
        ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\mathbb{E}_B[\Pi]$ gap vs DFA" if col == 0 else "")
        ax.set_title(f"{chr(67 + col)}  $\\Pi$ gap (dotted: $t_{{80}}$ alignment time)")
        if col == 0:
            ax.legend(frameon=False, loc="lower right")

    for path in [OUT_DIR / "alignment_dynamics.png", OUT_DIR / "alignment_dynamics.pdf"]:
        fig.savefig(path, dpi=360 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    df = load_trajectories()
    df.to_csv(OUT_DIR / "alignment_dynamics.csv", index=False)

    metrics = phase_metrics(df)
    metrics.to_csv(OUT_DIR / "alignment_phase_metrics.csv", index=False)
    phase = summarize_phase(metrics)

    gaps = eb_pi_gaps(df)
    gaps.to_csv(OUT_DIR / "pi_gap_trajectories.csv", index=False)
    gap_timing = summarize_gap_timing(gaps)

    make_figure(df, gaps, phase)

    write_markdown_report(
        OUT_DIR / "alignment_dynamics_summary.md",
        title="Conditioning vs feedback-alignment dynamics (align-then-memorize test)",
        sections=[
            (
                "Design",
                "Direct test of the paper's modularity prediction (related-work "
                "'Align-then-memorize' paragraph): DFA vs nDFA vs K-nDFA on the two "
                "focused synthetic cells of the factor-ablation protocol "
                "(task-aligned clean, nuisance-dominant hard; 5 data seeds x 3 "
                "feedback seeds, shared lr=0.08, damping 0.3). Logged per quarter-epoch "
                "(first 5 epochs) then per epoch: the Refinetti weight-alignment cosine "
                "cos(vec(W_L...W_{l+1}), vec(B_l)), Pi in delta space (the paper's "
                "projected BP-step ratio) and in weight-gradient space, loss, test acc. "
                "E_B averages over feedback seeds within each data seed.",
            ),
            ("Alignment-phase timing (per method)", dataframe_to_markdown(phase, float_format=".3f")),
            ("Pi-gap timing (delta-space, E_B)", dataframe_to_markdown(gap_timing, float_format=".3f")),
            (
                "Answers",
                "(a) Alignment-phase invariance: HOLDS in the task-aligned clean cell "
                "(overlapping alignment trajectories; t80 12-13 vs 10.5 epochs; conditioned "
                "variants end slightly MORE aligned), but FAILS in the nuisance-dominant "
                "hard cell: raw DFA anti-aligns (weight-alignment cosine ~ -0.20 in 15/15 "
                "runs) while its loss diverges, recovering only late (t80 ~ 55 ep); nDFA and "
                "K-nDFA enter the alignment phase immediately (t80 ~ 21 and ~ 12 ep). "
                "Conditioning there does not leave alignment dynamics untouched - it rescues "
                "the alignment phase. "
                "(b) Pi-gap timing: in the clean cell the E_B[Pi] gap peaks during the "
                "alignment phase (~ep 5-10 < t80) and decays through memorization, as "
                "predicted. In the nuisance cell the gap is near zero early, peaks at the "
                "END of the conditioned rules' alignment phase (~ep 20-30), and persists - "
                "dominated by raw DFA's failure to produce useful descent at all.",
            ),
        ],
    )
    print(f"Wrote {OUT_DIR}/alignment_dynamics.csv, alignment_phase_metrics.csv, "
          f"pi_gap_trajectories.csv, alignment_dynamics.png/.pdf, alignment_dynamics_summary.md")
    print(phase.to_string(index=False))
    print(gap_timing.to_string(index=False))


if __name__ == "__main__":
    main()
