"""Seed-level statistics for the conditioned-DFA synthetic suite (no new compute).

The paper's headline paired tests treat the 32 designed cells of each regime as
independent observations, but all cells share the same 5 data seeds. This
script re-does the inference at the data-seed level:

  1. For each regime and comparison (nDFA vs DFA, nDFA vs BP, K-nDFA vs DFA,
     K-nDFA vs BP), average the per-(cell, seed) accuracy delta over the 32
     cells within each seed, giving n=5 seed-level deltas.
  2. Paired t-test and Wilcoxon signed-rank on the 5 seed-level deltas.
  3. Hierarchical bootstrap 95% CI for the mean delta: resample cells with
     replacement, then seeds within each resampled cell (10,000 draws).

Method selection follows the honest LOSO scheme from
``reanalyze_synthetic_honest_selection.py``: for the conditioned rules the
feedback rank for each held-out seed is chosen on the remaining seeds; BP and
DFA use their single rank-0 configuration. This matches the LOSO columns of
``results/infodfa_honest_selection_reanalysis``.

It also computes the KFAC source-swap equivalence check from
``results/infodfa_kfac_control_v1``: the paired per-(cell, seed) difference
[K-nDFA with BP-error left factor] minus [K-nDFA with local DFA-error left
factor] at the final epoch, with a bootstrap CI over cells and a 0.5 pp
equivalence margin.

Outputs (all under ``results/infodfa_seedlevel_stats_v1``):
  - ``seedlevel_stats.csv``           regime x comparison seed-level tests
  - ``seedlevel_seed_deltas.csv``     the underlying n=5 per-seed mean deltas
  - ``kfac_source_swap_ci.csv``       KFAC source-swap equivalence numbers
  - ``seedlevel_stats.md``            markdown summary of both analyses
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("INFODFA_RESULTS", ROOT / "results")).resolve()
LEGACY_RESULTS = Path(os.environ.get("INFODFA_LEGACY_RESULTS", ROOT / "../Info-Man/results")).resolve()
OUT = RESULTS / "infodfa_seedlevel_stats_v1"

CELL = ["condition", "input_noise", "n_train", "train_label_noise"]
COND_ORDER = ["nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned"]
CONDITIONED = ["ndfa_random", "ndfa_random_kronecker"]
METHODS = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker"]
COMPARISONS = [
    ("ndfa_random", "dfa_random"),
    ("ndfa_random", "bp"),
    ("ndfa_random_kronecker", "dfa_random"),
    ("ndfa_random_kronecker", "bp"),
]
N_BOOT = 10_000
KFAC_SRC_REL = "infodfa_kfac_control_v1"
KN, KFACBP = "ndfa_random_kronecker", "ndfa_random_kronecker_bp"
EQUIV_MARGIN_PP = 0.5


def result_path(relative: str | Path) -> Path:
    rel = Path(relative)
    primary = RESULTS / rel
    legacy = LEGACY_RESULTS / rel
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Missing result artifact {rel}. Checked {primary} and {legacy}. "
        "Set INFODFA_RESULTS or INFODFA_LEGACY_RESULTS if the aggregate lives elsewhere."
    )


def load_final() -> pd.DataFrame:
    src = result_path("infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv")
    cols = CELL + ["method", "seed", "feedback_rank", "epoch", "test_acc"]
    df = pd.read_csv(src, usecols=cols)
    df = df[df["method"].isin(METHODS)].copy()
    df = df[df["epoch"] == df["epoch"].max()].copy()
    # average over any remaining nuisance dims (e.g. feedback_seed)
    df = df.groupby(CELL + ["method", "seed", "feedback_rank"], as_index=False)["test_acc"].mean()
    return df


def loso_values(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, method, seed) test accuracy under LOSO rank selection."""
    rows = []
    for keys, sub in df.groupby(CELL + ["method"]):
        *cell_vals, m = keys
        cell = dict(zip(CELL, cell_vals))
        if m not in CONDITIONED:
            s = sub[sub["feedback_rank"] == 0]
            for _, r in s.iterrows():
                rows.append({**cell, "method": m, "seed": r["seed"], "test_acc": r["test_acc"]})
            continue
        for held in sorted(sub["seed"].unique()):
            others = sub[sub["seed"] != held]
            best = others.groupby("feedback_rank")["test_acc"].mean().idxmax()
            held_val = sub[(sub["seed"] == held) & (sub["feedback_rank"] == best)]["test_acc"]
            if len(held_val):
                rows.append({**cell, "method": m, "seed": held, "test_acc": float(held_val.mean())})
    return pd.DataFrame(rows)


def hierarchical_bootstrap_ci(
    delta: pd.DataFrame, *, rng: np.random.Generator, n_boot: int = N_BOOT
) -> tuple[float, float]:
    """95% CI for the mean delta: resample cells, then seeds within cells.

    ``delta`` is a (n_cells x n_seeds) matrix of per-(cell, seed) deltas.
    """
    mat = delta.to_numpy()
    n_cells, n_seeds = mat.shape
    boot = np.empty(n_boot)
    for i in range(n_boot):
        cells = rng.integers(0, n_cells, size=n_cells)
        seeds = rng.integers(0, n_seeds, size=(n_cells, n_seeds))
        boot[i] = mat[cells[:, None], seeds].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def seedlevel_tests(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    piv = val.pivot_table(index=CELL + ["seed"], columns="method", values="test_acc")
    rng = np.random.default_rng(0)
    rows, seed_rows = [], []
    for cond in COND_ORDER:
        c = piv.xs(cond, level="condition")
        for method, reference in COMPARISONS:
            d = (c[method] - c[reference]).dropna() * 100  # pp
            # per-(cell, seed) matrix: rows = cells, cols = seeds
            mat = d.unstack("seed").dropna()
            seed_means = mat.mean(axis=0)  # n=5 seed-level mean deltas
            t_stat, t_p = stats.ttest_1samp(seed_means.values, 0.0)
            w_stat, w_p = stats.wilcoxon(seed_means.values)
            ci_low, ci_high = hierarchical_bootstrap_ci(mat, rng=rng)
            rows.append(
                {
                    "condition": cond,
                    "method": method,
                    "reference": reference,
                    "n_cells": int(mat.shape[0]),
                    "n_seeds": int(mat.shape[1]),
                    "mean_delta_pp": float(seed_means.mean()),
                    "seed_delta_min_pp": float(seed_means.min()),
                    "seed_delta_max_pp": float(seed_means.max()),
                    "hboot_ci_low_pp": ci_low,
                    "hboot_ci_high_pp": ci_high,
                    "t_stat": float(t_stat),
                    "t_p": float(t_p),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                }
            )
            for seed, value in seed_means.items():
                seed_rows.append(
                    {
                        "condition": cond,
                        "method": method,
                        "reference": reference,
                        "seed": float(seed),
                        "mean_delta_pp": float(value),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def kfac_source_swap() -> tuple[pd.DataFrame, pd.Series]:
    src = result_path(KFAC_SRC_REL)
    frames = []
    for f in glob.glob(str(src / "**/dfa_multioutput_results.csv"), recursive=True):
        df = pd.read_csv(f, usecols=CELL + ["method", "seed", "epoch", "test_acc"])
        frames.append(df[df["epoch"] == df["epoch"].max()])
    d = pd.concat(frames, ignore_index=True)
    piv = d.pivot_table(index=CELL + ["seed"], columns="method", values="test_acc")
    diff = (piv[KFACBP] - piv[KN]).dropna() * 100  # BP-error factor minus DFA-error factor, pp
    cell_diff = diff.groupby(level=CELL).mean()  # seed-averaged per-cell difference

    rng = np.random.default_rng(1)
    vals = cell_diff.to_numpy()
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boot[i] = vals[rng.integers(0, vals.size, size=vals.size)].mean()
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    summary = pd.Series(
        {
            "n_cells": int(cell_diff.size),
            "n_cell_seed_pairs": int(diff.size),
            "mean_delta_pp": float(diff.mean()),
            "boot_ci_low_pp": ci_low,
            "boot_ci_high_pp": ci_high,
            "mean_abs_cell_delta_pp": float(cell_diff.abs().mean()),
            "max_abs_cell_delta_pp": float(cell_diff.abs().max()),
            "max_abs_cell_seed_delta_pp": float(diff.abs().max()),
            "n_cells_within_margin": int((cell_diff.abs() <= EQUIV_MARGIN_PP).sum()),
            "all_cells_within_margin": bool((cell_diff.abs() <= EQUIV_MARGIN_PP).all()),
            "all_cell_seed_within_margin": bool((diff.abs() <= EQUIV_MARGIN_PP).all()),
            "equiv_margin_pp": EQUIV_MARGIN_PP,
        }
    )
    return cell_diff.reset_index(name="bp_minus_dfa_factor_pp"), summary


def write_markdown(tests: pd.DataFrame, kfac: pd.Series, path: Path) -> None:
    lines = [
        "# Seed-level statistics for the conditioned-DFA synthetic suite",
        "",
        "All 32 cells of a regime share the same 5 data seeds, so cell-level paired",
        "tests overstate the effective sample size. Here each regime x comparison is",
        "reduced to n=5 seed-level mean deltas (average over cells within each data",
        "seed) under LOSO rank selection; the CI is a hierarchical bootstrap",
        f"(cells, then seeds within cells, {N_BOOT:,} draws). With n=5 seeds the",
        "smallest attainable Wilcoxon p is 0.0625 (two-sided, all deltas same sign).",
        "",
        "| regime | comparison | mean delta (pp) | seed range (pp) | 95% hboot CI (pp) | t | t-p | Wilcoxon-p |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for _, r in tests.iterrows():
        lines.append(
            f"| {r['condition']} | {r['method']} vs {r['reference']} | {r['mean_delta_pp']:+.2f} | "
            f"[{r['seed_delta_min_pp']:+.2f}, {r['seed_delta_max_pp']:+.2f}] | "
            f"[{r['hboot_ci_low_pp']:+.2f}, {r['hboot_ci_high_pp']:+.2f}] | "
            f"{r['t_stat']:+.2f} | {r['t_p']:.2e} | {r['wilcoxon_p']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## KFAC source-swap equivalence (K-nDFA BP-error factor minus DFA-error factor)",
            "",
            f"- paired per-(cell, seed) mean delta: {kfac['mean_delta_pp']:+.4f} pp "
            f"(n={int(kfac['n_cell_seed_pairs'])} cell x seed pairs, {int(kfac['n_cells'])} cells)",
            f"- bootstrap 95% CI over cells: [{kfac['boot_ci_low_pp']:+.4f}, {kfac['boot_ci_high_pp']:+.4f}] pp",
            f"- mean |per-cell delta| (seed-averaged): {kfac['mean_abs_cell_delta_pp']:.4f} pp",
            f"- max |per-cell delta| (seed-averaged): {kfac['max_abs_cell_delta_pp']:.4f} pp",
            f"- max |per-(cell, seed) delta|: {kfac['max_abs_cell_seed_delta_pp']:.4f} pp",
            f"- seed-averaged cells within +/-{kfac['equiv_margin_pp']} pp margin: "
            f"{int(kfac['n_cells_within_margin'])}/{int(kfac['n_cells'])} "
            f"(all: {kfac['all_cells_within_margin']})",
            f"- all raw cell x seed pairs within +/-{kfac['equiv_margin_pp']} pp margin: {kfac['all_cell_seed_within_margin']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    val = loso_values(load_final())
    tests, seed_deltas = seedlevel_tests(val)
    tests.to_csv(OUT / "seedlevel_stats.csv", index=False)
    seed_deltas.to_csv(OUT / "seedlevel_seed_deltas.csv", index=False)
    cell_diff, kfac = kfac_source_swap()
    cell_diff.to_csv(OUT / "kfac_source_swap_ci.csv", index=False)
    kfac.to_frame("value").to_csv(OUT / "kfac_source_swap_summary.csv")
    write_markdown(tests, kfac, OUT / "seedlevel_stats.md")
    print((OUT / "seedlevel_stats.md").read_text())
    print(f"Wrote {OUT / 'seedlevel_stats.csv'}")
    print(f"Wrote {OUT / 'seedlevel_seed_deltas.csv'}")
    print(f"Wrote {OUT / 'kfac_source_swap_ci.csv'}")


if __name__ == "__main__":
    main()
