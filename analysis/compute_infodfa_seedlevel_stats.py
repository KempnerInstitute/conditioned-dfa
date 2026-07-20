"""Seed-level statistics for the conditioned-DFA synthetic suite (no new compute).

The paper's headline paired tests treat the 32 designed cells of each regime as
independent observations, but all cells share the same 5 data seeds. This
script re-does the inference at the data-seed level:

  1. For each regime, average the nDFA-vs-DFA per-(cell, seed) accuracy delta
     over the 32 cells within each seed, giving n=5 seed-level deltas.
  2. Paired t-test and Wilcoxon signed-rank on the 5 seed-level deltas.
  3. Descriptive hierarchical-bootstrap 95% interval for the mean delta:
     resample cells with replacement, then seeds within each resampled cell
     (10,000 draws).

The analysis fixes feedback rank 0 (full rank) for every method, matching the
revised paper's primary comparison and avoiding test-set rank selection.

Outputs (all under ``results/infodfa_seedlevel_stats_v1``):
  - ``seedlevel_stats.csv``           regime x comparison seed-level tests
  - ``seedlevel_seed_deltas.csv``     the underlying n=5 per-seed mean deltas
  - ``seedlevel_stats.md``            markdown summary
"""

from __future__ import annotations

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
METHODS = ["dfa_random", "ndfa_random"]
COMPARISONS = [
    ("ndfa_random", "dfa_random"),
]
N_BOOT = 10_000


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


def fixed_full_values(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(cell, method, seed) accuracy at fixed full-rank feedback."""
    return df[df["feedback_rank"].eq(0)].drop(columns="feedback_rank").copy()


def hierarchical_bootstrap_ci(
    delta: pd.DataFrame, *, rng: np.random.Generator, n_boot: int = N_BOOT
) -> tuple[float, float]:
    """Descriptive 95% interval: resample cells, then seeds within cells.

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


def write_markdown(tests: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Seed-level statistics for the conditioned-DFA synthetic suite",
        "",
        "All 32 cells of a regime share the same 5 data seeds, so cell-level paired",
        "tests overstate the effective sample size. Here each regime x comparison is",
        "reduced to n=5 seed-level mean deltas (average over cells within each data",
        "seed) at fixed full-rank feedback; the interval is a hierarchical bootstrap",
        f"(cells, then seeds within cells, {N_BOOT:,} draws). With n=5 seeds the",
        "smallest attainable Wilcoxon p is 0.0625 (two-sided, all deltas same sign).",
        "",
        "| regime | comparison | mean delta (pp) | seed range (pp) | descriptive 95% hboot interval (pp) | t | t-p | Wilcoxon-p |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for _, r in tests.iterrows():
        lines.append(
            f"| {r['condition']} | {r['method']} vs {r['reference']} | {r['mean_delta_pp']:+.2f} | "
            f"[{r['seed_delta_min_pp']:+.2f}, {r['seed_delta_max_pp']:+.2f}] | "
            f"[{r['hboot_ci_low_pp']:+.2f}, {r['hboot_ci_high_pp']:+.2f}] | "
            f"{r['t_stat']:+.2f} | {r['t_p']:.2e} | {r['wilcoxon_p']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    val = fixed_full_values(load_final())
    tests, seed_deltas = seedlevel_tests(val)
    tests.to_csv(OUT / "seedlevel_stats.csv", index=False)
    seed_deltas.to_csv(OUT / "seedlevel_seed_deltas.csv", index=False)
    write_markdown(tests, OUT / "seedlevel_stats.md")
    print((OUT / "seedlevel_stats.md").read_text())
    print(f"Wrote {OUT / 'seedlevel_stats.csv'}")
    print(f"Wrote {OUT / 'seedlevel_seed_deltas.csv'}")


if __name__ == "__main__":
    main()
