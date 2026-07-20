"""Paired statistical tests for the conditioned-DFA paper tables.

For each (regime, comparison) pair, this script computes:
  - paired t-statistic and two-sided p-value across the 32 cells of the regime
  - Wilcoxon signed-rank statistic and p-value (paired, non-parametric)
  - bootstrap 95% CI for the mean delta

The pairing is across (input_noise, n_train, train_label_noise) cells within
each condition, so each pair is the same data-generation seed evaluated under
two methods. Holm--Bonferroni correction is applied across the four comparisons
in the synthetic stress suite and across the two comparisons in the vision
sweep.

Output: ``results/infodfa_paper_tables_20260527/infodfa_statistical_tests.csv``
and a markdown summary suitable for the appendix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "infodfa_paper_tables_20260527"


COMPARISONS = [
    ("ndfa_random", "dfa_random"),
    ("ndfa_random_kronecker", "dfa_random"),
    ("ndfa_random", "bp"),
    ("ndfa_random_kronecker", "bp"),
]

SYNTHETIC_CELL_COLS = ["condition", "input_noise", "n_train", "train_label_noise"]
VISION_CELL_COLS = ["dataset", "n_train", "label_noise"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    synthetic = compute_tests(
        path=RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_best_by_method.csv",
        cell_cols=SYNTHETIC_CELL_COLS,
        group_col="condition",
    )
    vision = compute_tests(
        path=RESULTS / "infodfa_vision_noise_sweep_aggregate_v2" / "dfa_nmnc_best_by_method.csv",
        cell_cols=VISION_CELL_COLS,
        group_col="dataset",
    )
    combined = pd.concat([synthetic, vision], ignore_index=True)
    combined.to_csv(OUT_DIR / "infodfa_statistical_tests.csv", index=False)
    write_markdown(combined, OUT_DIR / "infodfa_statistical_tests.md")
    print(f"Wrote {OUT_DIR / 'infodfa_statistical_tests.csv'}")
    print(f"Wrote {OUT_DIR / 'infodfa_statistical_tests.md'}")


def compute_tests(*, path: Path, cell_cols: list[str], group_col: str) -> pd.DataFrame:
    if not path.exists():
        print(f"warning: {path} not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(path)
    key_cols = cell_cols + ["method"]
    counts = df.groupby(key_cols, dropna=False).size()
    if (counts > 1).any():
        max_count = int(counts.max())
        raise ValueError(
            f"{path} contains repeated cell/method rows (max count {max_count}); "
            "aggregate per cell before running the cell-level paper tests."
        )
    wide = df.pivot_table(
        index=cell_cols,
        columns="method",
        values="test_mean",
        aggfunc="mean",
    ).reset_index()
    rows = []
    rng = np.random.default_rng(0)
    for group, group_df in wide.groupby(group_col):
        for method, reference in COMPARISONS:
            if method not in group_df.columns or reference not in group_df.columns:
                continue
            paired = group_df[[method, reference]].dropna()
            if paired.shape[0] < 3:
                continue
            diffs = paired[method].values - paired[reference].values
            t_stat, t_p = stats.ttest_rel(paired[method], paired[reference], nan_policy="omit")
            w_stat, w_p = stats.wilcoxon(diffs)
            mean_delta = float(np.mean(diffs))
            ci_low, ci_high = bootstrap_ci(diffs, rng=rng)
            rows.append(
                {
                    "task_group": group,
                    "method": method,
                    "reference": reference,
                    "n_pairs": int(paired.shape[0]),
                    "mean_delta": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "t_stat": float(t_stat),
                    "t_p": float(t_p),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        # Holm-Bonferroni within each task_group
        out["t_p_holm"] = (
            out.groupby("task_group")["t_p"].transform(holm_bonferroni)
        )
        out["wilcoxon_p_holm"] = (
            out.groupby("task_group")["wilcoxon_p"].transform(holm_bonferroni)
        )
    return out


def bootstrap_ci(diffs: np.ndarray, *, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan")
    boot = np.empty(n_boot)
    n = diffs.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(diffs[idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def holm_bonferroni(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.values)
    adjusted = np.empty_like(p_values.values, dtype=float)
    n = len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        candidate = (n - rank + 1) * p_values.values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("# Statistical tests\n\nNo data.\n")
        return
    lines = [
        "# Paired statistical tests for conditioned-DFA paper tables",
        "",
        "Cells are paired across `(noise, sample, label-noise)` configurations within each ",
        "regime or dataset. Each row reports the paired t-statistic, Wilcoxon signed-rank ",
        "statistic, bootstrap 95% CI for the mean accuracy delta (in percent points), and ",
        "Holm-Bonferroni-corrected p-values within each task group.",
        "",
        "| group | method | reference | n | mean delta (pp) | 95% CI | t | t-p (Holm) | Wilcoxon-p (Holm) |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        ci = f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}]"
        lines.append(
            f"| {row['task_group']} | {row['method']} | {row['reference']} | {int(row['n_pairs'])} | "
            f"{100*row['mean_delta']:+.2f} | {ci} | {row['t_stat']:+.2f} | "
            f"{row['t_p_holm']:.2e} | {row['wilcoxon_p_holm']:.2e} |"
        )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
