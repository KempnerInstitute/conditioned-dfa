"""ColoredMNIST: means and Holm-corrected, seed-paired tests on both the standard
color-reversed test metric and the grayscale probe. Raw DFA's high seed variance
makes the paired gain over DFA non-significant after correction; the honest claim
is that conditioning stabilizes DFA to BP-level accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SRC = Path(os.environ["INFODFA_RESULTS"]) / "dfa_coloredmnist_v1/dfa_coloredmnist_results.csv"
METHODS = ["bp", "dfa_random", "dfa_random+norm_match", "ndfa_random", "ndfa_random_kronecker"]
COMPS = [("ndfa_random", "dfa_random"), ("ndfa_random_kronecker", "dfa_random"),
         ("ndfa_random", "bp"), ("ndfa_random_kronecker", "bp"), ("dfa_random+norm_match", "dfa_random")]


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    out = np.empty_like(ps)
    run = 0.0
    for rank, idx in enumerate(np.argsort(ps)):
        run = max(run, (len(ps) - rank) * ps[idx])
        out[idx] = min(run, 1.0)
    return out


def main() -> None:
    d = pd.read_csv(SRC)
    for metric in ["test_acc", "grayscale_acc"]:
        piv = d.pivot_table(index="seed", columns="method", values=metric) * 100
        print(f"\n## {metric} (n={piv.shape[0]} seeds)")
        for m in METHODS:
            print(f"  {m:24} {piv[m].mean():5.1f} +- {piv[m].sem():.1f}")
        ps = [stats.ttest_rel(piv[a], piv[b]).pvalue for a, b in COMPS]
        for (a, b), p, h in zip(COMPS, ps, holm(ps)):
            print(f"   {a} vs {b}: Δ={ (piv[a]-piv[b]).mean():+.2f}pp  Holm p={h:.3g}")


if __name__ == "__main__":
    main()
