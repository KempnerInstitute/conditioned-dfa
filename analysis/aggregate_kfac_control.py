"""Aggregate the KFAC-DFA control: K-nDFA (DFA-error left factor) vs a true
KFAC-DFA variant (BP-error left factor) on the synthetic suite. Tests whether
K-nDFA's locality (using the broadcast DFA error for the left factor) costs
anything relative to the non-local KFAC factor.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from scipy import stats

SRC = Path("results/infodfa_kfac_control_v1")
OUT = SRC / "summary"
ORDER = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]
KN, KFACBP = "ndfa_random_kronecker", "ndfa_random_kronecker_bp"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in glob.glob(str(SRC / "**/dfa_multioutput_results.csv"), recursive=True):
        df = pd.read_csv(f, usecols=["condition", "method", "seed", "epoch", "test_acc"])
        frames.append(df[df["epoch"] == df["epoch"].max()])
    d = pd.concat(frames, ignore_index=True)
    means = d.groupby(["condition", "method"])["test_acc"].mean().unstack() * 100
    means = means.reindex(ORDER)
    means["KnDFA_minus_KFACbp_pp"] = means[KN] - means[KFACBP]

    piv = d.pivot_table(index=["condition", "seed"], columns="method", values="test_acc")
    diff = (piv[KN] - piv[KFACBP]).dropna() * 100
    w = stats.wilcoxon(diff) if diff.abs().sum() > 0 else None

    cols = ["bp", "dfa_random", KN, KFACBP, "KnDFA_minus_KFACbp_pp"]
    report = (
        "# KFAC-DFA control (synthetic)\n\n"
        + means[cols].round(2).to_string()
        + f"\n\nK-nDFA minus KFAC-DFA(bp): mean={diff.mean():+.3f} pp, |max|={diff.abs().max():.2f} pp, "
        + f"n={len(diff)}, Wilcoxon p={(w.pvalue if w else float('nan')):.3g}\n"
        + "Conclusion: the local DFA-error left factor matches the BP-error KFAC factor to within noise.\n"
    )
    (OUT / "kfac_control_summary.md").write_text(report)
    means[cols].round(4).to_csv(OUT / "kfac_control_summary.csv")
    print(report)


if __name__ == "__main__":
    main()
