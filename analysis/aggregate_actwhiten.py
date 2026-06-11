"""Aggregate the decorrelation-baseline sweep (reviewer 'did you need a new rule?').

Compares raw DFA, dfa_actwhiten (ZCA activation decorrelation, precondition by
(C+lambda I)^{-1/2}), activity nDFA, and K-nDFA (full inverse, (C+lambda I)^{-1})
on the same 128-cell synthetic grid as the main sweep. The question is whether the
gain needs nDFA's full natural-gradient power or only generic decorrelation.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path("results/infodfa_actwhiten_synthetic_v1")
ORDER = ["dfa_random", "dfa_actwhiten", "ndfa_random", "ndfa_random_kronecker"]
REGIMES = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]


def main() -> None:
    rows = [pd.read_csv(f) for f in glob.glob(str(ROOT / "**/dfa_multioutput_results.csv"), recursive=True)]
    d = pd.concat(rows, ignore_index=True)
    d = d[d["epoch"] == d["epoch"].max()]
    piv = d.groupby(["condition", "method"])["test_acc"].mean().unstack().mul(100)
    piv = piv.reindex(index=[r for r in REGIMES if r in piv.index],
                      columns=[m for m in ORDER if m in piv.columns])
    piv["decorr-DFA"] = piv["dfa_actwhiten"] - piv["dfa_random"]
    piv["nDFA-decorr"] = piv["ndfa_random"] - piv["dfa_actwhiten"]
    out = [
        "# Decorrelation baseline vs nDFA (128-cell synthetic suite)\n",
        piv.round(2).to_string(),
        "",
        "decorr-DFA  = ZCA decorrelation (C^-1/2) gain over raw DFA.",
        "nDFA-decorr = extra gain from the full inverse (C^-1, natural gradient) over decorrelation.",
        "Reading: if nDFA-decorr is consistently positive, the gain needs the natural-gradient",
        "power, not just generic activation decorrelation; if ~0, decorrelation suffices.",
    ]
    text = "\n".join(out) + "\n"
    (ROOT / "actwhiten_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
