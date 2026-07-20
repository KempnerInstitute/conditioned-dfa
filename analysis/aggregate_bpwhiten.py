"""BP + input-whitening control. Applies nDFA's input-side whitening
(C_{l-1}+lambda I)^{-1} to the EXACT BP gradient and tunes lr per regime. If
whitening is the mechanism, it should help BP exactly where the analysis predicts
(anisotropic regimes) and not on the isotropic control. nDFA still beating
whitened BP in noisy regimes is the DFA-feedback regularization effect.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("INFODFA_RESULTS", ROOT / "results")).resolve()
BPW = Path(os.environ.get("INFODFA_BPWHITEN_RESULTS", RESULTS / "infodfa_bpwhiten_synthetic_v1")).resolve()
ORDER = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]

# Tuned BP values are the best-of-five-learning-rate BP controls reported in
# Table~\ref{tab:infodfa_bpwhiten}. The multioutput aggregate only stores the
# fixed sweep recipe used for trajectory panels, so keep the tuned baseline
# explicit and assert the deltas below.
TUNED_BP_PERCENT = {
    "nuisance_dominant": 27.9,
    "mixed_context": 43.8,
    "low_sample_noisy": 53.3,
    "task_aligned": 92.0,
}
EXPECTED_DELTAS = {
    "nuisance_dominant": 18.3,
    "mixed_context": 0.8,
    "low_sample_noisy": 7.1,
    "task_aligned": -0.1,
}
CONTEXT_PERCENT = {
    "nuisance_dominant": {"dfa": 14.7, "ndfa": 53.3, "k_ndfa": 56.8},
    "mixed_context": {"dfa": 25.5, "ndfa": 47.3, "k_ndfa": 50.0},
    "low_sample_noisy": {"dfa": 35.3, "ndfa": 65.7, "k_ndfa": 65.8},
    "task_aligned": {"dfa": 76.6, "ndfa": 90.9, "k_ndfa": 86.7},
}

def build_summary() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(BPW / "**" / "dfa_multioutput_results.csv"), recursive=True):
        lr = float(re.search(r"/lr_([0-9p]+)/", f).group(1).replace("p", "."))
        d = pd.read_csv(f, usecols=["condition", "seed", "epoch", "test_acc"])
        d = d[d["epoch"] == d["epoch"].max()]
        d["lr"] = lr
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f"No BP-precondition run CSVs found under {BPW}")
    bpw = pd.concat(rows).groupby(["condition", "lr"])["test_acc"].mean().reset_index()
    bpw_best = bpw.loc[bpw.groupby("condition")["test_acc"].idxmax()].set_index("condition")

    out = []
    for c in ORDER:
        bp_precond = float(bpw_best.loc[c, "test_acc"] * 100)
        delta = bp_precond - TUNED_BP_PERCENT[c]
        expected = EXPECTED_DELTAS[c]
        if abs(delta - expected) > 0.06:
            raise AssertionError(
                f"BP-precondition delta drift for {c}: got {delta:.2f} pp, expected {expected:.2f} pp"
            )
        out.append(
            {
                "regime": c,
                "bp": TUNED_BP_PERCENT[c],
                "bp_precond": round(bp_precond, 3),
                "delta": round(delta, 3),
                "bp_precond_lr": float(bpw_best.loc[c, "lr"]),
                "dfa": CONTEXT_PERCENT[c]["dfa"],
                "ndfa": CONTEXT_PERCENT[c]["ndfa"],
                "k_ndfa": CONTEXT_PERCENT[c]["k_ndfa"],
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    summary = build_summary()
    output = BPW / "bpwhiten_summary.csv"
    summary.to_csv(output, index=False)
    print(f"{'condition':18} {'BP':>6} {'BP+whiten':>10} {'DFA':>6} {'nDFA':>6} {'K-nDFA':>7}")
    for r in summary.itertuples(index=False):
        print(f"{r.regime:18} {r.bp:6.1f} {r.bp_precond:10.1f} {r.dfa:6.1f} {r.ndfa:6.1f} {r.k_ndfa:7.1f}")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
