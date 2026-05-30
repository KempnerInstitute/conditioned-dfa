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

import pandas as pd

BPW = "results/infodfa_bpwhiten_synthetic_v1"
MAIN = os.environ["INFODFA_RESULTS"] + "/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv"
ORDER = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]


def main() -> None:
    rows = []
    for f in glob.glob(f"{BPW}/**/dfa_multioutput_results.csv", recursive=True):
        lr = float(re.search(r"/lr_([0-9p]+)/", f).group(1).replace("p", "."))
        d = pd.read_csv(f, usecols=["condition", "seed", "epoch", "test_acc"])
        d = d[d["epoch"] == d["epoch"].max()]
        d["lr"] = lr
        rows.append(d)
    bpw = pd.concat(rows).groupby(["condition", "lr"])["test_acc"].mean().reset_index()
    bpw_best = bpw.loc[bpw.groupby("condition")["test_acc"].idxmax()].set_index("condition")

    m = pd.read_csv(MAIN, usecols=["condition", "method", "feedback_rank", "epoch", "test_acc"])
    m = m[(m["epoch"] == m["epoch"].max()) & (m["feedback_rank"] == 0)]
    mean = m.groupby(["condition", "method"])["test_acc"].mean().unstack() * 100

    print(f"{'condition':18} {'BP':>6} {'BP+whiten':>10} {'DFA':>6} {'nDFA':>6} {'K-nDFA':>7}")
    for c in ORDER:
        print(f"{c:18} {mean.loc[c,'bp']:6.1f} {bpw_best.loc[c,'test_acc']*100:10.1f} "
              f"{mean.loc[c,'dfa_random']:6.1f} {mean.loc[c,'ndfa_random']:6.1f} {mean.loc[c,'ndfa_random_kronecker']:7.1f}")


if __name__ == "__main__":
    main()
