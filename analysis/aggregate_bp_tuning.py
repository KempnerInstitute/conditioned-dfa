"""Aggregate the tuning-matched-BP synthetic sweep and compare to conditioned DFA.

Answers the reviewer concern that the synthetic "beats BP" claim used a single,
possibly under-tuned BP learning rate (lr=0.08). We re-ran BP over an lr grid at
every cell; here we pick the tuned BP per cell by leave-one-seed-out selection
over lr (unbiased), then compare best-conditioned vs tuned BP at the seed level.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BP_TUNE = Path("results/infodfa_bp_tuning_synthetic_v1").resolve()
MAIN = Path(os.environ["INFODFA_RESULTS"]) / "infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv"
OUT = Path(os.environ["INFODFA_RESULTS"]) / "infodfa_bp_tuning_summary"
CELL = ["condition", "input_noise", "n_train", "train_label_noise"]
COND_ORDER = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]


def load_bp_tuning():
    rows = []
    for f in glob.glob(str(BP_TUNE / "**/dfa_multioutput_results.csv"), recursive=True):
        m = re.search(r"/lr_([0-9p]+)/", f)
        lr = float(m.group(1).replace("p", "."))
        df = pd.read_csv(f, usecols=CELL + ["seed", "epoch", "test_acc"])
        df = df[df["epoch"] == df["epoch"].max()]
        df["lr"] = lr
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def tuned_bp_per_seed(bp):
    """LOSO over seeds: per (cell, held-out seed) pick best lr on the other seeds."""
    out = []
    for keys, sub in bp.groupby(CELL):
        cell = dict(zip(CELL, keys))
        seeds = sorted(sub["seed"].unique())
        for held in seeds:
            others = sub[sub["seed"] != held]
            best_lr = others.groupby("lr")["test_acc"].mean().idxmax()
            v = sub[(sub["seed"] == held) & (sub["lr"] == best_lr)]["test_acc"]
            if len(v):
                out.append({**cell, "seed": held, "tuned_bp": float(v.mean()), "best_lr": best_lr})
    return pd.DataFrame(out)


def load_conditioned():
    cols = CELL + ["method", "seed", "feedback_rank", "epoch", "test_acc"]
    df = pd.read_csv(MAIN, usecols=cols)
    df = df[df["epoch"] == df["epoch"].max()]
    df = df[df["feedback_rank"] == 0]  # honest-selection reanalysis: rank choice is immaterial
    df = df.groupby(CELL + ["method", "seed"], as_index=False)["test_acc"].mean()
    piv = df.pivot_table(index=CELL + ["seed"], columns="method", values="test_acc").reset_index()
    piv["best_cond"] = piv[["ndfa_random", "ndfa_random_kronecker"]].max(axis=1)
    return piv.rename(columns={"bp": "orig_bp"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bp = load_bp_tuning()
    tuned = tuned_bp_per_seed(bp)
    cond = load_conditioned()
    merged = cond.merge(tuned, on=CELL + ["seed"], how="inner")

    rows = []
    for c in COND_ORDER:
        s = merged[merged["condition"] == c]
        orig = 100 * s["orig_bp"].mean()
        tunedm = 100 * s["tuned_bp"].mean()
        best = 100 * s["best_cond"].mean()
        d = (s["best_cond"] - s["tuned_bp"]) * 100
        w = stats.wilcoxon(d) if len(d) > 5 and d.abs().sum() > 0 else None
        rows.append({
            "condition": c, "n": len(s),
            "orig_BP(lr.08)": orig, "tuned_BP": tunedm, "best_cond": best,
            "cond-origBP": best - orig, "cond-tunedBP": best - tunedm,
            "p(cond>tunedBP)": (w.pvalue if w else np.nan),
            "median_best_lr": float(s["best_lr"].median()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "bp_tuning_summary.csv", index=False)
    report = "# Tuning-matched BP vs conditioned DFA (synthetic)\n\n" + summary.round(2).to_string(index=False)
    (OUT / "bp_tuning_summary.md").write_text(report)
    print(report)
    print("\nlr usage across tuned-BP selections:")
    print(tuned["best_lr"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
