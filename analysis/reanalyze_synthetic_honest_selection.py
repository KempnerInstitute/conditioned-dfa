"""Honest re-analysis of the synthetic stress suite (no new compute).

The headline synthetic numbers selected the feedback rank per (cell, method) by
MAXIMIZING seed-averaged test accuracy -- i.e. model selection on the test set,
with only the conditioned rules getting the 5-way rank sweep. This script
recomputes the per-regime accuracies and the gains over DFA / BP under three
selection schemes, using only the existing per-seed runs:

  1. test_selected : the paper's scheme (argmax over ranks of test mean).
  2. fixed_full    : every method at feedback_rank 0 (no selection at all).
  3. loso          : leave-one-seed-out rank selection -- for each held-out seed,
                     pick the rank on the other seeds, evaluate on the held-out
                     seed. This is an unbiased estimate of the selected-rank
                     accuracy and removes the test-set-selection bias without
                     new compute.

It also runs seed-level paired tests (conditioned vs DFA, conditioned vs BP)
instead of the cell-averaged "n=32" pairing the reviewers flagged.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(os.environ.get("INFODFA_RESULTS", "../results")).resolve()
SRC = RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv"
OUT = RESULTS / "infodfa_honest_selection_reanalysis"

CELL = ["condition", "input_noise", "n_train", "train_label_noise"]
COND_ORDER = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]
CONDITIONED = ["ndfa_random", "ndfa_random_kronecker"]
METHODS = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker"]


def load_final():
    cols = CELL + ["method", "seed", "feedback_rank", "epoch", "test_acc"]
    df = pd.read_csv(SRC, usecols=cols)
    df = df[df["method"].isin(METHODS)].copy()
    df = df[df["epoch"] == df["epoch"].max()].copy()
    # average over any remaining nuisance dims (e.g. feedback_seed) per cell/method/seed/rank
    df = (df.groupby(CELL + ["method", "seed", "feedback_rank"], as_index=False)["test_acc"].mean())
    return df


def per_cell_method_value(df, scheme):
    """Return per (cell, method, seed) test_acc under the selection scheme."""
    rows = []
    for keys, sub in df.groupby(CELL + ["method"]):
        *cell_vals, m = keys
        cell = dict(zip(CELL, cell_vals))
        if m not in CONDITIONED:
            # BP / DFA: rank 0 only
            s = sub[sub["feedback_rank"] == 0]
            for _, r in s.iterrows():
                rows.append({**cell, "method": m, "seed": r["seed"], "test_acc": r["test_acc"]})
            continue
        seeds = sorted(sub["seed"].unique())
        if scheme == "fixed_full":
            s = sub[sub["feedback_rank"] == 0]
            for _, r in s.iterrows():
                rows.append({**cell, "method": m, "seed": r["seed"], "test_acc": r["test_acc"]})
        elif scheme == "test_selected":
            rank_mean = sub.groupby("feedback_rank")["test_acc"].mean()
            best = rank_mean.idxmax()
            s = sub[sub["feedback_rank"] == best]
            for _, r in s.iterrows():
                rows.append({**cell, "method": m, "seed": r["seed"], "test_acc": r["test_acc"]})
        elif scheme == "loso":
            for held in seeds:
                others = sub[sub["seed"] != held]
                rank_mean = others.groupby("feedback_rank")["test_acc"].mean()
                best = rank_mean.idxmax()
                held_val = sub[(sub["seed"] == held) & (sub["feedback_rank"] == best)]["test_acc"]
                if len(held_val):
                    rows.append({**cell, "method": m, "seed": held, "test_acc": float(held_val.mean())})
    return pd.DataFrame(rows)


def summarize(val):
    """Per condition: mean accuracy per method + gains vs DFA and BP (pp)."""
    out = []
    for cond in COND_ORDER:
        c = val[val["condition"] == cond]
        means = c.groupby("method")["test_acc"].mean() * 100
        best_cond = max(means.get("ndfa_random", np.nan), means.get("ndfa_random_kronecker", np.nan))
        out.append({
            "condition": cond,
            "BP": means.get("bp", np.nan),
            "DFA": means.get("dfa_random", np.nan),
            "nDFA": means.get("ndfa_random", np.nan),
            "K-nDFA": means.get("ndfa_random_kronecker", np.nan),
            "best_cond-DFA": best_cond - means.get("dfa_random", np.nan),
            "best_cond-BP": best_cond - means.get("bp", np.nan),
        })
    return pd.DataFrame(out)


def seed_level_tests(val):
    """Paired tests at the (cell, seed) level: best-conditioned vs DFA and vs BP."""
    piv = val.pivot_table(index=CELL + ["seed"], columns="method", values="test_acc")
    piv = piv.dropna(subset=["bp", "dfa_random"])
    piv["best_cond"] = piv[["ndfa_random", "ndfa_random_kronecker"]].max(axis=1)
    rows = []
    for cond in COND_ORDER:
        c = piv.xs(cond, level="condition")
        d_dfa = (c["best_cond"] - c["dfa_random"]).dropna() * 100
        d_bp = (c["best_cond"] - c["bp"]).dropna() * 100
        t_dfa = stats.wilcoxon(d_dfa) if len(d_dfa) > 5 and d_dfa.abs().sum() > 0 else None
        t_bp = stats.wilcoxon(d_bp) if len(d_bp) > 5 and d_bp.abs().sum() > 0 else None
        rows.append({
            "condition": cond, "n_pairs": len(d_dfa),
            "mean_gain_vs_DFA": d_dfa.mean(), "p_vs_DFA": (t_dfa.pvalue if t_dfa else np.nan),
            "mean_gain_vs_BP": d_bp.mean(), "p_vs_BP": (t_bp.pvalue if t_bp else np.nan),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_final()
    schemes = {}
    for scheme in ["test_selected", "fixed_full", "loso"]:
        val = per_cell_method_value(df, scheme)
        schemes[scheme] = summarize(val)
        if scheme == "loso":
            loso_val = val
    lines = ["# Honest synthetic re-analysis (no new compute)\n",
             f"Source: `{SRC}`  |  seeds={sorted(df['seed'].unique())}  ranks={sorted(df['feedback_rank'].unique())}\n"]
    for scheme, tab in schemes.items():
        lines.append(f"\n## Scheme: {scheme}\n")
        lines.append(tab.round(2).to_string(index=False))
    lines.append("\n## Seed-level paired tests (best conditioned), LOSO values\n")
    lines.append(seed_level_tests(loso_val).round(4).to_string(index=False))
    report = "\n".join(lines)
    (OUT / "honest_selection_reanalysis.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
