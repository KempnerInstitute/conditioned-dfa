"""Descriptive spatial-versus-channel effects in the complete crossed design.

A channel-shared field changes channel moments as well as spatial moments;
a stable channel-versus-raw gap therefore does not establish causal isolation.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import os

import sys

# Two layouts are supported:
#  (a) CIFAR-10 (default): alpha=0 anchor is the separate capable run, nonzero alphas
#      live under results/infodfa_spatialkron_nuisance/alpha{0.5,1.0,2.0}.
#  (b) Self-contained sweep root (e.g. CIFAR-100): all alphas (incl. 0.0) under one root,
#      passed as argv[1]. alpha=0 is alpha0.0/ inside that root.
if len(sys.argv) > 1:
    SWEEP_ROOT = Path(sys.argv[1])
    ALPHAS = [(a, SWEEP_ROOT / f"alpha{a}") for a in ("0.0", "0.5", "1.0", "2.0")]
else:
    CLEAN_DIR = Path("results/infodfa_capable_cifar10_v1")          # alpha = 0 anchor
    SWEEP_ROOT = Path("results/infodfa_spatialkron_nuisance")        # alpha{0.5,1.0,2.0}
    ALPHAS = [("0.0", CLEAN_DIR)] + [
        (a, SWEEP_ROOT / f"alpha{a}") for a in ("0.5", "1.0", "2.0")
    ]
KEYS = ["seed", "feedback_seed"]


def per_seed_final(method_dir: Path) -> pd.DataFrame:
    rows = [pd.read_csv(f) for f in glob.glob(str(method_dir / "*dfa_convnet_results.csv"))]
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    return d[d["epoch"] == d["epoch"].max()]


def paired(a_df: pd.DataFrame, b_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    keys = [k for k in KEYS if k in a_df.columns and k in b_df.columns]
    if keys:
        a = a_df.set_index(keys)["test_acc"]
        b = b_df.set_index(keys)["test_acc"]
        common = a.index.intersection(b.index)
        return a.loc[common].to_numpy(), b.loc[common].to_numpy()
    n = min(len(a_df), len(b_df))
    return a_df["test_acc"].to_numpy()[:n], b_df["test_acc"].to_numpy()[:n]


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = (a - b) * 100
    return {"mean": float(diff.mean()), "n_combinations": len(diff),
            "scope": "Descriptive complete crossed-design mean; no independent-run inference"}


def main() -> None:
    dataset = "CIFAR-100" if "cifar100" in str(SWEEP_ROOT).lower() else "CIFAR-10"
    lines = [f"# Spatial-Kronecker spatial-nuisance sweep (capable {dataset})\n"]
    table = []
    for alpha, d in ALPHAS:
        m = {name: per_seed_final(d / name) for name in
             ("bp", "dfa_random", "ndfa_random", "ndfa_spatial_kron")}
        means = {k: (v["test_acc"].mean() * 100 if not v.empty else float("nan")) for k, v in m.items()}
        row = {"alpha": alpha, **{k: round(means[k], 2) for k in means}}
        # D(alpha) = spatial-Kron - channel-nDFA
        if not m["ndfa_spatial_kron"].empty and not m["ndfa_random"].empty:
            a, b = paired(m["ndfa_spatial_kron"], m["ndfa_random"])
            s = diff_stats(a, b)
            row["D=spK-nDFA"] = f"{s['mean']:+.2f} (crossed-design mean)"
        # confound control: channel-nDFA - DFA (should stay flat in alpha)
        if not m["ndfa_random"].empty and not m["dfa_random"].empty:
            a, b = paired(m["ndfa_random"], m["dfa_random"])
            row["nDFA-DFA"] = f"{((a-b)*100).mean():+.2f}"
        table.append(row)

    df = pd.DataFrame(table)
    lines.append(df.to_string(index=False))
    lines += ["", "Effects describe the retained crossed seed combinations.",
              "Both channel and spatial moments can respond to channel-shared nuisance."]
    out = "\n".join(lines) + "\n"
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    target = Path(os.environ.get("NDFA_SPATIAL_SWEEP_OUTPUT", "results/ndfa_spatial_sweep_corrected_20260918"))
    target.mkdir(parents=True, exist_ok=True)
    (target / (dataset.lower()+"_spatialkron_sweep_summary.md")).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
