"""Aggregate the spatial-Kronecker spatial-nuisance amplitude sweep.

Tests the both-directions prediction: D(alpha) = acc(spatial-Kron) -
acc(channel-only nDFA) should rise from the clean -0.78pp (alpha=0) through zero
and turn positive as the class-independent low-frequency spatial nuisance grows.
The key anti-confound control is the channel-nDFA-minus-DFA gap, which should stay
roughly flat in alpha (channel whitening is blind to a channel-shared smooth field,
so a generic "denoising helps" effect would NOT show up there).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return {"mean": diff.mean(), "lo": lo, "hi": hi, "p": p, "n": len(diff)}


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
            row["D=spK-nDFA"] = f"{s['mean']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}] p={s['p']:.3g}"
        # confound control: channel-nDFA - DFA (should stay flat in alpha)
        if not m["ndfa_random"].empty and not m["dfa_random"].empty:
            a, b = paired(m["ndfa_random"], m["dfa_random"])
            row["nDFA-DFA"] = f"{((a-b)*100).mean():+.2f}"
        table.append(row)

    df = pd.DataFrame(table)
    lines.append(df.to_string(index=False))
    lines += [
        "",
        "Reading: if the kernel-patch spatial factor obeys the input-anisotropy law,",
        "D=spK-nDFA rises monotonically with alpha and crosses 0 (negative on clean,",
        "positive once spatial nuisance dominates the within-kernel covariance). The",
        "nDFA-DFA gap staying roughly flat rules out a generic denoising confound:",
        "channel whitening is blind to a channel-shared smooth field, so any spatial-Kron",
        "gain is attributable to the kernel-patch spatial factor alone.",
    ]
    out = "\n".join(lines) + "\n"
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    (SWEEP_ROOT / "spatialkron_sweep_summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
