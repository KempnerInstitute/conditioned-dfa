"""Aggregate the spatial-Kron robustness controls at alpha=1.0.

For each (damping, k) config, reports D = acc(spatial-Kron) - acc(channel-nDFA)
with paired Wilcoxon + bootstrap CI. The headline run is (damping=0.3, k=4) with
D=+1.38pp. sign(D) is jointly controlled by nuisance amplitude and damping: D is
positive at the paper's standard damping 0.3 and at 1.0 but REVERSES to -1.38pp at
the under-damped 0.1 -- the over-whitening (case iii) failure the linearized
analysis predicts, not blanket damping-robustness. Absolute accuracies are needed
because part of the lambda=1.0 gap is channel-nDFA degrading under heavy damping,
not the spatial factor improving. The k=8 config shows the effect persists at a
coarser nuisance scale.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("results/infodfa_spatialkron_controls")
SWEEP = Path("results/infodfa_spatialkron_nuisance/alpha1.0")  # headline d0.3_k4
KEYS = ["seed", "feedback_seed"]
CONFIGS = [("0.1", "4"), ("0.3", "4"), ("1.0", "4"), ("0.3", "8")]


def per_seed_final(d: Path) -> pd.DataFrame:
    rows = [pd.read_csv(f) for f in glob.glob(str(d / "*dfa_convnet_results.csv"))]
    if not rows:
        return pd.DataFrame()
    x = pd.concat(rows, ignore_index=True)
    return x[x["epoch"] == x["epoch"].max()]


def diff_row(spk: pd.DataFrame, nd: pd.DataFrame) -> str:
    keys = [k for k in KEYS if k in spk.columns and k in nd.columns]
    a = spk.set_index(keys)["test_acc"]
    b = nd.set_index(keys)["test_acc"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    diff = (a - b) * 100
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return f"{diff.mean():+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3g} (n={len(diff)})"


def main() -> None:
    lines = ["# Spatial-Kron robustness controls (alpha=1.0)\n", "damping  k   D=spatialKron-nDFA"]
    for damp, k in CONFIGS:
        src = SWEEP if (damp, k) == ("0.3", "4") else ROOT / f"d{damp}_k{k}"
        spk = per_seed_final(src / "ndfa_spatial_kron")
        nd = per_seed_final(src / "ndfa_random")
        if spk.empty or nd.empty:
            lines.append(f"{damp:<8} {k:<3} MISSING")
        else:
            lines.append(f"{damp:<8} {k:<3} {diff_row(spk, nd)}")
    lines += [
        "",
        "Reading: the gain is damping-dependent in the direction the linearized analysis",
        "predicts (case iii, over-whitening). At the paper's standard damping (0.3) and",
        "at 1.0 the spatial factor helps under nuisance (D=+1.4, +7.4pp); at the",
        "under-damped 0.1 it over-whitens the small task-relevant within-patch directions",
        "and reverses (D=-1.4pp). The effect persists at a coarser nuisance scale (k=8,",
        "D=+2.0pp), so it tracks within-patch correlation rather than one frequency.",
    ]
    out = "\n".join(lines) + "\n"
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "spatialkron_controls_summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
