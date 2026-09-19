"""Describe matched crossed-design spatial controls without independent-run tests.

The sign depends on damping and nuisance scale. These observations do not
identify which spectral directions cause that dependence.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import os

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
    return f"{diff.mean():+.2f} (descriptive crossed-design mean; {len(diff)} combinations)"



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
    lines += ["", "All effects are descriptive over the retained crossed seed design.",
              "Damping changes the observed sign; this alone does not identify its mechanism."]
    out = "\n".join(lines) + "\n"
    ROOT.mkdir(parents=True, exist_ok=True)
    target = Path(os.environ.get("NDFA_SPATIAL_CONTROLS_OUTPUT", "results/ndfa_spatial_controls_corrected_20260918"))
    target.mkdir(parents=True, exist_ok=True)
    (target / "spatialkron_controls_summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
