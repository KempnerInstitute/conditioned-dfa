"""Aggregate the spatial-Kronecker conv test (capable CIFAR-10 convnet).

Compares spatial-Kronecker conditioning (channel + kernel-spatial covariance
factors) against channel-only nDFA, raw DFA, and BP under an identical recipe,
to test whether the added kernel-spatial factor improves credit assignment on a
real capable convnet. Reports seed-level paired Wilcoxon (spatial-Kron vs nDFA)
with a bootstrap CI on the difference.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "results/infodfa_capable_cifar10_v1")
ORDER = ["bp", "ndfa_spatial_kron", "ndfa_random", "ndfa_random_kronecker", "dfa_random"]


def per_seed_final(method_dir: Path) -> pd.DataFrame:
    rows = [pd.read_csv(f) for f in glob.glob(str(method_dir / "*dfa_convnet_results.csv"))]
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d = d[d["epoch"] == d["epoch"].max()]
    return d


def main() -> None:
    frames = {m: per_seed_final(SRC / m) for m in ORDER}
    frames = {m: d for m, d in frames.items() if not d.empty}
    summary = []
    for m, d in frames.items():
        accs = d["test_acc"].to_numpy() * 100
        summary.append((m, accs.mean(), accs.std(ddof=1) / max(np.sqrt(len(accs)), 1), len(accs)))
    s = pd.DataFrame(summary, columns=["method", "mean", "sem", "n"]).set_index("method")
    s = s.reindex([m for m in ORDER if m in s.index])

    report = ["# Spatial-Kronecker conv test (capable CIFAR-10)\n", s.round(3).to_string(), ""]

    if "ndfa_spatial_kron" in frames and "ndfa_random" in frames:
        # Pair by (seed, feedback_seed) when available; else pool.
        keys = [k for k in ("seed", "feedback_seed") if k in frames["ndfa_random"].columns]
        a = frames["ndfa_spatial_kron"].set_index(keys)["test_acc"] if keys else frames["ndfa_spatial_kron"]["test_acc"]
        b = frames["ndfa_random"].set_index(keys)["test_acc"] if keys else frames["ndfa_random"]["test_acc"]
        if keys:
            common = a.index.intersection(b.index)
            a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        else:
            n = min(len(a), len(b))
            a, b = a.to_numpy()[:n], b.to_numpy()[:n]
        diff = (a - b) * 100
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try:
            _, p = stats.wilcoxon(a, b)
        except ValueError:
            p = float("nan")
        report += [
            f"\nspatial-Kron vs nDFA (channel-only), paired over {len(diff)} runs:",
            f"  mean diff = {diff.mean():+.3f} pp  (95% CI [{lo:+.3f}, {hi:+.3f}])",
            f"  Wilcoxon p = {p:.4g}",
            "  -> spatial factor helps" if lo > 0 else
            ("  -> spatial factor hurts" if hi < 0 else "  -> no significant difference from channel-only nDFA"),
        ]

    out = "\n".join(report) + "\n"
    (SRC / "spatial_kron_summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
