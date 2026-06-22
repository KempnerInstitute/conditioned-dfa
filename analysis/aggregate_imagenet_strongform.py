"""Aggregate the full-covariance ZCA ImageNet-100 substitution-depth diagnostic.

Builds a depth x conditioner table of ImageNet-100 top-1 for raw block-DFA,
diagonal nDFA, and full-covariance inverse-square-root/ZCA conditioning, all with the
BP-norm oracle dropped and a pretrained backbone. Includes the
deep-substitution LR check for full-covariance ZCA when present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MAIN = Path("results/imagenet100_strongform_v1")
LRCHK = Path("results/imagenet100_strongform_lrcheck_v1")
DEPTHS = ["layer4", "l34", "l234", "all"]
COND = {"dfa": "raw DFA", "ndfaDiag": "nDFA-diag", "ndfaFull": "full-cov ZCA"}
SEEDS = [0, 1, 2]  # seed 0 lives in the unsuffixed dir, others in <tag>_seed<N>


def seed_dir(tag: str, seed: int) -> Path:
    if seed == 0:
        # Deep full-covariance configs collapsed at lr=0.1; the headline (and
        # seeds 1-2) use the eta=0.01 lr-check runs, so those are seed 0 here.
        if tag in ("ndfaFull_l234", "ndfaFull_all"):
            return LRCHK / f"{tag}_lr01"
        return MAIN / tag
    return MAIN / f"{tag}_seed{seed}"


def final_best(csv: Path):
    d = pd.read_csv(csv)
    last = float(d.loc[d["epoch"] == d["epoch"].max(), "val_top1"].iloc[0])
    best = float(d["val_top1"].max())
    return last, best


def main() -> None:
    bp = final_best(MAIN / "bp/imagenet_credit_assignment.csv")
    print(f"BP reference: final={bp[0]:.2f}  best={bp[1]:.2f}\n")
    print(f"{'depth':>8} | " + " | ".join(f"{c:>22}" for c in COND.values()))
    for depth in DEPTHS:
        cells = []
        for key in COND:
            csv = MAIN / f"{key}_{depth}/imagenet_credit_assignment.csv"
            if csv.exists():
                last, best = final_best(csv)
                cells.append(f"{last:5.1f} (best {best:5.1f})")
            else:
                cells.append(f"{'--':>22}")
        print(f"{depth:>8} | " + " | ".join(f"{c:>22}" for c in cells))

    if LRCHK.exists():
        print("\nFull-cov ZCA deep-substitution LR check:")
        for d in sorted(LRCHK.glob("*/imagenet_credit_assignment.csv")):
            last, best = final_best(d)
            print(f"  {d.parent.name:24} final={last:5.1f}  best={best:5.1f}")

    print("\nMulti-seed final top-1 (mean +/- sem over available seeds):")
    rows = []
    for tag in ["bp"] + [f"{k}_{d}" for k in COND for d in DEPTHS]:
        finals = []
        for s in SEEDS:
            csv = seed_dir(tag, s) / "imagenet_credit_assignment.csv"
            if csv.exists():
                finals.append(final_best(csv)[0])
        if not finals:
            continue
        ser = pd.Series(finals)
        sem = ser.sem() if len(finals) > 1 else float("nan")
        rows.append({"tag": tag, "n_seeds": len(finals), "mean": ser.mean(),
                     "sem": sem, "finals": [round(f, 2) for f in finals]})
        print(f"  {tag:18} n={len(finals)}  {ser.mean():5.2f} +/- {sem:4.2f}"
              f"   {rows[-1]['finals']}")
    out = MAIN / "strongform_multiseed_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
