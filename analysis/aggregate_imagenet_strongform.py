"""Aggregate the strong-form ImageNet-100 substitution-depth sweep.

Builds a depth x conditioner table of ImageNet-100 top-1 for raw block-DFA,
diagonal nDFA, and full-covariance nDFA (the strong form), all with the BP-norm
oracle dropped and a pretrained backbone. Includes the deep-substitution LR
check for full-covariance nDFA when present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MAIN = Path("results/imagenet100_strongform_v1")
LRCHK = Path("results/imagenet100_strongform_lrcheck_v1")
DEPTHS = ["layer4", "l34", "l234", "all"]
COND = {"dfa": "raw DFA", "ndfaDiag": "nDFA-diag", "ndfaFull": "nDFA-full"}


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
        print("\nFull-cov nDFA deep-substitution LR check:")
        for d in sorted(LRCHK.glob("*/imagenet_credit_assignment.csv")):
            last, best = final_best(d)
            print(f"  {d.parent.name:24} final={last:5.1f}  best={best:5.1f}")


if __name__ == "__main__":
    main()
