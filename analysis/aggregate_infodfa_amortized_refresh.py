"""Aggregate the amortized covariance-refresh runs (cov-refresh-interval k in {1,10,50}).

Inputs (results/infodfa_amortized_refresh_v1):
  - synthetic_nuisance/k_{1,10,50}/dfa_multioutput_results.csv
    (nuisance_dominant, n_train 512, label-noise 0.2, input-noise 0.15)
  - fashion_mnist_noisy/k_{1,10,50}/dfa_nmnc_results.csv
    (fashion_mnist, n_train 10000, label-noise 0.4)
  Methods: bp, dfa_random, ndfa_random, ndfa_random_kronecker; full-rank
  feedback; 5 data seeds x 3 feedback seeds. The refresh interval k is
  encoded in the directory name (k=1 is the exact per-batch anchor; bp and
  dfa_random do not use the covariance, so their k rows are replicates).

Outputs (same directory):
  - amortized_refresh_summary.csv   cell x method x k final-epoch mean/sem
  - amortized_refresh_epochs.csv    cell x method x k per-epoch mean accuracy
  - amortized_refresh_aggregate.md  markdown report incl. deltas vs k=1
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.analysis import dataframe_to_markdown, write_markdown_report

RESULTS = ROOT / "results" / "infodfa_amortized_refresh_v1"
RUN_KEYS = ["cell", "refresh_k", "method", "seed", "feedback_seed"]


def load_all() -> pd.DataFrame:
    frames = []
    for path in sorted(
        glob.glob(str(RESULTS / "*" / "k_*" / "dfa_multioutput_results.csv"))
        + glob.glob(str(RESULTS / "*" / "k_*" / "dfa_nmnc_results.csv"))
    ):
        p = Path(path)
        match = re.fullmatch(r"k_(\d+)", p.parent.name)
        frame = pd.read_csv(path)
        frame["cell"] = p.parent.parent.name
        frame["refresh_k"] = int(match.group(1))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No result CSVs under {RESULTS}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_all()
    final = df.sort_values("epoch").groupby(RUN_KEYS, as_index=False).tail(1)

    summary = final.groupby(["cell", "method", "refresh_k"], as_index=False).agg(
        test_mean=("test_acc", "mean"),
        test_sem=("test_acc", "sem"),
        train_mean=("train_eval_acc", "mean"),
        n=("test_acc", "size"),
    )
    anchor = (
        summary[summary["refresh_k"] == 1]
        .set_index(["cell", "method"])["test_mean"]
        .rename("test_mean_k1")
    )
    summary = summary.join(anchor, on=["cell", "method"])
    summary["delta_vs_k1"] = summary["test_mean"] - summary["test_mean_k1"]
    summary.to_csv(RESULTS / "amortized_refresh_summary.csv", index=False)

    epochs = df.groupby(["cell", "method", "refresh_k", "epoch"], as_index=False).agg(
        test_mean=("test_acc", "mean"), test_sem=("test_acc", "sem")
    )
    epochs.to_csv(RESULTS / "amortized_refresh_epochs.csv", index=False)

    # Early-epoch deficit vs k=1 for the conditioned rules.
    cond = epochs[epochs["method"].isin(["ndfa_random", "ndfa_random_kronecker"])]
    wide = cond.pivot_table(index=["cell", "method", "epoch"], columns="refresh_k", values="test_mean").reset_index()
    wide = wide.rename(columns={1: "k1", 10: "k10", 50: "k50"})
    wide["k10_minus_k1"] = wide["k10"] - wide["k1"]
    wide["k50_minus_k1"] = wide["k50"] - wide["k1"]

    sections = [
        (
            "Protocol",
            "`--cov-refresh-interval k`: the damped covariance inverse used by nDFA/K-nDFA "
            "preconditioning is recomputed every k training batches and the cached inverse is "
            "reused in between. k=1 is the exact per-batch anchor (paper protocol). Cells: "
            "nuisance-dominant synthetic (n_train 512, label-noise 0.2, input-noise 0.15) and "
            "noisy Fashion-MNIST (n_train 10000, label-noise 0.4); 5 data seeds x 3 feedback "
            "seeds; bp/dfa_random do not use the covariance (their k rows are replicates).",
        ),
        ("Final accuracy by cell x method x refresh k", dataframe_to_markdown(summary, float_format=".4f")),
        ("Per-epoch deltas vs k=1 (conditioned rules)", dataframe_to_markdown(wide, float_format=".4f")),
    ]
    write_markdown_report(RESULTS / "amortized_refresh_aggregate.md", title="Amortized Covariance Refresh Aggregate", sections=sections)
    print(f"Wrote {RESULTS}/amortized_refresh_summary.csv, amortized_refresh_epochs.csv, amortized_refresh_aggregate.md")
    print(summary[summary["method"].isin(["ndfa_random", "ndfa_random_kronecker"])][
        ["cell", "method", "refresh_k", "test_mean", "test_sem", "delta_vs_k1"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
