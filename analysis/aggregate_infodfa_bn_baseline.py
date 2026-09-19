"""Aggregate the DFA+BatchNorm / BP+BatchNorm baseline runs and compare them,
cell-matched, against the paper's no-BN sweeps.

Inputs:
  - results/infodfa_bn_baseline_v1/synthetic/<condition>/ntrain_*/label_*/input_*/dfa_multioutput_results.csv
    (bp, dfa_random with --batchnorm; full-rank feedback rank 0; 5 data seeds x 3 feedback seeds)
  - results/infodfa_bn_baseline_v1/fashion_mnist/ntrain_*/label_*/dfa_nmnc_results.csv
    (same methods with --batchnorm on noisy Fashion-MNIST)
  - Legacy no-BN sweeps (cell-matched: identical grids, data seeds 0-4,
    feedback seeds 0-2, full-rank feedback_rank==0):
      <legacy>/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv
      <legacy>/infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_all.csv

Outputs (in results/infodfa_bn_baseline_v1):
  - bn_baseline_synthetic_cells.csv    per-cell method means (BN and no-BN) + deltas
  - bn_baseline_synthetic_regime.csv   per-regime means of the per-cell values
  - bn_baseline_fashion_cells.csv      per Fashion-MNIST cell method means + deltas
  - bn_baseline_aggregate.md           markdown report
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.analysis import dataframe_to_markdown, write_markdown_report

INPUT_RESULTS = ROOT / "results" / "infodfa_bn_baseline_v1"
RESULTS = Path(os.environ.get("INFODFA_BN_BASELINE_OUTPUT", INPUT_RESULTS))
LEGACY = Path(os.environ.get("INFODFA_LEGACY_RESULTS", ROOT / ".." / "Info-Man" / "results")).resolve()

SYN_CELL = ["condition", "input_noise", "n_train", "train_label_noise"]
VIS_CELL = ["n_train", "label_noise"]
COND_ORDER = ["nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned"]
LEGACY_METHODS = ["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker"]
RENAME = {
    "bp": "bp",
    "dfa_random": "dfa",
    "ndfa_random": "ndfa",
    "ndfa_random_kronecker": "kndfa",
}


def load_final(paths: list[str], run_keys: list[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(paths):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No input CSVs found under {paths[:1]}")
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in run_keys if k in df.columns]
    return df.sort_values("epoch").groupby(keys, as_index=False).tail(1)


def per_cell_method_means(final: pd.DataFrame, cell: list[str], suffix: str) -> pd.DataFrame:
    """Cell x method mean over (seed, feedback_seed); wide, one column per method."""
    means = final.groupby(cell + ["method"], as_index=False).agg(
        acc=("test_acc", "mean"), n=("test_acc", "size")
    )
    wide = means.pivot_table(index=cell, columns="method", values="acc").reset_index()
    wide = wide.rename(columns={m: f"{RENAME.get(m, m)}{suffix}" for m in RENAME})
    return wide


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    # --- synthetic suite ---
    syn_run_keys = SYN_CELL + ["method", "seed", "feedback_seed", "feedback_rank"]
    bn_syn = load_final(
        glob.glob(str(INPUT_RESULTS / "synthetic" / "*" / "ntrain_*" / "label_*" / "input_*" / "dfa_multioutput_results.csv")),
        syn_run_keys,
    )
    legacy_syn_all = pd.read_csv(
        LEGACY / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_all.csv",
        usecols=syn_run_keys + ["epoch", "test_acc"],
    )
    legacy_syn = (
        legacy_syn_all[
            (legacy_syn_all["feedback_rank"] == 0) & legacy_syn_all["method"].isin(LEGACY_METHODS)
        ]
        .sort_values("epoch")
        .groupby(syn_run_keys, as_index=False)
        .tail(1)
    )

    bn_wide = per_cell_method_means(bn_syn, SYN_CELL, "_bn")
    legacy_wide = per_cell_method_means(legacy_syn, SYN_CELL, "")
    cells = legacy_wide.merge(bn_wide, on=SYN_CELL, how="inner", validate="1:1")
    if len(cells) != 128:
        print(f"WARNING: matched {len(cells)} synthetic cells (expected 128)")
    cells["delta_dfa_bn_minus_dfa"] = cells["dfa_bn"] - cells["dfa"]
    cells["delta_ndfa_minus_dfa_bn"] = cells["ndfa"] - cells["dfa_bn"]
    cells["delta_kndfa_minus_dfa_bn"] = cells["kndfa"] - cells["dfa_bn"]
    cells["delta_bp_bn_minus_bp"] = cells["bp_bn"] - cells["bp"]
    cells.to_csv(RESULTS / "bn_baseline_synthetic_cells.csv", index=False)

    value_cols = [
        "bp", "dfa", "ndfa", "kndfa", "bp_bn", "dfa_bn",
        "delta_dfa_bn_minus_dfa", "delta_ndfa_minus_dfa_bn",
        "delta_kndfa_minus_dfa_bn", "delta_bp_bn_minus_bp",
    ]
    agg = {c: (c, "mean") for c in value_cols}
    agg["n_cells"] = ("dfa_bn", "size")
    regime = cells.groupby("condition", as_index=False).agg(**agg)
    # The grid reuses global seeds. Average feedback draws and the complete
    # grid within each seed before estimating replication uncertainty.
    seed_cells = per_cell_method_means(legacy_syn, SYN_CELL + ["seed"], "").merge(
        per_cell_method_means(bn_syn, SYN_CELL + ["seed"], "_bn"),
        on=SYN_CELL + ["seed"], validate="one_to_one")
    assert len(seed_cells) == 128 * 5
    seed_cells["delta_dfa_bn_minus_dfa"] = seed_cells.dfa_bn - seed_cells.dfa
    seed_cells["delta_ndfa_minus_dfa_bn"] = seed_cells.ndfa - seed_cells.dfa_bn
    contrasts = ["delta_dfa_bn_minus_dfa", "delta_ndfa_minus_dfa_bn"]
    seed_means = seed_cells.groupby(["condition", "seed"])[contrasts].mean().reset_index()
    assert seed_means.groupby("condition").size().eq(5).all()
    seed_means.to_csv(RESULTS / "bn_baseline_synthetic_seed_contrasts.csv", index=False)
    errors = seed_means.groupby("condition")[contrasts].sem().add_suffix("_sem")
    regime = regime.merge(errors, on="condition", validate="one_to_one")
    regime["replication_unit"] = "five global-seed means; fixed grid and feedback draws"
    regime["condition"] = pd.Categorical(regime["condition"], COND_ORDER, ordered=True)
    regime = regime.sort_values("condition")
    regime.to_csv(RESULTS / "bn_baseline_synthetic_regime.csv", index=False)

    # --- noisy Fashion-MNIST ---
    vis_run_keys = ["dataset"] + VIS_CELL + ["method", "seed", "feedback_seed", "feedback_rank"]
    bn_vis = load_final(
        glob.glob(str(INPUT_RESULTS / "fashion_mnist" / "ntrain_*" / "label_*" / "dfa_nmnc_results.csv")),
        vis_run_keys,
    )
    legacy_vis_all = pd.read_csv(
        LEGACY / "infodfa_vision_noise_sweep_aggregate_v2" / "dfa_nmnc_all.csv",
        usecols=vis_run_keys + ["epoch", "test_acc"],
    )
    legacy_vis = (
        legacy_vis_all[
            (legacy_vis_all["dataset"] == "fashion_mnist")
            & (legacy_vis_all["feedback_rank"] == 0)
            & legacy_vis_all["method"].isin(LEGACY_METHODS)
        ]
        .sort_values("epoch")
        .groupby(vis_run_keys, as_index=False)
        .tail(1)
    )
    bn_vis_wide = per_cell_method_means(bn_vis, VIS_CELL, "_bn")
    legacy_vis_wide = per_cell_method_means(legacy_vis, VIS_CELL, "")
    vis_cells = legacy_vis_wide.merge(bn_vis_wide, on=VIS_CELL, how="inner", validate="1:1")
    if len(vis_cells) != 12:
        print(f"WARNING: matched {len(vis_cells)} fashion cells (expected 12)")
    vis_cells["delta_dfa_bn_minus_dfa"] = vis_cells["dfa_bn"] - vis_cells["dfa"]
    vis_cells["delta_ndfa_minus_dfa_bn"] = vis_cells["ndfa"] - vis_cells["dfa_bn"]
    vis_cells["delta_bp_bn_minus_bp"] = vis_cells["bp_bn"] - vis_cells["bp"]
    vis_cells.to_csv(RESULTS / "bn_baseline_fashion_cells.csv", index=False)

    noisy_vis = vis_cells[vis_cells["label_noise"] > 0]
    sections = [
        (
            "Protocol",
            "BN runs: bp and dfa_random with `--batchnorm` (BatchNorm1d after each hidden "
            "linear, pre-activation), full-rank feedback (rank 0), 5 data seeds x 3 feedback "
            "seeds, otherwise the paper sweep protocol. Comparison partners are the paper's "
            "no-BN sweeps restricted to the SAME cells, SAME data seeds (0-4), SAME feedback "
            "seeds (0-2), and full-rank (`feedback_rank == 0`) rows; per-cell values are means "
            "over seed x feedback_seed at the final epoch. Reported contrast SEMs use five "
            "global-seed summaries after averaging the full grid and feedback draws; they "
            "are conditional on the fixed designed conditions and feedback set.",
        ),
        ("Synthetic: per-regime means (mean over the 32 cells per regime)", dataframe_to_markdown(regime, float_format=".4f")),
        ("Fashion-MNIST: per-cell values", dataframe_to_markdown(vis_cells, float_format=".4f")),
        (
            "Fashion-MNIST: mean over noisy cells (label_noise > 0)",
            dataframe_to_markdown(
                noisy_vis[["dfa", "ndfa", "dfa_bn", "delta_dfa_bn_minus_dfa", "delta_ndfa_minus_dfa_bn"]]
                .mean()
                .to_frame("mean_over_9_noisy_cells")
                .reset_index(names="quantity"),
                float_format=".4f",
            ),
        ),
        ("Synthetic: per-cell values", dataframe_to_markdown(cells, float_format=".4f")),
    ]
    write_markdown_report(RESULTS / "bn_baseline_aggregate.md", title="DFA+BatchNorm Baseline Aggregate", sections=sections)
    print(f"Wrote {RESULTS}/bn_baseline_*.csv and bn_baseline_aggregate.md")
    print(regime[["condition", "dfa", "dfa_bn", "ndfa", "delta_dfa_bn_minus_dfa", "delta_ndfa_minus_dfa_bn"]])


if __name__ == "__main__":
    main()
