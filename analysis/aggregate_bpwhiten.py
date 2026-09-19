"""Combine the tuned BP activity-preconditioning study with matched local controls.

Activity conditioning of the exact BP gradient tests a shared geometry effect.
It does not isolate a DFA-feedback regularization mechanism. BP tuning retains
its historical test-selection provenance; contextual local-rule means come
from the separately audited covariance-power replication, without hardcoding.
"""

from __future__ import annotations

import glob
import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path

import pandas as pd

if __package__:
    from .aggregate_actwhiten import ARCHIVED_COLUMN, build_summary as build_power_summary
else:
    from aggregate_actwhiten import ARCHIVED_COLUMN, build_summary as build_power_summary

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("INFODFA_RESULTS", ROOT / "results")).resolve()
BPW = Path(os.environ.get("INFODFA_BPWHITEN_RESULTS", RESULTS / "infodfa_bpwhiten_synthetic_v1")).resolve()
ORDER = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]

# Tuned BP values are the best-of-five-learning-rate BP controls reported in
# Table~\ref{tab:infodfa_bpwhiten}. The multioutput aggregate only stores the
# fixed sweep recipe used for trajectory panels, so keep the tuned baseline
# explicit and assert the deltas below.
TUNED_BP_PERCENT = {
    "nuisance_dominant": 27.9,
    "mixed_context": 43.8,
    "low_sample_noisy": 53.3,
    "task_aligned": 92.0,
}
EXPECTED_DELTAS = {
    "nuisance_dominant": 18.3,
    "mixed_context": 0.8,
    "low_sample_noisy": 7.1,
    "task_aligned": -0.1,
}

def build_summary(bp_root: Path | str | None = None,
                  power_root: Path | str | None = None) -> pd.DataFrame:
    """Attach audited local-rule means without changing the historical BP selection.

    ``table.attrs['cohort_manifest']`` records the matched cohort and the
    limitations of the BP provenance. This function writes no artifacts.
    """
    bp_root = Path(bp_root if bp_root is not None else BPW).resolve()
    rows = []
    source_files = []
    for f in sorted(glob.glob(str(bp_root / "**" / "dfa_multioutput_results.csv"), recursive=True)):
        lr = float(re.search(r"/lr_([0-9p]+)/", f).group(1).replace("p", "."))
        payload = Path(f).read_bytes()
        source_files.append({"path": str(Path(f).relative_to(bp_root)),
                             "sha256": hashlib.sha256(payload).hexdigest()})
        d = pd.read_csv(io.BytesIO(payload), usecols=["condition", "seed", "epoch", "test_acc"],
                        float_precision="round_trip")
        d = d[d["epoch"] == d["epoch"].max()]
        d["lr"] = lr
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f"No BP-precondition run CSVs found under {bp_root}")
    bpw = pd.concat(rows).groupby(["condition", "lr"])["test_acc"].mean().reset_index()
    bpw_best = bpw.loc[bpw.groupby("condition")["test_acc"].idxmax()].set_index("condition")

    power_table, power_manifest = build_power_summary(power_root)
    matched = power_table.set_index("regime")
    out = []
    for c in ORDER:
        bp_precond = float(bpw_best.loc[c, "test_acc"] * 100)
        delta = bp_precond - TUNED_BP_PERCENT[c]
        expected = EXPECTED_DELTAS[c]
        if abs(delta - expected) > 0.06:
            raise AssertionError(
                f"BP-precondition delta drift for {c}: got {delta:.2f} pp, expected {expected:.2f} pp"
            )
        out.append(
            {
                "regime": c,
                "bp": TUNED_BP_PERCENT[c],
                "bp_precond": bp_precond,
                "delta": delta,
                "bp_precond_lr": float(bpw_best.loc[c, "lr"]),
                "dfa": float(matched.loc[c, "dfa_random"]),
                "dfa_power_half": float(matched.loc[c, "dfa_actwhiten"]),
                "ndfa": float(matched.loc[c, "ndfa_random"]),
                "ndfa_minus_power_half_pp": float(matched.loc[c, "ndfa_minus_power_half_pp"]),
                ARCHIVED_COLUMN: float(matched.loc[c, ARCHIVED_COLUMN]),
            }
        )
    table = pd.DataFrame(out)
    table.attrs["cohort_manifest"] = {
        "schema_version": 1,
        "matched_local_rule_cohort": power_manifest,
        "bp_precondition_input_root": str(bp_root),
        "bp_precondition_source_files": source_files,
        "bp_precondition_selection": "maximum mean final test accuracy across five learning rates per regime",
        "tuned_bp_percent": TUNED_BP_PERCENT,
        "tuned_bp_provenance": "historical best-of-five-learning-rate manuscript BP controls, retained constants; not recomputed here",
        "comparison_limit": "BP rates are test-selected; local-rule columns use a fixed rate in a separate matched replication",
        "archive_exclusion": power_manifest["archive_exclusion"],
        "rounding": "CSV retains float precision; round display means and contrasts separately once",
    }
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp-root", type=Path, default=BPW)
    parser.add_argument("--power-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.bp_root, args.power_root)
    manifest = summary.attrs["cohort_manifest"]
    output = args.output_dir.resolve()
    sources = [Path(manifest["bp_precondition_input_root"]),
               Path(manifest["matched_local_rule_cohort"]["input_root"])]
    if any(output == source or source in output.parents for source in sources):
        raise ValueError("output directory must be outside the original source cohorts")
    names = ("bpwhiten_summary.csv", "bpwhiten_cohort_manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("refusing to overwrite existing BP-preconditioning report artifacts")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / names[0], index=False, mode="x")
    with (output / names[1]).open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(summary.drop(columns=ARCHIVED_COLUMN).to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(f"\nwrote {output / names[0]}; archived K column is excluded from paper claims")


if __name__ == "__main__":
    main()
