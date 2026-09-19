"""Audit the matched covariance-power update comparison at raw precision.

The power-half control applies G(C + ridge I)^(-1/2) to the weight update;
it does not decorrelate forward activations. Historical K-nDFA measurements
are retained for provenance only and excluded from current paper claims.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get(
    "INFODFA_ACTWHITEN_RESULTS", PROJECT_ROOT / "results/infodfa_actwhiten_synthetic_v1"
)).resolve()
REGIMES = ("nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned")
METHODS = ("dfa_random", "dfa_actwhiten", "ndfa_random", "ndfa_random_kronecker")
SEEDS = tuple(range(5))
FINAL_EPOCH = 14
N_TRAINS = (512, 1024, 2048, 4096)
LABEL_NOISES = (0.0, 0.1, 0.2, 0.4)
INPUT_NOISES = (0.05, 0.15)
ARCHIVED_COLUMN = "archived_k_ndfa_excluded"


def expected_cells() -> dict[str, tuple[str, int, float, float]]:
    """The recorded launch grid, rather than whichever files happen to exist."""
    cells = {}
    for condition, n_train, label, noise in itertools.product(
        REGIMES, N_TRAINS, LABEL_NOISES, INPUT_NOISES
    ):
        label_tag, noise_tag = str(label).replace(".", "p"), str(noise).replace(".", "p")
        relative = (f"{condition}/ntrain_{n_train}/label_{label_tag}/"
                    f"input_{noise_tag}/dfa_multioutput_results.csv")
        cells[relative] = (condition, n_train, label, noise)
    return cells


def build_summary(input_root: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return unrounded mean percentages and a strictly audited cohort manifest.

    Require all 128 cells and the complete 5-by-5 seed/feedback cross at epoch
    14 for all four methods. No endpoint selection, intermediate rounding, or
    inference about independent replication is performed.
    """
    root = Path(input_root if input_root is not None else ROOT).resolve()
    expected = expected_cells()
    observed = {str(path.relative_to(root)) for path in root.rglob("dfa_multioutput_results.csv")}
    if observed != set(expected):
        raise ValueError(
            "covariance-power cohort requires the exact 128-cell grid; "
            f"missing={sorted(set(expected) - observed)}, extra={sorted(observed - set(expected))}"
        )
    columns = ["condition", "method", "seed", "feedback_seed", "feedback_rank",
               "epoch", "n_train", "train_label_noise", "input_noise", "test_acc"]
    expected_keys = set(itertools.product(METHODS, SEEDS, SEEDS))
    parts, source_files = [], []
    cohort_digest = hashlib.sha256()
    for relative, (condition, n_train, label, noise) in sorted(expected.items()):
        path = root / relative
        # Hash the bytes actually parsed, including during concurrent replacement.
        payload = path.read_bytes()
        digest = hashlib.sha256(payload)
        cohort_digest.update(relative.encode() + b"\0" + digest.digest())
        source_files.append({"path": relative, "sha256": digest.hexdigest()})
        data = pd.read_csv(io.BytesIO(payload), usecols=columns, float_precision="round_trip")
        if data.empty:
            raise ValueError(f"empty covariance-power cell: {relative}")
        for column, value in (("condition", condition), ("n_train", n_train),
                              ("train_label_noise", label), ("input_noise", noise),
                              ("feedback_rank", 0)):
            if not data[column].eq(value).all():
                raise ValueError(f"wrong {column} in covariance-power cell: {relative}")
        for column in ("seed", "feedback_seed"):
            if not data[column].isin(SEEDS).all():
                raise ValueError(f"wrong {column} in covariance-power cell: {relative}")
        if (not data["epoch"].isin(range(FINAL_EPOCH + 1)).all()
                or data["epoch"].max() != FINAL_EPOCH):
            raise ValueError(f"wrong final epoch in covariance-power cell: {relative}")
        if not data["method"].isin(METHODS).all():
            raise ValueError(f"unexpected method in covariance-power cell: {relative}")
        final = data.loc[data["epoch"].eq(FINAL_EPOCH)].copy()
        keys = list(final[["method", "seed", "feedback_seed"]].itertuples(index=False, name=None))
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate final endpoint in covariance-power cell: {relative}")
        if set(keys) != expected_keys:
            raise ValueError(f"missing final seed/feedback endpoint in covariance-power cell: {relative}")
        if not final["test_acc"].map(lambda x: math.isfinite(x) and 0 <= x <= 1).all():
            raise ValueError(f"invalid final accuracy in covariance-power cell: {relative}")
        final["cell"] = relative
        parts.append(final)

    endpoints = pd.concat(parts, ignore_index=True)
    # Feedback draws are averaged within a model seed; balanced cells/seeds
    # receive equal weight. These levels are not independent draws of the task.
    seed_means = endpoints.groupby(["condition", "cell", "method", "seed"])["test_acc"].mean()
    means = seed_means.groupby(["condition", "method"]).mean().unstack().mul(100)
    table = means.reindex(index=REGIMES, columns=METHODS).rename(
        columns={"ndfa_random_kronecker": ARCHIVED_COLUMN}
    )
    table["power_half_minus_dfa_pp"] = table["dfa_actwhiten"] - table["dfa_random"]
    table["ndfa_minus_power_half_pp"] = table["ndfa_random"] - table["dfa_actwhiten"]
    table.index.name, table.columns.name = "regime", None
    table = table.reset_index()
    manifest = {
        "schema_version": 1,
        "input_root": str(root),
        "source_file_count": len(source_files),
        "cohort_csv_sha256": cohort_digest.hexdigest(),
        "cohort_digest_definition": "sorted relative path UTF-8 + null byte + raw SHA-256 digest per CSV",
        "source_files": source_files,
        "regimes": list(REGIMES), "cells_per_regime": 32,
        "model_seeds": list(SEEDS), "feedback_seeds": list(SEEDS),
        "feedback_rank": 0, "final_epoch": FINAL_EPOCH,
        "methods": list(METHODS), "endpoints_per_regime_method": 800,
        "total_final_endpoints": len(endpoints),
        "aggregation": "feedback mean within cell/model seed, then equally weighted balanced cells and model seeds",
        "metric": "final test_acc multiplied by 100; no checkpoint or hyperparameter selection",
        "rounding": "CSV retains float precision; round display means and contrasts separately once",
        "archive_exclusion": {
            "column": ARCHIVED_COLUMN,
            "reason": "historical mis-scaled error factor; excluded from current paper claims",
        },
        "source_provenance_limit": "retained CSVs and launch recipe; no contemporaneous full training-source hash manifest",
    }
    return table, manifest


def write_report(table: pd.DataFrame, manifest: dict, output_dir: Path | str) -> None:
    """Write fresh artifacts outside the original source cohort."""
    output = Path(output_dir).resolve()
    source = Path(manifest["input_root"])
    if output == source or source in output.parents:
        raise ValueError("output directory must be outside the original source cohort")
    names = ("actwhiten_summary.csv", "actwhiten_summary.md", "actwhiten_cohort_manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("refusing to overwrite existing covariance-power report artifacts")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / names[0], index=False, mode="x")
    display_columns = [column for column in table.columns if column != ARCHIVED_COLUMN]
    report = (
        "# Covariance-power update comparison (matched 128-cell cohort)\n\n"
        + table[display_columns].to_string(index=False, float_format=lambda value: f"{value:.1f}")
        + "\n\nEach displayed mean and contrast is rounded once from raw precision. "
        "Use the CSV for calculations. The power-half update does not transform forward activities. "
        "Archived K-nDFA values are retained only in the CSV and are excluded from paper claims.\n"
    )
    with (output / names[1]).open("x") as stream:
        stream.write(report)
    with (output / names[2]).open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="fresh report directory outside the retained source cohort")
    args = parser.parse_args()
    table, manifest = build_summary(args.input_root)
    write_report(table, manifest, args.output_dir)
    print(table.drop(columns=ARCHIVED_COLUMN).to_string(index=False, float_format=lambda value: f"{value:.1f}"))


if __name__ == "__main__":
    main()
