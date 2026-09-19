"""Reject cohort drift and double rounding in the matched covariance-power table."""

import csv
import hashlib
import itertools
import json
import sys

import pandas as pd
import pytest

from analysis import aggregate_actwhiten as power
from analysis import aggregate_bpwhiten as bp


# Non-round numbers deliberately reproduce the old double-rounding boundaries.
BASE_PERCENT = {
    "nuisance_dominant": (13.519500732421875, 47.549652099609375, 53.30816650390625, 56.81),
    "low_sample_noisy": (34.674224853515625, 61.994110107421875, 65.71841430664062, 65.91),
    "mixed_context": (25.3846435546875, 46.71588134765625, 47.482513427734375, 50.09),
    "task_aligned": (74.9639892578125, 91.0450439453125, 90.39614868164062, 83.26),
}


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path / "matched"
    # Independently enumerate the launch grid; do not use expected_cells().
    for regime, n_train, label, noise in itertools.product(
        BASE_PERCENT, (512, 1024, 2048, 4096), (0.0, 0.1, 0.2, 0.4), (0.05, 0.15)
    ):
        path = (root / regime / f"ntrain_{n_train}" /
                f"label_{str(label).replace('.', 'p')}" /
                f"input_{str(noise).replace('.', 'p')}" / "dfa_multioutput_results.csv")
        path.parent.mkdir(parents=True)
        rows = []
        for method_index, method in enumerate(
            ("dfa_random", "dfa_actwhiten", "ndfa_random", "ndfa_random_kronecker")
        ):
            for seed, feedback in itertools.product(range(5), range(5)):
                rows.append({
                    "condition": regime, "method": method, "seed": seed,
                    "feedback_seed": feedback, "feedback_rank": 0, "epoch": 14,
                    "n_train": n_train, "train_label_noise": label, "input_noise": noise,
                    "test_acc": BASE_PERCENT[regime][method_index] / 100
                    + (seed - 2) * .0001 + (feedback - 2) * .00001,
                })
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return root


def test_complete_matched_cohort_raw_precision_and_independent_source_digest(cohort, tmp_path):
    table, manifest = power.build_summary(cohort)
    indexed = table.set_index("regime")
    for regime, bases in BASE_PERCENT.items():
        for column, expected in zip(
            ("dfa_random", "dfa_actwhiten", "ndfa_random", power.ARCHIVED_COLUMN), bases
        ):
            assert indexed.loc[regime, column] == pytest.approx(expected, abs=2e-14)
        assert indexed.loc[regime, "ndfa_minus_power_half_pp"] == pytest.approx(
            bases[2] - bases[1], abs=2e-14
        )
    assert manifest["source_file_count"] == 128
    assert manifest["total_final_endpoints"] == 12800
    assert manifest["endpoints_per_regime_method"] == 800
    digest = hashlib.sha256()
    for path in sorted(cohort.rglob("dfa_multioutput_results.csv")):
        digest.update(str(path.relative_to(cohort)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert manifest["cohort_csv_sha256"] == digest.hexdigest()

    # An unusually good intermediate checkpoint must not select a new endpoint.
    path = next(cohort.rglob("dfa_multioutput_results.csv"))
    data = pd.read_csv(path, float_precision="round_trip")
    earlier = data.assign(epoch=0, test_acc=.999)
    pd.concat([earlier, data]).to_csv(path, index=False)
    unchanged, changed_manifest = power.build_summary(cohort)
    pd.testing.assert_frame_equal(table, unchanged)
    assert manifest["cohort_csv_sha256"] != changed_manifest["cohort_csv_sha256"]

    output = tmp_path / "report"
    power.write_report(table, manifest, output)
    restored = pd.read_csv(output / "actwhiten_summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(table, restored, check_exact=True)
    assert f"{indexed.loc['nuisance_dominant', 'dfa_actwhiten']:.1f}" == "47.5"
    assert f"{indexed.loc['task_aligned', 'dfa_actwhiten']:.1f}" == "91.0"
    assert f"{indexed.loc['task_aligned', 'ndfa_minus_power_half_pp']:.1f}" == "-0.6"
    report = (output / "actwhiten_summary.md").read_text()
    assert "-0.6" in report
    assert "excluded from paper claims" in report
    assert power.ARCHIVED_COLUMN not in report
    assert power.ARCHIVED_COLUMN in restored
    assert json.loads((output / "actwhiten_cohort_manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError, match="overwrite"):
        power.write_report(table, manifest, output)
    with pytest.raises(ValueError, match="outside"):
        power.write_report(table, manifest, cohort)


@pytest.mark.parametrize("change, match", [
    ("missing_file", "128-cell"), ("extra_file", "128-cell"),
    ("missing_endpoint", "missing final"), ("duplicate_endpoint", "duplicate final"),
    ("feedback_rank", "feedback_rank"), ("seed", "wrong seed"),
    ("feedback_seed", "feedback_seed"), ("old_epoch", "final epoch"),
    ("new_epoch", "final epoch"), ("condition", "condition"),
    ("input_noise", "input_noise"), ("test_acc", "accuracy"),
])
def test_invalid_or_drifted_cohort_is_rejected(cohort, change, match):
    path = sorted(cohort.rglob("dfa_multioutput_results.csv"))[0]
    if change == "missing_file":
        path.unlink()
    elif change == "extra_file":
        extra = cohort / "unexpected" / path.name
        extra.parent.mkdir()
        extra.write_bytes(path.read_bytes())
    else:
        data = pd.read_csv(path, float_precision="round_trip")
        if change == "missing_endpoint":
            data = data.iloc[1:]
        elif change == "duplicate_endpoint":
            data = pd.concat([data, data.iloc[:1]])
        elif change in ("old_epoch", "new_epoch"):
            data["epoch"] = 13 if change == "old_epoch" else 15
        else:
            bad = {"feedback_rank": 8, "seed": 5, "feedback_seed": 5,
                   "condition": "other", "input_noise": .99, "test_acc": float("inf")}
            data.loc[0, change] = bad[change]
        data.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        power.build_summary(cohort)


def test_bp_context_uses_the_same_raw_cohort_and_retains_bp_selection(cohort, tmp_path, monkeypatch):
    bp_root = tmp_path / "bp"
    for regime in BASE_PERCENT:
        for lr in (.02, .04, .08, .16, .32):
            path = bp_root / regime / f"lr_{str(lr).replace('.', 'p')}" / "dfa_multioutput_results.csv"
            path.parent.mkdir(parents=True)
            # Best final value has the historical expected BP contrast. An
            # earlier high value must not influence the chosen learning rate.
            best = bp.TUNED_BP_PERCENT[regime] + bp.EXPECTED_DELTAS[regime]
            final = best if lr == .32 else best - 1
            pd.DataFrame([
                {"condition": regime, "seed": seed, "epoch": epoch,
                 "test_acc": final / 100 if epoch == 14 else .999}
                for seed, epoch in itertools.product(range(5), (0, 14))
            ]).to_csv(path, index=False)

    combined = bp.build_summary(bp_root, cohort).set_index("regime")
    matched, manifest = power.build_summary(cohort)
    matched = matched.set_index("regime")
    for regime in BASE_PERCENT:
        assert combined.loc[regime, "bp"] == bp.TUNED_BP_PERCENT[regime]
        assert combined.loc[regime, "delta"] == pytest.approx(bp.EXPECTED_DELTAS[regime])
        assert combined.loc[regime, "bp_precond_lr"] == .32
        for column, original in (("dfa", "dfa_random"), ("dfa_power_half", "dfa_actwhiten"),
                                 ("ndfa", "ndfa_random"), (power.ARCHIVED_COLUMN, power.ARCHIVED_COLUMN)):
            assert combined.loc[regime, column] == matched.loc[regime, original]
    assert "k_ndfa" not in combined
    assert combined.attrs["cohort_manifest"]["matched_local_rule_cohort"] == manifest
    assert "retained constants" in combined.attrs["cohort_manifest"]["tuned_bp_provenance"]

    # Updating a raw matched measurement must update the context, not a stale
    # hardcoded constant, without altering any BP source or selected BP value.
    path = sorted(cohort.rglob("dfa_multioutput_results.csv"))[0]
    data = pd.read_csv(path, float_precision="round_trip")
    affected_regime = data.loc[0, "condition"]
    data.loc[data.method.eq("ndfa_random"), "test_acc"] += .032
    data.to_csv(path, index=False)
    after = bp.build_summary(bp_root, cohort).set_index("regime")
    assert after.loc[affected_regime, "ndfa"] - combined.loc[affected_regime, "ndfa"] == pytest.approx(.1)
    pd.testing.assert_frame_equal(after[["bp", "bp_precond", "delta", "bp_precond_lr"]],
                                  combined[["bp", "bp_precond", "delta", "bp_precond_lr"]])
    output = tmp_path / "bp_report"
    monkeypatch.setattr(sys, "argv", ["aggregate_bpwhiten", "--bp-root", str(bp_root),
                                     "--power-root", str(cohort), "--output-dir", str(output)])
    bp.main()
    exported = pd.read_csv(output / "bpwhiten_summary.csv", float_precision="round_trip").set_index("regime")
    pd.testing.assert_frame_equal(after, exported, check_exact=True)
    metadata = json.loads((output / "bpwhiten_cohort_manifest.json").read_text())
    assert metadata["archive_exclusion"]["column"] == power.ARCHIVED_COLUMN
    with pytest.raises(FileExistsError, match="overwrite"):
        bp.main()
