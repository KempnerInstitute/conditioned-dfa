"""Protect the main figure from later additions to the appendix cohorts."""

import numpy as np
import pandas as pd
import pytest

from analysis import make_error_kndfa_replication_figure as figure


@pytest.fixture
def figure_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(figure, "RESULTS", tmp_path)
    for label, seeds in figure.COHORTS.items():
        directory = tmp_path / figure.ANALYSIS_DIRS[label]
        directory.mkdir()
        rows = []
        for seed in (*seeds, max(seeds) + 1):
            for method_index, method in enumerate(figure.COLORS):
                for feedback_seed in figure.FEEDBACK_SEEDS:
                    rows.append({
                        "seed": seed, "method": method, "feedback_seed": feedback_seed,
                        "test_acc": .3 + .001 * seed + .01 * method_index + .001 * feedback_seed,
                        "test_loss": 1 + .001 * seed + .01 * method_index + .001 * feedback_seed,
                    })
        runs = pd.DataFrame(rows)
        means = runs.groupby(["seed", "method"], as_index=False).agg(
            test_acc=("test_acc", "mean"), test_loss=("test_loss", "mean"),
            n_feedback_seeds=("feedback_seed", "nunique"),
        )
        means.to_csv(directory / "confirmation_seed_means.csv", index=False)
        saved = runs if label == "ReLU MNIST" else runs.rename(
            columns={"test_acc": "final_test_acc", "test_loss": "final_test_loss"}
        )
        saved.to_csv(directory / "confirmation_runs.csv", index=False)
        if label == "ReLU MNIST":
            curves = pd.concat([
                runs.assign(step=step, val_acc=runs.test_acc + step / 1000)
                for step in (1, 100)
            ], ignore_index=True)
            curves.to_csv(directory / "confirmation_curves.csv", index=False)
    return tmp_path


def test_later_seeds_do_not_change_original_endpoints_or_curves(figure_sources):
    before = figure.load_seed_means()
    curves_before = figure.load_original_curves()
    for label, seeds in figure.COHORTS.items():
        assert set(before[label].seed) == set(seeds)
        directory = figure_sources / figure.ANALYSIS_DIRS[label]
        for path in directory.glob("*.csv"):
            data = pd.read_csv(path)
            later = ~data.seed.isin(seeds)
            for column in ("test_acc", "test_loss", "final_test_acc", "final_test_loss", "val_acc"):
                if column in data:
                    data.loc[later, column] = np.nan
            data.to_csv(path, index=False)
    after = figure.load_seed_means()
    for label in before:
        pd.testing.assert_frame_equal(before[label], after[label])
    pd.testing.assert_frame_equal(curves_before, figure.load_original_curves())
    assert set(curves_before.seed) == set(range(100, 108))


@pytest.mark.parametrize("filename", ["confirmation_runs.csv", "confirmation_curves.csv"])
def test_missing_feedback_observation_is_rejected(figure_sources, filename):
    path = figure_sources / figure.ANALYSIS_DIRS["ReLU MNIST"] / filename
    data = pd.read_csv(path)
    data.iloc[1:].to_csv(path, index=False)
    loader = figure.load_seed_means if filename == "confirmation_runs.csv" else figure.load_original_curves
    with pytest.raises(ValueError, match="original"):
        loader()


def test_duplicate_seed_method_summary_is_rejected(figure_sources):
    path = figure_sources / figure.ANALYSIS_DIRS["tanh MNIST"] / "confirmation_seed_means.csv"
    data = pd.read_csv(path)
    pd.concat([data, data.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicated original"):
        figure.load_seed_means()


def test_changed_original_aggregate_is_rejected(figure_sources):
    path = figure_sources / figure.ANALYSIS_DIRS["tanh Fashion"] / "confirmation_seed_means.csv"
    data = pd.read_csv(path)
    data.loc[0, "test_acc"] += .01
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        figure.load_seed_means()
