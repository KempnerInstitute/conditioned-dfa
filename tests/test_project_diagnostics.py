"""Smoke test for the Info-DFA project diagnostic."""

import numpy as np

from infogeo.project_diagnostics import (
    InfoDfaDiagnosticConfig,
    run_infodfa_diagnostic,
    summarize_infodfa_diagnostic,
)


def test_infodfa_project_diagnostic_smoke():
    cfg = InfoDfaDiagnosticConfig(
        n_train=96,
        n_test=80,
        hidden_dim=12,
        epochs=1,
        batch_size=32,
        eval_size=32,
        n_seeds=1,
        methods=("bp", "dfa_random", "ndfa_random_kronecker"),
    )
    df = run_infodfa_diagnostic(cfg)
    assert set(df["method"]) == {"bp", "dfa_random", "ndfa_random_kronecker"}
    assert set(df["epoch"]) == {0, 1}
    assert np.all(np.isfinite(df["hidden_projected_step"]))
    assert np.all(np.isfinite(df["fisher_trace"]))

    bp0 = df[(df["method"] == "bp") & (df["epoch"] == 0)].iloc[0]
    assert np.isclose(bp0["hidden_projected_step"], 1.0)
    assert bp0["test_acc"] >= 0.0


def test_infodfa_summary_extracts_final_by_method():
    cfg = InfoDfaDiagnosticConfig(
        n_train=64,
        n_test=40,
        hidden_dim=8,
        epochs=1,
        eval_size=16,
        n_seeds=1,
        methods=("bp", "dfa_random"),
    )
    df = run_infodfa_diagnostic(cfg)
    summary = summarize_infodfa_diagnostic(df)
    assert "method" in summary.columns
    assert set(summary["method"]) == {"bp", "dfa_random"}
