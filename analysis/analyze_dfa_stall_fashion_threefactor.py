"""Analyze the preregistered clean Fashion-MNIST three-factor replication."""

from __future__ import annotations

import analyze_dfa_stall_threefactor as analysis
import pandas as pd


analysis.DEV = analysis.RESULTS / "dfa_stall_fashion_threefactor_dev_v1"
analysis.CONFIRM = analysis.RESULTS / "dfa_stall_fashion_threefactor_confirmation_v1"
analysis.OUT = analysis.RESULTS / "dfa_stall_fashion_threefactor_analysis_v1"
analysis.DATASET_LABEL = "Fashion-MNIST"
analysis.FIGURE_STEM = "iclr_fig_fashion_threefactor_conditioning"
analysis.METHOD_ORDER = ["dfa", "ndfa", "endfa", "kndfa", "kndfa_bp"]
analysis.METHOD_LABEL["kndfa_bp"] = "K-nDFA (BP-error source)"
analysis.METHOD_TICK["kndfa_bp"] = "BP src."
analysis.COLORS["kndfa_bp"] = "#CC79A7"
analysis.CONTRASTS = [
    ("activity over raw", "ndfa", "dfa"),
    ("error over raw", "endfa", "dfa"),
    ("error after activity", "kndfa", "ndfa"),
    ("activity after error", "kndfa", "endfa"),
    ("both over raw", "kndfa", "dfa"),
    ("local versus BP-error source", "kndfa", "kndfa_bp"),
]


def score_preregistration() -> None:
    contrasts = pd.read_csv(analysis.OUT / "confirmation_contrasts.csv")

    def row(label: str, metric: str) -> pd.Series:
        match = contrasts[contrasts["contrast"].eq(label) & contrasts["metric"].eq(metric)]
        if len(match) != 1:
            raise ValueError(f"Expected one row for {label}/{metric}, found {len(match)}")
        return match.iloc[0]

    error_acc = row("error over raw", "test_acc")
    both_acc = row("error after activity", "test_acc")
    both_loss = row("error after activity", "test_loss")
    source_acc = row("local versus BP-error source", "test_acc")
    p6a = error_acc["mean_delta"] > 0 and int(error_acc["wins"]) >= 4
    p6b = both_acc["mean_delta"] > 0
    p6c = abs(source_acc["mean_delta"]) <= 0.5 and int(source_acc["wins"]) > 0
    lines = [
        "# P6 preregistration scorecard",
        "",
        f"- **P6a {'CONFIRMED' if p6a else 'REFUTED'}:** error nDFA minus raw DFA "
        f"= {error_acc['mean_delta']:+.3f} pp; positive in {int(error_acc['wins'])}/5 model seeds.",
        f"- **P6b {'CONFIRMED' if p6b else 'REFUTED'}:** K-nDFA minus activity nDFA "
        f"= {both_acc['mean_delta']:+.3f} pp accuracy and {both_loss['mean_delta']:+.4f} test loss.",
        f"- **P6c {'CONFIRMED' if p6c else 'REFUTED'}:** local K-nDFA minus the BP-error-source "
        f"control = {source_acc['mean_delta']:+.3f} pp; local wins in {int(source_acc['wins'])}/5 model seeds.",
        "",
        "The score uses the criteria recorded in PREDICTIONS.md before the development sweep.",
    ]
    (analysis.OUT / "preregistration_scorecard.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    analysis.main()
    score_preregistration()
    print((analysis.OUT / "preregistration_scorecard.md").read_text())
