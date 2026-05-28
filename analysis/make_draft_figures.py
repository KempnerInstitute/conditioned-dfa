"""Create publication-style figures from experiment result CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "drafts" / "figures"


LABELS = {
    "early_param_cosine": "early parameter cosine",
    "early_activity_cosine": "early activity cosine",
    "early_tangent_cosine": "early tangent cosine",
    "early_task_tangent_cosine": "early task-weighted tangent cosine",
    "delta_class_dprime2_early": "early class d-prime growth",
    "delta_fisher_trace_early": "early Fisher growth",
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_dfa_figure()
    print(f"Saved draft figures to {FIG_DIR}")


def make_dfa_figure() -> None:
    df = pd.read_csv(ROOT / "results" / "dfa_synthetic" / "dfa_synthetic_results.csv")
    run_summary = pd.read_csv(ROOT / "results" / "dfa_synthetic" / "dfa_run_summary.csv")
    scores = pd.read_csv(ROOT / "results" / "dfa_synthetic" / "dfa_predictor_scores.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax = axes[0, 0]
    acc_df = df[["seed", "feedback_seed", "feedback_scale", "method", "epoch", "test_acc"]].drop_duplicates()
    for method, group in acc_df.groupby("method"):
        curve = group.groupby("epoch")["test_acc"].mean()
        ax.plot(curve.index, curve.values, marker="o", linewidth=1.7, label=method.replace("_", " "))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Learning dynamics")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    layer1 = df[(df["layer"] == 1) & (df["method"] != "bp")]
    for method, group in layer1.groupby("method"):
        curve = group.groupby("epoch")["tangent_cosine"].mean()
        ax.plot(curve.index, curve.values, marker="o", linewidth=1.7, label=method.replace("_", " "))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tangent-projected cosine")
    ax.set_title("Alignment in information-bearing geometry")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    dfa_runs = run_summary[run_summary["method"] != "bp"]
    ax.scatter(dfa_runs["early_param_cosine"], dfa_runs["final_test_acc"], s=34, alpha=0.65, label="parameter")
    ax.scatter(dfa_runs["early_tangent_cosine"], dfa_runs["final_test_acc"], s=34, alpha=0.65, label="tangent")
    if "early_task_tangent_cosine" in dfa_runs:
        ax.scatter(
            dfa_runs["early_task_tangent_cosine"],
            dfa_runs["final_test_acc"],
            s=34,
            alpha=0.65,
            label="task tangent",
        )
    ax.set_xlabel("Early alignment")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Early alignment predicts eventual success")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    keep = [
        "early_param_cosine",
        "early_activity_cosine",
        "early_tangent_cosine",
        "early_task_tangent_cosine",
        "delta_class_dprime2_early",
        "delta_fisher_trace_early",
    ]
    score_subset = scores[scores["predictor"].isin(keep)].sort_values("r2")
    ax.barh(
        [LABELS.get(p, p) for p in score_subset["predictor"]],
        score_subset["r2"],
        color="#59a14f",
    )
    ax.set_xlabel("$R^2$ for predicting final accuracy")
    ax.set_title("Predictor comparison")
    ax.set_xlim(0, 1)

    fig.suptitle("DFA synthetic manifold benchmark: alignment, geometry, and learning", fontsize=14)
    fig.savefig(FIG_DIR / "dfa_preliminary_results.png", dpi=220)
    fig.savefig(FIG_DIR / "dfa_preliminary_results.pdf")
    plt.close(fig)


