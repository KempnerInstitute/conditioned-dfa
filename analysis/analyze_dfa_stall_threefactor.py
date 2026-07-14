"""Analyze the validation-selected DFA-Stall three-factor experiment.

The development protocol selects activity and error damping independently:

* activity nDFA selects ``lambda_A`` from activity-only validation runs;
* error nDFA selects ``lambda_E`` from error-only validation runs;
* K-nDFA combines those two single-factor choices without extra joint tuning.

Confirmation averages three feedback seeds within each of five fresh model/data-
order seeds before paired comparisons.  The MNIST test set is evaluated only at
the final training step; damping selection uses a fixed 5,000-example split from
the original training set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("INFODFA_RESULTS", ROOT / "results")).resolve()
DEV = RESULTS / "dfa_stall_threefactor_dev_v1"
CONFIRM = RESULTS / "dfa_stall_threefactor_confirmation_v1"
OUT = RESULTS / "dfa_stall_threefactor_analysis_v1"
PAPER_FIGURES = ROOT / "drafts" / "Info-DFA" / "figures"
DATASET_LABEL = "MNIST"
FIGURE_STEM = "iclr_fig_threefactor_conditioning"

METHOD_ORDER = ["dfa", "ndfa", "endfa", "kndfa"]
METHOD_LABEL = {
    "dfa": "DFA",
    "ndfa": "activity nDFA",
    "endfa": "error nDFA",
    "kndfa": "K-nDFA (both)",
}
METHOD_TICK = {
    "dfa": "DFA",
    "ndfa": "activity",
    "endfa": "error",
    "kndfa": "both",
}
COLORS = {
    "dfa": "#7F7F7F",
    "ndfa": "#009E73",
    "endfa": "#D55E00",
    "kndfa": "#6A3D9A",
}
CONTRASTS = [
    ("activity over raw", "ndfa", "dfa"),
    ("error over raw", "endfa", "dfa"),
    ("error after activity", "kndfa", "ndfa"),
    ("activity after error", "kndfa", "endfa"),
    ("both over raw", "kndfa", "dfa"),
]


def damping_from_dir(path: Path) -> float:
    match = re.search(r"_d([0-9]+(?:p[0-9]+)?)$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse damping from {path}")
    return float(match.group(1).replace("p", "."))


def load_development() -> tuple[pd.DataFrame, float, float]:
    frames: list[pd.DataFrame] = []
    for side, pattern, method in (
        ("activity", "activity_d*", "ndfa"),
        ("error", "error_d*", "endfa"),
    ):
        for directory in sorted(DEV.glob(pattern)):
            path = directory / "dfa_stall_comparison_summary.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame = frame[frame["method"].eq(method)].copy()
            frame["side"] = side
            frame["damping"] = damping_from_dir(directory)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No development summaries found under {DEV}")
    runs = pd.concat(frames, ignore_index=True)
    selection = (
        runs.groupby(["side", "method", "damping"], as_index=False)
        .agg(
            validation_acc_mean=("final_probe_acc", "mean"),
            validation_acc_sem=("final_probe_acc", lambda x: float(stats.sem(x))),
            validation_loss_mean=("final_probe_loss", "mean"),
            n=("seed", "nunique"),
        )
        .sort_values(["side", "damping"])
    )
    best = selection.sort_values(
        ["side", "validation_acc_mean", "validation_loss_mean"],
        ascending=[True, False, True],
    ).groupby("side", as_index=False).head(1)
    lambda_a = float(best.loc[best["side"].eq("activity"), "damping"].iloc[0])
    lambda_e = float(best.loc[best["side"].eq("error"), "damping"].iloc[0])
    return selection, lambda_a, lambda_e


def load_confirmation() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(CONFIRM.glob("fb*/dfa_stall_comparison_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No confirmation summaries found under {CONFIRM}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = set(METHOD_ORDER)
    if set(runs["method"].unique()) != expected:
        raise ValueError(f"Expected methods {expected}, found {set(runs['method'].unique())}")
    seed_means = (
        runs.groupby(["seed", "method"], as_index=False)
        .agg(
            test_acc=("final_test_acc", "mean"),
            test_loss=("final_test_loss", "mean"),
            validation_acc=("final_probe_acc", "mean"),
            grad_alignment=("mean_grad_alignment", "mean"),
            n_feedback_seeds=("feedback_seed", "nunique"),
        )
        .sort_values(["seed", "method"])
    )
    return runs, seed_means


def paired_contrasts(seed_means: pd.DataFrame) -> pd.DataFrame:
    wide = seed_means.pivot(index="seed", columns="method")
    rows: list[dict[str, float | int | str]] = []
    for label, method, reference in CONTRASTS:
        for metric, scale in (("test_acc", 100.0), ("test_loss", 1.0)):
            delta = (wide[metric][method] - wide[metric][reference]) * scale
            t = stats.ttest_1samp(delta, 0.0)
            w = stats.wilcoxon(delta)
            rows.append(
                {
                    "contrast": label,
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "n_seeds": int(delta.size),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "max_delta": float(delta.max()),
                    "wins": int((delta > 0).sum()),
                    "t_p": float(t.pvalue),
                    "wilcoxon_p": float(w.pvalue),
                }
            )
    return pd.DataFrame(rows)


def make_figure(selection: pd.DataFrame, seed_means: pd.DataFrame, lambda_a: float, lambda_e: float) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), constrained_layout=True)

    for side, method, ax in (("activity", "ndfa", axes[0]), ("error", "endfa", axes[1])):
        sub = selection[selection["side"].eq(side)].sort_values("damping")
        ax.errorbar(
            sub["damping"],
            100.0 * sub["validation_acc_mean"],
            yerr=100.0 * sub["validation_acc_sem"],
            color=COLORS[method],
            marker="o",
            lw=1.6,
            ms=3.5,
        )
        chosen = lambda_a if side == "activity" else lambda_e
        ax.axvline(chosen, color="0.25", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(rf"{side} damping $\lambda_{{{'A' if side == 'activity' else 'E'}}}$")
        ax.set_ylabel("validation accuracy (%)" if side == "activity" else "")
        ax.set_title(f"{'A' if side == 'activity' else 'B'}  {METHOD_LABEL[method]}", loc="left", fontweight="bold")
        ax.grid(alpha=0.2, lw=0.5)

    ax = axes[2]
    wide = seed_means.pivot(index="seed", columns="method", values="test_acc") * 100.0
    xs = np.arange(len(METHOD_ORDER))
    for _, row in wide[METHOD_ORDER].iterrows():
        ax.plot(xs, row.to_numpy(), color="0.78", lw=0.8, marker="o", ms=2.5, zorder=1)
    means = wide[METHOD_ORDER].mean(axis=0)
    sems = wide[METHOD_ORDER].sem(axis=0)
    ax.errorbar(
        xs,
        means.to_numpy(),
        yerr=sems.to_numpy(),
        fmt="none",
        ecolor="0.15",
        capsize=2,
        lw=1.1,
        zorder=3,
    )
    ax.scatter(xs, means.to_numpy(), c=[COLORS[m] for m in METHOD_ORDER], s=34, edgecolor="white", lw=0.6, zorder=4)
    ax.set_xticks(xs, [METHOD_TICK[m] for m in METHOD_ORDER], rotation=18)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("C  frozen confirmation", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, lw=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUT, PAPER_FIGURES):
        for ext in ("pdf", "png", "svg"):
            fig.savefig(directory / f"{FIGURE_STEM}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    selection: pd.DataFrame,
    seed_means: pd.DataFrame,
    contrasts: pd.DataFrame,
    lambda_a: float,
    lambda_e: float,
) -> None:
    method_summary = (
        seed_means.groupby("method", as_index=False)
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_sem=("test_acc", lambda x: float(stats.sem(x))),
            test_loss_mean=("test_loss", "mean"),
            test_loss_sem=("test_loss", lambda x: float(stats.sem(x))),
            n_seeds=("seed", "nunique"),
        )
    )
    method_summary["method"] = pd.Categorical(method_summary["method"], METHOD_ORDER, ordered=True)
    method_summary = method_summary.sort_values("method")
    lines = [
        f"# {DATASET_LABEL} DFA-Stall three-factor confirmation",
        "",
        f"Development selected activity damping lambda_A={lambda_a:g} and error damping lambda_E={lambda_e:g}.",
        "K-nDFA combines those independently selected single-factor values without extra joint tuning.",
        "Confirmation uses five fresh model/data-order seeds and three feedback seeds; feedback seeds are",
        "averaged within each model seed before paired comparisons. Damping selection uses a fixed",
        f"5,000-example training-validation split; the {DATASET_LABEL} test set is evaluated only at the final step.",
        "All conditioned hidden gradients are norm-matched to the raw-DFA layer norm.",
        "",
        "## Confirmation endpoints",
        "",
        method_summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Seed-level paired contrasts",
        "",
        contrasts.to_markdown(index=False, floatfmt=".5g"),
        "",
        "With five seed-level pairs, the smallest attainable two-sided Wilcoxon p-value is 0.0625.",
    ]
    (OUT / "threefactor_summary.md").write_text("\n".join(lines) + "\n")
    method_summary.to_csv(OUT / "confirmation_method_summary.csv", index=False)


def main() -> None:
    selection, lambda_a, lambda_e = load_development()
    runs, seed_means = load_confirmation()
    contrasts = paired_contrasts(seed_means)
    OUT.mkdir(parents=True, exist_ok=True)
    selection.to_csv(OUT / "damping_selection.csv", index=False)
    runs.to_csv(OUT / "confirmation_runs.csv", index=False)
    seed_means.to_csv(OUT / "confirmation_seed_means.csv", index=False)
    contrasts.to_csv(OUT / "confirmation_contrasts.csv", index=False)
    make_figure(selection, seed_means, lambda_a, lambda_e)
    write_report(selection, seed_means, contrasts, lambda_a, lambda_e)
    print((OUT / "threefactor_summary.md").read_text())


if __name__ == "__main__":
    main()
