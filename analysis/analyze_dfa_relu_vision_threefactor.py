"""Analyze the frozen P7 ReLU/softmax three-factor confirmation."""

from __future__ import annotations

import math
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEV = ROOT / "results" / "dfa_relu_mnist_dev_v1"
CONFIRM = ROOT / "results" / "dfa_relu_mnist_threefactor_confirmation_v1"
OUT = ROOT / "results" / "dfa_relu_mnist_threefactor_analysis_v1"
PAPER_FIGURES = ROOT / "drafts" / "Info-DFA" / "figures"
FIGURE_STEM = "iclr_fig_relu_threefactor_conditioning"

METHOD_ORDER = ["dfa", "ndfa", "endfa", "kndfa"]
METHOD_LABEL = {"dfa": "DFA", "ndfa": "activity", "endfa": "error", "kndfa": "both"}
COLORS = {"dfa": "#7F7F7F", "ndfa": "#009E73", "endfa": "#D55E00", "kndfa": "#6A3D9A"}
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
        raise ValueError(f"cannot parse damping from {path}")
    return float(match.group(1).replace("p", "."))


def load_development() -> tuple[pd.DataFrame, float, float]:
    frames = []
    for side, method in (("activity", "ndfa"), ("error", "endfa")):
        for directory in sorted(DEV.glob(f"{side}_d*")):
            path = directory / "relu_vision_endpoints.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame = frame[frame["method"].eq(method)].copy()
            frame["side"] = side
            frame["damping"] = damping_from_dir(directory)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no development runs under {DEV}")
    runs = pd.concat(frames, ignore_index=True)
    selection = runs.groupby(["side", "method", "damping"], as_index=False).agg(
        validation_acc_mean=("val_acc", "mean"),
        validation_acc_sem=("val_acc", sem),
        validation_loss_mean=("val_loss", "mean"),
        n=("seed", "nunique"),
    )
    best = selection.sort_values(
        ["side", "validation_acc_mean", "validation_loss_mean"],
        ascending=[True, False, True],
    ).groupby("side", as_index=False).head(1)
    lambda_a = float(best.loc[best["side"].eq("activity"), "damping"].iloc[0])
    lambda_e = float(best.loc[best["side"].eq("error"), "damping"].iloc[0])
    if not math.isclose(lambda_a, 3.0) or not math.isclose(lambda_e, 0.1):
        raise ValueError(f"frozen damping mismatch: selected {lambda_a:g}/{lambda_e:g}, expected 3/0.1")
    return selection, lambda_a, lambda_e


def load_confirmation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_paths = sorted(CONFIRM.glob("fb*/relu_vision_results.csv"))
    endpoint_paths = sorted(CONFIRM.glob("fb*/relu_vision_endpoints.csv"))
    if len(result_paths) != 3 or len(endpoint_paths) != 3:
        raise FileNotFoundError("expected three feedback-seed confirmation shards")
    curves = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    endpoints = pd.concat([pd.read_csv(path) for path in endpoint_paths], ignore_index=True)
    if set(endpoints["method"]) != set(METHOD_ORDER):
        raise ValueError("confirmation method set is incomplete")
    if endpoints["test_acc"].isna().any() or endpoints["test_loss"].isna().any():
        raise ValueError("confirmation is missing final test endpoints")
    seed_means = endpoints.groupby(["seed", "method"], as_index=False).agg(
        test_acc=("test_acc", "mean"),
        test_loss=("test_loss", "mean"),
        validation_acc=("val_acc", "mean"),
        validation_loss=("val_loss", "mean"),
        n_feedback_seeds=("feedback_seed", "nunique"),
    )
    if seed_means["seed"].nunique() != 8 or not seed_means["n_feedback_seeds"].eq(3).all():
        raise ValueError("expected eight model seeds crossed with three feedback seeds")
    return curves, endpoints, seed_means


def paired_contrasts(seed_means: pd.DataFrame) -> pd.DataFrame:
    wide = seed_means.pivot(index="seed", columns="method")
    rows = []
    for label, method, reference in CONTRASTS:
        for metric, scale in (("test_acc", 100.0), ("test_loss", 1.0)):
            delta = (wide[metric][method] - wide[metric][reference]) * scale
            rows.append(
                {
                    "contrast": label,
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "n_seeds": delta.size,
                    "mean_delta": delta.mean(),
                    "sem_delta": sem(delta),
                    "min_delta": delta.min(),
                    "max_delta": delta.max(),
                    "wins": int((delta > 0).sum()),
                    "t_p": stats.ttest_1samp(delta, 0.0).pvalue,
                    "wilcoxon_p": stats.wilcoxon(delta).pvalue,
                }
            )
    return pd.DataFrame(rows)


def make_figure(selection: pd.DataFrame, seed_means: pd.DataFrame, lambda_a: float, lambda_e: float) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), constrained_layout=True)
    for side, method, ax, letter in (
        ("activity", "ndfa", axes[0], "A"),
        ("error", "endfa", axes[1], "B"),
    ):
        sub = selection[selection["side"].eq(side)].sort_values("damping")
        ax.errorbar(
            sub["damping"],
            100 * sub["validation_acc_mean"],
            yerr=100 * sub["validation_acc_sem"],
            color=COLORS[method],
            marker="o",
            ms=3.5,
            lw=1.5,
        )
        chosen = lambda_a if side == "activity" else lambda_e
        ax.axvline(chosen, color="0.25", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(rf"{side} damping $\lambda_{{{'A' if side == 'activity' else 'E'}}}$")
        ax.set_ylabel("validation accuracy (%)" if side == "activity" else "")
        ax.set_title(f"{letter}  {side} selection", loc="left", fontweight="bold")
        ax.grid(alpha=0.2, lw=0.5)

    wide = seed_means.pivot(index="seed", columns="method", values="test_acc") * 100
    xs = np.arange(len(METHOD_ORDER))
    for _, row in wide[METHOD_ORDER].iterrows():
        axes[2].plot(xs, row, color="0.78", lw=0.75, marker="o", ms=2.5)
    means, errors = wide[METHOD_ORDER].mean(), wide[METHOD_ORDER].sem()
    axes[2].errorbar(xs, means, yerr=errors, fmt="none", ecolor="0.15", capsize=2, lw=1.1)
    axes[2].scatter(xs, means, c=[COLORS[m] for m in METHOD_ORDER], s=34, edgecolor="white", lw=0.6, zorder=4)
    axes[2].set_xticks(xs, [METHOD_LABEL[m] for m in METHOD_ORDER], rotation=18)
    axes[2].set_ylabel("test accuracy (%)")
    axes[2].set_title("C  frozen confirmation", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.2, lw=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUT, PAPER_FIGURES):
        for extension in ("pdf", "png", "svg"):
            fig.savefig(directory / f"{FIGURE_STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    selection: pd.DataFrame,
    curves: pd.DataFrame,
    endpoints: pd.DataFrame,
    seed_means: pd.DataFrame,
    contrasts: pd.DataFrame,
    lambda_a: float,
    lambda_e: float,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    method_summary = seed_means.groupby("method", as_index=False).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_sem=("test_acc", sem),
        test_loss_mean=("test_loss", "mean"),
        test_loss_sem=("test_loss", sem),
        n_seeds=("seed", "nunique"),
    )
    method_summary["method"] = pd.Categorical(method_summary["method"], METHOD_ORDER, ordered=True)
    method_summary = method_summary.sort_values("method")
    selection.to_csv(OUT / "damping_selection.csv", index=False)
    curves.to_csv(OUT / "confirmation_curves.csv", index=False)
    endpoints.to_csv(OUT / "confirmation_runs.csv", index=False)
    seed_means.to_csv(OUT / "confirmation_seed_means.csv", index=False)
    method_summary.to_csv(OUT / "confirmation_method_summary.csv", index=False)
    contrasts.to_csv(OUT / "confirmation_contrasts.csv", index=False)

    error_acc = contrasts.query("contrast == 'error over raw' and metric == 'test_acc'").iloc[0]
    error_loss = contrasts.query("contrast == 'error over raw' and metric == 'test_loss'").iloc[0]
    both_acc = contrasts.query("contrast == 'error after activity' and metric == 'test_acc'").iloc[0]
    both_loss = contrasts.query("contrast == 'error after activity' and metric == 'test_loss'").iloc[0]
    p7a = error_acc["mean_delta"] > 0 and error_acc["wins"] >= 6
    p7b = both_acc["mean_delta"] > 0 and both_acc["wins"] >= 6 and both_loss["mean_delta"] < 0
    lines = [
        "# P7 ReLU/softmax three-factor confirmation",
        "",
        f"Development selected lambda_A={lambda_a:g} and lambda_E={lambda_e:g} independently; K-nDFA adds no joint tuning.",
        "Confirmation uses eight fresh model/data-order seeds crossed with three feedback seeds. Feedback seeds are averaged within model seed before paired tests; the test set is evaluated only at the final update.",
        "",
        "## Confirmation endpoints",
        "",
        method_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired contrasts",
        "",
        contrasts.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## P7 scorecard",
        "",
        f"- **P7a {'CONFIRMED' if p7a else 'REFUTED'}:** error nDFA minus DFA is {error_acc['mean_delta']:+.3f} ± {error_acc['sem_delta']:.3f} pp, positive in {int(error_acc['wins'])}/8 seeds; test-loss difference {error_loss['mean_delta']:+.4f}.",
        f"- **P7b {'CONFIRMED' if p7b else 'REFUTED'}:** K-nDFA minus activity nDFA is {both_acc['mean_delta']:+.3f} ± {both_acc['sem_delta']:.3f} pp, positive in {int(both_acc['wins'])}/8 seeds; test-loss difference {both_loss['mean_delta']:+.4f}.",
        f"- **P7c {'CONFIRMED' if p7a and p7b else 'REFUTED'}:** both signs replicate under a two-hidden-layer ReLU MLP with softmax cross-entropy.",
        "",
        "With eight nonzero same-sign pairs, the exact two-sided Wilcoxon p-value is 0.0078125.",
    ]
    (OUT / "threefactor_summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "threefactor_summary.md").read_text())


def sem(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / np.sqrt(array.size))


def main() -> None:
    selection, lambda_a, lambda_e = load_development()
    curves, endpoints, seed_means = load_confirmation()
    contrasts = paired_contrasts(seed_means)
    make_figure(selection, seed_means, lambda_a, lambda_e)
    write_outputs(selection, curves, endpoints, seed_means, contrasts, lambda_a, lambda_e)


if __name__ == "__main__":
    main()
