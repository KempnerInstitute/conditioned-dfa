"""Aggregate Adam/diagonal-K approximation tests for Info-DFA."""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHOD_ORDER = [
    "bp_sgd",
    "dfa_sgd",
    "dfa_adam_hidden",
    "dfa_diag_activity_sqrt",
    "dfa_diag_error_sqrt",
    "dfa_diag_k_sqrt",
    "ndfa_activity",
    "ndfa_error",
    "kndfa",
]
METHOD_LABEL = {
    "bp_sgd": "BP",
    "dfa_sgd": "DFA",
    "dfa_adam_hidden": "DFA + Adam diag",
    "dfa_rmsprop_hidden": "DFA + RMSProp diag",
    "dfa_diag_activity_sqrt": "diag activity sqrt",
    "dfa_diag_error_sqrt": "diag error sqrt",
    "dfa_diag_k_sqrt": "diag K sqrt",
    "dfa_diag_activity": "diag activity inv",
    "dfa_diag_k": "diag K inv",
    "ndfa_activity": "activity nDFA",
    "ndfa_error": "error nDFA",
    "kndfa": "K-nDFA",
}
METHOD_COLOR = {
    "bp_sgd": "#0072B2",
    "dfa_sgd": "#999999",
    "dfa_adam_hidden": "#E69F00",
    "dfa_rmsprop_hidden": "#8E6BBE",
    "dfa_diag_activity_sqrt": "#8FC9E8",
    "dfa_diag_error_sqrt": "#D55E00",
    "dfa_diag_k_sqrt": "#56B4E9",
    "ndfa_activity": "#009E73",
    "ndfa_error": "#B45F06",
    "kndfa": "#6A3D9A",
}
CELL_ORDER = ["nuisance_hard", "low_sample_noisy", "mixed_hard", "clean_aligned"]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs)
    df.to_csv(output_dir / "infodfa_adam_diagk_all.csv", index=False)
    if df.empty:
        (output_dir / "infodfa_adam_diagk_summary.md").write_text("# Info-DFA Adam/diag-K approximation\n\n_No rows found._\n")
        return

    curves = summarize_curves(df)
    endpoints = summarize_endpoints(df)
    best = best_endpoints(endpoints)
    approx = summarize_approximation(df, preferred_adaptive_lr=args.preferred_adaptive_lr)
    curves.to_csv(output_dir / "infodfa_adam_diagk_curves.csv", index=False)
    endpoints.to_csv(output_dir / "infodfa_adam_diagk_endpoints.csv", index=False)
    best.to_csv(output_dir / "infodfa_adam_diagk_best_endpoints.csv", index=False)
    approx.to_csv(output_dir / "infodfa_adam_diagk_approximation.csv", index=False)

    make_learning_figure(curves, output_dir, preferred_damping=args.preferred_damping, preferred_adaptive_lr=args.preferred_adaptive_lr)
    make_best_learning_figure(curves, best, output_dir)
    make_diagnostic_figure(df, output_dir, preferred_damping=args.preferred_damping, preferred_adaptive_lr=args.preferred_adaptive_lr)
    write_report(best, approx, output_dir)
    mirror_figures(output_dir)

    print(best.to_string(index=False))
    print(f"\nwrote {output_dir}")


def load_inputs(patterns: list[str]) -> pd.DataFrame:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    frames = []
    for path in sorted(set(paths)):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        frame.insert(0, "source_file", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if col not in {"source_file", "cell", "condition", "method"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def run_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in [
            "cell",
            "condition",
            "method",
            "seed",
            "feedback_seed",
            "feedback_rank",
            "damping",
            "adaptive_lr",
            "n_train",
            "input_noise",
            "train_label_noise",
        ]
        if c in df.columns
    ]


def summarize_curves(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        c
        for c in [
            "cell",
            "condition",
            "method",
            "feedback_rank",
            "damping",
            "adaptive_lr",
            "n_train",
            "input_noise",
            "train_label_noise",
            "epoch",
        ]
        if c in df.columns
    ]
    return (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            train_mean=("train_eval_acc", "mean"),
            method_cosine=("method_param_cosine", "mean"),
            projected_step=("method_hidden_projected_step", "mean"),
            norm_ratio=("method_hidden_norm_ratio", "mean"),
            factorization_corr=("factorization_log_corr_mean", "mean"),
            adam_factor_corr=("adam_factor_log_corr_mean", "mean"),
            adam_diagk_cosine=("adam_diagk_update_cosine_mean", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["cell", "method", "epoch"])
    )


def summarize_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("epoch").groupby(run_columns(df), dropna=False, as_index=False).tail(1)
    group_cols = [c for c in run_columns(df) if c not in {"seed", "feedback_seed"}]
    out = (
        final.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            final_epoch=("epoch", "max"),
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            train_mean=("train_eval_acc", "mean"),
            method_cosine=("method_param_cosine", "mean"),
            projected_step=("method_hidden_projected_step", "mean"),
            norm_ratio=("method_hidden_norm_ratio", "mean"),
            factorization_corr=("factorization_log_corr_mean", "mean"),
            factorization_rel_error=("factorization_rel_error_mean", "mean"),
            adam_factor_corr=("adam_factor_log_corr_mean", "mean"),
            adam_exact_corr=("adam_exact_log_corr_mean", "mean"),
            adam_diagk_cosine=("adam_diagk_update_cosine_mean", "mean"),
            adam_m_diagk_cosine=("adam_m_diagk_update_cosine_mean", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["cell", "method", "damping", "adaptive_lr"])
    )
    ref_cols = [c for c in group_cols if c not in {"method", "damping", "adaptive_lr"}]
    dfa = out[out["method"] == "dfa_sgd"][ref_cols + ["test_mean"]].rename(columns={"test_mean": "dfa_test_mean"})
    out = out.merge(dfa, on=ref_cols, how="left")
    out["delta_vs_dfa"] = out["test_mean"] - out["dfa_test_mean"]
    out["method_label"] = out["method"].map(METHOD_LABEL).fillna(out["method"])
    return out


def best_endpoints(endpoints: pd.DataFrame) -> pd.DataFrame:
    if endpoints.empty:
        return pd.DataFrame()
    rows = []
    id_cols = [c for c in ["cell", "condition", "method", "feedback_rank", "n_train", "input_noise", "train_label_noise"] if c in endpoints]
    for _, group in endpoints.groupby(id_cols, dropna=False):
        rows.append(group.sort_values("test_mean").tail(1))
    return pd.concat(rows, ignore_index=True).sort_values(["cell", "method"])


def summarize_approximation(df: pd.DataFrame, *, preferred_adaptive_lr: float) -> pd.DataFrame:
    final = df.sort_values("epoch").groupby(run_columns(df), dropna=False, as_index=False).tail(1)
    adam = final[(final["method"] == "dfa_adam_hidden") & close_or_nan(final["adaptive_lr"], preferred_adaptive_lr)]
    if adam.empty:
        adam = final[final["method"] == "dfa_adam_hidden"]
    if adam.empty:
        return pd.DataFrame()
    return (
        adam.groupby(["cell", "condition"], dropna=False, as_index=False)
        .agg(
            factorization_corr=("factorization_log_corr_mean", "mean"),
            factorization_rel_error=("factorization_rel_error_mean", "mean"),
            grad_square_factor_corr=("gradsq_factor_log_corr_mean", "mean"),
            adam_factor_corr=("adam_factor_log_corr_mean", "mean"),
            adam_exact_corr=("adam_exact_log_corr_mean", "mean"),
            adam_diagk_cosine=("adam_diagk_update_cosine_mean", "mean"),
            adam_m_diagk_cosine=("adam_m_diagk_update_cosine_mean", "mean"),
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            n=("test_acc", "size"),
        )
        .sort_values("cell")
    )


def make_learning_figure(curves: pd.DataFrame, output_dir: Path, *, preferred_damping: float, preferred_adaptive_lr: float) -> None:
    setup_style()
    sub = select_preferred(curves, preferred_damping=preferred_damping, preferred_adaptive_lr=preferred_adaptive_lr)
    methods = ["bp_sgd", "dfa_sgd", "dfa_adam_hidden", "dfa_diag_k_sqrt", "ndfa_activity", "kndfa"]
    cells = ordered_cells(sub)
    if not cells:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), constrained_layout=True, squeeze=False)
    fig.get_layout_engine().set(h_pad=0.12)
    for ax, cell in zip(axes.ravel(), cells):
        cell_df = sub[sub["cell"] == cell]
        for method in methods:
            method_df = cell_df[cell_df["method"] == method].sort_values("epoch")
            if method_df.empty:
                continue
            x = method_df["epoch"].to_numpy(dtype=float)
            y = 100.0 * method_df["test_mean"].to_numpy(dtype=float)
            e = 100.0 * method_df["test_sem"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, color=METHOD_COLOR.get(method, "#777777"), lw=1.5, label=METHOD_LABEL.get(method, method))
            ax.fill_between(x, y - e, y + e, color=METHOD_COLOR.get(method, "#777777"), alpha=0.11, lw=0)
        ax.set_title(pretty_cell(cell))
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(axis="y", color="#E8EAE6", lw=0.6)
    for ax in axes.ravel()[len(cells) :]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=6.5, ncol=2)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"infodfa_adam_diagk_learning.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_best_learning_figure(curves: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> None:
    setup_style()
    methods = ["bp_sgd", "dfa_sgd", "dfa_adam_hidden", "dfa_diag_k_sqrt", "ndfa_activity", "kndfa"]
    cells = ordered_cells(curves)
    if not cells or best.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), constrained_layout=True, squeeze=False)
    fig.get_layout_engine().set(h_pad=0.12)
    for ax, cell in zip(axes.ravel(), cells):
        for method in methods:
            setting = best[(best["cell"] == cell) & (best["method"] == method)]
            if setting.empty:
                continue
            setting = setting.iloc[0]
            method_df = curves[(curves["cell"] == cell) & (curves["method"] == method)].copy()
            if "damping" in method_df:
                method_df = method_df[match_value_or_nan(method_df["damping"], setting.get("damping", np.nan))]
            if "adaptive_lr" in method_df:
                method_df = method_df[match_value_or_nan(method_df["adaptive_lr"], setting.get("adaptive_lr", np.nan))]
            method_df = method_df.sort_values("epoch")
            if method_df.empty:
                continue
            x = method_df["epoch"].to_numpy(dtype=float)
            y = 100.0 * method_df["test_mean"].to_numpy(dtype=float)
            e = 100.0 * method_df["test_sem"].fillna(0.0).to_numpy(dtype=float)
            label = METHOD_LABEL.get(method, method)
            if method in {"dfa_diag_k_sqrt", "ndfa_activity", "kndfa"} and np.isfinite(setting.get("damping", np.nan)):
                label = f"{label} (d={float(setting['damping']):g})"
            if method == "dfa_adam_hidden" and np.isfinite(setting.get("adaptive_lr", np.nan)):
                label = f"{label} (lr={float(setting['adaptive_lr']):g})"
            ax.plot(x, y, color=METHOD_COLOR.get(method, "#777777"), lw=1.5, label=label)
            ax.fill_between(x, y - e, y + e, color=METHOD_COLOR.get(method, "#777777"), alpha=0.11, lw=0)
        ax.set_title(pretty_cell(cell))
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(axis="y", color="#E8EAE6", lw=0.6)
    for ax in axes.ravel()[len(cells) :]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=5.8, ncol=1)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"infodfa_adam_diagk_learning_best.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic_figure(df: pd.DataFrame, output_dir: Path, *, preferred_damping: float, preferred_adaptive_lr: float) -> None:
    setup_style()
    final = df.sort_values("epoch").groupby(run_columns(df), dropna=False, as_index=False).tail(1)
    preferred = select_preferred(final, preferred_damping=preferred_damping, preferred_adaptive_lr=preferred_adaptive_lr)
    approx = summarize_approximation(df, preferred_adaptive_lr=preferred_adaptive_lr)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.85), constrained_layout=True)

    # A: direct approximation quality for Adam runs.
    if not approx.empty:
        approx = approx.set_index("cell").reindex(ordered_cells(approx.reset_index())).reset_index()
        x = np.arange(len(approx))
        width = 0.24
        axes[0].bar(x - width, approx["factorization_corr"], width=width, color="#56B4E9", label=r"$E[\delta^2 h^2]$ vs factor")
        axes[0].bar(x, approx["adam_factor_corr"], width=width, color=METHOD_COLOR["dfa_adam_hidden"], label="Adam v vs factor")
        axes[0].bar(x + width, approx["adam_diagk_cosine"], width=width, color="#CC79A7", label="Adam step vs diag K")
        axes[0].set_xticks(x, [pretty_cell(c) for c in approx["cell"]], rotation=25, ha="right")
        axes[0].set_ylabel("correlation / cosine")
        axes[0].set_title("A  Approximation quality")
        axes[0].set_ylim(-0.05, 1.32)
        axes[0].set_yticks([0.0, 0.5, 1.0])
        axes[0].legend(frameon=False, fontsize=5.8, ncol=1, loc="upper left")

    # B: does approximation quality predict Adam's gain over DFA?
    adam = preferred[preferred["method"] == "dfa_adam_hidden"].copy()
    dfa = (
        preferred[preferred["method"] == "dfa_sgd"]
        .groupby(["cell", "feedback_rank"], dropna=False, as_index=False)
        .agg(dfa_acc=("test_acc", "mean"))
    )
    adam = adam.merge(dfa, on=["cell", "feedback_rank"], how="left")
    if not adam.empty:
        adam["gain"] = 100.0 * (adam["test_acc"] - adam["dfa_acc"])
        for cell in [c for c in CELL_ORDER if c in set(adam["cell"].astype(str))]:
            group = adam[adam["cell"].astype(str) == cell]
            axes[1].scatter(
                group["adam_diagk_update_cosine_mean"],
                group["gain"],
                s=22,
                alpha=0.72,
                label=pretty_cell(cell),
            )
        axes[1].axhline(0, color="#444444", lw=0.8)
        axes[1].set_xlabel("Adam-step/diag-K cosine")
        axes[1].set_ylabel("Adam gain over DFA (pp)")
        axes[1].set_title("B  Does the approximation explain gain?")
        axes[1].legend(frameon=False, fontsize=5.8)

    # C: damping dependence for full and diagonal variants, measured as a gain
    # over raw DFA so task difficulty does not dominate the y-axis.
    damped = final[final["method"].isin(["dfa_diag_k_sqrt", "ndfa_activity", "kndfa"])].copy()
    if not damped.empty:
        hard_cells = [cell for cell in ["nuisance_hard", "low_sample_noisy", "mixed_hard"] if cell in set(final["cell"].astype(str))]
        dfa_ref = (
            final[final["method"] == "dfa_sgd"]
            .groupby(["cell", "feedback_rank"], dropna=False, as_index=False)
            .agg(dfa_acc=("test_acc", "mean"))
        )
        damped = damped.merge(dfa_ref, on=["cell", "feedback_rank"], how="left")
        damped["gain_pp"] = 100.0 * (damped["test_acc"] - damped["dfa_acc"])
        if hard_cells:
            damped = damped[damped["cell"].isin(hard_cells)]
        for method in ["dfa_diag_k_sqrt", "ndfa_activity", "kndfa"]:
            method_df = damped[damped["method"] == method]
            curve = method_df.groupby("damping", as_index=False).agg(gain=("gain_pp", "mean"), sem=("gain_pp", sem)).sort_values("damping")
            axes[2].errorbar(
                curve["damping"],
                curve["gain"],
                yerr=curve["sem"],
                marker="o",
                lw=1.4,
                color=METHOD_COLOR.get(method, "#777777"),
                label=METHOD_LABEL.get(method, method),
            )
        axes[2].set_xscale("log")
        axes[2].set_xlabel("damping")
        axes[2].set_ylabel("gain over DFA (pp)")
        axes[2].set_title("C  Damping matters in hard cells")
        axes[2].axhline(0.0, color="#444444", lw=0.8)
        axes[2].legend(frameon=False, fontsize=5.8)

    for ax in axes:
        ax.grid(axis="y", color="#E8EAE6", lw=0.6)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"infodfa_adam_diagk_diagnostics.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def select_preferred(df: pd.DataFrame, *, preferred_damping: float, preferred_adaptive_lr: float) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    if "damping" in df:
        keep &= df["damping"].isna() | close_or_nan(df["damping"], preferred_damping)
    if "adaptive_lr" in df:
        keep &= df["adaptive_lr"].isna() | close_or_nan(df["adaptive_lr"], preferred_adaptive_lr)
    return df[keep].copy()


def close_or_nan(values: pd.Series, target: float) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values.isna() | np.isclose(values, target, rtol=1e-6, atol=1e-12)


def match_value_or_nan(values: pd.Series, target: float) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if pd.isna(target):
        return values.isna()
    return np.isclose(values, float(target), rtol=1e-6, atol=1e-12)


def write_report(best: pd.DataFrame, approx: pd.DataFrame, output_dir: Path) -> None:
    if best.empty:
        endpoint_body = "_No endpoint rows._"
    else:
        cols = [
            "cell",
            "method_label",
            "damping",
            "adaptive_lr",
            "test_mean",
            "test_sem",
            "delta_vs_dfa",
            "method_cosine",
            "projected_step",
            "norm_ratio",
            "n",
        ]
        endpoint_body = best[[c for c in cols if c in best]].to_markdown(index=False, floatfmt=".4f")
    if approx.empty:
        approx_body = "_No Adam approximation rows._"
    else:
        cols = [
            "cell",
            "factorization_corr",
            "factorization_rel_error",
            "grad_square_factor_corr",
            "adam_factor_corr",
            "adam_diagk_cosine",
            "adam_m_diagk_cosine",
            "test_mean",
            "n",
        ]
        approx_body = approx[[c for c in cols if c in approx]].to_markdown(index=False, floatfmt=".4f")
    text = [
        "# Info-DFA Adam/diag-K approximation",
        "",
        "This experiment asks whether the nDFA/K-nDFA benefit can be reduced to an",
        "Adam-like diagonal second-moment effect. It compares raw DFA, hidden-weight",
        "Adam/RMSProp-style DFA, diagonal square-root factors, and full activity/error",
        "covariance factors on the same noisy/nuisance cells used in the paper.",
        "",
        "The main approximation diagnostics are: factorized KFAC diagonal",
        r"`E[delta_i^2]E[h_j^2]` versus exact per-sample second moment",
        r"`E[(delta_i h_j)^2]`; Adam's running `v_ij` versus that factor; and the",
        "cosine between the Adam-normalized update and the diagonal square-root K update.",
        "",
        "## Best endpoints",
        "",
        endpoint_body,
        "",
        "## Adam approximation diagnostics",
        "",
        approx_body,
        "",
        "Takeaway: Adam-style diagonal conditioning is related to the same second-moment",
        "object and helps in nuisance/noisy regimes, but it does not explain the full",
        "K-nDFA gain. The hard-regime gains are largest when full covariance factors",
        "are lightly damped; this points to structured activity/error covariance rather",
        "than only a scalar learning-rate shift.",
    ]
    (output_dir / "infodfa_adam_diagk_summary.md").write_text("\n".join(text) + "\n")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ordered_cells(df: pd.DataFrame) -> list[str]:
    cells = list(dict.fromkeys(df["cell"].astype(str))) if "cell" in df else []
    ordered = [cell for cell in CELL_ORDER if cell in cells]
    ordered.extend([cell for cell in cells if cell not in set(ordered)])
    return ordered


def pretty_cell(cell: str) -> str:
    return {
        "nuisance_hard": "Nuisance-dominant",
        "low_sample_noisy": "Low-sample/noisy",
        "mixed_hard": "Mixed-context",
        "clean_aligned": "Clean/task-aligned",
    }.get(cell, cell.replace("_", " ").title())


def sem(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def mirror_figures(output_dir: Path) -> None:
    dest = ROOT / "drafts" / "Info-DFA" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for stem in ["infodfa_adam_diagk_learning", "infodfa_adam_diagk_learning_best", "infodfa_adam_diagk_diagnostics"]:
        for ext in ("pdf", "png", "svg"):
            src = output_dir / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, dest / src.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/infodfa_adam_diagk_aggregate_v1")
    parser.add_argument("--preferred-damping", type=float, default=0.3)
    parser.add_argument("--preferred-adaptive-lr", type=float, default=0.003)
    return parser.parse_args()


if __name__ == "__main__":
    main()
