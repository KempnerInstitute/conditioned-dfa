"""Aggregate first-paper activity/error/K-nDFA factor ablations.

The focused ablation compares raw DFA, activity-side nDFA, error-side nDFA, and
K-nDFA on representative synthetic and noisy-vision cells, using long training
so the plots show saturation rather than only short-horizon endpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["bp", "dfa_random", "ndfa_random", "ndfa_random_error", "ndfa_random_kronecker"]
METHOD_LABEL = {
    "bp": "BP",
    "dfa_random": "DFA",
    "ndfa_random": "activity nDFA",
    "ndfa_random_error": "error nDFA",
    "ndfa_random_kronecker": "K-nDFA",
}
METHOD_COLOR = {
    "bp": "#222222",
    "dfa_random": "#0072B2",
    "ndfa_random": "#009E73",
    "ndfa_random_error": "#D55E00",
    "ndfa_random_kronecker": "#6A3D9A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", default="results/infodfa_factor_ablation_synthetic_v1")
    parser.add_argument("--vision-root", default="results/infodfa_factor_ablation_vision_v1")
    parser.add_argument("--output-dir", default="results/infodfa_factor_ablation_aggregate_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = load_family(Path(args.synthetic_root), filename="dfa_multioutput_results.csv", family="synthetic")
    vision = load_family(Path(args.vision_root), filename="dfa_nmnc_results.csv", family="vision")
    all_rows = pd.concat([synthetic, vision], ignore_index=True)
    all_rows.to_csv(output_dir / "infodfa_factor_ablation_all.csv", index=False)

    curves = summarize_curves(all_rows)
    endpoints = final_endpoints(all_rows)
    effect_curves = factor_effects(curves)
    effects = factor_effects(endpoints)
    curves.to_csv(output_dir / "infodfa_factor_ablation_curves.csv", index=False)
    endpoints.to_csv(output_dir / "infodfa_factor_ablation_endpoints.csv", index=False)
    effect_curves.to_csv(output_dir / "infodfa_factor_ablation_effect_curves.csv", index=False)
    effects.to_csv(output_dir / "infodfa_factor_ablation_effects.csv", index=False)

    write_report(endpoints, effects, output_dir)
    make_family_curves(curves, "synthetic", output_dir)
    make_family_curves(curves, "vision", output_dir)
    make_effect_curves_figure(effect_curves, output_dir)
    make_endpoint_figure(endpoints, output_dir)
    make_effects_figure(effects, output_dir)

    print(endpoints.to_string(index=False))
    if not effects.empty:
        print("\nFactor effects:")
        print(effects.to_string(index=False))
    print(f"\nwrote {output_dir}")


def load_family(root: Path, *, filename: str, family: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"**/{filename}")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        frame.insert(0, "family", family)
        frame.insert(1, "cell", infer_cell(path, frame, family))
        frame.insert(2, "source_file", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if col not in {"family", "cell", "source_file", "condition", "dataset", "method"}:
            try:
                out[col] = pd.to_numeric(out[col])
            except (TypeError, ValueError):
                pass
    return out.replace([np.inf, -np.inf], np.nan)


def infer_cell(path: Path, frame: pd.DataFrame, family: str) -> str:
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    if family == "synthetic" and "condition" in frame:
        row = frame.iloc[0]
        return (
            f"{row.get('condition', 'synthetic')}_n{int(row.get('n_train', 0))}"
            f"_ln{float(row.get('train_label_noise', 0.0)):.2g}_in{float(row.get('input_noise', 0.0)):.2g}"
        )
    if "dataset" in frame:
        row = frame.iloc[0]
        return f"{row.get('dataset', 'vision')}_n{int(row.get('n_train', 0))}_ln{float(row.get('label_noise', 0.0)):.2g}"
    return parent


def summarize_curves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [
        c
        for c in [
            "family",
            "cell",
            "condition",
            "dataset",
            "n_train",
            "input_noise",
            "train_label_noise",
            "label_noise",
            "feedback_rank",
            "method",
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
            n=("test_acc", "size"),
        )
        .sort_values(["family", "cell", "method", "epoch"])
    )


def final_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    run_cols = [c for c in ["family", "cell", "condition", "dataset", "n_train", "input_noise", "train_label_noise", "label_noise", "method", "seed", "feedback_seed", "feedback_rank"] if c in df.columns]
    final = df.sort_values("epoch").groupby(run_cols, dropna=False, as_index=False).tail(1)
    group_cols = [c for c in run_cols if c not in {"seed", "feedback_seed"}]
    out = (
        final.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            final_epoch=("epoch", "max"),
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            train_mean=("train_eval_acc", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["family", "cell", "method"])
    )
    dfa_ref_cols = [c for c in group_cols if c != "method"]
    ref = out[out["method"] == "dfa_random"][dfa_ref_cols + ["test_mean"]].rename(columns={"test_mean": "dfa_test_mean"})
    out = out.merge(ref, on=dfa_ref_cols, how="left")
    out["delta_vs_dfa"] = out["test_mean"] - out["dfa_test_mean"]
    out["method_label"] = out["method"].map(METHOD_LABEL).fillna(out["method"])
    return out


def factor_effects(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Compute the two-factor decomposition of activity/error conditioning.

    The four local-feedback corners form a small factorial design:
    DFA = no conditioning, activity nDFA = right/input factor only, error nDFA =
    left/error factor only, and K-nDFA = both factors. The most important
    quantity for the paper is not only the error-only effect, but the conditional
    error-side gain after activity conditioning is already present.
    """

    if endpoints.empty:
        return pd.DataFrame()
    id_cols = [
        c
        for c in [
            "family",
            "cell",
            "condition",
            "dataset",
            "n_train",
            "input_noise",
            "train_label_noise",
            "label_noise",
            "feedback_rank",
            "epoch",
            "final_epoch",
        ]
        if c in endpoints.columns
    ]
    needed = ["dfa_random", "ndfa_random", "ndfa_random_error", "ndfa_random_kronecker"]
    rows: list[dict[str, object]] = []
    for keys, group in endpoints.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = group.set_index("method")["test_mean"].to_dict()
        if not all(method in vals for method in needed):
            continue
        row = dict(zip(id_cols, keys))
        dfa = float(vals["dfa_random"])
        activity = float(vals["ndfa_random"])
        error = float(vals["ndfa_random_error"])
        kron = float(vals["ndfa_random_kronecker"])
        row.update(
            dfa=dfa,
            activity_ndfa=activity,
            error_ndfa=error,
            k_ndfa=kron,
            activity_gain=activity - dfa,
            error_only_gain=error - dfa,
            k_gain=kron - dfa,
            error_after_activity_gain=kron - activity,
            activity_after_error_gain=kron - error,
            interaction_gain=kron - activity - error + dfa,
            k_gain_over_best_single=kron - max(activity, error),
        )
        denom = kron - dfa
        row["activity_share_of_k_gain"] = (activity - dfa) / denom if abs(denom) > 1e-12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values([c for c in ["family", "cell"] if c in id_cols])


def sem(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def write_report(endpoints: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> None:
    if endpoints.empty:
        body = "_No rows found._"
    else:
        view_cols = [c for c in ["family", "cell", "method_label", "final_epoch", "test_mean", "test_sem", "delta_vs_dfa", "n"] if c in endpoints]
        body = endpoints[view_cols].to_markdown(index=False, floatfmt=".4f")
    if effects.empty:
        effect_body = "_No complete DFA/activity/error/K-nDFA cells found._"
    else:
        effect_cols = [
            c
            for c in [
                "family",
                "cell",
                "activity_gain",
                "error_only_gain",
                "error_after_activity_gain",
                "k_gain",
                "interaction_gain",
                "activity_share_of_k_gain",
            ]
            if c in effects
        ]
        effect_body = effects[effect_cols].to_markdown(index=False, floatfmt=".4f")
    lines = [
        "# Conditioned DFA Activity/Error Factor Ablation",
        "",
        "This focused ablation separates the two factors that make up K-nDFA:",
        "activity/input-side preconditioning (`ndfa_random`), error/local-delta-side",
        "preconditioning (`ndfa_random_error`), and both together",
        "(`ndfa_random_kronecker`). The runs use longer training than the main",
        "paper sweep so the curves show whether methods saturate or merely differ",
        "at an early stopping point.",
        "",
        "The factorial decomposition should be read as follows: `activity_gain` is",
        "activity nDFA minus DFA; `error_only_gain` is error nDFA minus DFA; and",
        "`error_after_activity_gain` is K-nDFA minus activity nDFA. Thus a negative",
        "error-only effect can coexist with a positive conditional error-side effect",
        "once the activity side has already corrected the dominant input geometry.",
        "",
        "## Factor effects",
        "",
        effect_body,
        "",
        "## Endpoints",
        "",
        body,
    ]
    (output_dir / "infodfa_factor_ablation_summary.md").write_text("\n".join(lines) + "\n")


def make_family_curves(curves: pd.DataFrame, family: str, output_dir: Path) -> None:
    sub = curves[curves["family"] == family].copy()
    if sub.empty:
        return
    setup_style()
    cells = list(dict.fromkeys(sub["cell"].astype(str)))
    ncols = 2
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.7 * nrows), constrained_layout=True, squeeze=False)
    for ax, cell in zip(axes.ravel(), cells):
        cell_df = sub[sub["cell"] == cell]
        for method in METHOD_ORDER:
            method_df = cell_df[cell_df["method"] == method].sort_values("epoch")
            if method_df.empty:
                continue
            ax.plot(
                method_df["epoch"],
                100.0 * method_df["test_mean"],
                color=METHOD_COLOR.get(method, "#777777"),
                label=METHOD_LABEL.get(method, method),
                lw=1.5,
            )
            if method_df["test_sem"].notna().any():
                y = 100.0 * method_df["test_mean"].to_numpy(dtype=float)
                e = 100.0 * method_df["test_sem"].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(method_df["epoch"], y - e, y + e, color=METHOD_COLOR.get(method, "#777777"), alpha=0.12, lw=0)
        ax.set_title(cell.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy (%)")
    for ax in axes.ravel()[len(cells) :]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=6.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"infodfa_factor_ablation_{family}_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_effect_curves_figure(effect_curves: pd.DataFrame, output_dir: Path) -> None:
    sub = effect_curves[effect_curves["family"] == "synthetic"].copy() if not effect_curves.empty else pd.DataFrame()
    if sub.empty or "epoch" not in sub.columns:
        return
    setup_style()
    order = ["nuisance_hard", "low_sample_noisy", "mixed_hard", "clean_aligned"]
    cells = [cell for cell in order if cell in set(sub["cell"].astype(str))]
    cells += [cell for cell in dict.fromkeys(sub["cell"].astype(str)) if cell not in set(cells)]
    ncols = 2
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.65 * nrows), constrained_layout=True, squeeze=False)
    series = [
        ("activity_gain", "activity gain", METHOD_COLOR["ndfa_random"], "-"),
        ("error_only_gain", "error alone", METHOD_COLOR["ndfa_random_error"], "-"),
        ("error_after_activity_gain", "error after activity", METHOD_COLOR["ndfa_random_kronecker"], "-"),
    ]
    for ax, cell in zip(axes.ravel(), cells):
        cell_df = sub[sub["cell"].astype(str).eq(cell)].sort_values("epoch")
        for col, label, color, linestyle in series:
            ax.plot(cell_df["epoch"], 100.0 * cell_df[col], color=color, linestyle=linestyle, lw=1.6, label=label)
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_title(cell.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.set_ylabel("effect vs reference (pp)")
        ax.grid(axis="y", color="#E8EAE6", lw=0.6)
    for ax in axes.ravel()[len(cells) :]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=6.8, loc="best")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"infodfa_factor_ablation_effect_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_endpoint_figure(endpoints: pd.DataFrame, output_dir: Path) -> None:
    if endpoints.empty:
        return
    setup_style()
    sub = endpoints[endpoints["method"].isin(METHOD_ORDER)].copy()
    sub["method_rank"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    cells = list(dict.fromkeys(sub["cell"].astype(str)))
    ncols = 2
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.7 * nrows), constrained_layout=True, squeeze=False)
    for ax, cell in zip(axes.ravel(), cells):
        cell_df = sub[sub["cell"] == cell].sort_values("method_rank")
        x = np.arange(cell_df.shape[0])
        ax.bar(
            x,
            100.0 * cell_df["test_mean"],
            yerr=100.0 * cell_df["test_sem"].fillna(0.0),
            color=[METHOD_COLOR.get(m, "#777777") for m in cell_df["method"]],
            width=0.72,
            error_kw={"lw": 0.7, "capsize": 2},
        )
        ax.set_xticks(x, cell_df["method_label"], rotation=25, ha="right")
        ax.set_ylabel("final test accuracy (%)")
        ax.set_title(cell.replace("_", " "))
    for ax in axes.ravel()[len(cells) :]:
        ax.axis("off")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"infodfa_factor_ablation_endpoints.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_effects_figure(effects: pd.DataFrame, output_dir: Path) -> None:
    if effects.empty:
        return
    setup_style()
    effects = effects.copy()
    effects["label"] = effects["cell"].astype(str).str.replace("_", " ", regex=False)
    y = np.arange(effects.shape[0])
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, 0.48 * effects.shape[0] + 1.1)), constrained_layout=True)
    bars = [
        ("activity_gain", "activity gain (nDFA - DFA)", METHOD_COLOR["ndfa_random"], -width),
        ("error_only_gain", "error alone (error nDFA - DFA)", METHOD_COLOR["ndfa_random_error"], 0.0),
        ("error_after_activity_gain", "error after activity (K-nDFA - nDFA)", METHOD_COLOR["ndfa_random_kronecker"], width),
    ]
    for col, label, color, offset in bars:
        ax.barh(y + offset, 100.0 * effects[col], height=width * 0.9, color=color, label=label)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(y, effects["label"])
    ax.set_xlabel("accuracy effect (percentage points)")
    ax.set_title("Activity/error factor decomposition")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=6.8, ncol=1)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"infodfa_factor_ablation_effects.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
        }
    )


if __name__ == "__main__":
    main()
