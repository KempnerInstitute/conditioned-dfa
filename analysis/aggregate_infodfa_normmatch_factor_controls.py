"""Aggregate norm-matched activity/error/K-nDFA factor controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "dfa_random",
    "dfa_random+norm_match",
    "ndfa_random",
    "ndfa_random+norm_match",
    "ndfa_random_error",
    "ndfa_random_error+norm_match",
    "ndfa_random_kronecker",
    "ndfa_random_kronecker+norm_match",
]
METHOD_LABEL = {
    "bp": "BP",
    "dfa_random": "DFA",
    "dfa_random+norm_match": "DFA + norm",
    "ndfa_random": "activity nDFA",
    "ndfa_random+norm_match": "activity nDFA + norm",
    "ndfa_random_error": "error nDFA",
    "ndfa_random_error+norm_match": "error nDFA + norm",
    "ndfa_random_kronecker": "K-nDFA",
    "ndfa_random_kronecker+norm_match": "K-nDFA + norm",
}
METHOD_COLOR = {
    "bp": "#222222",
    "dfa_random": "#999999",
    "dfa_random+norm_match": "#666666",
    "ndfa_random": "#009E73",
    "ndfa_random+norm_match": "#66C2A5",
    "ndfa_random_error": "#D55E00",
    "ndfa_random_error+norm_match": "#E69F00",
    "ndfa_random_kronecker": "#6A3D9A",
    "ndfa_random_kronecker+norm_match": "#AA80CF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/infodfa_normmatch_factor_controls_v1")
    parser.add_argument("--output-dir", default="results/infodfa_normmatch_factor_controls_aggregate_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_rows(input_root)
    all_rows.to_csv(output_dir / "infodfa_normmatch_factor_controls_all.csv", index=False)
    summary = summarize(all_rows)
    summary.to_csv(output_dir / "infodfa_normmatch_factor_controls_summary.csv", index=False)
    contrasts = compute_contrasts(summary)
    contrasts.to_csv(output_dir / "infodfa_normmatch_factor_controls_contrasts.csv", index=False)
    write_report(summary, contrasts, output_dir)
    make_summary_figure(summary, output_dir)
    make_contrast_figure(contrasts, output_dir)
    print(summary.to_string(index=False))
    if not contrasts.empty:
        print("\nContrasts:")
        print(contrasts.to_string(index=False))
    print(f"\nwrote {output_dir}")


def load_rows(input_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_root.glob("*/dfa_controls_results.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "cell", path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if col not in {"cell", "task", "method"}:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().any():
                out[col] = converted
    return out.replace([np.inf, -np.inf], np.nan)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["cell", "method"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", sem),
            train_mean=("train_acc", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["cell", "method"])
    )
    out["method_label"] = out["method"].map(METHOD_LABEL).fillna(out["method"])
    return out


def compute_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for cell, group in summary.groupby("cell"):
        vals = group.set_index("method")["test_mean"].to_dict()
        required = ["dfa_random", "dfa_random+norm_match", "ndfa_random+norm_match"]
        if not all(method in vals for method in required):
            continue
        row = {
            "cell": cell,
            "dfa": vals.get("dfa_random", np.nan),
            "dfa_norm": vals.get("dfa_random+norm_match", np.nan),
            "activity_ndfa": vals.get("ndfa_random", np.nan),
            "activity_ndfa_norm": vals.get("ndfa_random+norm_match", np.nan),
            "error_ndfa_norm": vals.get("ndfa_random_error+norm_match", np.nan),
            "k_ndfa": vals.get("ndfa_random_kronecker", np.nan),
            "k_ndfa_norm": vals.get("ndfa_random_kronecker+norm_match", np.nan),
        }
        row.update(
            scale_only_gain=row["dfa_norm"] - row["dfa"],
            activity_gain_after_norm=row["activity_ndfa_norm"] - row["dfa_norm"],
            k_gain_after_norm=row["k_ndfa_norm"] - row["dfa_norm"],
            norm_penalty_activity=row["activity_ndfa_norm"] - row["activity_ndfa"],
            norm_penalty_k=row["k_ndfa_norm"] - row["k_ndfa"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def sem(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def write_report(summary: pd.DataFrame, contrasts: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Norm-Matched Factor Controls",
        "",
        "These controls test whether activity/error/K-nDFA effects are just scalar",
        "learning-rate or per-layer norm changes. The `+ norm` variants rescale each",
        "local hidden gradient to a matched BP layer norm after applying the indicated",
        "preconditioner.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("_No rows found._")
    else:
        cols = ["cell", "method_label", "test_mean", "test_sem", "train_mean", "n"]
        lines.append(summary[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Contrasts", ""])
    if contrasts.empty:
        lines.append("_No complete contrast cells found._")
    else:
        cols = [
            "cell",
            "scale_only_gain",
            "activity_gain_after_norm",
            "k_gain_after_norm",
            "norm_penalty_activity",
            "norm_penalty_k",
        ]
        lines.append(contrasts[cols].to_markdown(index=False, floatfmt=".4f"))
    (output_dir / "infodfa_normmatch_factor_controls_summary.md").write_text("\n".join(lines) + "\n")


def make_summary_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    setup_style()
    cells = list(dict.fromkeys(summary["cell"].astype(str)))
    ncols = 2
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.8 * nrows), constrained_layout=True, squeeze=False)
    for ax, cell in zip(axes.ravel(), cells):
        sub = summary[summary["cell"] == cell].copy()
        sub["rank"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(-1)
        sub = sub[sub["rank"] >= 0].sort_values("rank")
        x = np.arange(sub.shape[0])
        ax.bar(
            x,
            100.0 * sub["test_mean"],
            yerr=100.0 * sub["test_sem"],
            color=[METHOD_COLOR.get(m, "#777777") for m in sub["method"]],
            width=0.72,
            error_kw={"lw": 0.7, "capsize": 2},
        )
        ax.set_xticks(x, sub["method_label"], rotation=35, ha="right")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(cell.replace("_", " "))
    for ax in axes.ravel()[len(cells) :]:
        ax.axis("off")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"infodfa_normmatch_factor_controls_summary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_contrast_figure(contrasts: pd.DataFrame, output_dir: Path) -> None:
    if contrasts.empty:
        return
    setup_style()
    contrasts = contrasts.copy()
    contrasts["label"] = contrasts["cell"].astype(str).str.replace("_", " ", regex=False)
    y = np.arange(contrasts.shape[0])
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.5 * contrasts.shape[0] + 1.0)), constrained_layout=True)
    bars = [
        ("scale_only_gain", "scale only: DFA+norm - DFA", METHOD_COLOR["dfa_random+norm_match"], -width),
        ("activity_gain_after_norm", "activity after norm", METHOD_COLOR["ndfa_random+norm_match"], 0.0),
        ("k_gain_after_norm", "K after norm", METHOD_COLOR["ndfa_random_kronecker+norm_match"], width),
    ]
    for col, label, color, offset in bars:
        ax.barh(y + offset, 100.0 * contrasts[col], height=width * 0.9, color=color, label=label)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(y, contrasts["label"])
    ax.set_xlabel("norm-matched effect (percentage points)")
    ax.set_title("Does conditioning survive layer-norm matching?")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=6.8, ncol=1)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"infodfa_normmatch_factor_controls_contrasts.{ext}", dpi=300, bbox_inches="tight")
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
        }
    )


if __name__ == "__main__":
    main()
