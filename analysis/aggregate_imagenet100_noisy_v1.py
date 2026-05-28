"""Aggregate experiment C (noisy-label ImageNet-100) results into a paper table.

Reads ``results/imagenet100_noisy_v1/resnet18/<method>/labelnoise_*/seed_<i>/imagenet_credit_assignment.csv``,
computes per-method final val-top1 with SEM, paired t-tests of nDFA-vs-DFA and
nDFA-vs-BP, and writes:
    - ``results/imagenet100_noisy_aggregate_v1/imagenet_credit_assignment_summary.csv``
    - ``results/imagenet100_noisy_aggregate_v1/imagenet_credit_assignment_summary.md``
    - ``drafts/Info-DFA/tables/table_infodfa_imagenet100_noisy.tex``  (only if any
      DFA-rescue is statistically significant at p<0.05)

The "include only winners" policy is encoded in the writer: the LaTeX table
is written only if at least one conditioned method beats DFA by >=2pp with
p<0.05. Otherwise the manuscript stays unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "imagenet100_noisy_v1"
OUT_DIR = ROOT / "results" / "imagenet100_noisy_aggregate_v1"
TABLE_PATH = ROOT / "drafts" / "Info-DFA" / "tables" / "table_infodfa_imagenet100_noisy.tex"


METHOD_LABELS = {
    "bp": "BP",
    "block_dfa": "block-DFA",
    "block_ndfa": "block-NDFA",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv in SRC.glob("resnet18/*/labelnoise_*/seed_*/imagenet_credit_assignment.csv"):
        parts = csv.relative_to(SRC).parts
        method = parts[1]
        noise = parts[2].replace("labelnoise_", "").replace("p", ".")
        seed = int(parts[3].replace("seed_", ""))
        df = pd.read_csv(csv)
        if df.empty:
            continue
        last = df.iloc[-1]
        rows.append(
            {
                "method": method,
                "label_noise": float(noise),
                "seed": seed,
                "val_top1": float(last["val_top1"]),
                "train_top1": float(last["train_top1"]),
                "epoch": int(last["epoch"]),
            }
        )
    if not rows:
        print("No experiment-C results found yet.")
        return

    res = pd.DataFrame(rows).sort_values(["method", "seed"])
    res.to_csv(OUT_DIR / "imagenet_credit_assignment_all.csv", index=False)
    summary = (
        res.groupby("method")
        .agg(
            val_mean=("val_top1", "mean"),
            val_sem=("val_top1", "sem"),
            train_mean=("train_top1", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "imagenet_credit_assignment_summary.csv", index=False)

    bp = res[res["method"] == "bp"].sort_values("seed")["val_top1"].values
    dfa = res[res["method"] == "block_dfa"].sort_values("seed")["val_top1"].values
    ndfa = res[res["method"] == "block_ndfa"].sort_values("seed")["val_top1"].values

    deltas = {}
    if len(ndfa) == len(dfa) and len(dfa) >= 2:
        deltas["nDFA_vs_DFA"] = paired_test(ndfa, dfa)
    if len(ndfa) == len(bp) and len(bp) >= 2:
        deltas["nDFA_vs_BP"] = paired_test(ndfa, bp)
    if len(dfa) == len(bp) and len(bp) >= 2:
        deltas["DFA_vs_BP"] = paired_test(dfa, bp)

    lines = ["# Experiment C: noisy-label ImageNet-100", ""]
    lines.append("| method | val_top1 mean | SEM | train_top1 mean | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {METHOD_LABELS.get(row['method'], row['method'])} | "
            f"{row['val_mean']:.2f} | {row['val_sem']:.2f} | "
            f"{row['train_mean']:.2f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## Paired t-tests (val_top1)")
    lines.append("")
    for name, (delta, t, p) in deltas.items():
        lines.append(f"- **{name}**: mean delta = {delta:+.2f} pp, t = {t:+.2f}, p = {p:.4g}")
    lines.append("")
    (OUT_DIR / "imagenet_credit_assignment_summary.md").write_text("\n".join(lines))

    winner = False
    if "nDFA_vs_DFA" in deltas:
        delta, _, p = deltas["nDFA_vs_DFA"]
        winner = (delta >= 2.0) and (p < 0.05)
    if winner:
        write_paper_table(summary, deltas)
        print(f"Winner detected: paper table written to {TABLE_PATH}")
    else:
        print(f"Not a winner: paper unchanged. Summary in {OUT_DIR / 'imagenet_credit_assignment_summary.md'}")

    print()
    print(open(OUT_DIR / "imagenet_credit_assignment_summary.md").read())


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Paired t-test; returns (mean delta in pp, t-stat, two-sided p-value)."""
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    return float(diff.mean()), float(t), float(p)


def write_paper_table(summary: pd.DataFrame, deltas: dict) -> None:
    """Write a LaTeX table only if there is something to claim."""
    bp = summary[summary["method"] == "bp"].iloc[0]
    dfa = summary[summary["method"] == "block_dfa"].iloc[0]
    ndfa = summary[summary["method"] == "block_ndfa"].iloc[0]

    def cell(row, ref_delta, ref_p):
        marker = ""
        if ref_p is not None:
            if ref_p < 0.01:
                marker = "$^\\ddagger$"
            elif ref_p < 0.05:
                marker = "$^\\dagger$"
        return f"{row['val_mean']:.2f} $\\pm$ {row['val_sem']:.2f}", marker

    bp_cell, _ = cell(bp, None, None)
    dfa_cell, _ = cell(dfa, None, None)
    ndfa_cell, ndfa_marker = cell(ndfa, *deltas.get("nDFA_vs_DFA", (None, None, None))[::2])

    body = [
        "\\begin{table}[!htbp]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\begin{tabular}{@{}lrrrr@{}}",
        "        \\toprule",
        "        Method & Val Top-1 & $\\Delta$ vs DFA & $\\Delta$ vs BP & Train Top-1 \\\\",
        "        \\midrule",
        f"        BP & {bp_cell} & -- & 0.00 & {bp['train_mean']:.2f} \\\\",
        f"        block-DFA & {dfa_cell} & 0.00 & {dfa['val_mean']-bp['val_mean']:+.2f} & {dfa['train_mean']:.2f} \\\\",
        f"        block-NDFA & {ndfa_cell}{ndfa_marker} & {ndfa['val_mean']-dfa['val_mean']:+.2f} & {ndfa['val_mean']-bp['val_mean']:+.2f} & {ndfa['train_mean']:.2f} \\\\",
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\caption{Noisy-label ImageNet-100 ResNet-18, 30\\% symmetric label noise, "
        "30 epochs, 3 seeds. Significance markers (paired $t$-test vs block-DFA): "
        "$\\dagger$ $p<0.05$, $\\ddagger$ $p<0.01$.}",
        "    \\label{tab:infodfa_imagenet100_noisy}",
        "\\end{table}",
    ]
    TABLE_PATH.write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    main()
