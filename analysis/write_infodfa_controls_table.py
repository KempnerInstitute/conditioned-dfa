"""Write the reviewer-controls LaTeX table from the controls runner output.

Consumes ``results/dfa_controls_<task>_v1/dfa_controls_summary.csv`` for
each task and produces ``drafts/Info-DFA/tables/table_infodfa_controls.tex``.

If the synthetic or vision results are not yet present, a placeholder row is
written so the LaTeX compiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE_OUT = ROOT / "drafts" / "Info-DFA" / "tables" / "table_infodfa_controls.tex"


METHOD_LABELS = {
    "bp": "BP",
    "bp+l2": "BP + L2 ($\\lambda$=1e-3)",
    "bp+label_smoothing": "BP + label smoothing (0.1)",
    "bp+early_stop": "BP + early stop",
    "dfa_random": "DFA",
    "dfa_random+norm_match": "DFA + norm match",
    "ndfa_random": "nDFA",
    "ndfa_random_kronecker": "K-nDFA",
}


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def fmt(value: float, sem: float) -> str:
    return f"{100.0*value:.2f} $\\pm$ {100.0*sem:.2f}"


def format_rows(df: pd.DataFrame, task_label: str) -> list[list[str]]:
    rows: list[list[str]] = []
    if df.empty:
        rows.append([task_label, "\\emph{results pending}", "--", "--", "--"])
        return rows
    bp = df.loc[df["method"] == "bp", "test_acc_mean"]
    dfa = df.loc[df["method"] == "dfa_random", "test_acc_mean"]
    bp_acc = float(bp.iloc[0]) if len(bp) else float("nan")
    dfa_acc = float(dfa.iloc[0]) if len(dfa) else float("nan")
    for method, label in METHOD_LABELS.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        acc = float(sub["test_acc_mean"].iloc[0])
        sem = float(sub["test_acc_sem"].iloc[0])
        gap_bp = f"{100.0*(acc - bp_acc):+.2f}" if np.isfinite(bp_acc) else "n/a"
        gap_dfa = f"{100.0*(acc - dfa_acc):+.2f}" if np.isfinite(dfa_acc) else "n/a"
        rows.append([task_label, label, fmt(acc, sem), gap_bp, gap_dfa])
        task_label = ""  # only label the first row per task
    return rows


def main() -> None:
    synthetic = read_summary(RESULTS / "dfa_controls_synthetic_nuisance_v1" / "dfa_controls_summary.csv")
    vision = read_summary(RESULTS / "dfa_controls_fashion_mnist_noisy_v1" / "dfa_controls_summary.csv")

    rows = []
    rows.extend(format_rows(synthetic, "Synthetic nuisance"))
    rows.extend(format_rows(vision, "Fashion-MNIST noisy"))

    body = [
        "\\begin{table}[!htbp]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\begin{tabular}{@{}llrrr@{}}",
        "        \\toprule",
        "        Task & Method & Test acc & $\\Delta$ BP & $\\Delta$ DFA \\\\",
        "        \\midrule",
    ]
    for row in rows:
        body.append("        " + " & ".join(row) + " \\\\")
    body.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\caption{Reviewer-control comparison. Three BP regularization variants test "
            "whether a stronger BP recipe closes the gap to conditioned DFA. \\texttt{DFA + norm match} "
            "rescales each layer's DFA gradient to match the BP gradient norm on a held-out evaluation batch, "
            "isolating the contribution of activity-covariance whitening (nDFA) from per-layer norm matching.}",
            "    \\label{tab:infodfa_controls}",
            "\\end{table}",
        ]
    )
    TABLE_OUT.write_text("\n".join(body) + "\n")
    print(f"Wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
