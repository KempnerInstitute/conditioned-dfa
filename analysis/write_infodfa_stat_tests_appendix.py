"""Convert paired-statistical-test CSV into an appendix-ready LaTeX block.

Reads ``results/infodfa_paper_tables_20260527/infodfa_statistical_tests.csv``
and writes ``drafts/Info-DFA/tables/table_infodfa_stat_tests.tex``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "infodfa_paper_tables_20260527" / "infodfa_statistical_tests.csv"
OUT = ROOT / "drafts" / "Info-DFA" / "tables" / "table_infodfa_stat_tests.tex"

GROUP_LABEL = {
    "nuisance_dominant": "Nuisance-dominant",
    "mixed_context": "Mixed-context",
    "low_sample_noisy": "Low-sample/noisy",
    "task_aligned": "Task-aligned",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}

METHOD_LABEL = {
    "ndfa_random": "nDFA",
    "ndfa_random_kronecker": "K-nDFA",
}

REF_LABEL = {
    "dfa_random": "DFA",
    "bp": "BP",
}


def fmt_p(p: float) -> str:
    if p < 1e-12:
        return "$<\\!10^{-12}$"
    if p < 0.01:
        text = f"{p:.1e}"
        coef, exp_str = text.split("e")
        exp = int(exp_str)
        return f"${float(coef):.1f}\\times 10^{{{exp}}}$"
    return f"{p:.3f}"


def main() -> None:
    df = pd.read_csv(SRC)
    GROUP_ORDER = [
        "nuisance_dominant",
        "mixed_context",
        "low_sample_noisy",
        "task_aligned",
        "fashion_mnist",
        "cifar10",
    ]
    df["group_rank"] = df["task_group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    df = df.sort_values(["group_rank", "method", "reference"])

    lines = [
        "\\begin{table}[!htbp]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\resizebox{\\textwidth}{!}{%",
        "        \\begin{tabular}{@{}llrrrcrr@{}}",
        "            \\toprule",
        "            Regime / dataset & Comparison & $n$ & $\\bar{\\Delta}$ (pp) & 95\\% CI (pp) & $t$ & $p$ (Holm) & $p_{\\mathrm{Wilcoxon}}$ \\\\",
        "            \\midrule",
    ]
    last_group = None
    for _, row in df.iterrows():
        group = GROUP_LABEL.get(row["task_group"], row["task_group"])
        if group == last_group:
            group_cell = ""
        else:
            group_cell = group
            last_group = group
        method = METHOD_LABEL.get(row["method"], row["method"])
        ref = REF_LABEL.get(row["reference"], row["reference"])
        comparison = f"{method} vs.\\ {ref}"
        ci = f"[{100*row['ci_low']:+.2f},\\,{100*row['ci_high']:+.2f}]"
        lines.append(
            f"            {group_cell} & {comparison} & {int(row['n_pairs'])} & "
            f"{100*row['mean_delta']:+.2f} & {ci} & {row['t_stat']:+.2f} & "
            f"{fmt_p(row['t_p_holm'])} & {fmt_p(row['wilcoxon_p_holm'])} \\\\"
        )
    lines.extend(
        [
            "            \\bottomrule",
            "        \\end{tabular}",
            "    }",
            "    \\caption{Paired statistical tests of conditioned-DFA gains. Cells are paired across "
            "(noise, sample, label-noise) configurations within each regime or dataset. We report the "
            "mean delta in percent points, a 2000-resample bootstrap 95\\% CI, the paired $t$ statistic, "
            "and Holm-corrected $p$-values for the paired $t$-test and the Wilcoxon signed-rank test. "
            "Conditioning gains over raw DFA are highly significant in every regime (Holm $p<10^{-7}$). "
            "Conditioning gains over BP are significant in nuisance-dominant, mixed-context, and "
            "low-sample/noisy regimes ($p<10^{-13}$), but conditioning \\emph{loses} to BP in the "
            "task-aligned synthetic control and shows no significant difference from BP on Fashion-MNIST and CIFAR-10.}",
            "    \\label{tab:infodfa_stat_tests}",
            "\\end{table}",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
