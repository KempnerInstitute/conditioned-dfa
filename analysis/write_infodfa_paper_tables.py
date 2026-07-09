"""Write paper-ready Info-DFA result tables from final aggregate CSVs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("INFODFA_RESULTS", ROOT / "results")).resolve()
LEGACY_RESULTS = Path(os.environ.get("INFODFA_LEGACY_RESULTS", ROOT / "../Info-Man/results")).resolve()
REPORT_DIR = RESULTS / "infodfa_paper_tables_20260527"
TABLE_DIR = Path(os.environ.get("INFODFA_TABLE_TEX_DIR", REPORT_DIR / "tex_tables")).resolve()
SOURCE_LOG: dict[str, Path] = {}


METHOD_LABELS = {
    "bp": "BP",
    "dfa_random": "DFA",
    "fa_random": "FA",
    "drtp_random": "DRTP",
    "ndfa_random": "nDFA",
    "ndfa_random_kronecker": "K-nDFA",
    "local_loss": "Local aux",
    "vnc": "VNC",
    "nmnc": "NMNC",
    "block_dfa": "block-DFA",
    "block_ndfa": "block-nDFA",
}

CONDITION_LABELS = {
    "nuisance_dominant": "Nuisance-dominant",
    "low_sample_noisy": "Low-sample/noisy",
    "mixed_context": "Mixed-context",
    "task_aligned": "Task-aligned",
}

DATASET_LABELS = {
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    synthetic = synthetic_summary()
    vision = vision_summary()
    hard = hard_cifar_summary()
    imagenet = imagenet_summary()
    kfac_source = kfac_source_summary(synthetic)
    seed_loso = seed_loso_summary()

    write_synthetic_split_table(synthetic)
    write_vision_split_table(vision)
    write_hard_cifar_table(hard)
    write_imagenet_table(imagenet)
    write_kfac_source_table(kfac_source)
    write_seed_loso_table(seed_loso)
    write_summary_report(synthetic, vision, hard, imagenet, kfac_source, seed_loso)

    print(f"Wrote Info-DFA table snapshots to {TABLE_DIR}")
    print(f"Wrote compact result summary to {REPORT_DIR}")


def result_path(relative: str | Path) -> Path:
    """Resolve final result artifacts from the primary or legacy result root."""

    rel = Path(relative)
    if rel.is_absolute():
        path = rel
    else:
        primary = RESULTS / rel
        legacy = LEGACY_RESULTS / rel
        if primary.exists():
            path = primary
        elif legacy.exists():
            path = legacy
        else:
            raise FileNotFoundError(
                f"Could not find result input {rel}. Checked {primary} and {legacy}. "
                "Set INFODFA_RESULTS or INFODFA_LEGACY_RESULTS if the aggregate is staged elsewhere."
            )
    SOURCE_LOG[str(rel)] = path
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_absolute():
        path = result_path(path)
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


def synthetic_summary() -> pd.DataFrame:
    df = read_csv("infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_best_by_method.csv")
    cell_cols = ["condition", "input_noise", "n_train", "train_label_noise"]
    wide = df.pivot_table(index=cell_cols, columns="method", values="test_mean", aggfunc="mean").reset_index()
    rows = []
    for condition in ["nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned"]:
        sub = wide[wide["condition"] == condition]
        # DEPRECATED post-hoc envelope: picks the better of nDFA / K-nDFA per
        # regime after seeing results. Kept only for the diagnostic markdown
        # report; no headline (tex) table uses these columns.
        best_conditioned_method = sub[["ndfa_random", "ndfa_random_kronecker"]].mean().idxmax()
        best_nc_method = sub[["vnc", "nmnc"]].mean().idxmax()
        best_conditioned_series = sub[best_conditioned_method]
        best_nc_series = sub[best_nc_method]
        rows.append(
            {
                "condition": condition,
                "bp": sub["bp"].mean(),
                "dfa": sub["dfa_random"].mean(),
                "ndfa": sub["ndfa_random"].mean(),
                "kndfa": sub["ndfa_random_kronecker"].mean(),
                "best_conditioned_posthoc": best_conditioned_series.mean(),
                "best_nc": best_nc_series.mean(),
                "delta_ndfa_dfa": (sub["ndfa_random"] - sub["dfa_random"]).mean(),
                "delta_kndfa_dfa": (sub["ndfa_random_kronecker"] - sub["dfa_random"]).mean(),
                "delta_best_conditioned_bp": (best_conditioned_series - sub["bp"]).mean(),
                "wins_ndfa_over_dfa": int((sub["ndfa_random"] > sub["dfa_random"]).sum()),
                "wins_kndfa_over_dfa": int((sub["ndfa_random_kronecker"] > sub["dfa_random"]).sum()),
                "wins_best_conditioned_dfa": int((best_conditioned_series > sub["dfa_random"]).sum()),
                "wins_best_conditioned_bp": int((best_conditioned_series > sub["bp"]).sum()),
                "n_cells": int(sub.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_synthetic_noise.csv", index=False)
    return out


def vision_summary() -> pd.DataFrame:
    df = read_csv("infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_best_by_method.csv")
    cell_cols = ["dataset", "n_train", "label_noise"]
    wide = df.pivot_table(index=cell_cols, columns="method", values="test_mean", aggfunc="mean").reset_index()
    rows = []
    for dataset in ["fashion_mnist", "cifar10"]:
        sub = wide[wide["dataset"] == dataset]
        # DEPRECATED post-hoc envelope: see synthetic_summary.
        best_conditioned_method = sub[["ndfa_random", "ndfa_random_kronecker"]].mean().idxmax()
        best_nc_method = sub[["vnc", "nmnc"]].mean().idxmax()
        best_conditioned_series = sub[best_conditioned_method]
        best_nc_series = sub[best_nc_method]
        rows.append(
            {
                "dataset": dataset,
                "bp": sub["bp"].mean(),
                "dfa": sub["dfa_random"].mean(),
                "ndfa": sub["ndfa_random"].mean(),
                "kndfa": sub["ndfa_random_kronecker"].mean(),
                "best_conditioned_posthoc": best_conditioned_series.mean(),
                "best_nc": best_nc_series.mean(),
                "delta_ndfa_dfa": (sub["ndfa_random"] - sub["dfa_random"]).mean(),
                "delta_kndfa_dfa": (sub["ndfa_random_kronecker"] - sub["dfa_random"]).mean(),
                "delta_best_conditioned_bp": (best_conditioned_series - sub["bp"]).mean(),
                "wins_ndfa_over_dfa": int((sub["ndfa_random"] > sub["dfa_random"]).sum()),
                "wins_kndfa_over_dfa": int((sub["ndfa_random_kronecker"] > sub["dfa_random"]).sum()),
                "wins_best_conditioned_dfa": int((best_conditioned_series > sub["dfa_random"]).sum()),
                "wins_best_conditioned_bp": int((best_conditioned_series > sub["bp"]).sum()),
                "n_cells": int(sub.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_vision_noise.csv", index=False)
    return out


def hard_cifar_summary() -> pd.DataFrame:
    df = read_csv("infodfa_hard_cifar100_confirm_aggregate_v2/dfa_convnet_summary.csv")
    order = ["bp", "local_loss", "ndfa_random", "ndfa_random_kronecker", "dfa_random", "drtp_random"]
    df = df.set_index("method").loc[order].reset_index()
    dfa = float(df.loc[df["method"] == "dfa_random", "test_mean"].iloc[0])
    df["delta_vs_dfa"] = df["test_mean"] - dfa
    df.to_csv(REPORT_DIR / "table_infodfa_hard_cifar.csv", index=False)
    return df


def imagenet_summary() -> pd.DataFrame:
    df = read_csv("imagenet100_strongform_v1/strongform_multiseed_summary.csv")
    lookup = df.set_index("tag")
    bp = lookup.loc["bp"]
    rows = []
    for depth, label in [
        ("layer4", "layer4 (late only)"),
        ("l34", "layer3+4"),
        ("l234", "layer2+3+4"),
        ("all", "all blocks"),
    ]:
        row = {"depth": depth, "label": label}
        for key, col in [("dfa", "raw"), ("ndfaDiag", "diag"), ("ndfaFull", "zca")]:
            tag = f"{key}_{depth}"
            item = lookup.loc[tag]
            row[f"{col}_mean"] = float(item["mean"])
            row[f"{col}_sem"] = float(item["sem"])
        row["bp_mean"] = float(bp["mean"])
        row["bp_sem"] = float(bp["sem"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_imagenet_boundary.csv", index=False)
    return out


def kfac_source_summary(synthetic: pd.DataFrame) -> pd.DataFrame:
    df = read_csv("infodfa_kfac_control_v1/summary/kfac_control_summary.csv")
    ndfa_ref = synthetic.set_index("condition")["ndfa"]
    rows = []
    for condition in ["nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned"]:
        row = df[df["condition"].eq(condition)].iloc[0]
        dfa_error = float(row["ndfa_random_kronecker"])
        bp_error = float(row["ndfa_random_kronecker_bp"])
        rows.append(
            {
                "condition": condition,
                "ndfa": 100.0 * float(ndfa_ref.loc[condition]),
                "kndfa_dfa_error_factor": dfa_error,
                "kndfa_bp_error_factor": bp_error,
                "bp_minus_dfa_error_pp": bp_error - dfa_error,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_kfac_source.csv", index=False)
    return out


def seed_loso_summary() -> pd.DataFrame:
    tests = read_csv("infodfa_honest_selection_reanalysis/cell_seed_loso_tests.csv")
    rows = []
    for condition in ["nuisance_dominant", "low_sample_noisy", "mixed_context", "task_aligned"]:
        row = tests[tests["condition"].eq(condition)].iloc[0]
        rows.append(
            {
                "condition": condition,
                "n_pairs": int(row["n_pairs"]),
                "n_positive_vs_bp": int(row["n_positive_vs_BP"]),
                "best_conditioned_minus_bp_pp": float(row["mean_gain_vs_BP"]),
                "p_value": float(row["p_vs_BP"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_seed_loso.csv", index=False)
    return out


def write_synthetic_split_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                CONDITION_LABELS[row["condition"]],
                pct(row["bp"]),
                pct(row["dfa"]),
                pct(row["ndfa"]),
                pct(row["kndfa"]),
                pct(row["best_nc"]),
                pp(row["delta_ndfa_dfa"]),
                pp(row["delta_kndfa_dfa"]),
                f"{int(row['wins_ndfa_over_dfa'])}/{int(row['n_cells'])}, {int(row['wins_kndfa_over_dfa'])}/{int(row['n_cells'])}",
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_synthetic_noise_split.tex",
        caption=(
            "Synthetic stress-suite summary with nDFA and K-nDFA reported as separate methods, "
            "rather than the per-cell maximum. Activity-only conditioning (nDFA) does the bulk of the work; "
            "the additional feedback-side whitening in K-nDFA gives a smaller and less consistent gain. "
            "Wins columns are reported as (nDFA over DFA)/(K-nDFA over DFA)."
        ),
        label="tab:infodfa_synthetic_noise_split",
        headers=[
            "Regime",
            "BP",
            "DFA",
            "nDFA",
            "K-nDFA",
            "Best NC",
            "$\\Delta$ nDFA",
            "$\\Delta$ K-nDFA",
            "Wins nDFA/K-nDFA",
        ],
        rows=rows,
    )


def write_vision_split_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                DATASET_LABELS[row["dataset"]],
                pct(row["bp"]),
                pct(row["dfa"]),
                pct(row["ndfa"]),
                pct(row["kndfa"]),
                pct(row["best_nc"]),
                pp(row["delta_ndfa_dfa"]),
                pp(row["delta_kndfa_dfa"]),
                f"{int(row['wins_ndfa_over_dfa'])}/{int(row['n_cells'])}, {int(row['wins_kndfa_over_dfa'])}/{int(row['n_cells'])}",
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_vision_noise_split.tex",
        caption=(
            "Noisy-label vision MLP sweep with nDFA and K-nDFA reported as separate methods. "
            "Both conditioned rules close most of the raw-DFA gap; the additional Kronecker "
            "feedback-side whitening does not yield a consistent further gain on these datasets."
        ),
        label="tab:infodfa_vision_noise_split",
        headers=[
            "Dataset",
            "BP",
            "DFA",
            "nDFA",
            "K-nDFA",
            "Best NC",
            "$\\Delta$ nDFA",
            "$\\Delta$ K-nDFA",
            "Wins nDFA/K-nDFA",
        ],
        rows=rows,
    )


def write_hard_cifar_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                METHOD_LABELS[row["method"]],
                f"{100.0 * row['test_mean']:.2f} $\\pm$ {100.0 * row['test_sem']:.2f}",
                f"{100.0 * row['train_mean']:.1f}",
                pp(row["delta_vs_dfa"]),
                str(int(row["n"])),
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_hard_cifar.tex",
        caption=(
            "Hard CIFAR-100 convnet confirmation. The conditioned local rules improve raw DFA by about "
            "8 percentage points, but remain below BP and the local auxiliary-loss baseline."
        ),
        label="tab:infodfa_hard_cifar",
        headers=["Method", "Test accuracy", "Train accuracy", "$\\Delta$ vs DFA", "$n$"],
        rows=rows,
        resize=False,
    )


def write_imagenet_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                row["label"],
                mean_sem(row["raw_mean"], row["raw_sem"]),
                mean_sem(row["diag_mean"], row["diag_sem"]),
                imagenet_zca_cell(row),
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_imagenet_boundary.tex",
        caption=(
            "ImageNet-100 substitution-depth boundary (full-covariance conditioner test). ResNet-18 "
            f"fine-tuned from pretrained weights (BP reference {float(df['bp_mean'].iloc[0]):.1f} $\\pm$ "
            f"{float(df['bp_sem'].iloc[0]):.1f}\\% top-1), local updates rescaled to unit norm "
            "(no BP-norm oracle), 90 epochs; mean $\\pm$ SEM over three seeds. Columns compare raw "
            "block-DFA, diagonal channel-nDFA, and the full channel-covariance ZCA conditioner "
            "(inverse square root, not the power-$1$ MLP nDFA operator). Late substitution costs "
            "$\\approx 6$--$7$\\,pp, but a substitution-depth cliff persists for every variant. "
            "$^{\\ast}$Full-cov ZCA at layer2+3+4 and all blocks uses the LR-tuned "
            "$\\eta{=}0.01$ runs; at the raw/diagonal baselines' $\\eta{=}0.1$, the same full-cov "
            "conditioner reaches only $51.0\\%$ and $38.6\\%$, respectively. Thus ImageNet all-layer "
            "credit assignment remains unsolved across the conditioner forms tested here."
        ),
        label="tab:infodfa_imagenet_boundary",
        headers=["Substitution depth", "raw block-DFA", "nDFA-diag", "full-cov ZCA"],
        rows=rows,
        align="@{}lrrr@{}",
        resize=False,
    )


def write_kfac_source_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                CONDITION_LABELS[row["condition"]],
                f"{row['ndfa']:.1f}",
                f"{row['kndfa_dfa_error_factor']:.1f}",
                f"{row['kndfa_bp_error_factor']:.1f}",
                f"{row['bp_minus_dfa_error_pp']:+.2f}",
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_kfac_source.tex",
        caption=(
            "No-weight-transport KFAC-source control. nDFA is the activity-only reference from the "
            "main synthetic suite. The two K-nDFA columns compare the usual local DFA-error left "
            "factor with an oracle BP-error left factor; $\\Delta$ is BP-error minus DFA-error "
            "source in percentage points. The average $\\Delta$ is $-0.007$\\,pp, so the BP-error "
            "factor does not materially improve the local DFA-error factor on this suite."
        ),
        label="tab:infodfa_kfac_source",
        headers=["Regime", "nDFA", "K-nDFA DFA-error factor", "K-nDFA BP-error factor", "$\\Delta$"],
        rows=rows,
        resize=False,
    )


def write_seed_loso_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                CONDITION_LABELS[row["condition"]],
                f"{int(row['n_positive_vs_bp'])}/{int(row['n_pairs'])}",
                f"{row['best_conditioned_minus_bp_pp']:+.2f}",
                p_value(row["p_value"]),
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_seed_loso.tex",
        caption=(
            "Matched cell-by-seed leave-one-seed-out robustness tests for the headline synthetic "
            "comparison. Values are best conditioned rule minus BP in percentage points after "
            "LOSO rank selection. The positive/pairs column counts held-out cell$\\times$seed "
            "observations where the selected conditioned rule exceeds BP. The paired units are "
            "matched cell$\\times$seed observations, not five independent seed-level draws, and "
            "the very small Wilcoxon values are reported only as robustness floors. The clean "
            "task-aligned control is negative, matching the anisotropy law rather than supporting "
            "a universal BP-replacement claim."
        ),
        label="tab:infodfa_seed_loso",
        headers=["Regime", "positive/pairs", "Best conditioned $-$ BP (pp)", "$p$"],
        rows=rows,
        resize=False,
    )


def write_table(
    path: Path,
    *,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    align: str | None = None,
    resize: bool = True,
) -> None:
    if align is None:
        align = "@{}" + "l" + "r" * (len(headers) - 1) + "@{}"
    body = []
    body.append("\\begin{table}[!htbp]")
    body.append("    \\centering")
    body.append("    \\small")
    body.append("    \\setlength{\\tabcolsep}{4pt}")
    if resize:
        body.append("    \\resizebox{\\textwidth}{!}{%")
        indent = "        "
    else:
        indent = "    "
    body.append(f"{indent}\\begin{{tabular}}{{{align}}}")
    body.append(f"{indent}    \\toprule")
    body.append(f"{indent}    " + " & ".join(headers) + " \\\\")
    body.append(f"{indent}    \\midrule")
    for row in rows:
        body.append(f"{indent}    " + " & ".join(escape_cell(v) for v in row) + " \\\\")
    body.append(f"{indent}    \\bottomrule")
    body.append(f"{indent}\\end{{tabular}}")
    if resize:
        body.append("    }")
    body.append(f"    \\caption{{{caption}}}")
    body.append(f"    \\label{{{label}}}")
    body.append("\\end{table}")
    path.write_text("\n".join(body) + "\n")


def write_summary_report(
    synthetic: pd.DataFrame,
    vision: pd.DataFrame,
    hard: pd.DataFrame,
    imagenet: pd.DataFrame,
    kfac_source: pd.DataFrame,
    seed_loso: pd.DataFrame,
) -> None:
    lines = [
        "# Info-DFA final result summary",
        "",
        "Generated from the final aggregate CSVs resolved via INFODFA_RESULTS / INFODFA_LEGACY_RESULTS.",
        "",
        "## Source inputs",
        "",
    ]
    for rel, path in sorted(SOURCE_LOG.items()):
        lines.append(f"- `{rel}` -> `{path}`")
    lines.extend(
        [
            "",
            "## Completed result sets",
            "",
            "- Synthetic noise/sample sweep v2: 128 cells, 8 methods per cell.",
            "- Vision noisy-label MLP sweep v2: Fashion-MNIST and CIFAR-10, 24 cells, 8 methods per cell.",
            "- Hard CIFAR-100 convnet confirmation v2: BP/local/DFA/nDFA/K-nDFA/DRTP comparison.",
            "- ImageNet-100 diagnostics: strong-form substitution-depth boundary and 90-epoch spatial head-to-head.",
            "",
            "## Synthetic stress suite",
            "",
            "The `best-cond` columns use the DEPRECATED post-hoc envelope (better of",
            "nDFA / K-nDFA selected after seeing results); they are diagnostic only and",
            "must not be quoted as headline numbers. Headline tables report nDFA and",
            "K-nDFA separately.",
            "",
            "| regime | BP | DFA | best-cond (post-hoc) | best NC | nDFA gain vs DFA | best-cond gain vs BP (post-hoc) | best-cond wins over DFA/BP (post-hoc) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in synthetic.iterrows():
        lines.append(
            f"| {CONDITION_LABELS[row['condition']]} | {pct(row['bp'])} | {pct(row['dfa'])} | "
            f"{pct(row['best_conditioned_posthoc'])} | {pct(row['best_nc'])} | {pp(row['delta_ndfa_dfa'])} | "
            f"{pp(row['delta_best_conditioned_bp'])} | {int(row['wins_best_conditioned_dfa'])}/{int(row['n_cells'])}, "
            f"{int(row['wins_best_conditioned_bp'])}/{int(row['n_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## Vision MLP sweep",
            "",
            "`best-cond` columns: same DEPRECATED post-hoc envelope caveat as above.",
            "",
            "| dataset | BP | DFA | best-cond (post-hoc) | best NC | nDFA gain vs DFA | best-cond gain vs BP (post-hoc) | best-cond wins over DFA/BP (post-hoc) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in vision.iterrows():
        lines.append(
            f"| {DATASET_LABELS[row['dataset']]} | {pct(row['bp'])} | {pct(row['dfa'])} | "
            f"{pct(row['best_conditioned_posthoc'])} | {pct(row['best_nc'])} | {pp(row['delta_ndfa_dfa'])} | "
            f"{pp(row['delta_best_conditioned_bp'])} | {int(row['wins_best_conditioned_dfa'])}/{int(row['n_cells'])}, "
            f"{int(row['wins_best_conditioned_bp'])}/{int(row['n_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## Hard CIFAR-100 convnet",
            "",
            "| method | test | train | gain vs DFA | n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in hard.iterrows():
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {100.0 * row['test_mean']:.2f} +/- {100.0 * row['test_sem']:.2f} | "
            f"{100.0 * row['train_mean']:.1f} | {pp(row['delta_vs_dfa'])} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## ImageNet-100 boundary",
            "",
            "| depth | raw block-DFA | nDFA-diag | full-cov ZCA |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in imagenet.iterrows():
        lines.append(
            f"| {row['label']} | {mean_sem(row['raw_mean'], row['raw_sem'])} | "
            f"{mean_sem(row['diag_mean'], row['diag_sem'])} | {imagenet_zca_cell(row)} |"
        )
    lines.extend(
        [
            "",
            "## Cell x seed LOSO synthetic robustness tests",
            "",
            "| regime | positive/pairs | best conditioned - BP (pp) | p |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in seed_loso.iterrows():
        lines.append(
            f"| {CONDITION_LABELS[row['condition']]} | {int(row['n_positive_vs_bp'])}/{int(row['n_pairs'])} | "
            f"{row['best_conditioned_minus_bp_pp']:+.2f} | {p_value(row['p_value'])} |"
        )
    lines.extend(
        [
            "",
            "## KFAC-source control",
            "",
            "| regime | nDFA | K-nDFA DFA-error factor | K-nDFA BP-error factor | BP-error - DFA-error (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in kfac_source.iterrows():
        lines.append(
            f"| {CONDITION_LABELS[row['condition']]} | {row['ndfa']:.1f} | "
            f"{row['kndfa_dfa_error_factor']:.1f} | {row['kndfa_bp_error_factor']:.1f} | "
            f"{row['bp_minus_dfa_error_pp']:+.2f} |"
        )
    lines.append("")
    (REPORT_DIR / "infodfa_final_result_summary.md").write_text("\n".join(lines))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}"


def pp(value: float) -> str:
    return f"{100.0 * float(value):+.1f}"


def pp_percent(value: float) -> str:
    return f"{float(value):+.2f}"


def mean_sem(mean: float, sem: float) -> str:
    return f"{float(mean):.1f} $\\pm$ {float(sem):.1f}"


def p_value(value: float) -> str:
    if float(value) < 1e-10:
        return "$<10^{-10}$"
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def imagenet_zca_cell(row: pd.Series) -> str:
    suffix = "$^{\\ast}$" if row["depth"] in {"l234", "all"} else ""
    return f"{float(row['zca_mean']):.1f}{suffix} $\\pm$ {float(row['zca_sem']):.1f}"


def escape_cell(value: object) -> str:
    text = str(value)
    replacements = {
        "%": "\\%",
        "&": "\\&",
        "_": "\\_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
