"""Write paper-ready Info-DFA result tables from final aggregate CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE_DIR = ROOT / "drafts" / "Info-DFA" / "tables"
REPORT_DIR = RESULTS / "infodfa_paper_tables_20260527"


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
    "mixed_context": "Mixed-context",
    "low_sample_noisy": "Low-sample/noisy",
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

    write_synthetic_table(synthetic)
    write_synthetic_split_table(synthetic)
    write_vision_table(vision)
    write_vision_split_table(vision)
    write_hard_cifar_table(hard)
    write_imagenet_table(imagenet)
    write_summary_report(synthetic, vision, hard, imagenet)

    print(f"Wrote Info-DFA paper tables to {TABLE_DIR}")
    print(f"Wrote compact result summary to {REPORT_DIR}")


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


def synthetic_summary() -> pd.DataFrame:
    df = read_csv(RESULTS / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_best_by_method.csv")
    cell_cols = ["condition", "input_noise", "n_train", "train_label_noise"]
    wide = df.pivot_table(index=cell_cols, columns="method", values="test_mean", aggfunc="max").reset_index()
    wide["best_ndfa"] = wide[["ndfa_random", "ndfa_random_kronecker"]].max(axis=1)
    wide["best_nc"] = wide[["vnc", "nmnc"]].max(axis=1)
    rows = []
    for condition in ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]:
        sub = wide[wide["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "bp": sub["bp"].mean(),
                "dfa": sub["dfa_random"].mean(),
                "ndfa": sub["ndfa_random"].mean(),
                "kndfa": sub["ndfa_random_kronecker"].mean(),
                "best_ndfa": sub["best_ndfa"].mean(),
                "best_nc": sub["best_nc"].mean(),
                "delta_ndfa_dfa": (sub["ndfa_random"] - sub["dfa_random"]).mean(),
                "delta_kndfa_dfa": (sub["ndfa_random_kronecker"] - sub["dfa_random"]).mean(),
                "delta_ndfa_bp": (sub["best_ndfa"] - sub["bp"]).mean(),
                "wins_ndfa_over_dfa": int((sub["ndfa_random"] > sub["dfa_random"]).sum()),
                "wins_kndfa_over_dfa": int((sub["ndfa_random_kronecker"] > sub["dfa_random"]).sum()),
                "wins_dfa": int((sub["best_ndfa"] > sub["dfa_random"]).sum()),
                "wins_bp": int((sub["best_ndfa"] > sub["bp"]).sum()),
                "n_cells": int(sub.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_synthetic_noise.csv", index=False)
    return out


def vision_summary() -> pd.DataFrame:
    df = read_csv(RESULTS / "infodfa_vision_noise_sweep_aggregate_v2" / "dfa_nmnc_best_by_method.csv")
    cell_cols = ["dataset", "n_train", "label_noise"]
    wide = df.pivot_table(index=cell_cols, columns="method", values="test_mean", aggfunc="max").reset_index()
    wide["best_ndfa"] = wide[["ndfa_random", "ndfa_random_kronecker"]].max(axis=1)
    wide["best_nc"] = wide[["vnc", "nmnc"]].max(axis=1)
    rows = []
    for dataset in ["fashion_mnist", "cifar10"]:
        sub = wide[wide["dataset"] == dataset]
        rows.append(
            {
                "dataset": dataset,
                "bp": sub["bp"].mean(),
                "dfa": sub["dfa_random"].mean(),
                "ndfa": sub["ndfa_random"].mean(),
                "kndfa": sub["ndfa_random_kronecker"].mean(),
                "best_ndfa": sub["best_ndfa"].mean(),
                "best_nc": sub["best_nc"].mean(),
                "delta_ndfa_dfa": (sub["ndfa_random"] - sub["dfa_random"]).mean(),
                "delta_kndfa_dfa": (sub["ndfa_random_kronecker"] - sub["dfa_random"]).mean(),
                "delta_ndfa_bp": (sub["best_ndfa"] - sub["bp"]).mean(),
                "wins_ndfa_over_dfa": int((sub["ndfa_random"] > sub["dfa_random"]).sum()),
                "wins_kndfa_over_dfa": int((sub["ndfa_random_kronecker"] > sub["dfa_random"]).sum()),
                "wins_dfa": int((sub["best_ndfa"] > sub["dfa_random"]).sum()),
                "wins_bp": int((sub["best_ndfa"] > sub["bp"]).sum()),
                "n_cells": int(sub.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_vision_noise.csv", index=False)
    return out


def hard_cifar_summary() -> pd.DataFrame:
    df = read_csv(RESULTS / "infodfa_hard_cifar100_confirm_aggregate_v2" / "dfa_convnet_summary.csv")
    order = ["bp", "local_loss", "ndfa_random", "ndfa_random_kronecker", "dfa_random", "drtp_random"]
    df = df.set_index("method").loc[order].reset_index()
    dfa = float(df.loc[df["method"] == "dfa_random", "test_mean"].iloc[0])
    df["delta_vs_dfa"] = df["test_mean"] - dfa
    df.to_csv(REPORT_DIR / "table_infodfa_hard_cifar.csv", index=False)
    return df


def imagenet_summary() -> pd.DataFrame:
    blend = read_csv(RESULTS / "imagenet100_infodfa_blend_diag_aggregate_20260525" / "imagenet_credit_assignment_summary.csv")
    stage = read_csv(RESULTS / "imagenet100_infodfa_stage_diag_aggregate_20260525" / "imagenet_credit_assignment_summary.csv")
    head = read_csv(RESULTS / "imagenet100_infodfa_spatial_headtohead90_aggregate_20260526" / "imagenet_credit_assignment_summary.csv")

    bp = float(
        blend[
            (blend["method"] == "block_dfa")
            & np.isclose(pd.to_numeric(blend["feedback_blend_gamma"], errors="coerce"), 0.0)
        ]["val_top1_mean"].iloc[0]
    )
    rows = [
        {
            "setting": "BP reference",
            "rule": "BP",
            "blocks": "all",
            "top1": bp,
            "sem": 0.0,
            "gap": 0.0,
            "reading": "Calibrated fine-tuning reference.",
        }
    ]

    for blocks, label, reading in [
        ("layer4", "layer4", "Late-stage DFA is nearly free."),
        ("layer3,layer4", "layer3+4", "Performance falls as substitution gets deeper."),
        ("layer1,layer2,layer3,layer4", "all blocks", "All-layer substitution fails badly."),
    ]:
        sub = head[
            (head["method"] == "block_dfa")
            & (head["feedback_modules"] == blocks)
            & (head["feedback_spatial_mode"] == "broadcast")
        ]
        if not sub.empty:
            mean = float(sub["val_top1_mean"].iloc[0])
            sem = float(sub["val_top1_sem"].iloc[0])
            rows.append(
                {
                    "setting": "90-epoch broadcast",
                    "rule": "block-DFA",
                    "blocks": label,
                    "top1": mean,
                    "sem": sem,
                    "gap": mean - bp,
                    "reading": reading,
                }
            )

    sub = stage[
        (stage["method"] == "block_dfa")
        & (stage["feedback_modules"] == "layer2,layer3,layer4")
        & np.isclose(pd.to_numeric(stage["feedback_scale"], errors="coerce"), 0.3)
    ]
    if not sub.empty:
        mean = float(sub["val_top1_mean"].iloc[0])
        rows.insert(
            3,
            {
                "setting": "60-epoch stage diag",
                "rule": "block-DFA",
                "blocks": "layer2+3+4",
                "top1": mean,
                "sem": 0.0,
                "gap": mean - bp,
                "reading": "Intermediate point on the depth cliff.",
            },
        )

    sub = stage[
        (stage["method"] == "block_ndfa")
        & (stage["feedback_modules"] == "layer1,layer2,layer3,layer4")
        & np.isclose(pd.to_numeric(stage["feedback_scale"], errors="coerce"), 0.3)
    ]
    if not sub.empty:
        mean = float(sub["val_top1_mean"].iloc[0])
        rows.append(
            {
                "setting": "60-epoch stage diag",
                "rule": "block-nDFA",
                "blocks": "all blocks",
                "top1": mean,
                "sem": 0.0,
                "gap": mean - bp,
                "reading": "Diagonal whitening does not fix ImageNet all-layer DFA.",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "table_infodfa_imagenet_boundary.csv", index=False)
    return out


def write_synthetic_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                CONDITION_LABELS[row["condition"]],
                pct(row["bp"]),
                pct(row["dfa"]),
                pct(row["best_ndfa"]),
                pct(row["best_nc"]),
                pp(row["delta_ndfa_dfa"]),
                pp(row["delta_ndfa_bp"]),
                f"{int(row['wins_dfa'])}/{int(row['n_cells'])}, {int(row['wins_bp'])}/{int(row['n_cells'])}",
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_synthetic_noise.tex",
        caption=(
            "Final synthetic stress-suite summary. Values are mean test accuracy across noise/sample cells. "
            "Best nDFA/K-nDFA compares the stronger of activity-only and Kronecker conditioning per cell; "
            "best NC compares VNC and NMNC. The final column reports wins over DFA and BP."
        ),
        label="tab:infodfa_synthetic_noise",
        headers=["Regime", "BP", "DFA", "Best nDFA/K", "Best NC", "$\\Delta$ vs DFA", "$\\Delta$ vs BP", "Wins DFA/BP"],
        rows=rows,
    )


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


def write_vision_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                DATASET_LABELS[row["dataset"]],
                pct(row["bp"]),
                pct(row["dfa"]),
                pct(row["best_ndfa"]),
                pct(row["best_nc"]),
                pp(row["delta_ndfa_dfa"]),
                pp(row["delta_ndfa_bp"]),
                f"{int(row['wins_dfa'])}/{int(row['n_cells'])}, {int(row['wins_bp'])}/{int(row['n_cells'])}",
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_vision_noise.tex",
        caption=(
            "Final noisy-label vision MLP sweep. nDFA/K-nDFA consistently rescue raw DFA, "
            "but BP remains the most frequent winner on these standard vision cells."
        ),
        label="tab:infodfa_vision_noise",
        headers=["Dataset", "BP", "DFA", "Best nDFA/K", "Best NC", "$\\Delta$ vs DFA", "$\\Delta$ vs BP", "Wins DFA/BP"],
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
        top1 = f"{row['top1']:.2f}"
        if float(row["sem"]) > 0:
            top1 = f"{row['top1']:.2f} $\\pm$ {row['sem']:.2f}"
        rows.append(
            [
                row["setting"],
                row["rule"],
                row["blocks"],
                top1,
                pp_percent(row["gap"]),
                row["reading"],
            ]
        )
    write_table(
        TABLE_DIR / "table_infodfa_imagenet_boundary.tex",
        caption=(
            "ImageNet-100 boundary evidence. Late-stage direct feedback is close to BP, "
            "but deeper substitution creates a large accuracy cliff and diagonal nDFA does not repair it."
        ),
        label="tab:infodfa_imagenet_boundary",
        headers=["Setting", "Rule", "Blocks", "Top-1", "$\\Delta$ vs BP", "Reading"],
        rows=rows,
        align="@{}lllrrp{0.31\\textwidth}@{}",
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
) -> None:
    lines = [
        "# Info-DFA final result summary",
        "",
        "Generated from the final v2 aggregate CSVs.",
        "",
        "## Completed result sets",
        "",
        "- Synthetic noise/sample sweep v2: 128 cells, 8 methods per cell.",
        "- Vision noisy-label MLP sweep v2: Fashion-MNIST and CIFAR-10, 24 cells, 8 methods per cell.",
        "- Hard CIFAR-100 convnet confirmation v2: BP/local/DFA/nDFA/K-nDFA/DRTP comparison.",
        "- ImageNet-100 diagnostics: substitution depth, blend, stage, and 90-epoch spatial head-to-head.",
        "",
        "## Synthetic stress suite",
        "",
        "| regime | BP | DFA | best nDFA/K | best NC | gain vs DFA | gain vs BP | wins over DFA/BP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in synthetic.iterrows():
        lines.append(
            f"| {CONDITION_LABELS[row['condition']]} | {pct(row['bp'])} | {pct(row['dfa'])} | "
            f"{pct(row['best_ndfa'])} | {pct(row['best_nc'])} | {pp(row['delta_ndfa_dfa'])} | "
            f"{pp(row['delta_ndfa_bp'])} | {int(row['wins_dfa'])}/{int(row['n_cells'])}, {int(row['wins_bp'])}/{int(row['n_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## Vision MLP sweep",
            "",
            "| dataset | BP | DFA | best nDFA/K | best NC | gain vs DFA | gain vs BP | wins over DFA/BP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in vision.iterrows():
        lines.append(
            f"| {DATASET_LABELS[row['dataset']]} | {pct(row['bp'])} | {pct(row['dfa'])} | "
            f"{pct(row['best_ndfa'])} | {pct(row['best_nc'])} | {pp(row['delta_ndfa_dfa'])} | "
            f"{pp(row['delta_ndfa_bp'])} | {int(row['wins_dfa'])}/{int(row['n_cells'])}, {int(row['wins_bp'])}/{int(row['n_cells'])} |"
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
            "| setting | rule | blocks | top-1 | gap vs BP | reading |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for _, row in imagenet.iterrows():
        sem = f" +/- {row['sem']:.2f}" if float(row["sem"]) > 0 else ""
        lines.append(
            f"| {row['setting']} | {row['rule']} | {row['blocks']} | {row['top1']:.2f}{sem} | "
            f"{pp_percent(row['gap'])} | {row['reading']} |"
        )
    lines.append("")
    (REPORT_DIR / "infodfa_final_result_summary.md").write_text("\n".join(lines))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}"


def pp(value: float) -> str:
    return f"{100.0 * float(value):+.1f}"


def pp_percent(value: float) -> str:
    return f"{float(value):+.2f}"


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
