#!/usr/bin/env python
"""Summarize the current Info-DFA experiment state.

The script is intentionally lightweight: it reads the aggregate CSV/Markdown
outputs already produced by the experiment pipeline and writes a compact audit
report for paper editing and job review.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/infodfa_experiment_audit_v1")


@dataclass(frozen=True)
class ResultFile:
    label: str
    path: Path
    role: str


RESULT_FILES = [
    ResultFile(
        "multioutput_main",
        Path("results/dfa_multioutput_synthetic_aggregate_v1/dfa_multioutput_best_by_method.csv"),
        "Eight-class stress suite; main K-nDFA/nDFA result.",
    ),
    ResultFile(
        "multioutput_diagnostics",
        Path("results/dfa_multioutput_synthetic_aggregate_v1/dfa_multioutput_diagnostic_tests.csv"),
        "Projected-step, rank, and nuisance-geometry diagnostic tests.",
    ),
    ResultFile(
        "multioutput_extra",
        Path("results/dfa_multioutput_extra_baselines_aggregate_v1/dfa_multioutput_best_by_method.csv"),
        "Stress suite with FA/VNC/NMNC/DRTP comparison baselines.",
    ),
    ResultFile(
        "nmnc_vision_mlp",
        Path("results/dfa_nmnc_aggregate_v1/dfa_nmnc_best_by_method.csv"),
        "MNIST/Fashion/CIFAR-10 MLP comparison with NMNC/VNC.",
    ),
    ResultFile(
        "kndfa_blend",
        Path("results/dfa_preconditioning_spectrum_aggregate_v1/dfa_preconditioning_spectrum_summary.csv"),
        "Raw-DFA to K-nDFA interpolation and damping sweep.",
    ),
    ResultFile(
        "rank_conditioning",
        Path("results/infodfa_rank_conditioning_v1/infodfa_rank_conditioning_report.md"),
        "Publication-facing rank/conditioning summary.",
    ),
    ResultFile(
        "convnet_extra",
        Path("results/dfa_convnet_extra_baselines_aggregate_v1/dfa_convnet_best_by_method.csv"),
        "CIFAR-10/CIFAR-100 stride-conv baselines.",
    ),
    ResultFile(
        "convnet_conditioning",
        Path("results/dfa_convnet_conditioning_aggregate_v1/dfa_convnet_best_by_method.csv"),
        "Full-data convnet rank/damping conditioning sweep.",
    ),
    ResultFile(
        "convnet_cifar100_harder",
        Path("results/dfa_convnet_cifar100_harder_matched_bp_aggregate_v1/dfa_convnet_best_by_method.csv"),
        "Harder CIFAR-100 convnet with matched BP learning-rate controls.",
    ),
    ResultFile(
        "imagenet_resnet50_bp",
        Path("results/imagenet_sota_resnet50_bp_aggregate_v1/imagenet_credit_assignment_summary.csv"),
        "ImageNet-1K ResNet-50 BP reference.",
    ),
    ResultFile(
        "imagenet_timm_resnet50d_bp",
        Path("results/imagenet_sota_timm_resnet50d_bp_aggregate_v1/imagenet_credit_assignment_summary.csv"),
        "ImageNet-1K timm ResNet-50d BP reference.",
    ),
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def _method_label(method: object) -> str:
    text = str(method)
    return {
        "bp": "BP",
        "dfa_random": "DFA",
        "ndfa_random": "nDFA",
        "ndfa_random_kronecker": "K-nDFA",
        "drtp_random": "DRTP",
        "fa_random": "FA",
        "vnc": "VNC",
        "nmnc": "NMNC",
        "local_loss": "local aux",
    }.get(text, text)


def _best_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    cols = [group_col, "method", "test_mean", "test_sem", "n"]
    optional = [
        "feedback_rank",
        "natural_damping",
        "nc_update_interval",
        "nc_manifold_rank",
        "lr",
        "feedback_mode",
        "feedback_normalization",
        "hidden_projected_step",
        "projected_step",
    ]
    cols.extend([col for col in optional if col in df.columns])
    out = df[cols].copy()
    out["method"] = out["method"].map(_method_label)
    return out.sort_values([group_col, "test_mean"], ascending=[True, False])


def _summarize_multoutput(md: list[str], root: Path) -> None:
    best = _read_csv(root / "results/dfa_multioutput_synthetic_aggregate_v1/dfa_multioutput_best_by_method.csv")
    diag = _read_csv(root / "results/dfa_multioutput_synthetic_aggregate_v1/dfa_multioutput_diagnostic_tests.csv")
    if best is None:
        return
    md.append("## Multi-Output Stress Suite")
    winners = (
        best.sort_values(["condition", "test_mean"], ascending=[True, False])
        .groupby("condition")
        .head(3)
    )
    for _, row in winners.iterrows():
        md.append(
            f"- {row['condition']}: {_method_label(row['method'])} "
            f"{_fmt_float(row['test_mean'])} +/- {_fmt_float(row['test_sem'])} "
            f"(rank={_fmt_float(row.get('effective_feedback_rank'), 0)}, n={int(row['n'])})."
        )
    if diag is not None:
        md.append("")
        md.append("Diagnostic checks:")
        for _, row in diag.iterrows():
            test = str(row.get("test", ""))
            if test in {
                "projected BP-step ratio > 0.1",
                "nuisance energy dominates design",
                "rank sufficient for full output-error space",
            }:
                md.append(
                    f"- {test}: mean_low={_fmt_float(row.get('mean_low'))}, "
                    f"mean_high={_fmt_float(row.get('mean_high'))}, "
                    f"delta={_fmt_float(row.get('delta_high_minus_low'))}."
                )
    md.append("")


def _summarize_best_by_method(md: list[str], root: Path, title: str, path: Path, group_col: str) -> None:
    df = _read_csv(root / path)
    if df is None:
        return
    md.append(f"## {title}")
    table = _best_table(df, group_col)
    keep_methods = {"BP", "DFA", "nDFA", "K-nDFA", "DRTP", "FA", "VNC", "NMNC", "local aux"}
    table = table[table["method"].isin(keep_methods)]
    for group, sub in table.groupby(group_col, sort=True):
        md.append(f"- {group}:")
        for _, row in sub.head(8).iterrows():
            extras = []
            for col in ("feedback_rank", "natural_damping", "lr", "nc_update_interval", "nc_manifold_rank"):
                if col in row and not pd.isna(row[col]):
                    extras.append(f"{col}={_fmt_float(row[col], 4 if col == 'lr' else 1)}")
            md.append(
                f"  - {row['method']}: {_fmt_float(row['test_mean'])} "
                f"+/- {_fmt_float(row['test_sem'])} (n={int(row['n'])}"
                + (", " + ", ".join(extras) if extras else "")
                + ")."
            )
    md.append("")


def _summarize_imagenet(md: list[str], root: Path) -> None:
    md.append("## ImageNet Status")
    resnet18_root = root / "results/imagenet_sota_resnet18_credit_v1/resnet18"
    rows = []
    smoke_rows = []
    for csv_path in sorted(resnet18_root.glob("*/lr_*/scale_*/damping_*/seed_*/fb_*/imagenet_credit_assignment.csv")):
        df = pd.read_csv(csv_path)
        method = csv_path.relative_to(resnet18_root).parts[0]
        last = df.tail(1).iloc[0]
        rows.append(
            {
                "arch": "resnet18",
                "method": method,
                "epochs_recorded": int(df["epoch"].max()) if "epoch" in df else len(df),
                "val_top1": float(last.get("val_top1", float("nan"))),
                "val_top5": float(last.get("val_top5", float("nan"))),
            }
        )
    smoke_root = root / "results/imagenet_block_feedback_smoke_20260522/resnet18"
    for csv_path in sorted(smoke_root.glob("*/lr_*/scale_*/damping_*/seed_*/fb_*/imagenet_credit_assignment.csv")):
        df = pd.read_csv(csv_path)
        method = csv_path.relative_to(smoke_root).parts[0]
        last = df.tail(1).iloc[0]
        smoke_rows.append(
            {
                "arch": "resnet18",
                "method": method,
                "epochs_recorded": int(df["epoch"].max()) if "epoch" in df else len(df),
                "val_top1": float(last.get("val_top1", float("nan"))),
                "val_top5": float(last.get("val_top5", float("nan"))),
            }
        )
    for path in [
        root / "results/imagenet_sota_resnet50_bp_aggregate_v1/imagenet_credit_assignment_summary.csv",
        root / "results/imagenet_sota_timm_resnet50d_bp_aggregate_v1/imagenet_credit_assignment_summary.csv",
    ]:
        df = _read_csv(path)
        if df is None:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "arch": row["arch"],
                    "method": row["method"],
                    "epochs_recorded": -1,
                    "val_top1": float(row["val_top1_mean"]),
                    "val_top5": float(row["val_top5_mean"]),
                }
            )
    if rows:
        image_df = pd.DataFrame(rows)
        for _, row in image_df.sort_values(["arch", "method"]).iterrows():
            epoch_text = "aggregate" if row["epochs_recorded"] < 0 else f"epoch {int(row['epochs_recorded'])}"
            md.append(
                f"- {row['arch']} {row['method']} ({epoch_text}): "
                f"top1={_fmt_float(row['val_top1'])}, top5={_fmt_float(row['val_top5'])}."
            )
        if smoke_rows:
            md.append("")
            md.append("Implementation smoke checks:")
            smoke_df = pd.DataFrame(smoke_rows)
            for _, row in smoke_df.sort_values(["arch", "method"]).iterrows():
                md.append(
                    f"- {row['arch']} {row['method']} smoke (epoch {int(row['epochs_recorded'])}, "
                    f"subsampled): top1={_fmt_float(row['val_top1'])}, top5={_fmt_float(row['val_top5'])}."
                )
            md.append(
                "- Smoke values are not accuracy evidence; they show the repaired block-feedback path "
                "can complete training steps beyond initialization."
            )
        md.append(
            "- Interpretation: ImageNet currently supports BP/local-aux references, "
            "but not a completed full-length block-DFA or block-nDFA scaling claim."
        )
    md.append("")


def _summarize_slurm(md: list[str], root: Path) -> None:
    logdir = root / "logs/slurm"
    if not logdir.exists():
        return
    md.append("## Slurm Log Audit")
    aggregate: dict[tuple[str, str], dict[str, int]] = {}
    examples: dict[tuple[str, str], list[str]] = {}
    for out_path in logdir.glob("*.out"):
        name = out_path.name
        if not (name.startswith("dfa") or name.startswith("imgnet") or name.startswith("infodfa")):
            continue
        match = re.match(r"(.+?)_(\d{8})(?:_(\d+))?\.out$", name)
        if not match:
            continue
        key = (match.group(1), match.group(2))
        err_path = logdir / f"{out_path.stem}.err"
        out_tail = out_path.read_text(errors="ignore")[-2000:]
        err_text = err_path.read_text(errors="ignore") if err_path.exists() else ""
        has_err = bool(err_text.strip())
        keyword_bad = bool(
            re.search(
                r"(?i)(traceback|exception|runtimeerror|outofmemory|cuda out of memory|"
                r"childfailederror|cancelled|killed|time limit|failed \(exitcode)",
                out_tail + "\n" + err_text,
            )
        )
        rec = aggregate.setdefault(key, {"outs": 0, "nonempty_err": 0, "keyword_bad": 0})
        rec["outs"] += 1
        rec["nonempty_err"] += int(has_err)
        rec["keyword_bad"] += int(keyword_bad)
        if (has_err or keyword_bad) and len(examples.setdefault(key, [])) < 2:
            examples[key].append(name)
    for (job_name, job_id), rec in sorted(aggregate.items(), key=lambda item: (int(item[0][1]), item[0][0])):
        if job_name.startswith(("dfa_synth", "dfa_multiout", "dfa_nmnc", "dfa_kndfa", "dfa_conv", "dfa_imgnet", "imgnet")):
            md.append(
                f"- {job_name} {job_id}: outs={rec['outs']}, "
                f"nonempty_err={rec['nonempty_err']}, keyword_bad={rec['keyword_bad']}"
                + (f" examples={', '.join(examples.get((job_name, job_id), []))}" if examples.get((job_name, job_id)) else "")
            )
    md.append("")


def write_audit(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    for item in RESULT_FILES:
        path = root / item.path
        row = {
            "label": item.label,
            "path": str(item.path),
            "exists": path.exists(),
            "role": item.role,
            "rows": None,
            "columns": None,
        }
        if path.suffix == ".csv" and path.exists():
            df = pd.read_csv(path)
            row["rows"] = len(df)
            row["columns"] = len(df.columns)
        inventory_rows.append(row)
    pd.DataFrame(inventory_rows).to_csv(output_dir / "infodfa_experiment_inventory.csv", index=False)

    md: list[str] = [
        "# Info-DFA Experiment Audit",
        "",
        "Generated from aggregate outputs and Slurm logs.",
        "",
        "## Result Inventory",
    ]
    for row in inventory_rows:
        status = "present" if row["exists"] else "missing"
        detail = ""
        if row["rows"] is not None:
            detail = f", rows={row['rows']}, columns={row['columns']}"
        md.append(f"- {row['label']}: {status}{detail}. {row['role']}")
    md.append("")

    _summarize_multoutput(md, root)
    _summarize_best_by_method(
        md,
        root,
        "Vision MLP / NMNC Comparison",
        Path("results/dfa_nmnc_aggregate_v1/dfa_nmnc_best_by_method.csv"),
        "dataset",
    )
    _summarize_best_by_method(
        md,
        root,
        "Convolutional CIFAR Baselines",
        Path("results/dfa_convnet_extra_baselines_aggregate_v1/dfa_convnet_best_by_method.csv"),
        "dataset",
    )
    _summarize_best_by_method(
        md,
        root,
        "Harder CIFAR-100 Matched-BP Convnet",
        Path("results/dfa_convnet_cifar100_harder_matched_bp_aggregate_v1/dfa_convnet_best_by_method.csv"),
        "dataset",
    )
    _summarize_imagenet(md, root)
    _summarize_slurm(md, root)

    md.extend(
        [
            "## Paper-Level Reading",
            "- Strongest supported claim: local feedback works when the hidden update has a non-trivial projected BP-step component and the task geometry is not swamped by nuisance variation.",
            "- Strongest method result: nDFA/K-nDFA are robust in the eight-class stress suite and competitive on MNIST/Fashion/CIFAR-10 MLPs.",
            "- Convnet result: channel-tied, normalized feedback is necessary; local auxiliary losses are a strong control; nDFA/K-nDFA narrow the BP gap but do not uniformly dominate all local controls.",
            "- ImageNet result: keep as reference/calibration only until matched full-length block-DFA/block-nDFA runs complete under the same optimized recipe.",
            "",
        ]
    )
    (output_dir / "infodfa_experiment_audit.md").write_text("\n".join(md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_audit(args.root.resolve(), args.output_dir)
    print(f"Wrote {args.output_dir / 'infodfa_experiment_audit.md'}")
    print(f"Wrote {args.output_dir / 'infodfa_experiment_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
