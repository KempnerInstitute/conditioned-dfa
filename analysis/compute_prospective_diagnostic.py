"""Prospective (pre-training) nuisance-energy diagnostic for conditioned DFA.

The paper's mechanism variable ``nuisance_energy_ratio`` is computed from the
synthetic generator's ground truth, so it cannot be used to decide, before
training, whether conditioning (nDFA) will rescue DFA on a new problem. This
script estimates the same quantity from information a practitioner actually has
before training: the training inputs, the (possibly noisy) training labels, and
a randomly initialized network.

Estimator, per presynaptic layer that nDFA preconditions (the input to each
hidden weight matrix, i.e. the raw input and each hidden ReLU activity at
init):

1. ``C`` = centered covariance of the layer activity over the training set.
2. Task subspace ``U`` = span of the centered class-conditional mean
   activations (between-class scatter eigenvectors; rank <= n_classes - 1).
3. Estimated nuisance-energy ratio = (tr C - tr(U^T C U)) / tr(U^T C U),
   the activity variance orthogonal to the estimated task subspace relative to
   the variance inside it. This is the empirical analog of the designed
   ``nuisance_energy_ratio`` (task-irrelevant / task-relevant energy).

Layers are combined by geometric mean; seeds (data seed = model-init seed,
matching the sweep protocol in ``slurm/infodfa_multioutput_noise_sweep.sbatch``)
by geometric mean as well. Two pre-registered variants and one baseline are
recorded alongside the primary estimator:

- ``ratio_bw``: between/within decomposition (tr C - tr B) / tr B with B the
  between-class scatter (energy weighted by class-mean separation, not just by
  the span);
- ``nuisance_topk``: fraction of the top-k (k=8) spectral energy of C that is
  orthogonal to the task subspace ("are the high-variance directions
  nuisance?");
- ``kappa``: damped condition number of C (damping 0.3, matching the runner's
  spectrum diagnostics) -- an anisotropy-only baseline with no task
  information, included to show the task projection matters.

Validation targets are the realized per-cell nDFA - DFA gains from the frozen
noise-sweep aggregates (the same best-by-method selection used by the paper
figures), plus the vision noise sweep (Fashion-MNIST / CIFAR-10 MLPs) as the
transfer test. Everything runs on CPU.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_multioutput_synthetic import make_multioutput_dataset  # noqa: E402
from experiments.run_dfa_nmnc_comparison import corrupt_labels as corrupt_labels_vision  # noqa: E402
from experiments.run_dfa_vision_baselines import load_vision_dataset  # noqa: E402
from infogeo.dfa import ManualMLP  # noqa: E402
from infogeo.analysis import dataframe_to_markdown, write_markdown_report  # noqa: E402


# Sweep protocol constants (slurm/infodfa_multioutput_noise_sweep.sbatch).
SYNTH_CONDITIONS = ["nuisance_dominant", "mixed_context", "low_sample_noisy", "task_aligned"]
SYNTH_N_TRAINS = [512, 1024, 2048, 4096]
SYNTH_LABEL_NOISES = [0.0, 0.1, 0.2, 0.4]
SYNTH_INPUT_NOISES = [0.05, 0.15]
SYNTH_INPUT_DIM = 64
SYNTH_N_CLASSES = 8
SYNTH_NUISANCE_DIM = 24
SYNTH_HIDDEN_DIMS = [256, 128]
SYNTH_N_SEEDS = 5

# Vision protocol constants (slurm/infodfa_vision_noise_sweep.sbatch).
VISION_DATASETS = ["fashion_mnist", "cifar10"]
VISION_N_TRAINS = [1000, 3000, 10000]
VISION_LABEL_NOISES = [0.0, 0.1, 0.2, 0.4]
VISION_HIDDEN = {"fashion_mnist": [512, 256], "cifar10": [1024, 512]}
VISION_N_TEST = {"fashion_mnist": 2000, "cifar10": 3000}
VISION_N_SEEDS = 5

DAMPING = 0.3  # natural-damping used by the runners' spectrum diagnostics
TOPK = 8
GAIN_THRESHOLD_PP = 5.0  # "conditioning helps" definition for the decision rule
# Primary estimator used for the figure, decision rule, and vision transfer.
# Selected on the synthetic tier only; the vision sweep stays a held-out test.
PRIMARY_PREDICTOR = "est_nuisance_ratio"

# House style (drafts/Info-DFA/scripts/make_iclr_figures.py).
REGIME_COLORS = {
    "nuisance_dominant": "#882255",
    "mixed_context": "#999933",
    "low_sample_noisy": "#332288",
    "task_aligned": "#CC6677",
}
REGIME_LABELS = {
    "nuisance_dominant": "Nuisance",
    "mixed_context": "Mixed",
    "low_sample_noisy": "Low sample",
    "task_aligned": "Task aligned",
}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(int(args.threads), 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synth = compute_synthetic_diagnostics(args)
    synth = attach_synthetic_gains(synth, Path(args.legacy_results))
    synth.to_csv(output_dir / "prospective_diagnostic_cells.csv", index=False)

    correlations = correlation_table(synth)
    correlations.to_csv(output_dir / "prospective_diagnostic_correlations.csv", index=False)

    rule_tables = []
    rules = {}
    for predictor in ("est_nuisance_ratio", "est_ratio_bw", "est_kappa"):
        rules[predictor] = decision_rule_analysis(synth, predictor=predictor)
        rule_tables.append(rules[predictor]["table"])
    rule = rules[PRIMARY_PREDICTOR]
    pd.concat(rule_tables, ignore_index=True).to_csv(output_dir / "prospective_decision_rule_loro.csv", index=False)

    vision = pd.DataFrame()
    vision_summary = pd.DataFrame()
    if not args.skip_vision:
        vision = compute_vision_diagnostics(args)
        vision = attach_vision_gains(vision, Path(args.legacy_results))
        vision.to_csv(output_dir / "prospective_diagnostic_vision.csv", index=False)
        vision_summary = vision_transfer_table(vision, rule["full_threshold_log10"])
        vision_summary.to_csv(output_dir / "prospective_vision_transfer.csv", index=False)

    make_figure(synth, vision, rule, output_dir)
    write_summary(synth, correlations, rule, vision, vision_summary, output_dir)
    print(f"Wrote prospective diagnostic outputs to {output_dir}")


# ---------------------------------------------------------------------------
# Diagnostic estimator
# ---------------------------------------------------------------------------


@dataclass
class LayerDiagnostic:
    ratio_span: float
    ratio_bw: float
    nuisance_topk: float
    kappa: float


def layer_diagnostic(activity: np.ndarray, labels: np.ndarray, *, damping: float = DAMPING, topk: int = TOPK) -> LayerDiagnostic:
    """Estimated nuisance-energy diagnostics for one layer's activity at init."""

    h = np.asarray(activity, dtype=np.float64)
    y = np.asarray(labels)
    n, dim = h.shape
    eps = 1e-12
    mean = h.mean(axis=0, keepdims=True)
    hc = h - mean
    cov = hc.T @ hc / max(n - 1, 1)
    total = float(np.trace(cov))

    # Between-class scatter of centered class-conditional means.
    classes = np.unique(y)
    between = np.zeros((dim, dim))
    for cls in classes:
        mask = y == cls
        weight = float(mask.mean())
        mc = hc[mask].mean(axis=0)
        between += weight * np.outer(mc, mc)
    evals_b, evecs_b = np.linalg.eigh(between)
    keep = evals_b > max(float(evals_b.max()), eps) * 1e-9
    max_rank = max(len(classes) - 1, 1)
    order = np.argsort(evals_b)[::-1]
    selected = [idx for idx in order if keep[idx]][:max_rank]
    basis = evecs_b[:, selected] if selected else np.zeros((dim, 1))

    task_energy = float(np.trace(basis.T @ cov @ basis))
    ratio_span = (total - task_energy) / max(task_energy, eps)
    tr_between = float(np.trace(between))
    ratio_bw = (total - tr_between) / max(tr_between, eps)

    evals_c, evecs_c = np.linalg.eigh(cov)
    order_c = np.argsort(evals_c)[::-1]
    evals_c = np.clip(evals_c[order_c], 0.0, None)
    evecs_c = evecs_c[:, order_c]
    k = min(topk, dim)
    proj = basis.T @ evecs_c[:, :k]  # (task_rank, k)
    inside = np.sum(proj * proj, axis=0)
    top_energy = float(np.sum(evals_c[:k]))
    nuisance_topk = float(np.sum(evals_c[:k] * (1.0 - inside)) / max(top_energy, eps))
    kappa = float((evals_c[0] + damping) / (evals_c[-1] + damping))
    return LayerDiagnostic(ratio_span, ratio_bw, nuisance_topk, kappa)


def model_diagnostics(model: ManualMLP, x: torch.Tensor, y: np.ndarray) -> dict[str, float]:
    """Geometric-mean diagnostics over the presynaptic layers nDFA preconditions."""

    with torch.no_grad():
        _, activations, _ = model.forward(x)
    per_layer: list[LayerDiagnostic] = []
    out: dict[str, float] = {}
    for layer_idx in range(model.n_hidden_layers):
        act = activations[layer_idx].detach().cpu().numpy()
        diag = layer_diagnostic(act, y)
        per_layer.append(diag)
        out[f"ratio_span_l{layer_idx}"] = diag.ratio_span
        out[f"kappa_l{layer_idx}"] = diag.kappa
    eps = 1e-12
    out["est_nuisance_ratio"] = float(np.exp(np.mean([np.log(max(d.ratio_span, eps)) for d in per_layer])))
    out["est_ratio_bw"] = float(np.exp(np.mean([np.log(max(d.ratio_bw, eps)) for d in per_layer])))
    out["est_nuisance_topk"] = float(np.mean([d.nuisance_topk for d in per_layer]))
    out["est_kappa"] = float(np.exp(np.mean([np.log(max(d.kappa, eps)) for d in per_layer])))
    return out


# ---------------------------------------------------------------------------
# Synthetic tier (128 cells x 5 seeds)
# ---------------------------------------------------------------------------


def compute_synthetic_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    cells = [
        (condition, n_train, label_noise, input_noise)
        for condition in SYNTH_CONDITIONS
        for n_train in SYNTH_N_TRAINS
        for label_noise in SYNTH_LABEL_NOISES
        for input_noise in SYNTH_INPUT_NOISES
    ]
    print(f"Computing prospective diagnostics for {len(cells)} synthetic cells", flush=True)
    for cell_idx, (condition, n_train, label_noise, input_noise) in enumerate(cells, start=1):
        seed_rows = []
        for seed in range(args.n_seeds):
            dataset = make_multioutput_dataset(
                condition=condition,
                n_train=n_train,
                n_test=8,  # test split unused; keep generation cheap without touching train draws
                input_dim=SYNTH_INPUT_DIM,
                n_classes=SYNTH_N_CLASSES,
                nuisance_dim=SYNTH_NUISANCE_DIM,
                input_noise=input_noise,
                train_label_noise=label_noise,
                test_label_noise=0.0,
                task_scale_override=None,
                nuisance_scale_override=None,
                seed=seed,
            )
            model = ManualMLP(
                input_dim=SYNTH_INPUT_DIM,
                hidden_dims=SYNTH_HIDDEN_DIMS,
                output_dim=SYNTH_N_CLASSES,
                seed=10_000 + seed,
                device="cpu",
            )
            x = torch.tensor(dataset.x_train, dtype=torch.float32)
            seed_rows.append(model_diagnostics(model, x, dataset.y_train))
        aggregated = aggregate_seed_rows(seed_rows)
        rows.append(
            {
                "condition": condition,
                "input_noise": float(input_noise),
                "n_train": float(n_train),
                "train_label_noise": float(label_noise),
                "n_seeds": float(args.n_seeds),
                **aggregated,
            }
        )
        if cell_idx % 16 == 0:
            print(f"  {cell_idx}/{len(cells)} cells", flush=True)
    return pd.DataFrame(rows)


def aggregate_seed_rows(seed_rows: list[dict[str, float]]) -> dict[str, float]:
    """Geometric mean across seeds for ratio-like keys, plus seed spread."""

    eps = 1e-12
    out: dict[str, float] = {}
    keys = seed_rows[0].keys()
    for key in keys:
        values = np.array([row[key] for row in seed_rows], dtype=float)
        if key.startswith("est_nuisance_topk"):
            out[key] = float(values.mean())
            out[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        else:
            logs = np.log10(np.clip(values, eps, None))
            out[key] = float(10.0 ** logs.mean())
            out[f"log10_{key}_sd"] = float(logs.std(ddof=1)) if len(logs) > 1 else 0.0
    return out


def attach_synthetic_gains(synth: pd.DataFrame, legacy_results: Path) -> pd.DataFrame:
    """Join the realized nDFA-DFA gains (paper's best-by-method selection)."""

    best = pd.read_csv(
        legacy_results / "infodfa_multioutput_noise_sweep_aggregate_v2" / "dfa_multioutput_best_by_method.csv"
    )
    cell_cols = ["condition", "input_noise", "n_train", "train_label_noise"]
    wide = best.pivot_table(index=cell_cols + ["nuisance_energy_ratio"], columns="method", values="test_mean", aggfunc="mean").reset_index()
    wide["gain_ndfa_pp"] = 100.0 * (wide["ndfa_random"] - wide["dfa_random"])
    wide["gain_kndfa_pp"] = 100.0 * (wide["ndfa_random_kronecker"] - wide["dfa_random"])
    wide = wide.dropna(subset=["gain_ndfa_pp"])
    merged = synth.merge(
        wide[cell_cols + ["nuisance_energy_ratio", "gain_ndfa_pp", "gain_kndfa_pp"]],
        on=cell_cols,
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"nuisance_energy_ratio": "designed_nuisance_ratio"})
    if merged["gain_ndfa_pp"].isna().any():
        missing = merged[merged["gain_ndfa_pp"].isna()][cell_cols]
        raise RuntimeError(f"Missing realized gains for cells:\n{missing}")
    return merged


# ---------------------------------------------------------------------------
# Vision tier (24 cells, transfer test)
# ---------------------------------------------------------------------------


def compute_vision_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset_name in VISION_DATASETS:
        for n_train in VISION_N_TRAINS:
            loader_args = argparse.Namespace(
                dataset=dataset_name,
                data_dir=args.data_dir,
                download=False,
                n_train=n_train,
                n_test=VISION_N_TEST[dataset_name],
                seed=0,
            )
            train_x, train_y, _, _ = load_vision_dataset(loader_args)
            for label_noise in VISION_LABEL_NOISES:
                noisy_y = corrupt_labels_vision(
                    train_y, n_classes=loader_args.n_classes, noise=label_noise, seed=loader_args.seed + 101
                )
                seed_rows = []
                for seed in range(VISION_N_SEEDS):
                    model = ManualMLP(
                        input_dim=train_x.shape[1],
                        hidden_dims=VISION_HIDDEN[dataset_name],
                        output_dim=loader_args.n_classes,
                        seed=10_000 + seed,
                        device="cpu",
                    )
                    seed_rows.append(model_diagnostics(model, train_x, noisy_y.numpy()))
                aggregated = aggregate_seed_rows(seed_rows)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "n_train": float(n_train),
                        "label_noise": float(label_noise),
                        "n_seeds": float(VISION_N_SEEDS),
                        **aggregated,
                    }
                )
                print(f"  vision {dataset_name} n_train={n_train} label_noise={label_noise}", flush=True)
    return pd.DataFrame(rows)


def attach_vision_gains(vision: pd.DataFrame, legacy_results: Path) -> pd.DataFrame:
    all_rows = pd.read_csv(legacy_results / "infodfa_vision_noise_sweep_aggregate_v2" / "dfa_nmnc_all.csv")
    final = all_rows.sort_values("epoch").groupby(
        ["dataset", "n_train", "label_noise", "method", "seed", "feedback_seed"], as_index=False
    ).tail(1)
    means = final.groupby(["dataset", "n_train", "label_noise", "method"])["test_acc"].mean().unstack("method")
    means["gain_ndfa_pp"] = 100.0 * (means["ndfa_random"] - means["dfa_random"])
    means["gain_kndfa_pp"] = 100.0 * (means["ndfa_random_kronecker"] - means["dfa_random"])
    means = means.reset_index()[["dataset", "n_train", "label_noise", "gain_ndfa_pp", "gain_kndfa_pp"]]
    return vision.merge(means, on=["dataset", "n_train", "label_noise"], how="left", validate="one_to_one")


def vision_transfer_table(vision: pd.DataFrame, threshold_log10: float) -> pd.DataFrame:
    rows = []
    x = np.log10(vision[PRIMARY_PREDICTOR].to_numpy(dtype=float))
    y = vision["gain_ndfa_pp"].to_numpy(dtype=float)
    rows.append(transfer_row("all 24 vision cells", x, y))
    for dataset_name, sub in vision.groupby("dataset"):
        xs = np.log10(sub[PRIMARY_PREDICTOR].to_numpy(dtype=float))
        rows.append(transfer_row(f"within {dataset_name} (12 cells)", xs, sub["gain_ndfa_pp"].to_numpy(dtype=float)))
        noise_rho = spearmanr(sub["label_noise"], sub["gain_ndfa_pp"]).statistic
        diag_noise_rho = spearmanr(sub["label_noise"], np.log10(sub[PRIMARY_PREDICTOR])).statistic
        rows.append(
            {
                "comparison": f"{dataset_name}: label-noise ordering (gain vs diagnostic)",
                "spearman": float(noise_rho),
                "pearson": float(diag_noise_rho),
                "n": int(len(sub)),
                "note": "spearman column: label_noise vs gain; pearson column: label_noise vs diagnostic",
            }
        )
    means = vision.groupby("dataset").agg(
        mean_gain=("gain_ndfa_pp", "mean"), mean_log10_diag=(PRIMARY_PREDICTOR, lambda v: float(np.log10(v).mean()))
    )
    ordering_ok = bool(
        (means.loc["fashion_mnist", "mean_gain"] > means.loc["cifar10", "mean_gain"])
        == (means.loc["fashion_mnist", "mean_log10_diag"] > means.loc["cifar10", "mean_log10_diag"])
    )
    rows.append(
        {
            "comparison": "dataset-level ordering (Fashion vs CIFAR-10)",
            "spearman": np.nan,
            "pearson": np.nan,
            "n": 2,
            "note": (
                f"gains: fashion {means.loc['fashion_mnist', 'mean_gain']:.1f}pp vs cifar {means.loc['cifar10', 'mean_gain']:.1f}pp; "
                f"log10 diagnostic: fashion {means.loc['fashion_mnist', 'mean_log10_diag']:.2f} vs cifar {means.loc['cifar10', 'mean_log10_diag']:.2f}; "
                f"ordering {'consistent' if ordering_ok else 'INCONSISTENT'}"
            ),
        }
    )
    above = (np.log10(vision[PRIMARY_PREDICTOR]) > threshold_log10).sum()
    helps = (vision["gain_ndfa_pp"] > GAIN_THRESHOLD_PP).sum()
    rows.append(
        {
            "comparison": "synthetic decision threshold applied to vision cells",
            "spearman": np.nan,
            "pearson": np.nan,
            "n": int(len(vision)),
            "note": f"{above}/{len(vision)} cells above threshold; {helps}/{len(vision)} cells with realized gain > {GAIN_THRESHOLD_PP:.0f}pp",
        }
    )
    return pd.DataFrame(rows)


def transfer_row(label: str, x: np.ndarray, y: np.ndarray) -> dict[str, float | str | int]:
    return {
        "comparison": label,
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
        "n": int(len(x)),
        "note": "log10 prospective diagnostic vs realized nDFA-DFA gain (pp)",
    }


# ---------------------------------------------------------------------------
# Validation: correlations and decision rule
# ---------------------------------------------------------------------------


PREDICTORS = [
    ("est_nuisance_ratio", "prospective nuisance-energy ratio (primary)", True),
    ("est_ratio_bw", "prospective between/within ratio (variant)", True),
    ("est_nuisance_topk", "prospective top-8 nuisance fraction (variant)", False),
    ("est_kappa", "kappa(C) anisotropy baseline (no task info)", True),
    ("designed_nuisance_ratio", "designed ratio (generator ground truth, post hoc)", True),
]


def correlation_table(synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label, log_scale in PREDICTORS:
        values = synth[column].to_numpy(dtype=float)
        x = np.log10(np.clip(values, 1e-12, None)) if log_scale else values
        y = synth["gain_ndfa_pp"].to_numpy(dtype=float)
        rows.append(
            {
                "predictor": column,
                "description": label,
                "scope": "all 128 cells",
                "spearman": float(spearmanr(x, y).statistic),
                "pearson": float(pearsonr(x, y).statistic),
                "n": int(len(synth)),
            }
        )
        for condition, sub in synth.groupby("condition"):
            xs = sub[column].to_numpy(dtype=float)
            xs = np.log10(np.clip(xs, 1e-12, None)) if log_scale else xs
            ys = sub["gain_ndfa_pp"].to_numpy(dtype=float)
            spearman = spearmanr(xs, ys).statistic if np.std(xs) > 0 else np.nan
            pearson = pearsonr(xs, ys).statistic if np.std(xs) > 0 else np.nan
            rows.append(
                {
                    "predictor": column,
                    "description": label,
                    "scope": f"within {condition}",
                    "spearman": float(spearman) if np.isfinite(spearman) else np.nan,
                    "pearson": float(pearson) if np.isfinite(pearson) else np.nan,
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def best_threshold(x: np.ndarray, helps: np.ndarray) -> float:
    """Threshold on x maximizing balanced accuracy of (x > t) vs helps."""

    order = np.argsort(x)
    xs = x[order]
    candidates = np.concatenate([[xs[0] - 1.0], (xs[:-1] + xs[1:]) / 2.0, [xs[-1] + 1.0]])
    best_t, best_score = candidates[0], -np.inf
    for t in candidates:
        pred = x > t
        score = balanced_accuracy(helps, pred)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    pos = truth.astype(bool)
    neg = ~pos
    tpr = float(pred[pos].mean()) if pos.any() else np.nan
    tnr = float((~pred[neg]).mean()) if neg.any() else np.nan
    parts = [v for v in (tpr, tnr) if np.isfinite(v)]
    return float(np.mean(parts)) if parts else np.nan


def decision_rule_analysis(synth: pd.DataFrame, *, predictor: str = PRIMARY_PREDICTOR) -> dict[str, object]:
    x = np.log10(np.clip(synth[predictor].to_numpy(dtype=float), 1e-12, None))
    helps = synth["gain_ndfa_pp"].to_numpy(dtype=float) > GAIN_THRESHOLD_PP
    conditions = synth["condition"].to_numpy()

    full_threshold = best_threshold(x, helps)
    full_pred = x > full_threshold
    rows = [
        {
            "predictor": predictor,
            "held_out_regime": "none (threshold fit on all 128 cells)",
            "threshold_log10": full_threshold,
            "test_n": int(len(x)),
            "test_accuracy": float((full_pred == helps).mean()),
            "test_balanced_accuracy": balanced_accuracy(helps, full_pred),
            "test_base_rate_helps": float(helps.mean()),
        }
    ]
    loro_correct = 0
    loro_total = 0
    for held_out in SYNTH_CONDITIONS:
        train_mask = conditions != held_out
        test_mask = ~train_mask
        t = best_threshold(x[train_mask], helps[train_mask])
        pred = x[test_mask] > t
        truth = helps[test_mask]
        loro_correct += int((pred == truth).sum())
        loro_total += int(test_mask.sum())
        rows.append(
            {
                "predictor": predictor,
                "held_out_regime": held_out,
                "threshold_log10": t,
                "test_n": int(test_mask.sum()),
                "test_accuracy": float((pred == truth).mean()),
                "test_balanced_accuracy": balanced_accuracy(truth, pred),
                "test_base_rate_helps": float(truth.mean()),
            }
        )
    rows.append(
        {
            "predictor": predictor,
            "held_out_regime": "pooled leave-one-regime-out",
            "threshold_log10": np.nan,
            "test_n": loro_total,
            "test_accuracy": loro_correct / loro_total,
            "test_balanced_accuracy": np.nan,
            "test_base_rate_helps": float(helps.mean()),
        }
    )
    return {
        "table": pd.DataFrame(rows),
        "full_threshold_log10": full_threshold,
        "loro_accuracy": loro_correct / loro_total,
        "base_rate": float(helps.mean()),
    }


# ---------------------------------------------------------------------------
# Figure and report
# ---------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_figure(synth: pd.DataFrame, vision: pd.DataFrame, rule: dict[str, object], output_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)

    for condition in SYNTH_CONDITIONS:
        sub = synth[synth["condition"] == condition]
        ax.scatter(
            sub[PRIMARY_PREDICTOR],
            sub["gain_ndfa_pp"],
            s=17,
            alpha=0.62,
            color=REGIME_COLORS[condition],
            edgecolors="none",
            label=REGIME_LABELS[condition],
        )
    if not vision.empty:
        fashion = vision[vision["dataset"] == "fashion_mnist"]
        cifar = vision[vision["dataset"] == "cifar10"]
        ax.scatter(
            fashion[PRIMARY_PREDICTOR], fashion["gain_ndfa_pp"],
            s=52, marker="*", facecolor="#222222", edgecolors="white", linewidths=0.4,
            label="Fashion-MNIST", zorder=4,
        )
        ax.scatter(
            cifar[PRIMARY_PREDICTOR], cifar["gain_ndfa_pp"],
            s=52, marker="*", facecolor="white", edgecolors="#222222", linewidths=0.8,
            label="CIFAR-10", zorder=4,
        )

    threshold = float(rule["full_threshold_log10"])
    ax.axvline(10.0 ** threshold, color="#888888", linewidth=0.9, linestyle=(0, (4, 2)))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.axhline(GAIN_THRESHOLD_PP, color="#BBBBBB", linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("prospective nuisance-energy ratio (pre-training estimate)")
    ax.set_ylabel("realized nDFA gain over DFA (pp)")

    x = np.log10(synth[PRIMARY_PREDICTOR].to_numpy(dtype=float))
    rho = spearmanr(x, synth["gain_ndfa_pp"]).statistic
    text = rf"synthetic: Spearman $\rho{{=}}{rho:.2f}$ ($n{{=}}{len(synth)}$)"
    if not vision.empty:
        xv = np.log10(vision[PRIMARY_PREDICTOR].to_numpy(dtype=float))
        rho_v = spearmanr(xv, vision["gain_ndfa_pp"]).statistic
        text += "\n" + rf"vision transfer: $\rho{{=}}{rho_v:.2f}$ ($n{{=}}{len(vision)}$)"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=6.8, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    ax.annotate(
        "decision threshold",
        (10.0 ** threshold, ax.get_ylim()[0]),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize=6.2,
        color="#666666",
    )
    ax.grid(color="#E8EAE6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right", ncol=2, handletextpad=0.2, columnspacing=0.8)

    for suffix in ("pdf", "png", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(output_dir / f"prospective_diagnostic.{suffix}", **kwargs)
    plt.close(fig)


def write_summary(
    synth: pd.DataFrame,
    correlations: pd.DataFrame,
    rule: dict[str, object],
    vision: pd.DataFrame,
    vision_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    x_all = np.log10(synth[PRIMARY_PREDICTOR].to_numpy(dtype=float))
    rho_all = spearmanr(x_all, synth["gain_ndfa_pp"]).statistic
    kappa_rho = spearmanr(np.log10(synth["est_kappa"]), synth["gain_ndfa_pp"]).statistic
    helps_rate = float((synth["gain_ndfa_pp"] > GAIN_THRESHOLD_PP).mean())
    vision_rho = np.nan
    if not vision.empty:
        vision_rho = spearmanr(np.log10(vision[PRIMARY_PREDICTOR]), vision["gain_ndfa_pp"]).statistic
    headline = (
        f"Across the 128 designed synthetic cells the pre-training estimate correlates moderately with "
        f"the realized nDFA-DFA gain (Spearman rho = {rho_all:.2f}), but it does not beat the task-blind "
        f"anisotropy baseline kappa(C) (rho = {kappa_rho:.2f}), so the class-mean task projection is not "
        f"adding predictive value here. Conditioning helps by > {GAIN_THRESHOLD_PP:.0f}pp in {helps_rate*100:.0f}% "
        f"of cells, so there is almost no 'does it help' signal to predict: the leave-one-regime-out decision "
        f"rule ({rule['loro_accuracy']:.3f} pooled accuracy) does not exceed the {rule['base_rate']:.3f} base rate "
        f"of an always-helps classifier. On the held-out vision sweep the estimate ranks the realized gains "
        f"BACKWARDS (rho = {vision_rho:.2f}) and gets the Fashion-vs-CIFAR dataset ordering wrong. "
        "Honest verdict: the prospective diagnostic is NOT paper-strength as a decision rule; the transfer test fails."
    )
    sections = [
        ("Headline (honest verdict)", headline),
        (
            "Method",
            "The prospective diagnostic estimates the paper's designed `nuisance_energy_ratio` "
            "using only pre-training information: training inputs, (noisy) training labels, and a "
            "randomly initialized network with the sweep's architecture and init seeds. For each "
            "presynaptic layer that nDFA preconditions, the activity covariance is split into energy "
            "inside vs orthogonal to the span of class-conditional mean activations; the ratio "
            "(orthogonal / inside) is combined across layers and seeds by geometric mean. "
            "`est_kappa` is a task-blind anisotropy baseline (damped condition number). No trained "
            "models, test data, or generator ground truth are used.",
        ),
        (
            "Caveats",
            "The 128 synthetic cells are a designed grid, not an i.i.d. population: cross-cell "
            "correlations are descriptive, and the leave-one-regime-out protocol only removes the "
            "most obvious circularity (fitting and evaluating the threshold on the same regime). "
            "The vision sweep is the meaningful transfer test because none of its cells were used "
            "to design the estimator or tune the threshold.",
        ),
        ("Correlations with realized nDFA-DFA gain (128 synthetic cells)", dataframe_to_markdown(correlations, float_format=".3f")),
        (
            "Decision rule",
            f"Rule: predict 'conditioning helps > {GAIN_THRESHOLD_PP:.0f}pp' when log10(diagnostic) exceeds a "
            "threshold fit by balanced accuracy. Base rate of 'helps' is "
            f"{rule['base_rate']:.3f}; leave-one-regime-out pooled accuracy is {rule['loro_accuracy']:.3f}.\n\n"
            + dataframe_to_markdown(rule["table"], float_format=".3f"),
        ),
    ]
    if not vision.empty:
        sections.append(("Vision transfer (Fashion-MNIST / CIFAR-10 MLP noise sweep)", dataframe_to_markdown(vision_summary, float_format=".3f")))
        sections.append(("Vision cells", dataframe_to_markdown(
            vision[["dataset", "n_train", "label_noise", "est_nuisance_ratio", "est_kappa", "gain_ndfa_pp", "gain_kndfa_pp"]],
            float_format=".3f",
        )))
    write_markdown_report(
        output_dir / "prospective_diagnostic_summary.md",
        title="Prospective Nuisance-Energy Diagnostic",
        sections=sections,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/infodfa_prospective_diagnostic_v1")
    parser.add_argument(
        "--legacy-results",
        default=str(ROOT.parent / "Info-Man" / "results"),
        help="Root holding the frozen noise-sweep aggregates (Info-Man results tree).",
    )
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--n-seeds", type=int, default=SYNTH_N_SEEDS)
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
