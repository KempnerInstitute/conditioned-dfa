"""Hard multi-output synthetic benchmarks for DFA/NMNC credit assignment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_synthetic import (  # noqa: E402
    feedback_mode_from_method,
    natural_mode_from_method,
    natural_precondition_gradients,
)
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.analysis import dataframe_to_markdown, predictor_scores, write_markdown_report  # noqa: E402
from infogeo.dfa import ManualMLP, gradient_cosines, init_fa_feedback, init_feedback  # noqa: E402
from infogeo.noise_correlation import (  # noqa: E402
    NoiseCorrelationStats,
    activity_pca_bases,
    noise_correlation_diagnostics,
    noise_correlation_feedback_update,
    zero_feedback,
)


NOISE_CORR_METHODS = {"vnc", "nmnc"}


@dataclass(frozen=True)
class MultiOutputDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    theta_train: np.ndarray
    theta_test: np.ndarray
    condition: str
    n_classes: int
    task_scale: float
    nuisance_scale: float
    nuisance_dim: int
    input_noise: float
    train_label_noise: float
    test_label_noise: float


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 1024
        args.n_test = 512
        args.epochs = 3
        args.n_seeds = 1
        args.n_feedback_seeds = 1
        args.hidden_dims = [128]
        args.conditions = [args.condition]
        args.feedback_ranks = [0, 1, 4]
        args.nc_update_intervals = [5]
        args.methods = ["bp", "dfa_random", "ndfa_random_kronecker", "vnc", "nmnc"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    if args.n_shards is not None:
        if args.shard_index is None:
            raise ValueError("--shard-index is required when --n-shards is set")
        specs = [spec for idx, spec in enumerate(specs) if idx % args.n_shards == args.shard_index]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    print(f"Running {len(specs)} multi-output DFA specs", flush=True)
    rows = []
    for spec_idx, spec in enumerate(specs, start=1):
        dataset = make_multioutput_dataset(
            condition=str(spec["condition"]),
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            n_classes=args.n_classes,
            nuisance_dim=args.nuisance_dim,
            input_noise=args.input_noise,
            train_label_noise=args.label_noise,
            test_label_noise=args.test_label_noise,
            task_scale_override=args.task_scale_override,
            nuisance_scale_override=args.nuisance_scale_override,
            seed=int(spec["seed"]),
        )
        rows.extend(run_one(dataset, spec=spec, args=args))
        if args.checkpoint_every > 0 and spec_idx % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(output_dir / "dfa_multioutput_results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "dfa_multioutput_results.csv"
    df.to_csv(csv_path, index=False)
    if not args.skip_analysis:
        plot_results(df, output_dir)
        write_report(df, output_dir)
    print(f"Saved {csv_path}")


def build_specs(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    specs: list[dict[str, float | int | str]] = []
    for seed in range(args.n_seeds):
        for condition in args.conditions:
            for method in args.methods:
                method_ranks = [0] if method == "bp" else args.feedback_ranks
                method_intervals = args.nc_update_intervals if method in NOISE_CORR_METHODS else [args.nc_update_intervals[0]]
                for feedback_rank in method_ranks:
                    for nc_update_interval in method_intervals:
                        feedback_seeds = [0] if method == "bp" else range(args.n_feedback_seeds)
                        for feedback_seed in feedback_seeds:
                            specs.append(
                                {
                                    "seed": seed,
                                    "condition": condition,
                                    "method": method,
                                    "feedback_seed": feedback_seed,
                                    "feedback_rank": feedback_rank,
                                    "nc_update_interval": nc_update_interval,
                                }
                            )
    return specs


def make_multioutput_dataset(
    *,
    condition: str,
    n_train: int,
    n_test: int,
    input_dim: int,
    n_classes: int,
    nuisance_dim: int,
    input_noise: float,
    train_label_noise: float,
    test_label_noise: float,
    task_scale_override: float | None,
    nuisance_scale_override: float | None,
    seed: int,
) -> MultiOutputDataset:
    condition = condition.lower().replace("-", "_")
    if condition == "task_aligned":
        task_scale, nuisance_scale, interaction = 1.3, 0.25, False
    elif condition == "nuisance_dominant":
        task_scale, nuisance_scale, interaction = 0.45, 2.0, False
    elif condition == "mixed_context":
        task_scale, nuisance_scale, interaction = 0.75, 1.2, True
    elif condition == "low_sample_noisy":
        task_scale, nuisance_scale, interaction = 0.7, 1.5, False
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if task_scale_override is not None:
        task_scale = float(task_scale_override)
    if nuisance_scale_override is not None:
        nuisance_scale = float(nuisance_scale_override)

    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(4 + nuisance_dim, input_dim)) / np.sqrt(4 + nuisance_dim)
    train = sample_multioutput_split(
        n_train,
        n_classes=n_classes,
        nuisance_dim=nuisance_dim,
        task_scale=task_scale,
        nuisance_scale=nuisance_scale,
        interaction=interaction,
        input_noise=input_noise,
        projection=projection,
        rng=rng,
    )
    test = sample_multioutput_split(
        n_test,
        n_classes=n_classes,
        nuisance_dim=nuisance_dim,
        task_scale=task_scale,
        nuisance_scale=nuisance_scale,
        interaction=interaction,
        input_noise=input_noise,
        projection=projection,
        rng=rng,
    )
    train_y = corrupt_labels(train[1], n_classes=n_classes, noise=train_label_noise, rng=rng)
    test_y = corrupt_labels(test[1], n_classes=n_classes, noise=test_label_noise, rng=rng)
    return MultiOutputDataset(
        x_train=train[0].astype(np.float32),
        y_train=train_y.astype(np.int64),
        theta_train=train[2].astype(np.float32),
        x_test=test[0].astype(np.float32),
        y_test=test_y.astype(np.int64),
        theta_test=test[2].astype(np.float32),
        condition=condition,
        n_classes=n_classes,
        task_scale=task_scale,
        nuisance_scale=nuisance_scale,
        nuisance_dim=nuisance_dim,
        input_noise=input_noise,
        train_label_noise=train_label_noise,
        test_label_noise=test_label_noise,
    )


def corrupt_labels(y: np.ndarray, *, n_classes: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly replace a fraction of labels with an incorrect class."""

    y = np.asarray(y, dtype=np.int64).copy()
    noise = float(noise)
    if noise <= 0:
        return y
    if noise >= 1:
        noise = 1.0
    mask = rng.random(y.shape[0]) < noise
    if not np.any(mask):
        return y
    offsets = rng.integers(1, n_classes, size=int(mask.sum()))
    y[mask] = (y[mask] + offsets) % n_classes
    return y


def sample_multioutput_split(
    n_samples: int,
    *,
    n_classes: int,
    nuisance_dim: int,
    task_scale: float,
    nuisance_scale: float,
    interaction: bool,
    input_noise: float,
    projection: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    nuisance = rng.normal(size=(n_samples, nuisance_dim))
    y = np.floor(theta / (2.0 * np.pi) * n_classes).astype(int) % n_classes
    if interaction:
        context = (nuisance[:, 0] > 0.0).astype(int)
        y = (y + context * (n_classes // 2)) % n_classes
    task = np.column_stack([np.cos(theta), np.sin(theta), np.cos(2.0 * theta), np.sin(2.0 * theta)])
    base = np.column_stack([task_scale * task, nuisance_scale * nuisance])
    x = base @ projection
    if input_noise > 0:
        x = x + rng.normal(scale=input_noise, size=x.shape)
    return x, y, theta


def run_one(
    dataset: MultiOutputDataset,
    *,
    spec: dict[str, float | int | str],
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    method = str(spec["method"])
    seed = int(spec["seed"])
    feedback_seed = int(spec["feedback_seed"])
    feedback_rank = int(spec["feedback_rank"])
    nc_update_interval = int(spec["nc_update_interval"])

    model = ManualMLP(
        input_dim=dataset.x_train.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=dataset.n_classes,
        seed=10_000 + seed,
        device=device,
        batchnorm=args.batchnorm,
    )
    x_train = torch.tensor(dataset.x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long, device=device)
    x_test = torch.tensor(dataset.x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long, device=device)

    feedback = initialize_feedback(model, method=method, seed=seed, feedback_seed=feedback_seed, feedback_rank=feedback_rank, args=args)
    precond_cache = {} if args.cov_refresh_interval > 1 else None
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    eval_n = min(args.eval_size, len(x_train))
    pca_n = min(args.pca_size, len(x_train))
    bases = activity_pca_bases(model, x_train[:pca_n], rank=args.nc_manifold_rank)
    stats = empty_stats()
    rows = []
    step = 0
    for epoch in range(args.epochs + 1):
        bases = activity_pca_bases(model, x_train[:pca_n], rank=args.nc_manifold_rank)
        rows.append(
            evaluate(
                model,
                x_train[:eval_n],
                y_train[:eval_n],
                x_test,
                y_test,
                dataset=dataset,
                method=method,
                seed=seed,
                feedback_seed=feedback_seed,
                feedback_rank=feedback_rank,
                nc_update_interval=nc_update_interval,
                feedback=feedback,
                bases=bases,
                stats=stats,
                epoch=epoch,
                args=args,
            )
        )
        if epoch == args.epochs:
            break
        for batch in minibatches(len(x_train), args.batch_size, rng):
            xb = x_train[batch]
            yb = y_train[batch]
            if method in NOISE_CORR_METHODS and feedback is not None:
                if step % args.nc_pca_update_interval == 0:
                    bases = activity_pca_bases(model, x_train[:pca_n], rank=args.nc_manifold_rank)
                if step % nc_update_interval == 0:
                    feedback, stats = noise_correlation_feedback_update(
                        model,
                        xb,
                        feedback,
                        mode=method,
                        bases=bases,
                        eta=args.nc_feedback_lr,
                        noise_scale=args.nc_noise_scale,
                        manifold_rank=args.nc_manifold_rank,
                        rng=rng,
                        antithetic=not args.nc_no_antithetic,
                        normalize_by_variance=not args.nc_covariance_scaled,
                    )
            model.training = True  # batch-stats BatchNorm during updates (no-op without --batchnorm)
            gradients = compute_gradients(
                model,
                xb,
                yb,
                method=method,
                feedback=feedback,
                args=args,
                precond_cache=precond_cache,
                refresh_precond=step % max(args.cov_refresh_interval, 1) == 0,
            )
            model.training = False
            model.apply_gradients(gradients, lr=args.lr)
            step += 1
    print(
        f"{dataset.condition:18s} {method:22s} seed={seed} fb={feedback_seed} "
        f"rank={feedback_rank} b={nc_update_interval} test={model.accuracy(x_test, y_test):.3f}",
        flush=True,
    )
    return rows


def initialize_feedback(
    model: ManualMLP,
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    feedback_rank: int,
    args: argparse.Namespace,
) -> list[torch.Tensor] | None:
    if method in ("bp", "bp_whitened"):
        return None
    if method in NOISE_CORR_METHODS and args.nc_init == "zero":
        return zero_feedback(model)
    if method.startswith("fa_"):
        return init_fa_feedback(
            model,
            seed=20_000 + 100 * seed + feedback_seed,
            scale=args.feedback_scale,
            rank=None if feedback_rank <= 0 else feedback_rank,
        )
    return init_feedback(
        model,
        mode="random" if method in NOISE_CORR_METHODS else feedback_mode_from_method(method),
        seed=20_000 + 100 * seed + feedback_seed,
        scale=args.feedback_scale,
        rank=None if feedback_rank <= 0 else feedback_rank,
    )


def compute_gradients(
    model: ManualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    method: str,
    feedback,
    args: argparse.Namespace,
    precond_cache: dict | None = None,
    refresh_precond: bool = True,
):
    if method == "bp":
        return model.bp_gradients(x, y)
    if method == "bp_whitened":
        # BP gradient with the SAME input-side whitening as nDFA. If nDFA ~ this,
        # the gain is the natural-gradient reconditioning of \S3.1, not the DFA pathway.
        g = model.bp_gradients(x, y)
        return natural_precondition_gradients(model, g, x, damping=args.natural_damping, mode="activity")
    if method == "fa_sign":
        # Sign-symmetry baseline: layerwise feedback whose weights share the sign
        # of the forward weights (magnitude 1), recomputed each step from the
        # current weights (Liao et al. 2016; Xiao et al. 2018).
        sign_fb = [torch.sign(model.weights[i + 1]) for i in range(model.n_hidden_layers)]
        return model.fa_gradients(x, y, sign_fb)
    if feedback is None:
        raise ValueError(f"{method} requires feedback")
    if method.startswith("drtp_"):
        gradients = model.target_projection_gradients(x, y, feedback)
    elif method.startswith("fa_"):
        gradients = model.fa_gradients(x, y, feedback)
    else:
        gradients = model.dfa_gradients(x, y, feedback)
    if method.startswith("ndfa_"):
        error_deltas = None
        mode = natural_mode_from_method(method)
        if method == "ndfa_random_kronecker_bp":
            # True-KFAC-DFA control: K-nDFA but the left (error-side) factor uses
            # the BP error covariance instead of DFA's own broadcast-error deltas.
            mode = "kronecker"
            error_deltas = model.bp_gradients(x, y).deltas
        gradients = natural_precondition_gradients(
            model,
            gradients,
            x,
            damping=args.natural_damping,
            mode=mode,
            error_deltas=error_deltas,
            cache=precond_cache,
            refresh=refresh_precond,
        )
    if method == "dfa_actwhiten":
        # Decorrelation baseline: DFA with ZCA-whitened presynaptic activity,
        # i.e. preconditioning by (C+lambda I)^{-1/2} instead of nDFA's full inverse.
        gradients = natural_precondition_gradients(
            model, gradients, x, damping=args.natural_damping, mode="activity_sqrt",
        )
    return gradients


def evaluate(
    model: ManualMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    dataset: MultiOutputDataset,
    method: str,
    seed: int,
    feedback_seed: int,
    feedback_rank: int,
    nc_update_interval: int,
    feedback,
    bases,
    stats: NoiseCorrelationStats,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    diagnostics = {
        "param_cosine": np.nan,
        "activity_cosine_mean": np.nan,
        "activity_angle_deg_mean": np.nan,
        "projected_step_ratio_mean": np.nan,
        "activity_norm_ratio_mean": np.nan,
        "manifold_gradient_alpha_mean": np.nan,
        "manifold_dim_fraction_mean": np.nan,
        "manifold_condition_margin_mean": np.nan,
    }
    if feedback is not None:
        local = compute_gradients(model, x_eval, y_eval, method=method, feedback=feedback, args=args)
        diagnostics.update({k: float(v) for k, v in gradient_cosines(bp, local).items() if np.isscalar(v)})
        diagnostics.update(noise_correlation_diagnostics(bp, local, bases))
        if args.covariance_diagnostics:
            diagnostics.update(covariance_diagnostics(model, x_eval, bp, local, damping=args.natural_damping))
    elif args.covariance_diagnostics:
        diagnostics.update(covariance_diagnostics(model, x_eval, bp, bp, damping=args.natural_damping))
    return {
        "condition": dataset.condition,
        "method": method,
        "seed": float(seed),
        "feedback_seed": float(feedback_seed),
        "feedback_rank": float(feedback_rank),
        "nc_update_interval": float(nc_update_interval),
        "nc_manifold_rank": float(args.nc_manifold_rank),
        "epoch": float(epoch),
        "n_classes": float(dataset.n_classes),
        "n_train": float(args.n_train),
        "n_test": float(args.n_test),
        "task_scale": float(dataset.task_scale),
        "nuisance_scale": float(dataset.nuisance_scale),
        "nuisance_dim": float(dataset.nuisance_dim),
        "input_noise": float(dataset.input_noise),
        "train_label_noise": float(dataset.train_label_noise),
        "test_label_noise": float(dataset.test_label_noise),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval),
        "test_acc": model.accuracy(x_test, y_test),
        "feedback_norm": stats.mean_feedback_norm,
        "feedback_update_norm": stats.mean_update_norm,
        "noise_norm": stats.mean_noise_norm,
        "delta_output_norm": stats.mean_delta_output_norm,
        **diagnostics,
    }


def covariance_diagnostics(
    model: ManualMLP,
    x: torch.Tensor,
    bp: object,
    local: object,
    *,
    damping: float,
) -> dict[str, float]:
    """Layer-averaged covariance spectra for the local update mechanism."""

    with torch.no_grad():
        _, activations, _ = model.forward(x)
    presynaptic_stats = []
    local_delta_stats = []
    bp_delta_stats = []
    for layer_idx in range(model.n_hidden_layers):
        presynaptic_stats.append(spectrum_stats(activations[layer_idx], damping=damping))
        local_delta_stats.append(spectrum_stats(local.deltas[layer_idx], damping=damping))
        bp_delta_stats.append(spectrum_stats(bp.deltas[layer_idx], damping=damping))
    return {
        **prefix_mean_stats("pre_activity", presynaptic_stats),
        **prefix_mean_stats("local_error", local_delta_stats),
        **prefix_mean_stats("bp_error", bp_delta_stats),
    }


def prefix_mean_stats(prefix: str, stats: list[dict[str, float]]) -> dict[str, float]:
    if not stats:
        return {}
    keys = sorted(stats[0])
    return {f"{prefix}_{key}_mean": float(np.nanmean([item[key] for item in stats])) for key in keys}


def spectrum_stats(values: torch.Tensor, *, damping: float, eps: float = 1e-12) -> dict[str, float]:
    x = values.detach().flatten(start_dim=1).float()
    if x.shape[0] <= 1 or x.shape[1] == 0:
        return {"condition": np.nan, "effective_rank": np.nan, "top1_fraction": np.nan, "trace": np.nan}
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=0, keepdim=True)
    try:
        singular = torch.linalg.svdvals(x.cpu())
    except RuntimeError:
        cov = (x.T @ x) / max(int(x.shape[0]) - 1, 1)
        cov = 0.5 * (cov + cov.T)
        eig = robust_eigvalsh(cov)
    else:
        eig = singular.square() / max(int(x.shape[0]) - 1, 1)
    eig = torch.sort(torch.nan_to_num(eig, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)).values
    trace = eig.sum().clamp_min(eps)
    probs = eig / trace
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    effective_rank = torch.exp(entropy)
    top1 = eig[-1] / trace
    damp = max(float(damping), eps)
    condition = (eig[-1] + damp) / (eig[0] + damp)
    return {
        "condition": float(condition.detach().cpu().item()),
        "effective_rank": float(effective_rank.detach().cpu().item()),
        "top1_fraction": float(top1.detach().cpu().item()),
        "trace": float(trace.detach().cpu().item()),
    }


def robust_eigvalsh(cov: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    for jitter in (0.0, 1e-8, 1e-6, 1e-4):
        try:
            return torch.linalg.eigvalsh(cov + jitter * eye)
        except RuntimeError:
            continue
    diag = torch.diagonal(cov).clamp_min(0.0)
    return diag


def empty_stats() -> NoiseCorrelationStats:
    return NoiseCorrelationStats(float("nan"), float("nan"), float("nan"), float("nan"))


def summarize_final(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("epoch").groupby(
        ["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval", "nc_manifold_rank"],
        as_index=False,
    ).tail(1)
    return (
        final.groupby(["condition", "method", "feedback_rank", "nc_update_interval"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            train_mean=("train_eval_acc", "mean"),
            angle=("activity_angle_deg_mean", "mean"),
            step=("projected_step_ratio_mean", "mean"),
            alpha=("manifold_gradient_alpha_mean", "mean"),
            margin=("manifold_condition_margin_mean", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["condition", "method", "feedback_rank", "nc_update_interval"])
    )


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_final(df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for condition, sub in summary.groupby("condition"):
        best = sub.sort_values("test_mean").groupby("method", as_index=False).tail(1).sort_values("test_mean")
        axes[0, 0].barh([f"{condition}:{m}" for m in best["method"]], best["test_mean"], alpha=0.72)
    axes[0, 0].set_xlabel("Best final test accuracy")
    axes[0, 0].set_title("Best method/settings")

    for method in ["dfa_random", "ndfa_random", "ndfa_random_kronecker", "vnc", "nmnc", "drtp_random"]:
        sub = summary[summary["method"] == method]
        if sub.empty:
            continue
        curve = sub.groupby("feedback_rank")["test_mean"].mean()
        axes[0, 1].plot(curve.index, curve.values, marker="o", label=method)
    axes[0, 1].set_xlabel("Feedback rank")
    axes[0, 1].set_ylabel("Final test accuracy")
    axes[0, 1].set_title("True multi-output rank sweep")
    axes[0, 1].legend(frameon=False, fontsize=7)

    final = df.sort_values("epoch").groupby(
        ["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval"],
        as_index=False,
    ).tail(1)
    nc = final[final["method"].isin(["vnc", "nmnc"])]
    for method, color in [("vnc", "#E15759"), ("nmnc", "#F28E2B")]:
        sub = nc[nc["method"] == method]
        axes[1, 0].scatter(sub["manifold_condition_margin_mean"], sub["test_acc"], s=18, alpha=0.55, color=color, label=method)
        axes[1, 1].scatter(sub["projected_step_ratio_mean"], sub["test_acc"], s=18, alpha=0.55, color=color, label=method)
    axes[1, 0].axvline(0.0, color="0.65", linewidth=0.8)
    axes[1, 0].set_xlabel(r"Manifold-gradient margin $\alpha-d/n$")
    axes[1, 0].set_ylabel("Final test accuracy")
    axes[1, 0].set_title("Manifold condition")
    axes[1, 1].axvline(0.0, color="0.65", linewidth=0.8)
    axes[1, 1].set_xlabel("Projected BP-step ratio")
    axes[1, 1].set_ylabel("Final test accuracy")
    axes[1, 1].set_title("Useful step")
    axes[1, 0].legend(frameon=False, fontsize=7)
    fig.savefig(output_dir / "dfa_multioutput_summary.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_final(df)
    scores = predictor_scores(
        df.sort_values("epoch").groupby(
            ["condition", "method", "seed", "feedback_seed", "feedback_rank", "nc_update_interval"],
            as_index=False,
        ).tail(1),
        target="test_acc",
        predictors=[
            "feedback_rank",
            "activity_angle_deg_mean",
            "projected_step_ratio_mean",
            "manifold_gradient_alpha_mean",
            "manifold_condition_margin_mean",
        ],
    )
    write_markdown_report(
        output_dir / "dfa_multioutput_analysis.md",
        title="DFA Multi-Output Synthetic Analysis",
        sections=[
            (
                "Design",
                "Multi-output synthetic tasks use 8 class labels on a latent circle with nuisance dimensions. Conditions separate task-aligned, nuisance-dominant, mixed-context, and low-sample noisy regimes. Unlike binary synthetic tasks, feedback ranks above 2 are meaningful.",
            ),
            ("Summary", dataframe_to_markdown(summary, float_format=".4f")),
            ("Predictors", dataframe_to_markdown(scores, float_format=".4f")),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/dfa_multioutput_synthetic_v1")
    parser.add_argument("--condition", default="task_aligned")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "drtp_random", "vnc", "nmnc"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--feedback-ranks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--nuisance-dim", type=int, default=24)
    parser.add_argument("--input-noise", type=float, default=0.05)
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of training labels replaced by an incorrect class.")
    parser.add_argument("--test-label-noise", type=float, default=0.0, help="Optional test-label corruption for stress tests; defaults to clean evaluation.")
    parser.add_argument("--task-scale-override", type=float, default=None)
    parser.add_argument("--nuisance-scale-override", type=float, default=None)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--pca-size", type=int, default=1024)
    parser.add_argument("--nc-update-intervals", type=int, nargs="+", default=[10])
    parser.add_argument("--nc-pca-update-interval", type=int, default=20)
    parser.add_argument("--nc-manifold-rank", type=int, default=64)
    parser.add_argument("--nc-noise-scale", type=float, default=0.05)
    parser.add_argument("--nc-feedback-lr", type=float, default=0.5)
    parser.add_argument("--nc-init", choices=["zero", "random"], default="zero")
    parser.add_argument("--nc-no-antithetic", action="store_true")
    parser.add_argument("--nc-covariance-scaled", action="store_true")
    parser.add_argument("--covariance-diagnostics", action="store_true")
    parser.add_argument("--batchnorm", action="store_true", help="Insert BatchNorm1d after each hidden linear layer (pre-activation); default off.")
    parser.add_argument("--cov-refresh-interval", type=int, default=1, help="Recompute the damped covariance inverse for ndfa preconditioning every k training batches (1 = every batch, the default behavior).")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    args.conditions = args.conditions if args.conditions is not None else [args.condition]
    return args


if __name__ == "__main__":
    main()
