"""Research benchmark for DFA as information-geometric manifold learning."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.analysis import dataframe_to_markdown, predictor_scores, write_markdown_report
from infogeo.dfa import (
    Gradients,
    ManualMLP,
    error_second_moment,
    finite_difference_hidden_tangents,
    gradient_cosines,
    init_fa_feedback,
    init_feedback,
    rayleigh_quotient_scores,
    tangent_projected_cosines,
)
from infogeo.geometry import class_dprime2, effective_dimension, inv_sqrtm_psd, principal_subspace, stable_covariance
from infogeo.synthetic import make_manifold_split, manifold_features, task_boundary_weights


SOLVE_STATS = {
    "solve_damping_escalations": 0,
    "solve_lstsq_fallbacks": 0,
    "solve_max_damping_multiplier": 1.0,
}


def reset_solve_stats() -> None:
    SOLVE_STATS["solve_damping_escalations"] = 0
    SOLVE_STATS["solve_lstsq_fallbacks"] = 0
    SOLVE_STATS["solve_max_damping_multiplier"] = 1.0


def solve_stats_dict() -> dict[str, float]:
    return {key: float(value) for key, value in SOLVE_STATS.items()}


def _record_solve(multiplier: float, *, used_lstsq: bool = False) -> None:
    if multiplier > 1.0:
        SOLVE_STATS["solve_damping_escalations"] += 1
    if used_lstsq:
        SOLVE_STATS["solve_lstsq_fallbacks"] += 1
    SOLVE_STATS["solve_max_damping_multiplier"] = max(
        float(SOLVE_STATS["solve_max_damping_multiplier"]),
        float(multiplier),
    )


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 512
        args.n_test = 512
        args.epochs = 6
        args.n_seeds = 2
        args.n_feedback_seeds = 2
        args.hidden_dim = 48
        args.hidden_dims = [48]
        args.task_frequencies = [args.task_frequency]
        args.input_noises = [args.input_noise]
        args.n_hidden_layers_list = [args.n_hidden_layers]
        args.feedback_scales = [0.5, 1.0]
        args.feedback_ranks = [0]
        args.manifolds = [args.manifold]
        args.methods = ["bp", "dfa_random", "dfa_bp_aligned", "dfa_negative_bp", "ndfa_random"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    specs = build_run_specs(args)
    if args.n_shards is not None:
        if args.shard_index is None:
            raise ValueError("--shard-index is required when --n-shards is set")
        if args.shard_index < 0 or args.shard_index >= args.n_shards:
            raise ValueError("--shard-index must be in [0, n_shards)")
        specs = [spec for idx, spec in enumerate(specs) if idx % args.n_shards == args.shard_index]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    print(f"Running {len(specs)} DFA synthetic run specs", flush=True)
    for spec_idx, spec in enumerate(specs, start=1):
        dataset = make_manifold_split(
            manifold=spec["manifold"],
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            noise=spec["input_noise"],
            task_frequency=spec["task_frequency"],
            seed=spec["seed"],
        )
        rows.extend(
            run_one_method(
                dataset,
                method=spec["method"],
                seed=int(spec["seed"]),
                feedback_seed=int(spec["feedback_seed"]),
                feedback_scale=float(spec["feedback_scale"]),
                feedback_rank=int(spec["feedback_rank"]),
                hidden_dim=int(spec["hidden_dim"]),
                n_hidden_layers=int(spec["n_hidden_layers"]),
                task_frequency=int(spec["task_frequency"]),
                input_noise=float(spec["input_noise"]),
                args=args,
            )
        )
        if args.checkpoint_every > 0 and spec_idx % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(output_dir / "dfa_synthetic_results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "dfa_synthetic_results.csv"
    df.to_csv(csv_path, index=False)
    if args.skip_analysis:
        print(f"\nSaved shard results to {csv_path}")
        return

    plot_dfa_results(df, output_dir)
    write_dfa_analysis(df, output_dir, early_epoch=args.early_epoch)

    last = df.sort_values("epoch").groupby(run_columns(), as_index=False).tail(1)
    summary = last.groupby("method")[["test_acc", "param_cosine", "tangent_cosine"]].mean(numeric_only=True)
    print(summary.to_string(float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved results to {csv_path}")
    print(f"Saved analysis to {output_dir / 'dfa_synthetic_analysis.md'}")


def build_run_specs(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    """Enumerate training runs once so large sweeps can be sharded safely."""

    specs: list[dict[str, float | int | str]] = []
    for seed in range(args.n_seeds):
        for manifold in args.manifolds:
            for task_frequency in args.task_frequencies:
                for input_noise in args.input_noises:
                    for hidden_dim in args.hidden_dims:
                        for n_hidden_layers in args.n_hidden_layers_list:
                            for method in args.methods:
                                feedback_seeds = (
                                    [0]
                                    if method == "bp" or "bp_aligned" in method or "negative_bp" in method
                                    else range(args.n_feedback_seeds)
                                )
                                feedback_scales = [1.0] if method == "bp" else args.feedback_scales
                                for feedback_seed in feedback_seeds:
                                    for feedback_scale in feedback_scales:
                                        feedback_ranks = [0] if method == "bp" else args.feedback_ranks
                                        for feedback_rank in feedback_ranks:
                                            specs.append(
                                                {
                                                    "seed": int(seed),
                                                    "manifold": str(manifold),
                                                    "task_frequency": int(task_frequency),
                                                    "input_noise": float(input_noise),
                                                    "hidden_dim": int(hidden_dim),
                                                    "n_hidden_layers": int(n_hidden_layers),
                                                    "method": str(method),
                                                    "feedback_seed": int(feedback_seed),
                                                    "feedback_scale": float(feedback_scale),
                                                    "feedback_rank": int(feedback_rank),
                                                }
                                            )
    return specs


def run_one_method(
    dataset,
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    feedback_scale: float,
    feedback_rank: int,
    hidden_dim: int,
    n_hidden_layers: int,
    task_frequency: int,
    input_noise: float,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    reset_solve_stats()
    torch.manual_seed(seed)
    model = ManualMLP(
        input_dim=dataset.x_train.shape[1],
        hidden_dims=[hidden_dim] * n_hidden_layers,
        output_dim=2,
        seed=10_000 + seed,
        device=device,
    )

    feature_fn = lambda z: manifold_features(z, dataset.projection, manifold=dataset.manifold, noise=0.0).astype(np.float32)
    basis_z = dataset.z_train[: min(args.eval_size, len(dataset.z_train))]
    init_tangents = finite_difference_hidden_tangents(model, basis_z, feature_fn)
    tangent_bases = [principal_subspace(flatten_tangent_samples(tangent), args.tangent_rank) for tangent in init_tangents]

    feedback = None
    if method != "bp":
        if method.startswith("fa_"):
            feedback = init_fa_feedback(
                model,
                seed=20_000 + 100 * seed + feedback_seed,
                scale=feedback_scale,
                rank=None if feedback_rank <= 0 else feedback_rank,
            )
        else:
            feedback_mode = feedback_mode_from_method(method)
            feedback = init_feedback(
                model,
                mode=feedback_mode,
                tangent_bases=tangent_bases,
                seed=20_000 + 100 * seed + feedback_seed,
                scale=feedback_scale,
                rank=None if feedback_rank <= 0 else feedback_rank,
            )

    x_train = torch.tensor(dataset.x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long, device=device)
    x_test = torch.tensor(dataset.x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long, device=device)

    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    rows = []
    for epoch in range(args.epochs + 1):
        rows.extend(
            evaluate_model(
                model,
                dataset,
                method=method,
                seed=seed,
                feedback_seed=feedback_seed,
                feedback_scale=feedback_scale,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                task_frequency=task_frequency,
                input_noise=input_noise,
                epoch=epoch,
                feedback_rank=feedback_rank,
                feedback=feedback,
                args=args,
            )
        )
        if epoch == args.epochs:
            break
        if method_has_dynamic_feedback(method):
            feedback = init_feedback(
                model,
                mode=feedback_mode_from_method(method),
                tangent_bases=tangent_bases,
                seed=20_000 + 100 * seed + feedback_seed,
                scale=feedback_scale,
                rank=None if feedback_rank <= 0 else feedback_rank,
            )
        for batch_idx in minibatches(len(dataset.x_train), args.batch_size, rng):
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            if method == "bp":
                gradients = model.bp_gradients(xb, yb)
            else:
                assert feedback is not None
                if method.startswith("drtp_"):
                    gradients = model.target_projection_gradients(xb, yb, feedback)
                elif method.startswith("fa_"):
                    gradients = model.fa_gradients(xb, yb, feedback)
                else:
                    gradients = model.dfa_gradients(xb, yb, feedback)
                if method.startswith("ndfa_"):
                    gradients = natural_precondition_gradients(
                        model,
                        gradients,
                        xb,
                        damping=args.natural_damping,
                        error_damping=args.natural_error_damping,
                        mode=natural_mode_from_method(method),
                    )
            model.apply_gradients(gradients, lr=args.lr)

    final_train_acc = model.accuracy(x_train, y_train)
    final_test_acc = model.accuracy(x_test, y_test)
    print(
        f"{method:18s} manifold={dataset.manifold:10s} seed={seed} fb={feedback_seed} "
        f"scale={feedback_scale:0.2f} rank={feedback_rank} "
        f"k={task_frequency} noise={input_noise:0.2f} width={hidden_dim} depth={n_hidden_layers} "
        f"train={final_train_acc:0.3f} test={final_test_acc:0.3f}"
    )
    return rows


def evaluate_model(
    model: ManualMLP,
    dataset,
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    feedback_scale: float,
    hidden_dim: int,
    n_hidden_layers: int,
    task_frequency: int,
    input_noise: float,
    epoch: int,
    feedback_rank: int,
    feedback,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    device = model.device
    eval_n = min(args.eval_size, len(dataset.x_train))
    x_eval_np = dataset.x_train[:eval_n]
    y_eval_np = dataset.y_train[:eval_n]
    z_eval = dataset.z_train[:eval_n]
    x_eval = torch.tensor(x_eval_np, dtype=torch.float32, device=device)
    y_eval = torch.tensor(y_eval_np, dtype=torch.long, device=device)
    x_train = torch.tensor(dataset.x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long, device=device)
    x_test = torch.tensor(dataset.x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long, device=device)

    train_acc = model.accuracy(x_train, y_train)
    test_acc = model.accuracy(x_test, y_test)
    bp = model.bp_gradients(x_eval, y_eval)
    alignments = {"param_cosine": np.nan}
    tangent_alignments: dict[str, float] = {}
    task_tangent_alignments: dict[str, float] = {}
    task_weights = task_boundary_weights(z_eval, manifold=dataset.manifold, task_frequency=task_frequency)
    if feedback is not None:
        if method.startswith("drtp_"):
            dfa = model.target_projection_gradients(x_eval, y_eval, feedback)
        elif method.startswith("fa_"):
            dfa = model.fa_gradients(x_eval, y_eval, feedback)
        else:
            dfa = model.dfa_gradients(x_eval, y_eval, feedback)
        alignments = gradient_cosines(bp, dfa)
        feature_fn = lambda z: manifold_features(z, dataset.projection, manifold=dataset.manifold, noise=0.0).astype(np.float32)
        tangents = finite_difference_hidden_tangents(model, z_eval, feature_fn)
        tangent_alignments = tangent_projected_cosines(bp, dfa, tangents)
        task_tangent_alignments = tangent_projected_cosines(bp, dfa, tangents, sample_weights=task_weights)
    else:
        feature_fn = lambda z: manifold_features(z, dataset.projection, manifold=dataset.manifold, noise=0.0).astype(np.float32)
        tangents = finite_difference_hidden_tangents(model, z_eval, feature_fn)

    hidden = [h.detach().cpu().numpy() for h in model.hidden_activations(x_eval)]
    _, activations, _ = model.forward(x_eval)
    input_activations = [a.detach().cpu().numpy() for a in activations[:-1]]
    rows = []
    for layer_idx, (h, tangent) in enumerate(zip(hidden, tangents), start=1):
        residual_cov = class_residual_covariance(h, y_eval_np)
        inv_sqrt_cov = inv_sqrtm_psd(residual_cov, ridge=1e-5)
        whitened_tangent = tangent @ inv_sqrt_cov.T
        rq = rayleigh_quotient_scores(
            input_activations[layer_idx - 1],
            model.weights[layer_idx - 1].detach().cpu().numpy(),
        )
        rows.append(
            {
                "seed": float(seed),
                "feedback_seed": float(feedback_seed),
                "feedback_scale": float(feedback_scale),
                "feedback_rank": float(feedback_rank),
                "hidden_dim": float(hidden_dim),
                "n_hidden_layers": float(n_hidden_layers),
                "task_frequency": float(task_frequency),
                "input_noise": float(input_noise),
                "manifold": dataset.manifold,
                "method": method,
                "epoch": float(epoch),
                "layer": float(layer_idx),
                "loss": bp.loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "param_cosine": float(alignments.get("param_cosine", np.nan)),
                "activity_cosine": float(alignments.get(f"activity_cosine_l{layer_idx}", np.nan)),
                "tangent_cosine": float(tangent_alignments.get(f"tangent_cosine_l{layer_idx}", np.nan)),
                "task_tangent_cosine": float(task_tangent_alignments.get(f"tangent_cosine_l{layer_idx}", np.nan)),
                "raw_jacobian_norm": mean_sample_frobenius_sq(tangent),
                "fisher_trace": mean_sample_frobenius_sq(whitened_tangent),
                "weight_rayleigh": float(np.mean(rq)),
                "class_dprime2": class_dprime2(h, y_eval_np),
                "effective_dim": effective_dimension(h),
                **solve_stats_dict(),
            }
        )
    return rows


def mean_sample_frobenius_sq(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return float(np.mean(np.sum(values * values, axis=1)))
    if values.ndim == 3:
        return float(np.mean(np.sum(values * values, axis=(1, 2))))
    raise ValueError("values must have shape (n, h) or (n, k, h)")


def class_residual_covariance(hidden: np.ndarray, labels: np.ndarray) -> np.ndarray:
    residuals = []
    for label in np.unique(labels):
        group = hidden[labels == label]
        if group.shape[0] >= 2:
            residuals.append(group - group.mean(axis=0, keepdims=True))
    if not residuals:
        return stable_covariance(hidden, shrinkage=0.05, ridge=1e-5)
    return stable_covariance(np.vstack(residuals), shrinkage=0.05, ridge=1e-5, center=False)


def natural_precondition_gradients(
    model: ManualMLP,
    gradients: Gradients,
    x: torch.Tensor,
    *,
    damping: float,
    error_damping: float | None = None,
    mode: str = "activity",
    error_deltas: Sequence[torch.Tensor] | None = None,
    cache: dict | None = None,
    refresh: bool = True,
) -> Gradients:
    """Local natural-DFA proxies using Kronecker-style preconditioning.

    ``error_damping`` controls the left factor independently; by default it
    reuses ``damping`` for backward compatibility. ``error_deltas`` overrides
    the deltas used for the error-side (left) factor.
    Passing the BP deltas gives a true-KFAC-DFA control: the K-nDFA rule but with
    the left factor built from the BP error covariance rather than DFA's own.

    ``cache``/``refresh`` implement amortized covariance refresh: with a dict
    ``cache``, the damped inverses are recomputed from the current batch only
    when ``refresh`` is True and reused otherwise. ``cache=None`` (the default)
    keeps the original every-batch behavior exactly.
    """

    if cache is not None and not cache:
        refresh = True  # never apply an uninitialized cache
    need_forward = cache is None or refresh or mode == "activity_sqrt"
    activations = model.forward(x)[1] if need_forward else None
    new_weights = [grad.clone() for grad in gradients.weights]
    new_biases = [grad.clone() for grad in gradients.biases]
    left_deltas = gradients.deltas if error_deltas is None else error_deltas
    left_damping = damping if error_damping is None else float(error_damping)
    for layer_idx in range(model.n_hidden_layers):
        grad = gradients.weights[layer_idx]
        if mode == "activity_sqrt":
            # Decorrelation baseline: ZCA-whiten the presynaptic activity,
            # i.e. precondition by (C+lambda I)^{-1/2} (power 1/2) instead of nDFA's
            # full inverse (C+lambda I)^{-1} (power 1). Tests whether the gain needs
            # the inverse-second-moment power or only generic activation decorrelation.
            activity = activations[layer_idx].detach()
            cov = activity.T @ activity / max(activity.shape[0], 1)
            eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
            inv_sqrt = torch.as_tensor(
                inv_sqrtm_psd((cov + damping * eye).cpu().numpy()),
                dtype=cov.dtype, device=cov.device,
            )
            grad = grad @ inv_sqrt
        if mode in {"activity", "kronecker"}:
            if cache is None:
                activity = activations[layer_idx].detach()
                cov = activity.T @ activity / max(activity.shape[0], 1)
                grad = damped_solve(cov, grad.T, damping=damping).T
            else:
                key = ("activity", layer_idx)
                if refresh:
                    activity = activations[layer_idx].detach()
                    cov = activity.T @ activity / max(activity.shape[0], 1)
                    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
                    cache[key] = damped_solve(cov, eye, damping=damping)
                grad = (cache[key] @ grad.T).T
        if mode in {"error", "kronecker"}:
            if cache is None:
                delta = left_deltas[layer_idx].detach()
                cov = error_second_moment(delta, normalization_count=delta.shape[0])
                grad = damped_solve(cov, grad, damping=left_damping)
            else:
                key = ("error", layer_idx)
                if refresh:
                    delta = left_deltas[layer_idx].detach()
                    cov = error_second_moment(delta, normalization_count=delta.shape[0])
                    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
                    cache[key] = damped_solve(cov, eye, damping=left_damping)
                grad = cache[key] @ grad
        new_weights[layer_idx] = grad
    return Gradients(new_weights, new_biases, gradients.deltas, gradients.loss)


def damped_solve(cov: torch.Tensor, rhs: torch.Tensor, *, damping: float) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    base = max(float(damping), 1e-6)
    last_error: RuntimeError | None = None
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        matrix = cov + (base * multiplier) * eye
        try:
            solution = torch.linalg.solve(matrix, rhs)
        except RuntimeError as exc:
            last_error = exc
            continue
        if torch.isfinite(solution).all():
            _record_solve(multiplier)
            return solution
    try:
        _record_solve(10000.0, used_lstsq=True)
        return torch.linalg.lstsq(cov + (base * 10000.0) * eye, rhs).solution
    except RuntimeError:
        if last_error is not None:
            raise last_error
        raise


def flatten_tangent_samples(tangent: np.ndarray) -> np.ndarray:
    tangent = np.asarray(tangent)
    if tangent.ndim == 2:
        return tangent
    if tangent.ndim == 3:
        return tangent.reshape(-1, tangent.shape[-1])
    raise ValueError("tangent must be 2D or 3D")


def feedback_mode_from_method(method: str) -> str:
    mode = method
    for prefix in ("dfa_", "ndfa_", "drtp_", "fa_"):
        if mode.startswith(prefix):
            mode = mode[len(prefix) :]
            break
    if mode.endswith("_dynamic"):
        mode = mode[: -len("_dynamic")]
    if mode.endswith("_kronecker_bp"):
        mode = mode[: -len("_kronecker_bp")]
    elif mode.endswith("_activity") or mode.endswith("_error") or mode.endswith("_kronecker"):
        mode = mode.rsplit("_", maxsplit=1)[0]
    return mode


def method_has_dynamic_feedback(method: str) -> bool:
    return "_dynamic" in method


def natural_mode_from_method(method: str) -> str:
    if method.endswith("_error"):
        return "error"
    if method.endswith("_kronecker"):
        return "kronecker"
    return "activity"


def write_dfa_analysis(df: pd.DataFrame, output_dir: Path, *, early_epoch: int) -> None:
    run_df = summarize_runs(df, early_epoch=early_epoch)
    run_df.to_csv(output_dir / "dfa_run_summary.csv", index=False)

    predictors = [
        "early_param_cosine",
        "early_activity_cosine",
        "early_tangent_cosine",
        "early_task_tangent_cosine",
        "early_fisher_trace",
        "early_weight_rayleigh",
        "early_class_dprime2",
        "delta_fisher_trace_early",
        "delta_weight_rayleigh_early",
        "delta_class_dprime2_early",
    ]
    scores = predictor_scores(run_df[run_df["method"] != "bp"], target="final_test_acc", predictors=predictors)
    scores.to_csv(output_dir / "dfa_predictor_scores.csv", index=False)

    by_method = (
        run_df.groupby("method")[
            [
                "final_test_acc",
                "early_param_cosine",
                "early_tangent_cosine",
                "early_task_tangent_cosine",
                "early_weight_rayleigh",
                "delta_class_dprime2_early",
            ]
        ]
        .mean()
        .reset_index()
    )
    best = scores.sort_values("r2", ascending=False).head(8)
    body = (
        "This benchmark tests the DFA draft's most actionable claim: early alignment should predict "
        "later learning. Each run is summarized by final test accuracy and by early alignment/geometry "
        f"features averaged through epoch {early_epoch}."
    )
    write_markdown_report(
        output_dir / "dfa_synthetic_analysis.md",
        title="DFA Synthetic Analysis",
        sections=[
            ("Design", body),
            ("Mean Run Summary By Method", dataframe_to_markdown(by_method, float_format=".4f")),
            ("Predictors Of Final Test Accuracy", dataframe_to_markdown(best, float_format=".4f")),
        ],
    )


def summarize_runs(df: pd.DataFrame, *, early_epoch: int) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(run_columns()):
        key_dict = dict(zip(run_columns(), keys))
        max_epoch = group["epoch"].max()
        final = group[group["epoch"] == max_epoch].mean(numeric_only=True)
        early = group[(group["epoch"] > 0) & (group["epoch"] <= early_epoch)].mean(numeric_only=True)
        baseline = group[group["epoch"] == 0].mean(numeric_only=True)
        if early.empty:
            early = group[group["epoch"] == min(max_epoch, 1)].mean(numeric_only=True)
        row = {
            **key_dict,
            "final_test_acc": float(final["test_acc"]),
            "final_train_acc": float(final["train_acc"]),
            "early_param_cosine": float(early.get("param_cosine", np.nan)),
            "early_activity_cosine": float(early.get("activity_cosine", np.nan)),
            "early_tangent_cosine": float(early.get("tangent_cosine", np.nan)),
            "early_task_tangent_cosine": float(early.get("task_tangent_cosine", np.nan)),
            "early_fisher_trace": float(early.get("fisher_trace", np.nan)),
            "early_weight_rayleigh": float(early.get("weight_rayleigh", np.nan)),
            "early_class_dprime2": float(early.get("class_dprime2", np.nan)),
            "early_effective_dim": float(early.get("effective_dim", np.nan)),
            "delta_fisher_trace_early": float(early.get("fisher_trace", np.nan) - baseline.get("fisher_trace", np.nan)),
            "delta_weight_rayleigh_early": float(
                early.get("weight_rayleigh", np.nan) - baseline.get("weight_rayleigh", np.nan)
            ),
            "delta_class_dprime2_early": float(
                early.get("class_dprime2", np.nan) - baseline.get("class_dprime2", np.nan)
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def minibatches(n_samples: int, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield indices[start : start + batch_size]


def plot_dfa_results(df: pd.DataFrame, output_dir: Path) -> None:
    run_df = summarize_runs(df, early_epoch=2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    acc_df = df[run_columns() + ["epoch", "test_acc"]].drop_duplicates()
    for method, group in acc_df.groupby("method"):
        curve = group.groupby("epoch")["test_acc"].mean()
        axes[0, 0].plot(curve.index, curve.values, marker="o", label=method)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Test accuracy")
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=7)

    layer1 = df[(df["layer"] == 1) & (df["method"] != "bp")]
    for method, group in layer1.groupby("method"):
        curve = group.groupby("epoch")["tangent_cosine"].mean()
        axes[0, 1].plot(curve.index, curve.values, marker="o", label=method)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Layer 1 tangent-projected BP/DFA cosine")
    axes[0, 1].legend(frameon=False, fontsize=7)

    dfa_runs = run_df[run_df["method"] != "bp"]
    axes[1, 0].scatter(dfa_runs["early_param_cosine"], dfa_runs["final_test_acc"], alpha=0.65, label="param")
    axes[1, 0].scatter(dfa_runs["early_tangent_cosine"], dfa_runs["final_test_acc"], alpha=0.65, label="tangent")
    axes[1, 0].scatter(
        dfa_runs["early_task_tangent_cosine"],
        dfa_runs["final_test_acc"],
        alpha=0.65,
        label="task tangent",
    )
    axes[1, 0].set_xlabel("Early alignment")
    axes[1, 0].set_ylabel("Final test accuracy")
    axes[1, 0].legend(frameon=False, fontsize=8)

    by_method = run_df.groupby("method")["final_test_acc"].mean().sort_values()
    axes[1, 1].barh(by_method.index, by_method.values)
    axes[1, 1].set_xlabel("Final test accuracy")
    axes[1, 1].set_xlim(0.0, 1.0)

    fig.savefig(output_dir / "dfa_synthetic_summary.png", dpi=180)
    plt.close(fig)


def run_columns() -> list[str]:
    return [
        "seed",
        "feedback_seed",
        "feedback_scale",
        "feedback_rank",
        "hidden_dim",
        "n_hidden_layers",
        "task_frequency",
        "input_noise",
        "manifold",
        "method",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke benchmark.")
    parser.add_argument("--output-dir", default="results/dfa_synthetic")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "bp",
            "dfa_random",
            "dfa_bp_aligned",
            "dfa_bp_orthogonal",
            "dfa_negative_bp",
            "dfa_tangent_biased",
            "dfa_tangent_orthogonal",
            "dfa_bp_aligned_dynamic",
            "dfa_bp_orthogonal_dynamic",
            "ndfa_random",
            "ndfa_random_error",
            "ndfa_random_kronecker",
            "drtp_random",
        ],
    )
    parser.add_argument("--manifold", default="circle", choices=["circle", "torus", "swiss_roll", "low_rank"])
    parser.add_argument("--manifolds", nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--feedback-scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--feedback-ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--input-noise", type=float, default=0.05, help="Deprecated alias for --input-noises.")
    parser.add_argument("--input-noises", type=float, nargs="+", default=None)
    parser.add_argument("--task-frequency", type=int, default=2, help="Deprecated alias for --task-frequencies.")
    parser.add_argument("--task-frequencies", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64, help="Deprecated alias for --hidden-dims.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--n-hidden-layers", type=int, default=2, help="Deprecated alias for --n-hidden-layers-list.")
    parser.add_argument("--n-hidden-layers-list", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-epoch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--tangent-rank", type=int, default=1)
    parser.add_argument("--natural-damping", type=float, default=1e-2)
    parser.add_argument("--natural-error-damping", type=float, default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based shard index for Slurm array runs.")
    parser.add_argument("--n-shards", type=int, default=None, help="Total number of run-spec shards.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional smoke-test limit on run specs.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Write partial CSV after this many run specs.")
    parser.add_argument("--skip-analysis", action="store_true", help="Only write raw results; aggregate shards later.")
    args = parser.parse_args()
    args.input_noises = args.input_noises if args.input_noises is not None else [args.input_noise]
    args.task_frequencies = args.task_frequencies if args.task_frequencies is not None else [args.task_frequency]
    args.hidden_dims = args.hidden_dims if args.hidden_dims is not None else [args.hidden_dim]
    args.n_hidden_layers_list = (
        args.n_hidden_layers_list if args.n_hidden_layers_list is not None else [args.n_hidden_layers]
    )
    args.manifolds = args.manifolds if args.manifolds is not None else [args.manifold]
    return args


if __name__ == "__main__":
    main()
