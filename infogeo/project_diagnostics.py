"""Lightweight project-level diagnostic for Info-DFA.

A fast smoke test for the core Info-DFA claims that does not replace the
publication sweeps. Used by ``experiments/run_project_diagnostics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .dfa import (
    Gradients,
    ManualMLP,
    finite_difference_hidden_tangents,
    gradient_cosines,
    init_feedback,
    tangent_projected_cosines,
)
from .geometry import class_dprime2, fisher_from_jacobian, principal_subspace, stable_covariance
from .synthetic import make_manifold_split, manifold_features, task_boundary_weights


@dataclass(frozen=True)
class InfoDfaDiagnosticConfig:
    """Configuration for the Info-DFA project smoke diagnostic."""

    n_train: int = 256
    n_test: int = 256
    input_dim: int = 10
    hidden_dim: int = 24
    n_hidden_layers: int = 1
    output_dim: int = 2
    manifold: str = "torus"
    task_frequency: int = 2
    input_noise: float = 0.04
    epochs: int = 3
    batch_size: int = 64
    eval_size: int = 96
    tangent_rank: int = 2
    lr: float = 0.08
    feedback_rank: int = 0
    feedback_scale: float = 1.0
    natural_damping: float = 0.3
    n_seeds: int = 2
    methods: tuple[str, ...] = (
        "bp",
        "dfa_random",
        "dfa_tangent_biased",
        "dfa_tangent_orthogonal",
        "ndfa_random",
        "ndfa_random_kronecker",
    )


def run_infodfa_diagnostic(config: InfoDfaDiagnosticConfig | None = None) -> pd.DataFrame:
    """Run a compact Info-DFA alignment and preconditioning test."""

    cfg = config or InfoDfaDiagnosticConfig()
    rows: list[dict[str, float | int | str]] = []
    for seed in range(cfg.n_seeds):
        dataset = make_manifold_split(
            manifold=cfg.manifold,
            n_train=cfg.n_train,
            n_test=cfg.n_test,
            input_dim=cfg.input_dim,
            noise=cfg.input_noise,
            task_frequency=cfg.task_frequency,
            seed=seed,
        )
        for method in cfg.methods:
            model = ManualMLP(
                input_dim=dataset.x_train.shape[1],
                hidden_dims=[cfg.hidden_dim] * cfg.n_hidden_layers,
                output_dim=cfg.output_dim,
                seed=10_000 + seed,
            )
            feature_fn = lambda z, dataset=dataset: manifold_features(
                z,
                dataset.projection,
                manifold=dataset.manifold,
                noise=0.0,
            ).astype(np.float32)
            z_eval = dataset.z_train[: min(cfg.eval_size, dataset.z_train.shape[0])]
            tangents = finite_difference_hidden_tangents(model, z_eval, feature_fn)
            tangent_bases = [
                principal_subspace(_flatten_tangent_samples(tangent), cfg.tangent_rank)
                for tangent in tangents
            ]
            feedback = None if method == "bp" else _init_project_feedback(model, method, tangent_bases, cfg, seed)

            x_train = torch.tensor(dataset.x_train, dtype=torch.float32)
            y_train = torch.tensor(dataset.y_train, dtype=torch.long)
            rng = np.random.default_rng(30_000 + seed)
            for epoch in range(cfg.epochs + 1):
                rows.append(_evaluate_infodfa_epoch(model, dataset, method, feedback, epoch, seed, cfg, feature_fn))
                if epoch == cfg.epochs:
                    break
                for batch_idx in _minibatches(dataset.x_train.shape[0], cfg.batch_size, rng):
                    raw = _method_gradients(model, method, x_train[batch_idx], y_train[batch_idx], feedback, cfg)
                    model.apply_gradients(raw, lr=cfg.lr)
    return pd.DataFrame(rows)


def summarize_infodfa_diagnostic(dfa: pd.DataFrame) -> pd.DataFrame:
    """Compact final-by-method summary for reports."""

    if dfa.empty:
        return dfa
    final = dfa.sort_values("epoch").groupby(["method", "seed"], as_index=False).tail(1)
    return (
        final.groupby("method", as_index=False)[
            [
                "test_acc",
                "hidden_projected_step",
                "hidden_param_cosine",
                "tangent_cosine",
                "fisher_trace",
                "class_dprime2",
            ]
        ]
        .mean(numeric_only=True)
        .sort_values("test_acc", ascending=False)
    )


def _evaluate_infodfa_epoch(
    model: ManualMLP,
    dataset,
    method: str,
    feedback: Sequence[torch.Tensor] | None,
    epoch: int,
    seed: int,
    cfg: InfoDfaDiagnosticConfig,
    feature_fn,
) -> dict[str, float | int | str]:
    n_eval = min(cfg.eval_size, dataset.x_train.shape[0], dataset.z_train.shape[0])
    x_eval = torch.tensor(dataset.x_train[:n_eval], dtype=torch.float32)
    y_eval = torch.tensor(dataset.y_train[:n_eval], dtype=torch.long)
    bp = model.bp_gradients(x_eval, y_eval)
    local = _method_gradients(model, method, x_eval, y_eval, feedback, cfg)
    cosines = gradient_cosines(bp, local)

    z_eval = dataset.z_train[:n_eval]
    tangents = finite_difference_hidden_tangents(model, z_eval, feature_fn)
    task_weights = task_boundary_weights(z_eval, manifold=dataset.manifold, task_frequency=cfg.task_frequency)
    tangent_cosines = tangent_projected_cosines(bp, local, tangents)
    task_tangent_cosines = tangent_projected_cosines(bp, local, tangents, sample_weights=task_weights)

    x_test = torch.tensor(dataset.x_test, dtype=torch.float32)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long)
    hidden = model.hidden_activations(x_eval)[-1].detach().cpu().numpy()
    hidden_cov = stable_covariance(hidden, shrinkage=0.05, ridge=1e-6)
    jacobian = np.transpose(tangents[-1], (0, 2, 1)) if tangents[-1].ndim == 3 else tangents[-1][:, :, None]
    fisher = fisher_from_jacobian(jacobian, hidden_cov, ridge=1e-6)
    return {
        "project": "Info-DFA",
        "method": method,
        "seed": seed,
        "epoch": epoch,
        "test_acc": model.accuracy(x_test, y_test),
        "loss": local.loss,
        "hidden_projected_step": _hidden_weight_projected_step(bp, local),
        "hidden_weight_norm_ratio": _hidden_weight_norm_ratio(local, bp),
        "hidden_param_cosine": float(cosines.get("param_cosine", np.nan)),
        "activity_cosine": float(cosines.get("activity_cosine_l1", np.nan)),
        "tangent_cosine": float(tangent_cosines.get("tangent_cosine_l1", np.nan)),
        "task_tangent_cosine": float(task_tangent_cosines.get("tangent_cosine_l1", np.nan)),
        "fisher_trace": float(np.mean(np.trace(fisher, axis1=-2, axis2=-1))),
        "class_dprime2": class_dprime2(hidden, y_eval.detach().cpu().numpy()),
    }


def _method_gradients(
    model: ManualMLP,
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    feedback: Sequence[torch.Tensor] | None,
    cfg: InfoDfaDiagnosticConfig,
) -> Gradients:
    if method == "bp":
        return model.bp_gradients(x, y)
    if feedback is None:
        raise ValueError(f"{method} requires feedback")
    raw = model.dfa_gradients(x, y, feedback)
    if method == "ndfa_random":
        return _natural_precondition_gradients(model, raw, x, damping=cfg.natural_damping, mode="activity")
    if method == "ndfa_random_kronecker":
        return _natural_precondition_gradients(model, raw, x, damping=cfg.natural_damping, mode="kronecker")
    return raw


def _init_project_feedback(
    model: ManualMLP,
    method: str,
    tangent_bases: Sequence[np.ndarray],
    cfg: InfoDfaDiagnosticConfig,
    seed: int,
) -> list[torch.Tensor]:
    if method in {"dfa_tangent_biased", "dfa_tangent_orthogonal"}:
        mode = method.removeprefix("dfa_")
    else:
        mode = "random"
    return init_feedback(
        model,
        mode=mode,
        tangent_bases=tangent_bases,
        seed=20_000 + seed,
        scale=cfg.feedback_scale,
        rank=None if cfg.feedback_rank <= 0 else cfg.feedback_rank,
    )


def _natural_precondition_gradients(
    model: ManualMLP,
    gradients: Gradients,
    x: torch.Tensor,
    *,
    damping: float,
    mode: str,
) -> Gradients:
    _, activations, _ = model.forward(x)
    weights = [grad.clone() for grad in gradients.weights]
    biases = [grad.clone() for grad in gradients.biases]
    for layer_idx in range(model.n_hidden_layers):
        grad = gradients.weights[layer_idx]
        if mode in {"activity", "kronecker"}:
            activity = activations[layer_idx].detach()
            cov = activity.T @ activity / max(activity.shape[0], 1)
            grad = _damped_solve(cov, grad.T, damping=damping).T
        if mode in {"error", "kronecker"}:
            delta = gradients.deltas[layer_idx].detach()
            cov = delta.T @ delta / max(delta.shape[0], 1)
            grad = _damped_solve(cov, grad, damping=damping)
        weights[layer_idx] = grad
    return Gradients(weights, biases, gradients.deltas, gradients.loss)


def _damped_solve(cov: torch.Tensor, rhs: torch.Tensor, *, damping: float) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    matrix = cov + max(float(damping), 1e-6) * eye
    return torch.linalg.solve(matrix, rhs)


def _hidden_weight_projected_step(reference: Gradients, estimate: Gradients) -> float:
    ref = torch.cat([grad.flatten() for grad in reference.weights[:-1]])
    est = torch.cat([grad.flatten() for grad in estimate.weights[:-1]])
    denom = torch.sum(ref * ref).clamp_min(1e-12)
    return float((torch.sum(ref * est) / denom).detach().cpu().item())


def _hidden_weight_norm_ratio(estimate: Gradients, reference: Gradients) -> float:
    est = torch.cat([grad.flatten() for grad in estimate.weights[:-1]])
    ref = torch.cat([grad.flatten() for grad in reference.weights[:-1]])
    return float((torch.linalg.norm(est) / torch.linalg.norm(ref).clamp_min(1e-12)).detach().cpu().item())


def _flatten_tangent_samples(tangent: np.ndarray) -> np.ndarray:
    tangent = np.asarray(tangent, dtype=float)
    if tangent.ndim == 2:
        return tangent
    if tangent.ndim == 3:
        return tangent.reshape(-1, tangent.shape[-1])
    raise ValueError("tangent must have shape (n, h) or (n, k, h)")


def _minibatches(n_samples: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield order[start : start + batch_size]
