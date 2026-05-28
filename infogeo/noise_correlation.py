"""Noise-correlation feedback estimators for local learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from infogeo.dfa import Gradients, ManualMLP, _torch_cosine


@dataclass
class NoiseCorrelationStats:
    mean_feedback_norm: float
    mean_update_norm: float
    mean_noise_norm: float
    mean_delta_output_norm: float


def activity_pca_bases(
    model: ManualMLP,
    x: torch.Tensor,
    *,
    rank: int,
) -> list[torch.Tensor]:
    """Estimate one global PCA activity basis per hidden layer."""

    hidden = model.hidden_activations(x)
    bases: list[torch.Tensor] = []
    for h in hidden:
        h_centered = h.detach() - h.detach().mean(dim=0, keepdim=True)
        n_samples, hidden_dim = h_centered.shape
        n_components = int(min(max(rank, 1), n_samples, hidden_dim))
        if n_components <= 0:
            bases.append(torch.zeros(hidden_dim, 0, dtype=h.dtype, device=h.device))
            continue
        try:
            _, _, vh = torch.linalg.svd(h_centered, full_matrices=False)
            basis = vh[:n_components].T.contiguous()
        except RuntimeError:
            basis = torch.zeros(hidden_dim, n_components, dtype=h.dtype, device=h.device)
        bases.append(basis)
    return bases


def zero_feedback(model: ManualMLP) -> list[torch.Tensor]:
    """Feedback matrices with the shape expected by ``ManualMLP.dfa_gradients``."""

    return [
        torch.zeros(model.output_dim, hidden_dim, dtype=torch.float32, device=model.device)
        for hidden_dim in model.hidden_dims
    ]


def noise_correlation_feedback_update(
    model: ManualMLP,
    x: torch.Tensor,
    feedback: Sequence[torch.Tensor],
    *,
    mode: str,
    bases: Sequence[torch.Tensor] | None,
    eta: float,
    noise_scale: float,
    manifold_rank: int,
    rng: np.random.Generator,
    antithetic: bool = True,
    normalize_by_variance: bool = True,
) -> tuple[list[torch.Tensor], NoiseCorrelationStats]:
    """Update feedback matrices by correlating hidden perturbations with output changes.

    ``mode='vnc'`` injects isotropic full-space perturbations. ``mode='nmnc'``
    restricts perturbations to the supplied activity PCA bases. Perturbation
    energy is matched by scaling VNC with ``sqrt(d / n)``.
    """

    if mode not in {"vnc", "nmnc"}:
        raise ValueError("mode must be 'vnc' or 'nmnc'")
    if mode == "nmnc" and bases is None:
        raise ValueError("NMNC requires activity bases")
    if len(feedback) != model.n_hidden_layers:
        raise ValueError("feedback must contain one matrix per hidden layer")

    with torch.no_grad():
        base_logits, activations, _ = model.forward(x)
        updated: list[torch.Tensor] = []
        update_norms: list[float] = []
        feedback_norms: list[float] = []
        noise_norms: list[float] = []
        delta_norms: list[float] = []
        for layer_idx in range(model.n_hidden_layers):
            hidden = activations[layer_idx + 1].detach()
            noise = sample_hidden_noise(
                hidden.shape,
                mode=mode,
                basis=None if bases is None else bases[layer_idx],
                noise_scale=noise_scale,
                manifold_rank=manifold_rank,
                rng=rng,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if antithetic:
                plus = forward_from_hidden(model, layer_idx, hidden + noise)
                minus = forward_from_hidden(model, layer_idx, hidden - noise)
                delta_y = 0.5 * (plus - minus)
            else:
                plus = forward_from_hidden(model, layer_idx, hidden + noise)
                delta_y = plus - base_logits
            estimate = (delta_y.T @ noise) / max(int(x.shape[0]), 1)
            if normalize_by_variance:
                estimate = estimate / max(
                    perturbation_variance(
                        mode=mode,
                        hidden_dim=int(hidden.shape[1]),
                        manifold_rank=manifold_rank,
                        noise_scale=noise_scale,
                    ),
                    1e-12,
                )
            old = feedback[layer_idx].to(model.device)
            new = (1.0 - eta) * old + eta * estimate
            updated.append(new.detach())
            update_norms.append(float(torch.linalg.norm(estimate).item()))
            feedback_norms.append(float(torch.linalg.norm(new).item()))
            noise_norms.append(float(torch.linalg.norm(noise, dim=1).mean().item()))
            delta_norms.append(float(torch.linalg.norm(delta_y, dim=1).mean().item()))
    return updated, NoiseCorrelationStats(
        mean_feedback_norm=float(np.mean(feedback_norms)) if feedback_norms else float("nan"),
        mean_update_norm=float(np.mean(update_norms)) if update_norms else float("nan"),
        mean_noise_norm=float(np.mean(noise_norms)) if noise_norms else float("nan"),
        mean_delta_output_norm=float(np.mean(delta_norms)) if delta_norms else float("nan"),
    )


def perturbation_variance(
    *,
    mode: str,
    hidden_dim: int,
    manifold_rank: int,
    noise_scale: float,
) -> float:
    rank = int(min(max(manifold_rank, 1), hidden_dim))
    if mode == "nmnc":
        return float(noise_scale * noise_scale)
    return float(noise_scale * noise_scale * rank / max(hidden_dim, 1))


def sample_hidden_noise(
    shape: torch.Size | tuple[int, int],
    *,
    mode: str,
    basis: torch.Tensor | None,
    noise_scale: float,
    manifold_rank: int,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_samples, hidden_dim = int(shape[0]), int(shape[1])
    if mode == "nmnc":
        if basis is None or basis.numel() == 0:
            return torch.zeros(n_samples, hidden_dim, dtype=dtype, device=device)
        basis = basis.to(device=device, dtype=dtype)
        rank = int(basis.shape[1])
        z = rng.normal(size=(n_samples, rank)).astype(np.float32)
        return noise_scale * (torch.tensor(z, dtype=dtype, device=device) @ basis.T)
    rank = int(min(max(manifold_rank, 1), hidden_dim))
    sigma = noise_scale * np.sqrt(rank / max(hidden_dim, 1))
    z = rng.normal(scale=sigma, size=(n_samples, hidden_dim)).astype(np.float32)
    return torch.tensor(z, dtype=dtype, device=device)


def forward_from_hidden(model: ManualMLP, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
    """Continue a forward pass after hidden layer ``layer_idx``."""

    h = hidden.to(model.device)
    for weight_idx in range(layer_idx + 1, len(model.weights)):
        h = h @ model.weights[weight_idx].T + model.biases[weight_idx]
        if weight_idx < len(model.weights) - 1:
            h = torch.relu(h)
    return h


def noise_correlation_diagnostics(
    bp: Gradients,
    local: Gradients,
    bases: Sequence[torch.Tensor] | None,
) -> dict[str, float]:
    """Gradient diagnostics used by the NMNC/VNC comparison paper."""

    activity_cosines: list[float] = []
    projected_steps: list[float] = []
    norm_ratios: list[float] = []
    alphas: list[float] = []
    alpha_thresholds: list[float] = []
    for layer_idx in range(len(bp.deltas) - 1):
        true = bp.deltas[layer_idx].detach()
        estimate = local.deltas[layer_idx].detach()
        activity_cosines.append(_torch_cosine(true.flatten(), estimate.flatten()))
        true_norm2 = torch.sum(true * true).clamp_min(1e-12)
        projected_steps.append(float((torch.sum(true * estimate) / true_norm2).item()))
        norm_ratios.append(float((torch.linalg.norm(estimate) / torch.linalg.norm(true).clamp_min(1e-12)).item()))
        if bases is not None and layer_idx < len(bases) and bases[layer_idx].numel():
            basis = bases[layer_idx].to(device=true.device, dtype=true.dtype)
            projected = (true @ basis) @ basis.T
            alphas.append(float((torch.sum(projected * projected) / true_norm2).item()))
            alpha_thresholds.append(float(basis.shape[1] / max(true.shape[1], 1)))
    out = {
        "activity_cosine_mean": _nanmean(activity_cosines),
        "activity_angle_deg_mean": _angle_from_cosine(_nanmean(activity_cosines)),
        "projected_step_ratio_mean": _nanmean(projected_steps),
        "activity_norm_ratio_mean": _nanmean(norm_ratios),
        "manifold_gradient_alpha_mean": _nanmean(alphas),
        "manifold_dim_fraction_mean": _nanmean(alpha_thresholds),
    }
    out["manifold_condition_margin_mean"] = out["manifold_gradient_alpha_mean"] - out["manifold_dim_fraction_mean"]
    return out


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _angle_from_cosine(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))
