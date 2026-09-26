"""Regime-adaptive conditioning for DFA: the rule measures its own regime.

Two mechanisms, both computed from quantities that are already local to the
layer update:

1. **Task-orientation gate (activity side).** The linearized analysis says
   activity conditioning pays exactly when credit lives in low-variance
   activity directions. In the linear model the raw update factorizes as
   ``G ~ M C_A``, so projecting ``G``'s rows onto ``C_A``'s eigenbasis and
   dividing each direction's energy by ``(lambda_i + eps)^2`` recovers the
   task loading ``||M_i||^2`` per eigendirection. The orientation statistic

       alpha = sum_i w_i lambda_i / mean(lambda),   w = normalized loading,

   is >> 1 when the task rides the high-variance directions (conditioning
   unnecessary) and << 1 when high-variance directions are nuisance
   (conditioning pays). The gate ``g = 1 / (1 + alpha**(1/temp))`` blends the
   raw and conditioned updates per layer. The conditioned direction uses the
   parameter-free spectral-relative damping ``lambda_A = 0.01 lambda_max(C_A)``,
   the geometric center of the selected-damping-to-spectrum ratios observed
   across the tuned factor studies.

2. **Spectral-relative error damping (error side).** Selected error dampings
   vary by 300x across studies while the error spectra vary just as much; a
   trajectory probe shows same-architecture studies coincide near
   ``lambda_E ~ lambda_max(C_E)``. The adaptive rule sets
   ``lambda_E = c_error * lambda_max(C_E)`` per layer per refresh, with one
   dimensionless ``c_error`` intended to transfer across tasks and scales
   (it is invariant to any rescaling of the error signal, which also removes
   the mean-loss-normalization failure mode by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from infogeo.dfa import Gradients, ManualMLP, error_second_moment


@dataclass
class AdaptiveDiagnostics:
    """Per-layer gate and damping decisions for logging."""

    alpha: list[float]
    gate: list[float]
    lambda_a: list[float]
    lambda_e: list[float]


def _sym_eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sym = 0.5 * (cov + cov.t()).double()
    jitter = 1e-10 * float(sym.trace()) / sym.shape[0]
    sym = sym + jitter * torch.eye(sym.shape[0], dtype=sym.dtype, device=sym.device)
    values, vectors = torch.linalg.eigh(sym)
    return values.clamp_min(0.0), vectors


def _damped_solve(cov: torch.Tensor, rhs: torch.Tensor, damping: float) -> torch.Tensor:
    sym = 0.5 * (cov + cov.t())
    eye = torch.eye(sym.shape[0], dtype=sym.dtype, device=sym.device)
    base = max(float(damping), 1e-8)
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        try:
            out = torch.linalg.solve(sym + (base * multiplier) * eye, rhs)
        except RuntimeError:
            continue
        if torch.isfinite(out).all():
            return out
    raise RuntimeError("adaptive damped solve failed")


def task_orientation(
    grad: torch.Tensor,
    cov: torch.Tensor,
    *,
    stat: str = "deconv",
    activity: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    top1_ref: float = 0.15,
) -> tuple[float, float]:
    """Return (alpha, lambda_a_auto) for one layer.

    ``alpha`` > 1 reads "task rides the high-variance directions" (gate toward
    raw DFA); ``alpha`` < 1 reads "high-variance directions are nuisance"
    (gate toward conditioning). ``lambda_a_auto`` is the mean active
    eigenvalue tr(C)/effrank(C).

    ``stat`` selects the candidate statistic:
    - ``deconv``: energy of the update's rows in the activity eigenbasis,
      deconvolved by ``(lambda_i + mean_lambda)^2`` (linear-theory task map).
    - ``scatter``: between-class scatter of the presynaptic activity along
      each eigendirection, normalized per direction (requires ``activity``
      and ``labels``); uses the same information channel the broadcast error
      already carries.
    - ``top1``: the retrospectively validated concentration statistic
      top-1 eigenvalue fraction of C_A, mapped through ``top1_ref``.
    """

    values, vectors = _sym_eigh(cov)
    tr = float(values.sum())
    if tr <= 0:
        return 1.0, 1e-8
    # Spectral-relative activity damping: selected lambda_A across the three
    # factor studies sits at ~0.001-0.05 x lambda_max, so 0.01 x lambda_max is
    # the parameter-free default (same form as the error side).
    lambda_a = max(0.01 * float(values.max()), 1e-10)
    mean_lambda = tr / values.numel()
    if stat == "top1":
        top1 = float(values.max()) / tr
        alpha = (top1 / max(top1_ref, 1e-6)) ** 2
        return alpha, lambda_a
    if stat == "scatter":
        if activity is None or labels is None:
            raise ValueError("scatter statistic requires activity and labels")
        act = activity.double()
        mu = act.mean(dim=0, keepdim=True)
        loading = torch.zeros_like(values)
        for cls in labels.unique():
            sel = act[labels == cls]
            if sel.shape[0] == 0:
                continue
            diff = (sel.mean(dim=0, keepdim=True) - mu) @ vectors
            loading = loading + (sel.shape[0] / act.shape[0]) * diff.squeeze(0).pow(2)
        loading = loading / (values + mean_lambda)
    else:
        energy = (grad.double() @ vectors).pow(2).sum(dim=0)
        loading = energy / (values + mean_lambda).pow(2)
    total = float(loading.sum())
    if total <= 0:
        return 1.0, lambda_a
    weights = loading / total
    alpha = float((weights * values).sum() / max(mean_lambda, 1e-30))
    return alpha, lambda_a


def adaptive_condition_gradients(
    model: ManualMLP,
    raw: Gradients,
    x: torch.Tensor,
    *,
    mode: str,
    c_error: float = 1.0,
    gate_temp: float = 1.0,
    gate_center: float = 1.0,
    gate_stat: str = "deconv",
    labels: torch.Tensor | None = None,
    norm_match: bool = True,
) -> tuple[Gradients, AdaptiveDiagnostics]:
    """Adaptively conditioned copy of ``raw``.

    ``mode`` is ``activity_gate``, ``error_spectral``, or ``kronecker`` (both).
    Per-layer norm matching to the raw gradient norm mirrors the fixed-damping
    protocol so accuracy differences reflect direction, not scale.
    """

    if mode not in {"activity_gate", "error_spectral", "kronecker"}:
        raise ValueError(f"unknown adaptive mode: {mode}")
    activations = [x.detach()] + [h.detach() for h in model.hidden_activations(x)]
    weights = [w.clone() for w in raw.weights]
    diag = AdaptiveDiagnostics([], [], [], [])
    for layer_idx in range(model.n_hidden_layers):
        grad = weights[layer_idx]
        raw_norm = float(grad.norm().clamp_min(1e-12))
        alpha = float("nan")
        gate = float("nan")
        lambda_a = float("nan")
        lambda_e = float("nan")
        if mode in {"activity_gate", "kronecker"}:
            activity = activations[layer_idx]
            cov_a = activity.t() @ activity / max(activity.shape[0], 1)
            alpha, lambda_a = task_orientation(
                grad, cov_a, stat=gate_stat, activity=activity, labels=labels
            )
            ratio = max(alpha, 1e-12) / max(gate_center, 1e-12)
            gate = 1.0 / (1.0 + ratio ** (1.0 / max(gate_temp, 1e-6)))
            conditioned = _damped_solve(cov_a, grad.t(), lambda_a).t()
            conditioned = conditioned * (raw_norm / float(conditioned.norm().clamp_min(1e-12)))
            grad = (1.0 - gate) * grad + gate * conditioned
        if mode in {"error_spectral", "kronecker"}:
            delta = raw.deltas[layer_idx].detach()
            cov_e = error_second_moment(delta, normalization_count=delta.shape[0])
            top = float(torch.linalg.matrix_norm(0.5 * (cov_e + cov_e.t()), ord=2))
            lambda_e = max(c_error * top, 1e-10)
            grad = _damped_solve(cov_e, grad, lambda_e)
        if norm_match:
            grad = grad * (raw_norm / float(grad.norm().clamp_min(1e-12)))
        weights[layer_idx] = grad
        diag.alpha.append(alpha)
        diag.gate.append(gate)
        diag.lambda_a.append(lambda_a)
        diag.lambda_e.append(lambda_e)
    conditioned = Gradients(
        weights,
        [b.clone() for b in raw.biases],
        raw.deltas,
        raw.loss,
        raw.bn_gammas,
        raw.bn_betas,
    )
    return conditioned, diag
