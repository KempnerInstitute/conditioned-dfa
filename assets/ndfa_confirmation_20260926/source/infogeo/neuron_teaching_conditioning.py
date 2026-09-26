"""Batched rank-one implementation of a scalar-teacher activity metric.

This mathematical prototype uses one uncentered teaching column per output
neuron, which may differ from its actual gated local error. It applies the
classical Sherman--Morrison identity to a shared activity solve; it introduces
neither a new identity nor a new credit signal. The conditional statistic is
not identified observation noise, and the shared dense activity statistics are
not strictly synaptic. No performance, biological-locality, or learning benefit
is established by implementing this operator.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from infogeo.local_preconditioning import condition_local_update


_DTYPES = {torch.float32, torch.float64}
_BACKENDS = {"auto", "feature", "sample"}


def _validate_inputs(activity: torch.Tensor, per_example_error: torch.Tensor,
                     teaching_signal: torch.Tensor) -> None:
    matrices = (("activity", activity), ("per_example_error", per_example_error),
                ("teaching_signal", teaching_signal))
    for name, matrix in matrices:
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if matrix.ndim != 2 or min(matrix.shape) == 0:
            raise ValueError(f"{name} must be a nonempty two-dimensional tensor")
        if matrix.layout != torch.strided or matrix.dtype not in _DTYPES:
            raise TypeError(f"{name} must be a dense float32 or float64 tensor")
        if matrix.shape[0] != activity.shape[0]:
            raise ValueError(f"{name} and activity must have the same sample count")
        if matrix.dtype != activity.dtype:
            raise TypeError(f"{name} and activity must have the same dtype")
        if matrix.device != activity.device:
            raise ValueError(f"{name} and activity must be on the same device")
    if teaching_signal.shape != per_example_error.shape:
        raise ValueError("teaching_signal and per_example_error must have the same shape")
    for name, matrix in matrices:
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError(f"{name} must contain only finite values")


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values; no fallback was applied")


def _normalize_teachers(teaching: torch.Tensor, gamma: float
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Normalize the ridge jointly with each column. Squaring the original T or
    # its scale would overflow for otherwise valid, very large descriptors.
    teacher64 = teaching.double()
    magnitude = teacher64.abs().amax(dim=0)
    nonzero = magnitude > 0
    root_gamma = torch.full_like(magnitude, math.sqrt(gamma))
    scale = torch.maximum(magnitude, root_gamma)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (teacher64 / scale).to(teaching.dtype)
    normalized_gamma = (root_gamma / scale).square()
    return normalized, normalized_gamma, nonzero


def _check_base_residual(activity64: torch.Tensor, targets64: torch.Tensor,
                         solved64: torch.Tensor, fitted64: torch.Tensor,
                         delta: float, dtype: torch.dtype) -> None:
    # A backward-error check detects a failed shared solve; it does not certify
    # forward accuracy for an ill-conditioned Gram matrix.
    n = activity64.shape[0]
    residual = (fitted64 - targets64).T @ activity64 / n + delta * solved64
    scale = ((torch.linalg.vector_norm(fitted64, dim=0)
              + torch.linalg.vector_norm(targets64, dim=0))
             * torch.linalg.vector_norm(activity64) / n
             + delta * torch.linalg.vector_norm(solved64, dim=1))
    _finite(residual, "shared base solve residual")
    _finite(scale, "shared base solve residual scale")
    tolerance = 64 * torch.finfo(dtype).eps * max(activity64.shape)
    bound = tolerance * scale.clamp_min(torch.finfo(torch.float64).tiny)
    if bool((torch.linalg.vector_norm(residual, dim=1) > bound).any()):
        raise FloatingPointError("shared base solve failed its residual check; no fallback was applied")


@torch.no_grad()
def neuron_teaching_residual_activity_update(
    activity: torch.Tensor,
    per_example_error: torch.Tensor,
    teaching_signal: torch.Tensor,
    *,
    penalty: float,
    teaching_ridge: float,
    damping: float,
    backend: str = "auto",
) -> torch.Tensor:
    """Return a (p,d) update with an individual scalar-teacher metric per row.

    Inputs are A=(n,d), D=(n,p), and T=(n,p), with finite float32/float64 values
    and a shared dtype/device. D contains actual local errors before mean-loss
    normalization; T may contain distinct pre-gate teaching signals. For each
    neuron j, define g_j=D_j.T A/n, v_j=T_j.T A/n, and
    s_j=mean(T_j^2)+teaching_ridge. This returns

        u_j = g_j [C_A + penalty*(C_A-v_j.T v_j/s_j) + damping*I]^-1.

    The rank-one term is zero when T_j=0, including s_j=0. With alpha=1+penalty
    and P=(alpha*C_A+damping*I)^-1, Sherman--Morrison gives

        u_j = g_j P + penalty*(g_j P v_j.T)/(s_j-penalty*v_j P v_j.T) * v_j P.

    One call to the existing activity conditioner solves the stacked D and
    normalized T right-hand sides with delta=damping/alpha. Both ``feature``
    and ``sample`` backends share that single factorization; ``auto`` selects
    sample space when n<d. There are no per-neuron Gram factorizations or SVDs.

    The denominator is evaluated as a sum of nonnegative ridge-fit energies,
    not by subtracting nearly equal moments. Teacher columns and their ridges
    are jointly rescaled internally. Float64 products/reductions and a base
    solve residual check improve numerical behavior, but the shared solve
    retains the input dtype. These safeguards do not restore forward accuracy
    lost to an ill-conditioned base system. Unrepresentable scaled damping,
    nonfinite intermediates, or a failed residual check raise; the method does
    not change damping or silently switch solvers. Returned dtype/device match
    A. Penalty zero calls ordinary conditioning directly for bitwise equality.

    This initial API is uncentered, with no intercept or degrees-of-freedom
    correction. No norm matching is applied. If T_j=D_j, per-row norm matching
    collapses the direction to ordinary activity conditioning at retuned
    damping delta; whole-layer matching may preserve different row gains.
    Distinct T_j can escape that degeneracy but need not do so. For paired
    A=[X;-X], T=[t;-t] and a bias-free ReLU gate without exact-zero preactivations,
    D=T*gate gives G=V/2: per-row directions still collapse despite T!=D.
    No activation model is imposed by this API, and no learning benefit or
    strictly synaptic noise-cancellation mechanism is established.
    """
    _validate_inputs(activity, per_example_error, teaching_signal)
    beta = _nonnegative(penalty, "penalty")
    gamma = _nonnegative(teaching_ridge, "teaching_ridge")
    ridge = _nonnegative(damping, "damping")
    if ridge == 0:
        raise ValueError("damping must be positive and finite")
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
    if beta == 0:
        return condition_local_update(activity, per_example_error,
                                      activity_damping=ridge, backend=backend)
    alpha = 1 + beta
    inverse_alpha = 1 / alpha
    delta = ridge * inverse_alpha
    represented_delta = torch.tensor(delta, dtype=activity.dtype).item()
    if delta == 0 or represented_delta == 0 or not math.isfinite(represented_delta):
        raise FloatingPointError("scaled activity damping is not representable in the input dtype")

    teaching, normalized_gamma, nonzero = _normalize_teachers(teaching_signal, gamma)
    if not bool(nonzero.any()):
        base = condition_local_update(activity, per_example_error,
                                      activity_damping=delta, backend=backend)
        result = (base.double() * inverse_alpha).to(activity.dtype)
        _finite(result, "neuron teaching update")
        return result

    p = per_example_error.shape[1]
    targets = torch.cat((per_example_error, teaching), dim=1)
    solved = condition_local_update(activity, targets, activity_damping=delta, backend=backend)
    activity64, targets64, solved64 = activity.double(), targets.double(), solved.double()
    fitted = activity64 @ solved64.T
    _finite(fitted, "shared base fitted samples")
    _check_base_residual(activity64, targets64, solved64, fitted, delta, activity.dtype)
    r_gradient, teacher_weights = solved64[:p], solved64[p:]
    fitted_teacher = fitted[:, p:]
    teacher_residual = targets64[:, p:] - fitted_teacher

    # For w=(C_A+delta I)^-1 v, the positive denominator is
    # alpha*den = alpha*gamma + alpha*mean((t-Aw)^2)
    #             + damping*||w||^2 + mean((Aw)^2) + delta*||w||^2.
    # Keeping alpha*den avoids forming a huge correction and dividing it later.
    denominator = (alpha * normalized_gamma
                   + alpha * teacher_residual.square().mean(dim=0)
                   + (math.sqrt(ridge) * teacher_weights).square().sum(dim=1)
                   + fitted_teacher.square().mean(dim=0)
                   + (math.sqrt(delta) * teacher_weights).square().sum(dim=1))
    _finite(denominator, "scaled rank-one denominator")
    if bool((denominator[nonzero] <= 0).any()):
        raise FloatingPointError("nonpositive rank-one denominator; no fallback was applied")
    cross_moment = targets64[:, p:].T @ activity64 / activity.shape[0]
    coupling = (r_gradient * cross_moment).sum(dim=1)
    safe_denominator = torch.where(nonzero, denominator, torch.ones_like(denominator))
    coefficient = (beta / alpha) * coupling / safe_denominator
    coefficient = torch.where(nonzero, coefficient, torch.zeros_like(coefficient))
    result64 = inverse_alpha * r_gradient + coefficient[:, None] * teacher_weights
    _finite(result64, "neuron teaching update")
    result = result64.to(activity.dtype)
    _finite(result, "neuron teaching update")
    return result
