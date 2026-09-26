"""Mathematical prototype of teaching-conditioned activity penalties.

For same-batch activities A, actual local errors D, and a separate teaching
descriptor T, this conditions G=D.T A/n by C_A + beta*S_gamma + lambda*I.
Without an intercept, S_gamma=C_A-C_AT(C_T+gamma*I)^-1 C_TA. This classical
regularized conditional statistic is not an identified noise covariance.
With an unpenalized intercept, S_gamma instead uses centered moments while the
base C_A and numerator G remain uncentered. No degrees-of-freedom correction,
norm matching, data generation, or training is performed here.

This is not a validated learning method or a strictly synaptic mechanism:
it uses batch statistics across presynaptic units. Same-batch fitting can
protect nuisance variation or interpolate the activities. In particular,
unregularized descriptors spanning all sample rows make S_gamma zero.

A future per-neuron scalar pre-gate T_j, distinct from gated D_j, avoids a
high-dimensional descriptor's full-row-rank interpolation and need not require
hidden-layer class labels. Distinct T_j and D_j can escape the scalar T=D
directional collapse. Different feedback directions can then define different
neuron metrics, unlike a full-rank multivariate projection preserving only the
shared output-error column space. A batched rank-one Sherman--Morrison method
could share a base C_A solve; that is a prospective implementation, not a new
matrix identity, locality result, or validated novelty claim.

Specifically, without an intercept, let P=((1+beta)*C_A+lambda*I)^-1,
v_j=A.T t_j/n, s_j=t_j.T t_j/n+gamma, and g_j=D_j.T A/n. For s_j>0,
u_j=g_j P + beta*(g_j P v_j)/(s_j-beta*v_j.T P v_j) * v_j.T P.
The denominator is positive because lambda>0. If s_j=0, t_j=0 and
u_j=g_j P directly. This permits shared solves with the G and V right-hand
sides. Per-row norm matching would remove the scalar T_j=D_j gain; matching
only the whole-layer norm can retain relative gains between neurons.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import torch

from infogeo.local_preconditioning import condition_local_update


_DTYPES = {torch.float32, torch.float64}
_BACKENDS = {"auto", "feature", "sample"}


def _matrix(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional tensor")
    if value.layout != torch.strided or value.dtype not in _DTYPES:
        raise TypeError(f"{name} must be a dense float32 or float64 tensor")


def _inputs(activity: torch.Tensor, teaching_signal: torch.Tensor,
            per_example_error: torch.Tensor | None = None) -> None:
    matrices = [("activity", activity), ("teaching_signal", teaching_signal)]
    if per_example_error is not None:
        matrices.append(("per_example_error", per_example_error))
    for name, value in matrices:
        _matrix(value, name)
        if value.shape[0] != activity.shape[0]:
            raise ValueError(f"{name} and activity must have the same sample count")
        if value.dtype != activity.dtype:
            raise TypeError(f"{name} and activity must have the same dtype")
        if value.device != activity.device:
            raise ValueError(f"{name} and activity must be on the same device")
    for name, value in matrices:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _options(penalty: float, teaching_ridge: float, intercept: bool,
             rank_rtol: float | None) -> tuple[float, float, float | None]:
    beta = _nonnegative(penalty, "penalty")
    gamma = _nonnegative(teaching_ridge, "teaching_ridge")
    if not isinstance(intercept, bool):
        raise TypeError("intercept must be a boolean")
    rtol = None if rank_rtol is None else _nonnegative(rank_rtol, "rank_rtol")
    return beta, gamma, rtol


class _Spectrum(NamedTuple):
    basis: torch.Tensor
    residual_eigenvalues: torch.Tensor
    reflector: torch.Tensor | None
    zero_residual: bool


def _reflect(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return matrix - 2 * vector[:, None] * (vector @ matrix)[None, :]


def _spectrum(teaching_signal: torch.Tensor, gamma: float, intercept: bool,
              rank_rtol: float | None) -> _Spectrum:
    n, q = teaching_signal.shape
    descriptor = teaching_signal / math.sqrt(n)
    reflector = None
    if intercept:
        # An implicit Householder contrast avoids an n x n projector and keeps
        # the constant coordinate separate even for a wide descriptor. Center
        # relative to one row first so a constant descriptor is exactly zero.
        descriptor = descriptor - descriptor[:1]
        descriptor = descriptor - descriptor.mean(dim=0, keepdim=True)
        reflector = descriptor.new_full((n,), 1 / math.sqrt(n))
        reflector[0] += 1
        reflector = reflector / torch.linalg.vector_norm(reflector)
        descriptor = _reflect(descriptor, reflector)[1:]
    m = descriptor.shape[0]
    if m == 0:
        return _Spectrum(descriptor.new_empty((0, 0)),
                         descriptor.new_empty((0,), dtype=torch.float64), reflector, True)
    if not bool(torch.isfinite(descriptor).all()):
        raise FloatingPointError("centering the teaching descriptor produced non-finite values")
    basis, singular_values, _ = torch.linalg.svd(descriptor, full_matrices=False)
    if not bool(torch.isfinite(singular_values).all()):
        raise FloatingPointError("teaching SVD produced non-finite singular values")
    if gamma == 0:
        rtol = max(n, q) * torch.finfo(teaching_signal.dtype).eps if rank_rtol is None else rank_rtol
        largest = singular_values[0]
        keep = singular_values > 0 if bool(largest == 0) else singular_values / largest > rtol
        basis = basis[:, keep]
        residual = singular_values.new_zeros((basis.shape[1],), dtype=torch.float64)
        return _Spectrum(basis, residual, reflector, basis.shape[1] == m)
    # Compute gamma/(sigma^2+gamma) directly: 1-h loses small positive residual
    # eigenvalues. Float64 hypot avoids squared-singular-value overflow and
    # permits positive ridges below float32's representable range.
    root_ridge = torch.full_like(singular_values, math.sqrt(gamma), dtype=torch.float64)
    residual = (root_ridge / torch.hypot(singular_values.double(), root_ridge)).square()
    return _Spectrum(basis, residual.clamp(0, 1), reflector, False)


def _apply(matrix: torch.Tensor, spectrum: _Spectrum, beta: float,
           *, inverse: bool) -> torch.Tensor:
    if spectrum.zero_residual:
        return matrix
    rotated = matrix if spectrum.reflector is None else _reflect(matrix, spectrum.reflector)
    coordinates = rotated if spectrum.reflector is None else rotated[1:]
    basis = spectrum.basis
    weights = (1 + beta * spectrum.residual_eigenvalues).sqrt()
    scale = math.sqrt(1 + beta)
    if inverse:
        weights = weights.reciprocal()
        scale = 1 / scale
    coefficients = basis.T @ coordinates
    transformed = basis @ (weights.to(matrix.dtype)[:, None] * coefficients)
    if basis.shape[1] < coordinates.shape[0]:
        transformed = transformed + scale * (coordinates - basis @ coefficients)
    if spectrum.reflector is not None:
        transformed = _reflect(torch.cat((rotated[:1], transformed)), spectrum.reflector)
    if not bool(torch.isfinite(transformed).all()):
        raise FloatingPointError("teaching conditioning produced non-finite samples")
    return transformed


@torch.no_grad()
def teaching_conditioning_samples(
    activity: torch.Tensor,
    teaching_signal: torch.Tensor,
    *,
    penalty: float,
    teaching_ridge: float,
    intercept: bool = False,
    rank_rtol: float | None = None,
) -> torch.Tensor:
    """Return W^1/2 A with moment C_A + penalty*S_gamma, as a tensor.

    A has shape (n,d), T has shape (n,q); both must be finite dense float32/64
    tensors with the same dtype/device. W=I+penalty*(I-H), with
    H=T(T.T T+n*teaching_ridge*I)^-1 T.T. With ``intercept=True``, H additionally
    preserves the constant subspace and fits centered T with the same ridge.
    At ridge zero a numerical pseudoprojection replaces the inverse, retaining
    singular values above ``rank_rtol * sigma_max``; the default relative
    threshold is max(n,q)*dtype_epsilon. This tolerance is unused for positive
    ridge, where all singular values receive continuous shrinkage.

    This is the additive Schur-complement metric, not raw ridge residual SSE:
    S_gamma = residual.T residual/n + teaching_ridge*B.T B for ridge-fit B.
    There is no automatic degrees-of-freedom correction. Using original D with
    these samples generally changes G: use ``teaching_conditioning_transforms``
    or ``teaching_residual_activity_update`` to preserve the actual update.
    Penalty zero and a zero residual return the original activity object.
    """
    _inputs(activity, teaching_signal)
    beta, gamma, rtol = _options(penalty, teaching_ridge, intercept, rank_rtol)
    if beta == 0:
        return activity
    return _apply(activity, _spectrum(teaching_signal, gamma, intercept, rtol), beta,
                  inverse=False)


@torch.no_grad()
def teaching_conditioning_transforms(
    activity: torch.Tensor,
    per_example_error: torch.Tensor,
    teaching_signal: torch.Tensor,
    *,
    penalty: float,
    teaching_ridge: float,
    intercept: bool = False,
    rank_rtol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (W^1/2 A, W^-1/2 D), preserving D.T A/n on the same batch.

    D has shape (n,p) and contains local errors before mean-loss normalization;
    T is an independently specified (n,q) descriptor, which may differ from D.
    All inputs share a floating dtype/device. Metric and rank conventions are
    those of ``teaching_conditioning_samples``. Only the metric changes: these
    transformed errors must not supply a replacement error moment to a
    two-sided conditioner. Original input tensors are never modified.
    """
    _matrix(per_example_error, "per_example_error")
    _inputs(activity, teaching_signal, per_example_error)
    beta, gamma, rtol = _options(penalty, teaching_ridge, intercept, rank_rtol)
    if beta == 0:
        return activity, per_example_error
    spectrum = _spectrum(teaching_signal, gamma, intercept, rtol)
    return (_apply(activity, spectrum, beta, inverse=False),
            _apply(per_example_error, spectrum, beta, inverse=True))


@torch.no_grad()
def teaching_residual_activity_update(
    activity: torch.Tensor,
    per_example_error: torch.Tensor,
    teaching_signal: torch.Tensor,
    *,
    penalty: float,
    teaching_ridge: float,
    damping: float,
    backend: str = "auto",
    intercept: bool = False,
    rank_rtol: float | None = None,
) -> torch.Tensor:
    """Return G(C_A+penalty*S_gamma+damping*I)^-1, with no norm matching.

    Damping is a positive finite scalar. Activity-based damping must be computed
    from the original A. The feature/sample/auto solve reuses the existing exact
    local conditioner with reciprocal sample transforms. Penalty zero calls
    ordinary conditioning on the unchanged inputs for bitwise equivalence.

    S_gamma is a conditional statistic, not necessarily true noise. At positive
    teaching ridge it includes the ridge-fit coefficient penalty, and cannot
    be replaced by raw fit-residual covariance. With no intercept and scalar
    T=D, this update has the same own-norm-matched direction as ordinary
    activity conditioning with damping/(1+penalty). A distinct T can change
    that direction. No loss-descent, strict-locality, or performance claim is
    made by this mathematical prototype.
    """
    ridge = _nonnegative(damping, "damping")
    if ridge == 0:
        raise ValueError("damping must be positive and finite")
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
    transformed_activity, transformed_error = teaching_conditioning_transforms(
        activity, per_example_error, teaching_signal, penalty=penalty,
        teaching_ridge=teaching_ridge, intercept=intercept, rank_rtol=rank_rtol,
    )
    return condition_local_update(transformed_activity, transformed_error,
                                  activity_damping=ridge, backend=backend)
