"""Exact feature- or sample-space conditioning of a local outer-product update.

The sample-space backend uses the push-through identity for a damped Gram
matrix. It changes the linear algebra, not the conditioner, its damping, or
the observations used to estimate its second moments. In particular, it does
not reuse an inverse from an earlier minibatch.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


_MODES = {"activity", "error", "kronecker"}
_BACKENDS = {"auto", "feature", "sample"}
_DTYPES = {torch.float32, torch.float64}


def _positive_damping(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    damping = float(value)
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return damping


def _ridge_solve(
    samples: torch.Tensor,
    rhs: torch.Tensor,
    damping: float,
    *,
    sample_space: bool,
) -> torch.Tensor:
    """Solve one exact damped second-moment system without changing its ridge."""
    n = samples.shape[0]
    gram = samples @ samples.T if sample_space else samples.T @ samples
    gram = gram / n
    system = gram + damping * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(system, rhs)


def condition_local_update(
    activity: torch.Tensor,
    per_example_error: torch.Tensor,
    *,
    activity_damping: float,
    error_damping: float | None = None,
    mode: str = "activity",
    backend: str = "auto",
) -> torch.Tensor:
    """Condition ``G = per_example_error.T @ activity / n`` exactly.

    Args:
        activity: Finite float32/float64 presynaptic samples, shape ``(n, din)``.
        per_example_error: Finite errors before mean-loss normalization, shape
            ``(n, dout)``, with the same dtype and device as ``activity``.
        activity_damping: Strictly positive ridge for the activity second moment.
        error_damping: Strictly positive error ridge; ``None`` reuses the activity
            ridge. Supplied dampings are validated even when their factor is unused.
        mode: ``activity`` returns ``G (A.T A/n + lambda_A I)^-1``; ``error``
            returns ``(D.T D/n + lambda_E I)^-1 G``; ``kronecker`` applies both.
        backend: ``feature`` solves in feature coordinates; ``sample`` solves
            only ``n x n`` systems. ``auto`` chooses sample space for each factor
            independently when ``n`` is smaller than its feature dimension.

    Returns:
        A ``(dout, din)`` tensor with the input dtype and device. The operator
        includes the minibatch mean once. It does not apply norm matching.

    The sample-space expression for both factors is
    ``D.T @ solve(K_E + lambda_E I, solve(K_A + lambda_A I, A)) / n``.
    The two sample systems generally do not commute. To reduce solve work when
    ``dout < din``, we transpose this expression and solve in the reverse order
    with ``D`` as the right-hand side. No feature-size Gram matrix is formed
    for a factor selected to use sample space.

    Singular systems or numerical failures raise an exception. There is no
    damping escalation, pseudoinverse, diagonal fallback, or stale-factor cache.
    """
    if not isinstance(activity, torch.Tensor) or not isinstance(per_example_error, torch.Tensor):
        raise TypeError("activity and per_example_error must be torch tensors")
    if activity.ndim != 2 or per_example_error.ndim != 2:
        raise ValueError("activity and per_example_error must be two-dimensional")
    if any(size == 0 for size in (*activity.shape, *per_example_error.shape)):
        raise ValueError("sample and feature dimensions must be nonzero")
    if activity.shape[0] != per_example_error.shape[0]:
        raise ValueError("activity and per_example_error must have the same sample count")
    if activity.dtype not in _DTYPES or per_example_error.dtype not in _DTYPES:
        raise TypeError("activity and per_example_error must use float32 or float64")
    if activity.dtype != per_example_error.dtype:
        raise TypeError("activity and per_example_error must have the same dtype")
    if activity.device != per_example_error.device:
        raise ValueError("activity and per_example_error must be on the same device")
    if not bool(torch.isfinite(activity).all() & torch.isfinite(per_example_error).all()):
        raise ValueError("activity and per_example_error must contain only finite values")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
    lambda_a = _positive_damping(activity_damping, "activity_damping")
    lambda_e = lambda_a if error_damping is None else _positive_damping(error_damping, "error_damping")

    n, din = activity.shape
    dout = per_example_error.shape[1]
    use_activity = mode in {"activity", "kronecker"}
    use_error = mode in {"error", "kronecker"}
    sample_a = use_activity and (backend == "sample" or (backend == "auto" and n < din))
    sample_e = use_error and (backend == "sample" or (backend == "auto" and n < dout))

    # The list is ordered for applying the operators to A on the right.
    sample_factors = []
    if sample_a:
        sample_factors.append((activity, lambda_a))
    if sample_e:
        sample_factors.append((per_example_error, lambda_e))
    if din <= dout:
        rhs = activity
        for samples, damping in sample_factors:
            rhs = _ridge_solve(samples, rhs, damping, sample_space=True)
        update = per_example_error.T @ rhs / n
    else:
        rhs = per_example_error
        for samples, damping in reversed(sample_factors):
            rhs = _ridge_solve(samples, rhs, damping, sample_space=True)
        update = rhs.T @ activity / n

    if use_activity and not sample_a:
        update = _ridge_solve(activity, update.T, lambda_a, sample_space=False).T
    if use_error and not sample_e:
        update = _ridge_solve(per_example_error, update, lambda_e, sample_space=False)
    if not bool(torch.isfinite(update).all()):
        raise RuntimeError("conditioning produced a non-finite update; damping was not changed")
    return update
