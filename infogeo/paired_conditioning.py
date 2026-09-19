"""Experimental conditioning by within-pair activity variation.

This is an activity-side metric intervention, not a new credit generator or
an exact natural gradient. Task preservation is an assumption about the views.
The sample-space solve is the standard Woodbury identity.
"""

from __future__ import annotations

import math

import torch

from infogeo.local_preconditioning import condition_local_update


@torch.no_grad()
def right_condition_from_samples(
    gradient: torch.Tensor,
    samples: torch.Tensor,
    *,
    damping: float,
    backend: str = "auto",
) -> torch.Tensor:
    """Return G (S.T S / n + damping I)^-1 without requiring G's rows in S.

    Unlike the same-batch outer-product push-through shortcut, the arbitrary
    right-hand side matters here: a training update need not lie in the span
    of the paired differences used to estimate nuisance covariance. ``auto``
    solves in the smaller of sample and feature spaces. Sample-space Woodbury
    subtraction can lose accuracy at very small damping; use a dense reference
    and float64 when checking that regime. Damping is never silently changed.
    """
    if backend not in {"auto", "feature", "sample"}:
        raise ValueError("backend must be auto, feature, or sample")
    if gradient.ndim != 2 or samples.ndim != 2 or min(*gradient.shape, *samples.shape) < 1:
        raise ValueError("gradient and samples must be nonempty matrices")
    if gradient.shape[1] != samples.shape[1]:
        raise ValueError("gradient and samples must share the feature dimension")
    if gradient.dtype not in {torch.float32, torch.float64} or gradient.dtype != samples.dtype:
        raise ValueError("gradient and samples must share float32 or float64 dtype")
    if gradient.device != samples.device:
        raise ValueError("gradient and samples must share a device")
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    if not torch.isfinite(gradient).all() or not torch.isfinite(samples).all():
        raise ValueError("gradient and samples must be finite")
    n, width = samples.shape
    selected = ("sample" if n < width else "feature") if backend == "auto" else backend
    if selected == "feature":
        matrix = samples.T @ samples / n
        matrix.diagonal().add_(damping)
        result = torch.linalg.solve(matrix, gradient.T).T
    else:
        matrix = samples @ samples.T
        matrix.diagonal().add_(n * damping)
        coefficients = torch.linalg.solve(matrix, samples @ gradient.T)
        result = (gradient - coefficients.T @ samples) / damping
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite conditioned update; inspect scales and damping")
    return result


@torch.no_grad()
def paired_activity_update(
    gradient: torch.Tensor,
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    *,
    damping: float,
    backend: str = "auto",
) -> torch.Tensor:
    """Condition with C_delta = mean((a1-a2)(a1-a2).T) / 2.

    Under conditionally independent views of the same latent identity, the
    population factor is the expected within-identity covariance. Correlated
    or task-changing views do not generally have that interpretation. This
    function has no labels or backpropagated errors; estimating a dense factor
    still requires layer-wide, minibatch statistics.
    """
    if first_view.shape != second_view.shape:
        raise ValueError("paired views must have identical shapes")
    if first_view.dtype != second_view.dtype or first_view.device != second_view.device:
        raise ValueError("paired views must share dtype and device")
    differences = (first_view - second_view) / math.sqrt(2.0)
    return right_condition_from_samples(gradient, differences, damping=damping, backend=backend)


@torch.no_grad()
def view_regularized_activity_update(
    activity: torch.Tensor,
    per_example_error: torch.Tensor,
    *,
    penalty: float,
    damping: float,
    backend: str = "auto",
) -> torch.Tensor:
    """Return G (C_A + penalty C_delta + damping I)^-1.

    Rows contain all first views followed by their corresponding second views;
    errors precede mean-loss normalization. C_A uses every row and C_delta is
    half the mean outer product of within-pair differences. The result minimizes
    a local ridge-regression objective with an additional penalty on the change
    in the proposed update's response across views. It does not make the full
    network invariant, supply new credit, or guarantee true-loss descent.

    An orthogonal sum/difference transform, followed by reciprocal scaling of
    activities and errors, retains G exactly while changing only C_A. This
    permits the existing sample-space solver without Woodbury subtraction or
    extra samples. The transformed error moment must NOT be used as the original
    error factor in a two-sided conditioner. At penalty=0 this calls ordinary
    activity conditioning directly. Damping is supplied from the original
    activities, not recomputed from the transformed samples.
    """
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise TypeError("penalty must be a finite nonnegative real number")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if activity.ndim != 2 or per_example_error.ndim != 2:
        raise ValueError("activity and errors must be matrices")
    if activity.shape[0] < 2 or activity.shape[0] % 2:
        raise ValueError("activity must contain two equal, nonempty view halves")
    if activity.shape[0] != per_example_error.shape[0]:
        raise ValueError("activity and errors must have the same sample count")
    if activity.dtype not in {torch.float32, torch.float64}:
        raise TypeError("activity and errors must use float32 or float64")
    if activity.dtype != per_example_error.dtype or activity.device != per_example_error.device:
        raise ValueError("activity and errors must share dtype and device")
    if penalty == 0:
        return condition_local_update(activity, per_example_error,
                                      activity_damping=damping, mode="activity", backend=backend)
    first, second = activity.chunk(2)
    first_error, second_error = per_example_error.chunk(2)
    scale = math.sqrt(1 + 2 * penalty)
    transformed_activity = torch.cat([(first + second) / math.sqrt(2),
                                      (first - second) * (scale / math.sqrt(2))])
    transformed_error = torch.cat([(first_error + second_error) / math.sqrt(2),
                                   (first_error - second_error) / (scale * math.sqrt(2))])
    return condition_local_update(transformed_activity, transformed_error,
                                  activity_damping=damping, mode="activity", backend=backend)
