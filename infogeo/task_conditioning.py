"""Experimental activity conditioning using supervised within-class variation.

The grouping labels identify a task, not an observed noise realization. The
within-class residual is a classical scatter estimate; it is not guaranteed to
contain only nuisance variation. Multimodal classes, nonlinear task structure,
and incorrect labels can make suppressing this variation harmful.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from infogeo.local_preconditioning import condition_local_update


_FLOAT_DTYPES = {torch.float32, torch.float64}
_LABEL_DTYPES = {torch.int32, torch.int64}


def _validate_labels(labels: torch.Tensor) -> None:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch tensor")
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a nonempty one-dimensional tensor")
    if labels.dtype not in _LABEL_DTYPES:
        raise TypeError("labels must use int32 or int64 grouping identifiers")


def _validate_matrix(matrix: torch.Tensor, name: str) -> None:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional tensor")
    if matrix.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_options(penalty: float, df_correction: bool) -> float:
    if isinstance(penalty, bool) or not isinstance(penalty, Real):
        raise TypeError("penalty must be a finite nonnegative real number")
    beta = float(penalty)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if not isinstance(df_correction, bool):
        raise TypeError("df_correction must be a boolean")
    return beta


def _validate_inputs(activity: torch.Tensor, labels: torch.Tensor) -> None:
    _validate_matrix(activity, "activity")
    _validate_labels(labels)
    if activity.shape[0] != labels.shape[0]:
        raise ValueError("activity and labels must have the same sample count")
    if activity.device != labels.device:
        raise ValueError("activity and labels must be on the same device")


def _grouping(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Allocate by the number of observed groups, never max(label) or a supplied
    # total class count. Negative and nonconsecutive identifiers are valid.
    _, inverse, counts = torch.unique(labels, sorted=True, return_inverse=True,
                                      return_counts=True)
    return inverse, counts


def _projection(matrix: torch.Tensor, inverse: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
    means = matrix.new_zeros((counts.numel(), matrix.shape[1]))
    means.index_add_(0, inverse, matrix)
    means = means / counts.to(dtype=matrix.dtype).unsqueeze(1)
    return means[inverse]


def _residual_scale(n: int, groups: int, df_correction: bool) -> float:
    degrees = n - groups
    if degrees == 0:
        return 0.0
    return n / degrees if df_correction else 1.0


def _transform_scale(beta: float, residual_scale: float) -> float:
    squared = 1.0 + beta * residual_scale
    if not math.isfinite(squared):
        raise ValueError("penalty and residual correction produce a non-finite scale")
    return math.sqrt(squared)


@torch.no_grad()
def task_conditioning_diagnostics(
    labels: torch.Tensor, *, df_correction: bool = True,
) -> dict[str, int | float | bool]:
    """Describe the batch support for the within-class covariance estimate.

    ``residual_scale`` is zero when every group is a singleton, in which case
    the proposed penalty is skipped. The n/(n-K) correction estimates a common
    within-class covariance under independent samples with class-specific means
    and common covariance; it does not establish a noise interpretation.
    """
    _validate_labels(labels)
    _validate_options(0.0, df_correction)
    n = labels.numel()
    groups = torch.unique(labels).numel()
    return {
        "sample_count": n,
        "present_class_count": groups,
        "residual_degrees_of_freedom": n - groups,
        "residual_scale": _residual_scale(n, groups, df_correction),
        "penalty_available": n > groups,
    }


@torch.no_grad()
def task_conditioning_samples(
    activity: torch.Tensor,
    labels: torch.Tensor,
    *,
    penalty: float,
    df_correction: bool = True,
) -> torch.Tensor:
    """Return samples with second moment C_A + penalty * s * C_W_raw.

    Let P project sample rows onto their observed class means, R=(I-P)A, and
    C_W_raw=R.T R/n. This returns P A + sqrt(1+penalty*s) R, where s=n/(n-K)
    with the degrees-of-freedom correction, otherwise s=1. Only K groups present
    in this batch are used. If n=K, or penalty=0, ``activity`` is returned
    unchanged. The correction assumes independent samples and a common
    within-class covariance when interpreted as an unbiased covariance estimate.

    This function changes only metric samples. For an arbitrary update G, use
    an arbitrary-right-hand-side conditioner with these samples and the original
    damping. Using original errors with these transformed activities would
    change G; ``task_residual_activity_update`` preserves the numerator instead.
    """
    _validate_inputs(activity, labels)
    beta = _validate_options(penalty, df_correction)
    if beta == 0:
        return activity
    inverse, counts = _grouping(labels)
    residual_scale = _residual_scale(activity.shape[0], counts.numel(), df_correction)
    if residual_scale == 0:
        return activity
    scale = _transform_scale(beta, residual_scale)
    projected = _projection(activity, inverse, counts)
    transformed = projected + scale * (activity - projected)
    if not bool(torch.isfinite(transformed).all()):
        raise FloatingPointError("task conditioning produced non-finite metric samples")
    return transformed


@torch.no_grad()
def task_residual_activity_update(
    activity: torch.Tensor,
    per_example_error: torch.Tensor,
    labels: torch.Tensor,
    *,
    penalty: float,
    damping: float,
    backend: str = "auto",
    df_correction: bool = True,
) -> torch.Tensor:
    """Return G (C_A + penalty * s * C_W_raw + damping I)^-1 exactly.

    G=D.T A/n uses local errors before mean-loss normalization. With P, R, and s
    as in ``task_conditioning_samples``, the update minimizes

        ||U A.T - D.T||_F^2/(2n)
        + penalty*s*||U R.T||_F^2/(2n) + damping*||U||_F^2/2.

    Orthogonality of P and I-P permits reciprocal sample-row transforms of A
    and D. Their cross moment remains G, enabling the existing exact sample-space
    solve with no Woodbury subtraction. The transformed D second moment must not
    replace the original error moment in a two-sided conditioner. Damping must
    come from the original activities if it is scaled by an activity statistic.

    At penalty=0 or with all-singleton groups, this calls ordinary nDFA directly
    for bitwise equality. It applies no update norm matching and provides no
    guarantee of true-loss descent or unsupervised noise identification.
    """
    _validate_inputs(activity, labels)
    _validate_matrix(per_example_error, "per_example_error")
    if activity.shape[0] != per_example_error.shape[0]:
        raise ValueError("activity and per_example_error must have the same sample count")
    if activity.dtype != per_example_error.dtype:
        raise TypeError("activity and per_example_error must have the same dtype")
    if activity.device != per_example_error.device:
        raise ValueError("activity and per_example_error must be on the same device")
    beta = _validate_options(penalty, df_correction)
    if beta == 0:
        return condition_local_update(activity, per_example_error,
                                      activity_damping=damping, backend=backend)
    inverse, counts = _grouping(labels)
    residual_scale = _residual_scale(activity.shape[0], counts.numel(), df_correction)
    if residual_scale == 0:
        return condition_local_update(activity, per_example_error,
                                      activity_damping=damping, backend=backend)
    scale = _transform_scale(beta, residual_scale)
    projected_activity = _projection(activity, inverse, counts)
    projected_error = _projection(per_example_error, inverse, counts)
    transformed_activity = projected_activity + scale * (activity - projected_activity)
    transformed_error = projected_error + (per_example_error - projected_error) / scale
    return condition_local_update(transformed_activity, transformed_error,
                                  activity_damping=damping, backend=backend)
