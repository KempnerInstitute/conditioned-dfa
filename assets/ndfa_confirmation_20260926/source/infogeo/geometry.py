"""Linear algebra utilities for representation geometry under noise."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


def stable_covariance(
    x: np.ndarray,
    *,
    shrinkage: float = 1e-3,
    ridge: float = 1e-8,
    center: bool = True,
) -> np.ndarray:
    """Return a shrinkage covariance estimate for row-major observations."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_samples, n_features)")
    if x.shape[0] < 2:
        scale = float(np.mean(np.square(x))) if x.size else 1.0
        return (scale + ridge) * np.eye(x.shape[1])

    x_centered = x - x.mean(axis=0, keepdims=True) if center else x
    cov = (x_centered.T @ x_centered) / max(x_centered.shape[0] - 1, 1)
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    return (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(cov.shape[0]) + ridge * np.eye(cov.shape[0])


def inv_sqrtm_psd(matrix: np.ndarray, *, ridge: float = 1e-8) -> np.ndarray:
    """Inverse square root of a symmetric positive-semidefinite matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, ridge)
    return (vecs * vals**-0.5) @ vecs.T


def psd_inverse(matrix: np.ndarray, *, ridge: float = 1e-8) -> np.ndarray:
    """Stable inverse of a symmetric positive-semidefinite matrix."""

    matrix = np.asarray(matrix, dtype=float)
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, ridge)
    return (vecs * vals**-1.0) @ vecs.T


def orthonormal_basis(vectors: np.ndarray, *, rtol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal column basis spanning the supplied columns."""

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    if vectors.ndim != 2:
        raise ValueError("vectors must be one- or two-dimensional")
    if vectors.size == 0:
        return np.zeros((vectors.shape[0], 0))

    u, s, _ = np.linalg.svd(vectors, full_matrices=False)
    if s.size == 0:
        return np.zeros((vectors.shape[0], 0))
    keep = s > (rtol * max(s[0], 1.0))
    return u[:, keep]


def projection_matrix(basis: np.ndarray) -> np.ndarray:
    """Orthogonal projection matrix for a column-space basis."""

    q = orthonormal_basis(basis)
    return q @ q.T


def project_onto_subspace(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project row-major vectors onto the column span of basis."""

    vectors = np.asarray(vectors, dtype=float)
    q = orthonormal_basis(basis)
    if q.shape[1] == 0:
        return np.zeros_like(vectors)
    return vectors @ q @ q.T


def subspace_fraction(vectors: np.ndarray, basis: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Fraction of squared vector norm lying in a subspace."""

    vectors = np.asarray(vectors, dtype=float)
    projected = project_onto_subspace(vectors, basis)
    numerator = np.sum(projected * projected, axis=-1)
    denominator = np.sum(vectors * vectors, axis=-1)
    return numerator / np.maximum(denominator, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray, *, axis: int | None = None, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity with safe zero-norm handling."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    numerator = np.sum(a * b, axis=axis)
    denominator = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return numerator / np.maximum(denominator, eps)


def principal_subspace(samples: np.ndarray, n_components: int) -> np.ndarray:
    """Principal column basis of row-major samples."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_features)")
    if n_components < 1:
        return np.zeros((samples.shape[1], 0))
    centered = samples - samples.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:n_components].T


def fisher_from_jacobian(
    jacobian: np.ndarray,
    covariance: np.ndarray | None = None,
    *,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Compute J^T Sigma^{-1} J for one or many Jacobians.

    `jacobian` may have shape `(n_features, latent_dim)` or
    `(n_points, n_features, latent_dim)`.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim not in (2, 3):
        raise ValueError("jacobian must be 2D or 3D")
    n_features = jacobian.shape[-2]
    inv_cov = np.eye(n_features) if covariance is None else psd_inverse(covariance, ridge=ridge)
    if jacobian.ndim == 2:
        return jacobian.T @ inv_cov @ jacobian
    return np.einsum("nfa,fg,ngb->nab", jacobian, inv_cov, jacobian)


def effective_dimension(values: np.ndarray, *, from_covariance: bool = False, eps: float = 1e-12) -> float:
    """Participation-ratio dimension from samples, covariance, or eigenvalues."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        eigvals = np.maximum(values, 0.0)
    elif from_covariance:
        eigvals = np.maximum(np.linalg.eigvalsh(0.5 * (values + values.T)), 0.0)
    else:
        cov = stable_covariance(values, shrinkage=0.0, ridge=0.0)
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    numerator = float(np.sum(eigvals) ** 2)
    denominator = float(np.sum(eigvals**2))
    return numerator / max(denominator, eps)


def class_dprime2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    shrinkage: float = 1e-2,
    ridge: float = 1e-6,
) -> float:
    """Mean pairwise noise-whitened class separation."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2:
        return 0.0

    if covariance is None:
        residuals = []
        for cls in classes:
            x_cls = x[y == cls]
            residuals.append(x_cls - x_cls.mean(axis=0, keepdims=True))
        covariance = stable_covariance(np.vstack(residuals), shrinkage=shrinkage, ridge=ridge, center=False)
    inv_cov = psd_inverse(covariance, ridge=ridge)

    means = {cls: x[y == cls].mean(axis=0) for cls in classes}
    distances: list[float] = []
    for a, b in itertools.combinations(classes, 2):
        delta = means[a] - means[b]
        distances.append(float(delta.T @ inv_cov @ delta))
    return float(np.mean(distances)) if distances else 0.0


def mean_of_dicts(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average numeric values across dictionaries."""

    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
