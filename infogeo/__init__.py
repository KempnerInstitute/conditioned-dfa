"""Conditioned DFA: geometry and DFA utilities."""

from .geometry import (
    class_dprime2,
    cosine_similarity,
    effective_dimension,
    fisher_from_jacobian,
    inv_sqrtm_psd,
    orthonormal_basis,
    principal_subspace,
    projection_matrix,
    stable_covariance,
    subspace_fraction,
)

__all__ = [
    "class_dprime2",
    "cosine_similarity",
    "effective_dimension",
    "fisher_from_jacobian",
    "inv_sqrtm_psd",
    "orthonormal_basis",
    "principal_subspace",
    "projection_matrix",
    "stable_covariance",
    "subspace_fraction",
]
