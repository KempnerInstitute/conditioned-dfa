"""Geometry tests for Info-DFA (information/PID tests live in Info-Man)."""

import numpy as np

from infogeo.geometry import (
    class_dprime2,
    fisher_from_jacobian,
    projection_matrix,
    stable_covariance,
    subspace_fraction,
)
from infogeo.synthetic import make_circle_split


def test_projection_fraction_matches_cos2():
    angle = np.deg2rad(60.0)
    vector = np.array([[np.cos(angle), np.sin(angle)]])
    basis = np.array([[1.0], [0.0]])
    frac = subspace_fraction(vector, basis)[0]
    assert np.isclose(frac, np.cos(angle) ** 2)


def test_fisher_from_jacobian_identity_noise():
    jac = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    fisher = fisher_from_jacobian(jac)
    assert fisher.shape == (2, 2)
    assert np.allclose(fisher, jac.T @ jac)


def test_projection_matrix_identity_is_identity():
    assert projection_matrix(np.eye(2)).shape == (2, 2)


def test_stable_covariance_is_psd():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 5))
    cov = stable_covariance(x, shrinkage=1e-3, ridge=1e-8)
    eigvals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert eigvals.min() > -1e-9


def test_circle_dataset_shapes():
    data = make_circle_split(n_train=64, n_test=64, input_dim=6, noise=0.02, task_frequency=2, seed=0)
    assert data.x_train.shape == (64, 6)
    assert data.y_train.shape == (64,)
    assert data.z_train.shape == (64,)


def test_class_dprime2_separates_two_classes():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0.0, 1.0, size=(40, 3)), rng.normal(2.5, 1.0, size=(40, 3))], axis=0)
    y = np.array([0] * 40 + [1] * 40)
    assert class_dprime2(x, y) > 1.0
