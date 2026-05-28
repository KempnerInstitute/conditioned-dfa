import numpy as np
import torch

from infogeo.dfa import (
    ManualMLP,
    finite_difference_hidden_tangents,
    init_feedback,
    local_pca_tangent_spaces,
    rayleigh_quotient_scores,
    tangent_projected_cosines,
)
from infogeo.geometry import principal_subspace
from infogeo.synthetic import circle_features, make_circle_split, make_manifold_split, manifold_features


def test_manual_dfa_one_update_changes_loss():
    data = make_circle_split(n_train=128, n_test=64, seed=2)
    model = ManualMLP(data.x_train.shape[1], [16], 2, seed=3)
    feature_fn = lambda z: circle_features(z, data.projection, noise=0.0).astype(np.float32)
    tangents = finite_difference_hidden_tangents(model, data.z_train[:64], feature_fn)
    bases = [principal_subspace(tangent, 1) for tangent in tangents]
    feedback = init_feedback(model, mode="random", tangent_bases=bases, seed=4)

    x = torch.tensor(data.x_train[:64], dtype=torch.float32)
    y = torch.tensor(data.y_train[:64], dtype=torch.long)
    before = model.dfa_gradients(x, y, feedback).loss
    gradients = model.dfa_gradients(x, y, feedback)
    model.apply_gradients(gradients, lr=0.1)
    after = model.dfa_gradients(x, y, feedback).loss
    assert np.isfinite(before)
    assert np.isfinite(after)
    assert abs(after - before) > 1e-8


def test_multidimensional_manifold_tangents_and_feedback_modes():
    data = make_manifold_split(manifold="torus", n_train=96, n_test=32, input_dim=10, seed=5)
    model = ManualMLP(data.x_train.shape[1], [12], 2, seed=6)
    feature_fn = lambda z: manifold_features(z, data.projection, manifold=data.manifold, noise=0.0).astype(np.float32)
    tangents = finite_difference_hidden_tangents(model, data.z_train[:24], feature_fn)
    assert tangents[0].shape == (24, 2, 12)

    basis = [principal_subspace(tangents[0].reshape(-1, tangents[0].shape[-1]), 2)]
    feedback = init_feedback(model, mode="tangent_orthogonal", tangent_bases=basis, rank=1, seed=7)
    assert feedback[0].shape == (2, 12)
    assert np.linalg.matrix_rank(feedback[0].detach().cpu().numpy()) == 1

    x = torch.tensor(data.x_train[:24], dtype=torch.float32)
    y = torch.tensor(data.y_train[:24], dtype=torch.long)
    bp = model.bp_gradients(x, y)
    dfa = model.dfa_gradients(x, y, feedback)
    scores = tangent_projected_cosines(bp, dfa, tangents)
    assert np.isfinite(scores["tangent_cosine_l1"])


def test_local_pca_tangent_spaces_shape():
    rng = np.random.default_rng(8)
    hidden = rng.normal(size=(40, 6))
    tangents = local_pca_tangent_spaces(hidden, n_neighbors=8, rank=3)
    assert tangents.shape == (40, 3, 6)
    assert np.all(np.isfinite(tangents))


def test_rayleigh_quotient_prefers_high_variance_axis():
    rng = np.random.default_rng(9)
    inputs = np.column_stack([3.0 * rng.normal(size=120), 0.2 * rng.normal(size=120)])
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = rayleigh_quotient_scores(inputs, weights, relative=False)
    assert scores[0] > scores[1]
