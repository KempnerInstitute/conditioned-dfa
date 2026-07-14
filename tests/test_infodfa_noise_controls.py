import numpy as np
import torch
import torch.nn as nn

from experiments.run_dfa_multioutput_synthetic import (
    corrupt_labels,
    covariance_diagnostics,
    make_multioutput_dataset,
)
from experiments.run_dfa_synthetic import natural_precondition_gradients
from experiments.run_dfa_stall_comparison import exact_bp_hidden_deltas
from infogeo.dfa import ManualMLP, error_second_moment, init_feedback


def test_error_second_moment_undoes_mean_loss_scaling():
    per_example = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-2.0, 0.5], [0.0, 4.0]])
    batch = per_example.shape[0]
    stored_delta = per_example / batch
    expected = per_example.T @ per_example / batch

    actual = error_second_moment(stored_delta, normalization_count=batch)

    assert torch.allclose(actual, expected)


def test_error_second_moment_is_invariant_to_repeated_batch_normalization():
    per_example = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    repeated = per_example.repeat_interleave(3, dim=0)

    small = error_second_moment(per_example / len(per_example), normalization_count=len(per_example))
    large = error_second_moment(repeated / len(repeated), normalization_count=len(repeated))

    assert torch.allclose(small, large)


def test_error_damping_is_independent_of_activity_damping():
    model = ManualMLP(input_dim=5, hidden_dims=[7], output_dim=3, seed=2, device="cpu")
    x = torch.randn(16, 5)
    y = torch.arange(16) % 3
    feedback = init_feedback(model, seed=4, scale=1.0)
    raw = model.dfa_gradients(x, y, feedback)

    activity_low = natural_precondition_gradients(
        model, raw, x, damping=0.2, error_damping=0.1, mode="activity"
    )
    activity_high = natural_precondition_gradients(
        model, raw, x, damping=0.2, error_damping=10.0, mode="activity"
    )
    error_low = natural_precondition_gradients(
        model, raw, x, damping=0.2, error_damping=0.1, mode="error"
    )
    error_high = natural_precondition_gradients(
        model, raw, x, damping=0.2, error_damping=10.0, mode="error"
    )

    assert torch.allclose(activity_low.weights[0], activity_high.weights[0])
    assert not torch.allclose(error_low.weights[0], error_high.weights[0])


def test_stall_bp_error_source_matches_per_example_autograd():
    class TinyTanhMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 5), nn.Linear(5, 6), nn.Linear(6, 3)])
            self.preacts = []

        def forward(self, x):
            preact_1 = self.layers[0](x)
            preact_2 = self.layers[1](torch.tanh(preact_1))
            self.preacts = [preact_1, preact_2]
            return torch.sigmoid(self.layers[2](torch.tanh(preact_2)))

    torch.manual_seed(7)
    model = TinyTanhMLP()
    x = torch.randn(6, 4)
    labels = torch.arange(6) % 3
    targets = torch.nn.functional.one_hot(labels, num_classes=3).float()
    predictions = model(x)
    expected_by_layer = [[] for _ in model.preacts]
    for idx in range(x.shape[0]):
        loss = torch.nn.functional.binary_cross_entropy(predictions[idx], targets[idx], reduction="sum")
        full_deltas = torch.autograd.grad(loss, model.preacts, retain_graph=True)
        for layer_idx, full_delta in enumerate(full_deltas):
            expected_by_layer[layer_idx].append(full_delta[idx])

    actual = exact_bp_hidden_deltas(model, predictions.detach() - targets)

    for actual_layer, expected_rows in zip(actual, expected_by_layer):
        assert torch.allclose(actual_layer, torch.stack(expected_rows), atol=1e-6, rtol=1e-5)


def test_multioutput_label_noise_and_scale_overrides():
    clean = make_multioutput_dataset(
        condition="nuisance_dominant",
        n_train=96,
        n_test=64,
        input_dim=16,
        n_classes=8,
        nuisance_dim=6,
        input_noise=0.05,
        train_label_noise=0.0,
        test_label_noise=0.0,
        task_scale_override=0.9,
        nuisance_scale_override=1.7,
        seed=3,
    )
    noisy = make_multioutput_dataset(
        condition="nuisance_dominant",
        n_train=96,
        n_test=64,
        input_dim=16,
        n_classes=8,
        nuisance_dim=6,
        input_noise=0.05,
        train_label_noise=1.0,
        test_label_noise=0.0,
        task_scale_override=0.9,
        nuisance_scale_override=1.7,
        seed=3,
    )

    assert clean.task_scale == 0.9
    assert clean.nuisance_scale == 1.7
    assert clean.input_noise == 0.05
    assert noisy.train_label_noise == 1.0
    assert np.array_equal(clean.x_train, noisy.x_train)
    assert np.all(clean.y_train != noisy.y_train)
    assert np.array_equal(clean.y_test, noisy.y_test)


def test_corrupt_labels_never_keeps_corrupted_class():
    rng = np.random.default_rng(9)
    y = np.arange(24) % 8
    out = corrupt_labels(y, n_classes=8, noise=1.0, rng=rng)
    assert np.all(out != y)
    assert set(out).issubset(set(range(8)))


def test_covariance_diagnostics_are_finite_for_mlp_updates():
    model = ManualMLP(input_dim=10, hidden_dims=[12], output_dim=4, seed=0, device="cpu")
    x = torch.randn(32, 10)
    y = torch.arange(32) % 4
    feedback = init_feedback(model, seed=1, scale=1.0)
    bp = model.bp_gradients(x, y)
    local = model.dfa_gradients(x, y, feedback)
    diag = covariance_diagnostics(model, x, bp, local, damping=0.3)

    expected = {
        "pre_activity_condition_mean",
        "pre_activity_effective_rank_mean",
        "local_error_condition_mean",
        "local_error_effective_rank_mean",
        "bp_error_top1_fraction_mean",
    }
    assert expected.issubset(diag)
    assert all(np.isfinite(diag[key]) for key in expected)
