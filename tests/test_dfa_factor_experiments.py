from types import SimpleNamespace

import torch

from experiments.run_dfa_factorial_synthetic import anisotropize_feedback, condition_gradients
from infogeo.dfa import ManualMLP, init_feedback


def test_anisotropic_feedback_preserves_each_matrix_norm():
    model = ManualMLP(12, [9, 7], 4, seed=1)
    feedback = init_feedback(model, seed=2)
    scaled = anisotropize_feedback(feedback, ratio=100.0, seed=3)

    assert len(scaled) == len(feedback)
    for original, transformed in zip(feedback, scaled):
        torch.testing.assert_close(transformed.norm(), original.norm(), rtol=1e-6, atol=1e-6)
        assert not torch.allclose(transformed, original)


def test_factor_conditioners_norm_match_hidden_weight_gradients():
    model = ManualMLP(10, [8, 6], 3, seed=4)
    feedback = init_feedback(model, seed=5)
    x = torch.randn(16, 10, generator=torch.Generator().manual_seed(6))
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(7))
    args = SimpleNamespace(activity_damping=0.3, error_damping=1.0)
    raw = model.dfa_gradients(x, y, feedback)

    for method in ("ndfa", "endfa", "kndfa"):
        conditioned = condition_gradients(model, raw, x, method=method, args=args)
        for layer_idx in range(model.n_hidden_layers):
            torch.testing.assert_close(
                conditioned.weights[layer_idx].norm(),
                raw.weights[layer_idx].norm(),
                rtol=1e-5,
                atol=1e-6,
            )
        torch.testing.assert_close(conditioned.weights[-1], raw.weights[-1])
