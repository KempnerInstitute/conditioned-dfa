import pytest
import torch
import torch.nn.functional as F

from infogeo.decorrelated_dfa import DecorrelatedGradients, DecorrelatedManualMLP
from infogeo.dfa import ManualMLP, init_feedback


def _model(hidden=(4, 3), *, backend="low_rank", dtype=torch.float64):
    model = DecorrelatedManualMLP(5, hidden, 3, seed=120, decor_lr=1e-3,
                                 decor_backend=backend, dtype=dtype)
    generator = torch.Generator().manual_seed(121)
    # Nonidentity, nonsymmetric maps expose omitted/transposed pullbacks.
    for decorator in model.decorators:
        width = decorator.num_features
        decorator.decor_weight = (torch.eye(width, dtype=dtype)
                                  + .1 * torch.randn(width, width, generator=generator, dtype=dtype))
        decorator.running_mean = .2 * torch.randn(width, generator=generator, dtype=dtype)
    for index, bias in enumerate(model.biases):
        model.biases[index] = .1 * torch.randn(bias.shape, generator=generator, dtype=dtype)
    return model


def _data(dtype=torch.float64):
    generator = torch.Generator().manual_seed(122)
    x = torch.randn(9, 5, generator=generator, dtype=dtype) + .4
    return x, torch.arange(len(x)) % 3


def _reference(model, x):
    weights = [weight.detach().clone().requires_grad_() for weight in model.weights]
    biases = [bias.detach().clone().requires_grad_() for bias in model.biases]
    matrices = [decorator.decor_weight.detach().clone() for decorator in model.decorators]
    means = [decorator.running_mean.detach().clone() for decorator in model.decorators]
    raw, transformed, preactivations, states = [], [], [], []
    activity = x.clone()
    for index, (weight, bias, matrix, mean, decorator) in enumerate(zip(weights, biases, matrices, means, model.decorators)):
        raw.append(activity)
        batch_mean = activity.mean(0)
        center = batch_mean if model.training else mean
        inputs = (activity - center) @ matrix
        transformed.append(inputs)
        z = inputs @ weight.T + bias
        preactivations.append(z)
        if model.training:
            count = min(len(x), int(decorator.sample_fraction * len(x)) + 1)
            sample = inputs.detach()[:count]
            covariance = sum(torch.outer(row, row) for row in sample) / count
            gain = torch.stack([
                source.detach().square().sum().sqrt() / (target.square().sum() + 1e-8).sqrt()
                for source, target in zip(activity[:count], sample)
            ]).mean()
            states.append((gain * matrix - decorator.lr * (covariance @ (gain * matrix)),
                           (1 - decorator.mean_momentum) * mean + decorator.mean_momentum * batch_mean.detach()))
        else:
            states.append((matrix, mean))
        activity = torch.relu(z) if index < len(weights) - 1 else z
    return activity, weights, biases, preactivations, raw, transformed, states


@pytest.mark.parametrize("hidden", [(), (4,), (4, 3)])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("backend", ["dense", "low_rank"])
def test_exact_bp_matches_independent_autograd_with_classifier_decorator_and_correct_single_state_update(hidden, training, backend):
    model = _model(hidden, backend=backend)
    model.training = training
    x, y = _data()
    logits, weights, biases, preactivations, raw, inputs, expected_states = _reference(model, x)
    expected_loss = F.cross_entropy(logits, y)
    expected_gradients = torch.autograd.grad(expected_loss, [*weights, *biases, *preactivations])
    old_matrices = [decorator.decor_weight for decorator in model.decorators]
    old_means = [decorator.running_mean for decorator in model.decorators]

    actual = model.bp_gradients(x, y)

    assert isinstance(actual, DecorrelatedGradients)
    assert len(model.decorators) == len(model.weights)
    assert actual.loss == pytest.approx(expected_loss.item(), abs=1e-13)
    torch.testing.assert_close(actual.forward_cache.logits, logits, rtol=2e-12, atol=2e-13)
    for actual_gradient, expected in zip([*actual.weights, *actual.biases, *actual.deltas], expected_gradients):
        torch.testing.assert_close(actual_gradient, expected, rtol=2e-11, atol=2e-13)
    for index, (decorator, (expected_matrix, expected_mean)) in enumerate(zip(model.decorators, expected_states)):
        torch.testing.assert_close(decorator.decor_weight, expected_matrix, rtol=2e-12, atol=2e-13)
        torch.testing.assert_close(decorator.running_mean, expected_mean, rtol=2e-12, atol=2e-13)
        assert actual.forward_cache.decorator_caches[index].matrix is old_matrices[index]
        if not training:
            assert decorator.decor_weight is old_matrices[index]
            assert decorator.running_mean is old_means[index]
    for actual_input, expected in zip(model.conditioning_activities(actual), inputs):
        torch.testing.assert_close(actual_input, expected, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("training", [False, True])
def test_dfa_uses_fixed_gated_preactivation_teacher_and_cached_transformed_outer_products(training):
    model = _model()
    model.training = training
    x, y = _data()
    feedback = init_feedback(model, seed=123, scale=.25)
    saved_feedback = [matrix.clone() for matrix in feedback]
    raw = model.dfa_gradients(x, y, feedback)
    cache = raw.forward_cache
    output_delta = (cache.logits.softmax(1) - F.one_hot(y, 3)) / len(y)

    for index in range(len(model.weights)):
        if index < model.n_hidden_layers:
            expected_delta = (output_delta @ feedback[index].double()) * (cache.preactivations[index] > 0)
        else:
            expected_delta = output_delta
        torch.testing.assert_close(raw.deltas[index], expected_delta, rtol=0, atol=0)
        torch.testing.assert_close(raw.weights[index], expected_delta.T @ cache.linear_inputs[index], rtol=0, atol=0)
        torch.testing.assert_close(raw.biases[index], expected_delta.sum(0), rtol=0, atol=0)
    for actual, expected in zip(feedback, saved_feedback):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Injecting into decorated coordinates would be a different credit rule.
    transformed_teacher = (output_delta @ feedback[0].double()) @ cache.decorator_caches[1].matrix.T
    if training:
        transformed_teacher -= transformed_teacher.mean(0)
    transformed_teacher *= cache.preactivations[0] > 0
    assert not torch.allclose(raw.deltas[0], transformed_teacher)


def test_cached_gradients_and_conditioning_survive_new_forward_parameter_update_and_evaluation():
    model = _model()
    model.training = True
    x, y = _data()
    feedback = init_feedback(model, seed=123)
    dfa = model.dfa_gradients(x, y, feedback)
    cache = dfa.forward_cache
    original_inputs = model.conditioning_activities(dfa)
    copied_inputs = [value.clone() for value in original_inputs]
    original_bp = model.bp_gradients_from_cache(cache, y)
    state_after_first = [(decorator.decor_weight, decorator.running_mean) for decorator in model.decorators]
    repeated_dfa = model.dfa_gradients_from_cache(cache, y, feedback)
    for original, actual in zip(dfa.weights, repeated_dfa.weights):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    for decorator, (matrix, mean) in zip(model.decorators, state_after_first):
        assert decorator.decor_weight is matrix
        assert decorator.running_mean is mean

    model.apply_gradients(dfa, lr=.02)
    for decorator, (matrix, mean) in zip(model.decorators, state_after_first):
        assert decorator.decor_weight is matrix
        assert decorator.running_mean is mean
    model.forward(2 * x - .2)
    model.training = False
    eval_state = [(decorator.decor_weight, decorator.running_mean) for decorator in model.decorators]
    model.forward(x)
    recomputed_bp = model.bp_gradients_from_cache(cache, y)
    for actual, expected in zip(recomputed_bp.weights, original_bp.weights):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for current, original, saved in zip(model.conditioning_activities(dfa), original_inputs, copied_inputs):
        assert current is original
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
    for decorator, (matrix, mean) in zip(model.decorators, eval_state):
        assert decorator.decor_weight is matrix
        assert decorator.running_mean is mean


def test_forward_interface_exposes_actual_linear_inputs_and_raw_hidden_activities():
    model = _model()
    model.training = False
    x, _ = _data()
    logits, inputs, preactivations = model.forward(x)
    hidden = model.hidden_activations(x)

    assert len(inputs) == len(model.weights) + 1
    assert inputs[-1] is logits
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        torch.testing.assert_close(preactivations[index], inputs[index] @ weight.T + bias)
    for actual, z in zip(hidden, preactivations[:-1]):
        torch.testing.assert_close(actual, z.relu())
    assert not torch.allclose(inputs[0], x)


def test_default_parameter_initialization_feedback_and_count_match_manual_mlp():
    base = ManualMLP(5, [4, 3], 3, seed=120)
    model = DecorrelatedManualMLP(5, [4, 3], 3, seed=120)
    for actual, expected in zip(model.weights + model.biases, base.weights + base.biases):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(init_feedback(model, seed=8), init_feedback(base, seed=8)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.bn_gamma == model.bn_beta == []
    count = lambda m: sum(w.numel() + b.numel() for w, b in zip(m.weights, m.biases))
    assert count(model) == count(base)
    assert model.decorrelation_state_numel == sum(width * width + width for width in [5, 4, 3])


def test_foreign_cache_and_unsupported_credit_or_normalization_paths_are_rejected():
    model, other = _model(), _model()
    x, y = _data()
    foreign = other.bp_gradients(x, y)
    with pytest.raises(ValueError):
        model.conditioning_activities(foreign)
    with pytest.raises(ValueError):
        model.bp_gradients_from_cache(foreign.forward_cache, y)
    with pytest.raises(NotImplementedError):
        model.fa_gradients(x, y, [])
    with pytest.raises(NotImplementedError):
        model.target_projection_gradients(x, y, [])
    with pytest.raises(NotImplementedError):
        DecorrelatedManualMLP(5, [4], 3, batchnorm=True)


def test_invalid_direct_feedback_is_rejected_before_advancing_state():
    model = _model()
    model.training = True
    x, y = _data()
    state = [(decorator.decor_weight, decorator.running_mean) for decorator in model.decorators]
    with pytest.raises(ValueError):
        model.dfa_gradients(x, y, [torch.ones(3, 4)])
    with pytest.raises(ValueError):
        model.dfa_gradients(x, y, [torch.ones(3, 4), torch.ones(4, 3)])
    for decorator, (matrix, mean) in zip(model.decorators, state):
        assert decorator.decor_weight is matrix
        assert decorator.running_mean is mean
