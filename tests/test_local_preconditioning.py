"""Algebraic and API checks for exact local-update conditioning."""

import pytest
import torch

from infogeo.local_preconditioning import condition_local_update


def dense_reference(activity, error, lambda_a, lambda_e, mode):
    """Explicit feature-space reference, independent of backend selection."""
    n = activity.shape[0]
    update = error.T @ activity / n
    if mode in {"activity", "kronecker"}:
        covariance = activity.T @ activity / n
        inverse = torch.linalg.inv(covariance + lambda_a * torch.eye(activity.shape[1], dtype=activity.dtype))
        update = update @ inverse
    if mode in {"error", "kronecker"}:
        covariance = error.T @ error / n
        inverse = torch.linalg.inv(covariance + lambda_e * torch.eye(error.shape[1], dtype=error.dtype))
        update = inverse @ update
    return update


def samples(n, din, dout, *, seed=31, dtype=torch.float64):
    rng = torch.Generator().manual_seed(seed)
    activity = torch.randn(n, din, generator=rng, dtype=dtype)
    # Errors include sample-dependent gates, rather than a fixed output-space map.
    direct = torch.randn(n, 3, generator=rng, dtype=dtype) @ torch.randn(3, dout, generator=rng, dtype=dtype)
    gates = torch.sigmoid(torch.randn(n, dout, generator=rng, dtype=dtype))
    return activity, direct * gates


@pytest.mark.parametrize("shape", [(7, 4, 5), (4, 9, 8), (6, 3, 11), (6, 11, 3), (1, 5, 4), (5, 5, 5)])
@pytest.mark.parametrize("mode", ["activity", "error", "kronecker"])
@pytest.mark.parametrize("backend", ["feature", "sample", "auto"])
def test_exact_equivalence_with_sample_dependent_errors(shape, mode, backend):
    activity, error = samples(*shape)
    actual = condition_local_update(activity, error, activity_damping=0.13, error_damping=0.71, mode=mode, backend=backend)
    expected = dense_reference(activity, error, 0.13, 0.71, mode)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    assert actual.shape == (shape[2], shape[1])
    assert actual.dtype == activity.dtype


@pytest.mark.parametrize("mode", ["activity", "error", "kronecker"])
def test_rank_deficient_and_zero_samples(mode):
    activity, error = samples(3, 8, 6)
    activity[:, 2:] = activity[:, :1]
    error[:, 1:] = error[:, :1]
    for a, d in [(activity, error), (activity * 0, error), (activity, error * 0)]:
        expected = dense_reference(a, d, 0.2, 0.6, mode)
        for backend in ["feature", "sample", "auto"]:
            actual = condition_local_update(a, d, activity_damping=0.2, error_damping=0.6, mode=mode, backend=backend)
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("mode", ["activity", "error", "kronecker"])
@pytest.mark.parametrize("backend", ["feature", "sample", "auto"])
def test_repeating_batch_does_not_change_mean_or_conditioner(mode, backend):
    activity, error = samples(3, 8, 5)
    kwargs = dict(activity_damping=0.2, error_damping=0.6, mode=mode, backend=backend)
    small = condition_local_update(activity, error, **kwargs)
    repeated = condition_local_update(activity.repeat_interleave(4, dim=0), error.repeat_interleave(4, dim=0), **kwargs)
    torch.testing.assert_close(small, repeated, rtol=2e-11, atol=2e-12)


def test_default_error_damping_and_single_sample_normalization():
    activity = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    error = torch.tensor([[3.0, -1.0, 2.0]], dtype=torch.float64)
    # With one observation, both damped solves act along a single data direction.
    expected = (error.T @ activity) / ((activity.square().sum() + 0.4) * (error.square().sum() + 0.4))
    actual = condition_local_update(activity, error, activity_damping=0.4, mode="kronecker")
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape,expected_sizes", [((6, 3, 11), [6, 3]), ((6, 11, 3), [6, 3]), ((4, 9, 8), [4, 4]), ((9, 4, 5), [4, 5])])
def test_auto_selects_each_factor_independently(monkeypatch, shape, expected_sizes):
    activity, error = samples(*shape)
    real_solve = torch.linalg.solve
    sizes = []

    def record_solve(system, rhs):
        sizes.append(system.shape[0])
        return real_solve(system, rhs)

    monkeypatch.setattr(torch.linalg, "solve", record_solve)
    condition_local_update(activity, error, activity_damping=0.2, error_damping=0.6, mode="kronecker", backend="auto")
    assert sizes == expected_sizes


def test_float32_noncontiguous_inputs():
    activity, error = samples(5, 14, 18, dtype=torch.float32)
    activity, error = activity[:, ::2], error[:, ::2]
    actual = condition_local_update(activity, error, activity_damping=0.3, error_damping=0.8, mode="kronecker", backend="sample")
    expected = dense_reference(activity.double(), error.double(), 0.3, 0.8, "kronecker")
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("name,value", [("activity_damping", 0), ("activity_damping", -1), ("activity_damping", float("nan")), ("activity_damping", float("inf")), ("error_damping", 0), ("error_damping", -1), ("error_damping", float("nan")), ("error_damping", float("inf"))])
def test_rejects_invalid_damping(name, value):
    activity, error = samples(3, 4, 5)
    kwargs = dict(activity_damping=0.3, error_damping=0.8)
    kwargs[name] = value
    with pytest.raises(ValueError, match="positive and finite"):
        condition_local_update(activity, error, **kwargs)


@pytest.mark.parametrize("value", [True, "0.3", complex(0.3, 0.1)])
def test_rejects_non_real_damping(value):
    activity, error = samples(3, 4, 5)
    with pytest.raises(TypeError, match="real number"):
        condition_local_update(activity, error, activity_damping=value)


@pytest.mark.parametrize("name,value", [("mode", "joint"), ("backend", "approximate")])
def test_rejects_unknown_operator(name, value):
    activity, error = samples(3, 4, 5)
    with pytest.raises(ValueError, match=name):
        condition_local_update(activity, error, activity_damping=0.3, **{name: value})


@pytest.mark.parametrize("kind", ["rank", "empty_batch", "empty_feature", "sample_count", "dtype", "integer", "complex", "half", "nan_activity", "inf_error"])
def test_rejects_invalid_tensors(kind):
    activity, error = samples(3, 4, 5)
    expected_exception = ValueError
    if kind == "rank":
        activity = activity.unsqueeze(0)
    elif kind == "empty_batch":
        activity, error = activity[:0], error[:0]
    elif kind == "empty_feature":
        error = error[:, :0]
    elif kind == "sample_count":
        error = error[:2]
    elif kind == "dtype":
        error = error.float()
        expected_exception = TypeError
    elif kind in {"integer", "complex", "half"}:
        activity = activity.to({"integer": torch.int64, "complex": torch.complex128, "half": torch.float16}[kind])
        expected_exception = TypeError
    elif kind == "nan_activity":
        activity[0, 0] = float("nan")
    else:
        error[0, 0] = float("inf")
    with pytest.raises(expected_exception):
        condition_local_update(activity, error, activity_damping=0.3)


def test_rejects_non_tensor_and_device_mismatch():
    activity, error = samples(3, 4, 5)
    with pytest.raises(TypeError, match="torch tensors"):
        condition_local_update(activity.tolist(), error, activity_damping=0.3)
    with pytest.raises(ValueError, match="same device"):
        condition_local_update(activity, torch.empty_like(error, device="meta"), activity_damping=0.3)


def test_numerical_failure_does_not_change_damping(monkeypatch):
    activity, error = samples(3, 4, 5)
    calls = []

    def fail(system, rhs):
        calls.append(system)
        raise RuntimeError("injected numerical failure")

    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(RuntimeError, match="injected numerical failure"):
        condition_local_update(activity, error, activity_damping=0.3, backend="sample")
    assert len(calls) == 1


def test_rejects_nonfinite_solver_result(monkeypatch):
    activity, error = samples(3, 4, 5)
    monkeypatch.setattr(torch.linalg, "solve", lambda system, rhs: torch.full_like(rhs, float("nan")))
    with pytest.raises(RuntimeError, match="non-finite update"):
        condition_local_update(activity, error, activity_damping=0.3)
