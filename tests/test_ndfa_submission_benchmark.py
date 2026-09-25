"""Integration invariants for the new, unfrozen submission benchmark path."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F

SPEC = importlib.util.spec_from_file_location("submission_runner", Path(__file__).parents[1] / "experiments/run_ndfa_submission_benchmark.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def args(tmp_path, method="bp", **changes):
    parsed = r.parse_args(["--output-dir", str(tmp_path / method), "--method", method,
                           "--dataset", "fixture", "--stage", "fixture", "--fixture-train", "10",
                           "--fixture-validation", "7", "--fixture-side", "5", "--batch-size", "4",
                           "--hidden-dims", "7", "5", "--epochs", "2", "--lr", ".01"])
    for key, value in changes.items():
        setattr(parsed, key, value)
    return parsed


@pytest.mark.parametrize("method", r.METHODS)
def test_complete_endpoint_full_state_roundtrip_and_no_dropped_identities(tmp_path, method):
    a = args(tmp_path, method)
    r.configure(a)
    data = r.load_data(a)
    model, endpoint = r.train_one(a, data)
    assert endpoint["status"] == "complete"
    assert endpoint["completed_updates"] == 6
    assert endpoint["completed_epochs"] == 2
    assert endpoint["final_validation"]["validation_examples"] == 7
    checkpoint = torch.load(Path(a.output_dir) / "final.pt", weights_only=False)
    assert checkpoint["completed_updates"] == 6
    assert checkpoint["order_sha256"] == endpoint["order_sha256"]
    assert r.sha256(Path(a.output_dir) / "final.pt") == endpoint["final_checkpoint_sha256"]
    manifest = json.loads((Path(a.output_dir) / "manifest.json").read_text())
    assert not manifest["test_evaluated"]
    assert not manifest["data"]["official_test_loaded"]
    assert set(manifest["source_sha256"]) == set(r.SOURCE_FILES)
    restored = r.restore_model(a, data, checkpoint["model"])
    old = r.evaluate(model, data.validation, data.validation_labels, data, a)
    new = r.evaluate(restored, data.validation, data.validation_labels, data, a)
    assert old == new
    if method.endswith("decorrelation"):
        assert len(checkpoint["model"]["decorators"]) == 3
        assert any(torch.count_nonzero(value["running_mean"]) for value in checkpoint["model"]["decorators"])
    if method.endswith("batchnorm"):
        assert len(checkpoint["model"]["bn_running_mean"]) == 2
        assert any(torch.count_nonzero(value) for value in checkpoint["model"]["bn_running_mean"])
    with pytest.raises(FileExistsError):
        r.train_one(a, data)


def test_initialization_feedback_order_and_augmentation_are_paired_across_rules(tmp_path):
    records = []
    for method in r.METHODS:
        a = args(tmp_path, method, epochs=1)
        _, result = r.run(a)
        assert result["status"] == "complete"
        manifest = json.loads((Path(a.output_dir) / "manifest.json").read_text())
        records.append((result, manifest))
    for key in ("initial_parameter_sha256", "initial_feedback_sha256"):
        assert len({manifest[key] for _, manifest in records}) == 1
    for key in ("order_sha256", "augmentation_sha256"):
        assert len({result[key] for result, _ in records}) == 1
    assert len({json.dumps(manifest["data"], sort_keys=True) for _, manifest in records}) == 1


@pytest.mark.parametrize("method,mode", [("bp_activity", "activity"), ("dfa_activity", "activity"),
                                         ("dfa_error", "error"), ("dfa_kronecker", "kronecker")])
def test_conditioning_equals_independent_feature_solve_from_actual_unaveraged_errors(tmp_path, method, mode):
    a = args(tmp_path, method, activity_rho=.7, error_rho=.2)
    data = r.load_data(a)
    model, feedback = r.make_model(a, data)
    x, y, _ = r.augmented_batch(data, torch.arange(4), torch.Generator().manual_seed(6), "cpu")
    model.training = True
    raw = model.bp_gradients(x, y) if method.startswith("bp") else model.dfa_gradients(x, y, feedback)
    activities = [value.clone() for value in model.last_activities]
    result = r.gradients(model, feedback, x, y, a)
    for index in range(model.n_hidden_layers):
        activity = activities[index].double()
        error = raw.deltas[index].double() * len(x)
        g = error.T @ activity / len(x)
        torch.testing.assert_close(g.float(), raw.weights[index], rtol=2e-6, atol=2e-7)
        ridge_a = max(a.activity_rho * float(activities[index].square().mean()), a.damping_floor)
        ridge_e = max(a.error_rho * float((raw.deltas[index] * len(x)).square().mean()), a.damping_floor)
        expected = g
        if mode in {"activity", "kronecker"}:
            expected = torch.linalg.solve(activity.T @ activity / len(x) + ridge_a * torch.eye(activity.shape[1]), expected.T).T
        if mode in {"error", "kronecker"}:
            expected = torch.linalg.solve(error.T @ error / len(x) + ridge_e * torch.eye(error.shape[1]), expected)
        expected *= raw.weights[index].double().norm() / expected.norm()
        torch.testing.assert_close(result.weights[index], expected.float(), rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(result.weights[index].double().norm(), raw.weights[index].double().norm(), rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(result.weights[-1], raw.weights[-1], rtol=0, atol=0)
    for actual, expected in zip(result.biases, raw.biases):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bp_is_exact_mean_cross_entropy_gradient(tmp_path):
    a = args(tmp_path)
    data = r.load_data(a)
    model, feedback = r.make_model(a, data)
    x, y, _ = r.augmented_batch(data, torch.arange(4), torch.Generator().manual_seed(6), "cpu")
    weights = [value.clone().requires_grad_() for value in model.weights]
    biases = [value.clone().requires_grad_() for value in model.biases]
    hidden = x
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        hidden = hidden @ weight.T + bias
        if index < len(weights) - 1:
            hidden = hidden.relu()
    loss = F.cross_entropy(hidden, y)
    expected = torch.autograd.grad(loss, weights + biases)
    actual = r.gradients(model, feedback, x, y, a)
    for actual_value, expected_value in zip(actual.weights + actual.biases, expected):
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("method", ["bp_batchnorm", "dfa_batchnorm", "bp_decorrelation", "dfa_decorrelation"])
def test_evaluation_preserves_running_state_and_train_mode(tmp_path, method):
    a = args(tmp_path, method)
    data = r.load_data(a)
    model, feedback = r.make_model(a, data)
    model.training = True
    x, y, _ = r.augmented_batch(data, torch.arange(4), torch.Generator().manual_seed(6), "cpu")
    r.gradients(model, feedback, x, y, a)
    before = r.tensors_hash(r.state_tensors(model))
    cache = list(model._bn_cache)
    r.evaluate(model, data.validation, data.validation_labels, data, a)
    assert before == r.tensors_hash(r.state_tensors(model))
    assert model.training
    assert all(a is b for a, b in zip(cache, model._bn_cache))


def test_numerical_failure_is_recorded_with_state_and_history(tmp_path, monkeypatch):
    a = args(tmp_path)
    data = r.load_data(a)
    original = r.gradients
    calls = 0
    def fail_after_update(*positional):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("controlled numerical fixture")
        return original(*positional)
    monkeypatch.setattr(r, "gradients", fail_after_update)
    _, result = r.train_one(a, data)
    assert result["status"] == "numerical_failure"
    assert result["completed_updates"] == 1
    assert (Path(a.output_dir) / "failure.pt").is_file()
    assert (Path(a.output_dir) / "history.json").is_file()
    assert not (Path(a.output_dir) / "endpoint.json").exists()


def test_structural_failure_propagates_without_becoming_numerical(tmp_path, monkeypatch):
    a = args(tmp_path)
    def structural(*positional):
        raise RuntimeError("structural fixture")
    monkeypatch.setattr(r, "gradients", structural)
    with pytest.raises(RuntimeError, match="structural fixture"):
        r.run(a)
    assert (Path(a.output_dir) / "manifest.json").is_file()
    assert not (Path(a.output_dir) / "failure.json").exists()


@pytest.mark.parametrize("method", ["bp_decorrelation", "dfa_decorrelation"])
def test_finite_weights_with_overflowed_hidden_activity_are_numerical(tmp_path, method):
    a = args(tmp_path, method)
    r.configure(a)
    data = r.load_data(a)
    model, feedback = r.make_model(a, data)
    model.training = True
    model.weights[0].fill_(1e38)
    assert all(torch.isfinite(t).all() for t in r.state_tensors(model))
    x = torch.arange(4 * data.input_dim, dtype=torch.float32).reshape(4, data.input_dim)
    with pytest.raises(FloatingPointError, match="activity must be finite"):
        r.gradients(model, feedback, x, torch.arange(4), a)


def test_decorrelated_evaluation_overflow_restores_training_mode(tmp_path):
    a = args(tmp_path, "bp_decorrelation")
    r.configure(a)
    data = r.load_data(a)
    data.channel_mean.zero_()
    data.channel_std.fill_(1)
    model, _ = r.make_model(a, data)
    model.training = True
    model.weights[0].fill_(1e38)
    images = torch.full((4, 3, 5, 5), 255, dtype=torch.uint8)
    with pytest.raises(FloatingPointError, match="activity must be finite"):
        r.evaluate(model, images, torch.arange(4), data, a)
    assert model.training is True


@pytest.mark.parametrize("message", ["signal must be finite", "activity must be finite"])
def test_only_explicit_nonfinite_decorrelation_guards_are_translated(tmp_path, message):
    with pytest.raises(FloatingPointError, match=message):
        r.translate_decorrelation_numerical_failure(args(tmp_path, "bp_decorrelation"), ValueError(message))
    assert r.translate_decorrelation_numerical_failure(args(tmp_path, "bp"), ValueError(message)) is None


@pytest.mark.parametrize("message", ["activity must be a nonempty batch with num_features columns",
                                      "signal must have the cached forward's output shape",
                                      "activity and state must share a device"])
def test_decorrelation_structural_guards_are_not_translated(tmp_path, message):
    assert r.translate_decorrelation_numerical_failure(args(tmp_path, "bp_decorrelation"), ValueError(message)) is None


def test_official_loader_requests_only_training_split(tmp_path, monkeypatch):
    seen = []
    class StopLoading(Exception):
        pass
    def loader(**kwargs):
        seen.append(kwargs)
        raise StopLoading
    module = types.ModuleType("torchvision.datasets")
    module.CIFAR10 = loader
    monkeypatch.setitem(sys.modules, "torchvision.datasets", module)
    a = args(tmp_path, dataset="cifar10", stage="development")
    with pytest.raises(StopLoading):
        r.load_data(a)
    assert len(seen) == 1 and seen[0]["train"] is True


def test_fixture_normalization_excludes_validation(tmp_path):
    a = args(tmp_path)
    data = r.load_data(a)
    expected = data.train.double() / 255.
    torch.testing.assert_close(data.channel_mean.double(), expected.mean((0, 2, 3)), rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(data.channel_std.double(), expected.std((0, 2, 3), correction=0), rtol=1e-7, atol=1e-8)


def test_throughput_is_fixture_only_and_retains_exact_timed_count(tmp_path):
    a = args(tmp_path, stage="throughput", benchmark_steps=5, benchmark_warmup=2)
    _, result = r.run(a)
    assert result["completed_updates"] == 7
    assert len(result["benchmark_step_seconds"]) == 5
    assert result["final_validation"] is None
    assert result["benchmark_mean_step_seconds"] > 0
    with pytest.raises(SystemExit):
        r.parse_args(["--output-dir", str(tmp_path), "--benchmark-steps", "2"])


@pytest.mark.parametrize("flag,value", [("--lr", "nan"), ("--error-rho", "0"), ("--activity-rho", "-1"),
                                        ("--decor-lr", "inf"), ("--batch-size", "1")])
def test_invalid_cli_rejected(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        r.parse_args(["--output-dir", str(tmp_path), flag, value])


def test_reference_checkpoint_preserves_full_initial_state_and_rng(tmp_path):
    a = args(tmp_path, "dfa_decorrelation", epochs=1)
    data = r.load_data(a)
    initial_model, initial_feedback = r.make_model(a, data)
    expected = r.tensors_hash(r.state_tensors(initial_model))
    _, endpoint = r.run(a, data, reference_dir=tmp_path / "reference")
    assert endpoint["status"] == "complete"
    reference = torch.load(tmp_path / "reference/initial.pt", weights_only=False)
    restored = r.restore_model(a, data, reference["model"])
    assert r.tensors_hash(r.state_tensors(restored)) == expected
    assert r.tensors_hash(reference["feedback"]) == r.tensors_hash(initial_feedback)
    assert torch.equal(reference["order_rng_state"], torch.Generator().manual_seed(a.order_seed).get_state())
    assert torch.equal(reference["augmentation_rng_state"], torch.Generator().manual_seed(a.augmentation_seed).get_state())
    manifest = json.loads((tmp_path / "reference/manifest.json").read_text())
    assert manifest["initial_checkpoint_sha256"] == r.sha256(tmp_path / "reference/initial.pt")
    assert reference["data_provenance"] == data.provenance


@pytest.mark.parametrize("method,overflow", [("bp_activity", "activity"), ("dfa_error", "error"), ("dfa_kronecker", "error")])
def test_overflowing_finite_samples_are_numerical_failure(tmp_path, monkeypatch, method, overflow):
    a = args(tmp_path, method)
    data = r.load_data(a)
    model, feedback = r.make_model(a, data)
    x, y, _ = r.augmented_batch(data, torch.arange(4), torch.Generator().manual_seed(6), "cpu")
    raw = model.bp_gradients(x, y) if method.startswith("bp") else model.dfa_gradients(x, y, feedback)
    if overflow == "activity":
        model.last_activities[0] = torch.full_like(model.last_activities[0], 1e20)
    else:
        raw.deltas[0] = torch.full_like(raw.deltas[0], 1e20)
    monkeypatch.setattr(model, "bp_gradients" if method.startswith("bp") else "dfa_gradients", lambda *positional: raw)
    with pytest.raises(FloatingPointError, match="second moment"):
        r.gradients(model, feedback, x, y, a)


def test_json_update_is_atomic_and_invalid_content_preserves_existing_file(tmp_path):
    path = tmp_path / "record.json"
    r.write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        r.write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}
    r.write_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 2}
    assert list(tmp_path.iterdir()) == [path]
