"""Protect scientific comparisons in the new long-horizon runner."""
import argparse
import copy
import json

import pytest
import torch

from scripts.ndfa_strengthening_20260925 import baseline_development as study
from experiments import run_ndfa_submission_benchmark as base


@pytest.mark.parametrize("credit", ["bp", "dfa"])
@pytest.mark.parametrize("bn", [False, True])
def test_conditioned_gradient_dense_reference_and_single_bn_forward(credit, bn):
    torch.set_num_threads(1)
    args = base.parse_args(["--output-dir", "unused"])
    args.method = credit
    model = base.CachedManualMLP(7, [5, 4], 3, seed=16, batchnorm=bn)
    model.training = True
    reference = copy.deepcopy(model)
    feedback = base.init_feedback(model, seed=17, scale=.1)
    x = torch.randn(11, 7, generator=torch.Generator().manual_seed(18))
    y = torch.arange(11) % 3
    raw = reference.bp_gradients(x, y) if credit == "bp" else reference.dfa_gradients(x, y, feedback)
    case = dict(credit=credit, normalization="bn" if bn else "none", operator="activity", rho=3.)
    actual = study.gradient(model, feedback, x, y, args, case)
    for layer in range(model.n_hidden_layers):
        a = reference.last_activities[layer].double()
        damping = 3 * float(reference.last_activities[layer].square().mean())
        moment = a.T @ a / len(a) + damping * torch.eye(a.shape[1], dtype=torch.double)
        expected = torch.linalg.solve(moment, raw.weights[layer].double().T).T
        expected *= raw.weights[layer].double().norm() / expected.norm()
        torch.testing.assert_close(actual.weights[layer].double(), expected, rtol=3e-5, atol=1e-7)
    torch.testing.assert_close(actual.weights[-1], raw.weights[-1], rtol=0, atol=0)
    for a, b in zip(actual.biases, raw.biases):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name in ["bn_running_mean", "bn_running_var"]:
        for a, b in zip(getattr(model, name), getattr(reference, name)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer", ["sgd_momentum", "adamw"])
def test_paired_streams_and_validation_do_not_change_bn_state(tmp_path, optimizer):
    config = dict(data_dir="unused", threads=1, hidden_dims=[8,4], batch_size=8,
                  epochs=2, feedback_scale=.1, split_seed=19, weight_decay=1e-4,
                  warmup_epochs=0, evaluate_every_epochs=1)
    cases = [dict(id=str(i), credit="dfa", normalization="bn", operator=op,
                  rho=3, optimizer=optimizer, lr=.001) for i,op in enumerate(["none","activity"])]
    results = [study.train(config, c, 20, tmp_path/c["id"], device="cpu", fixture=True) for c in cases]
    assert all(r["status"] == "complete" and r["completed_updates"] == 8 for r in results)
    for key in ["initial_parameter_sha256", "initial_feedback_sha256", "order_sha256", "augmentation_sha256"]:
        assert results[0][key] == results[1][key]
    for c in cases:
        saved = torch.load(tmp_path/c["id"]/"final.pt", weights_only=False)
        assert saved["optimizer_state"]["state"]
        args = argparse.Namespace(**saved["args"])
        data = base.load_data(args)
        model = base.restore_model(args, data, saved["model"])
        before = base.tensors_hash(base.state_tensors(model))
        state=data.on("cpu")
        base.evaluate(model, state["validation"], state["validation_labels"], data, args)
        assert base.tensors_hash(base.state_tensors(model)) == before
        manifest=json.loads((tmp_path/c["id"]/"manifest.json").read_text())
        assert manifest["data"]["official_test_loaded"] is False
        assert manifest["data"]["substantive_learning_evidence"] is False


def test_learning_rate_schedule_endpoints():
    assert study.learning_rate(1., 0, 200, 5) == .2
    assert study.learning_rate(1., 4, 200, 5) == 1.
    assert study.learning_rate(1., 5, 200, 5) == 1.
    assert study.learning_rate(1., 199, 200, 5) == .01
