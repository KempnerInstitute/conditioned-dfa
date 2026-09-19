"""Run one unchanged legacy trajectory, then export its final inference state."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_ndfa_paired_views as legacy

STATE_KEYS = ("weights", "biases", "bn_gamma", "bn_beta", "bn_running_mean", "bn_running_var")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_tensors(model):
    return {key: [value.detach().cpu().clone() for value in getattr(model, key)] for key in STATE_KEYS}


def tensor_digest(tensor):
    return hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode() +
                          tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_fingerprint(model):
    return {"training": model.training, "bn_eps": model.bn_eps, "bn_momentum": model.bn_momentum,
            "tensors": {key: [tensor_digest(value) for value in getattr(model, key)] for key in STATE_KEYS},
            "bn_cache": [None if entry is None else [tensor_digest(entry[0]), tensor_digest(entry[1]), entry[2]]
                         for entry in model._bn_cache]}


def rng_fingerprint(device):
    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256(numpy_state[1].tobytes() + repr((numpy_state[0], *numpy_state[2:])).encode()).hexdigest()
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(), "numpy": numpy_digest,
            "cpu": tensor_digest(torch.get_rng_state()),
            "cuda": tensor_digest(torch.cuda.get_rng_state(device)) if torch.device(device).type == "cuda" else None}


@torch.no_grad()
def validation_logits(model, data, args):
    was_training = model.training
    values = []
    try:
        model.training = False
        with legacy.preserve_batchnorm_state(model):
            for start in range(0, len(data.validation), 256):
                inputs = data.validation[start:start + 256]
                if args.dataset != "synthetic":
                    inputs = legacy.standardize_images(inputs.float().div(255.), data).flatten(1)
                values.append(model.forward(inputs.to(args.device))[0].detach().cpu())
    finally:
        model.training = was_training
    return torch.cat(values)


def metrics_from_logits(logits, labels):
    values = logits.double().numpy()
    labels_np = labels.numpy()
    maxima = values.max(axis=1)
    losses = maxima + np.log(np.exp(values - maxima[:, None]).sum(axis=1)) - values[np.arange(len(values)), labels_np]
    return {"validation_loss": float(losses.mean()),
            "validation_accuracy": float((values.argmax(axis=1) == labels_np).mean())}


def restore_model(checkpoint, device="cpu"):
    config = checkpoint["model"]
    model = legacy.ManualMLP(config["input_dim"], config["hidden_dims"], config["output_dim"],
                             device=device, batchnorm=config["batchnorm"], bn_eps=config["bn_eps"],
                             bn_momentum=config["bn_momentum"], seed=0)
    for key in STATE_KEYS:
        setattr(model, key, [value.to(device).clone() for value in checkpoint["state"][key]])
    model.training = checkpoint["training_mode_at_capture"]
    return model


def export_final(model, data, args, endpoint, output):
    """Export after training; preserve all live model, BN-cache and global RNG state."""
    started = time.monotonic()
    before_state, before_rng = model_fingerprint(model), rng_fingerprint(args.device)
    logits = validation_logits(model, data, args)
    labels = data.validation_labels.detach().cpu().clone()
    require(bool(torch.isfinite(logits).all()), "nonfinite final validation logits")
    if args.dataset == "cifar10":
        permutation = torch.randperm(50000, generator=torch.Generator().manual_seed(args.split_seed))
        indices = permutation[:args.validation_size]
    elif args.dataset == "synthetic":
        indices = torch.arange(len(labels))
    else:
        raise ValueError("export adapter supports CIFAR-10 and synthetic verification fixtures only")
    metrics = metrics_from_logits(logits, labels)
    require(metrics["validation_accuracy"] == endpoint["validation_accuracy"], "export accuracy differs from recorded endpoint")
    require(abs(metrics["validation_loss"] - endpoint["validation_loss"]) <= 2e-6 * max(1., abs(endpoint["validation_loss"])),
            "export loss differs from recorded endpoint")
    preprocessing = None if data.channel_mean is None else {
        "channel_mean": data.channel_mean.cpu().clone(), "channel_std": data.channel_std.cpu().clone(),
        "source": "selected training pool before augmentation", "uint8_scale": 255., "validation_used": False}
    checkpoint = {"schema_version": 1, "stage": "final validation-only inference state",
                  "method": endpoint["method"], "seed": endpoint["seed"], "step": endpoint["step"],
                  "args": vars(args), "model": {"input_dim": data.input_dim, "hidden_dims": model.hidden_dims,
                  "output_dim": model.output_dim, "batchnorm": model.batchnorm,
                  "bn_eps": model.bn_eps, "bn_momentum": model.bn_momentum},
                  "training_mode_at_capture": model.training, "state": state_tensors(model),
                  "preprocessing": preprocessing, "parameter_count": legacy.parameter_count(model),
                  "validation_indices": indices, "checkpoint_scope": "Inference state only; training sampler/view generators are not returned by the legacy trainer"}
    for key, tensors in checkpoint["state"].items():
        require(all(bool(torch.isfinite(value).all()) for value in tensors), f"nonfinite checkpoint state: {key}")
    torch.save(checkpoint, output / "final_state.pt")
    torch.save({"schema_version": 1, "method": endpoint["method"], "seed": endpoint["seed"],
                "step": endpoint["step"], "logits": logits, "labels": labels, "indices": indices,
                "official_test_evaluated": False}, output / "final_validation.pt")
    after_state, after_rng = model_fingerprint(model), rng_fingerprint(args.device)
    require(before_state == after_state, "final export changed live model or BN state/cache")
    require(before_rng == after_rng, "final export consumed a global RNG stream")
    receipt = {"method": endpoint["method"], "seed": endpoint["seed"], "step": endpoint["step"],
               "model_and_bn_state_unchanged": True, "global_rng_state_unchanged": True,
               "before_model_fingerprint": before_state, "after_model_fingerprint": after_state,
               "before_global_rng": before_rng, "after_global_rng": after_rng,
               "independent_float64_metrics": metrics, "legacy_recorded_metrics": endpoint,
               "loss_absolute_tolerance": 2e-6 * max(1., abs(endpoint["validation_loss"])),
               "final_state_sha256": sha256(output / "final_state.pt"),
               "final_validation_sha256": sha256(output / "final_validation.pt"),
               "export_seconds": time.monotonic() - started, "official_test_evaluated": False}
    write_json(output / "final_export.json", receipt)
    return receipt


def audit_saved_case(output, command, recorded_endpoint, recorded_standardization):
    """CPU-only audit of final saved tensors, identities and independent logit metrics."""
    receipt = json.loads((output / "final_export.json").read_text())
    require(receipt["legacy_recorded_metrics"] == recorded_endpoint, "export receipt endpoint differs from trajectory")
    require(receipt["loss_absolute_tolerance"] == 2e-6 * max(1., abs(recorded_endpoint["validation_loss"])), "loss tolerance changed")
    require(receipt["export_seconds"] > 0 and np.isfinite(receipt["export_seconds"]), "invalid export timing")
    require(sha256(output / "final_state.pt") == receipt["final_state_sha256"], "changed checkpoint")
    require(sha256(output / "final_validation.pt") == receipt["final_validation_sha256"], "changed validation predictions")
    state = torch.load(output / "final_state.pt", map_location="cpu", weights_only=True)
    validation = torch.load(output / "final_validation.pt", map_location="cpu", weights_only=True)
    for record in (receipt, state, validation):
        require(record["method"] == command["method"] and record["seed"] == command["seed"] and record["step"] == command["steps"], "wrong saved final identity/horizon")
    require(state["args"] == command["args"], "checkpoint args changed")
    require(state["parameter_count"] == command["parameter_count"], "checkpoint parameter count changed")
    require(state["model"] == {"input_dim": 3072, "hidden_dims": command["hidden_dims"], "output_dim": 10,
                               "batchnorm": True, "bn_eps": 1e-5, "bn_momentum": .1}, "saved architecture changed")
    expected_dims = [3072, *command["hidden_dims"], 10]
    expected_shapes = {
        "weights": [(out_dim, in_dim) for in_dim, out_dim in zip(expected_dims[:-1], expected_dims[1:])],
        "biases": [(dim,) for dim in expected_dims[1:]],
        **{key: [(dim,) for dim in command["hidden_dims"]] for key in STATE_KEYS[2:]}}
    for key, shapes in expected_shapes.items():
        tensors = state["state"][key]
        require([tuple(value.shape) for value in tensors] == shapes, "wrong saved tensor shapes")
        require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in tensors), "nonfinite or wrong-dtype saved state")
        require([tensor_digest(value) for value in tensors] == receipt["after_model_fingerprint"]["tensors"][key], "checkpoint/live-state fingerprint mismatch")
    require(receipt["before_model_fingerprint"] == receipt["after_model_fingerprint"] and receipt["before_global_rng"] == receipt["after_global_rng"], "export changed live state")
    require(receipt["model_and_bn_state_unchanged"] and receipt["global_rng_state_unchanged"], "state preservation failed")
    logits, labels = validation["logits"], validation["labels"]
    require(tuple(logits.shape) == (2000, 10) and logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()), "invalid saved logits")
    require(tuple(labels.shape) == (2000,) and labels.dtype == torch.int64 and bool(((labels >= 0) & (labels < 10)).all()), "invalid saved labels")
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(command["args"]["split_seed"]))[:2000]
    require(torch.equal(indices, validation["indices"]) and torch.equal(indices, state["validation_indices"]), "wrong validation identities")
    require(state["preprocessing"]["validation_used"] is False, "validation-derived preprocessing")
    for key in ("channel_mean", "channel_std"):
        value = state["preprocessing"][key]
        require(tuple(value.shape) == (3,) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "invalid preprocessing")
    require(bool((state["preprocessing"]["channel_std"] > 0).all()), "invalid preprocessing scale")
    require(state["preprocessing"]["channel_mean"].tolist() == recorded_standardization["mean"] and
            state["preprocessing"]["channel_std"].tolist() == recorded_standardization["std"] and
            recorded_standardization["validation_used"] is False, "checkpoint preprocessing differs from training manifest")
    metrics = metrics_from_logits(logits, labels)
    require(metrics == receipt["independent_float64_metrics"], "independent metrics changed")
    require(metrics["validation_accuracy"] == receipt["legacy_recorded_metrics"]["validation_accuracy"], "saved prediction accuracy mismatch")
    require(abs(metrics["validation_loss"] - receipt["legacy_recorded_metrics"]["validation_loss"]) <= receipt["loss_absolute_tolerance"], "saved prediction loss mismatch")
    require(validation["official_test_evaluated"] is False and receipt["official_test_evaluated"] is False, "test evaluation prohibited")
    return {"step": command["steps"], "validation_count": len(labels), "metrics": metrics,
            "checkpoint_sha256": receipt["final_state_sha256"], "predictions_sha256": receipt["final_validation_sha256"],
            "model_and_bn_state_unchanged": True, "global_rng_state_unchanged": True}


def main():
    args = legacy.parse_args()
    require(len(args.methods) == len(args.seeds) == 1, "one fixed method and seed per child")
    require(args.methods[0] in ("bp", "dfa", "ndfa"), "unexpected method")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable; no fallback")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = legacy.load_data(args)
    source_keys = ("experiments/run_ndfa_paired_views.py", "infogeo/dfa.py",
                   "infogeo/local_preconditioning.py", "infogeo/paired_conditioning.py")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
                "stage": "independent seed confirmation; validation only", "test_evaluated": False,
                "torch": str(torch.__version__), "python": sys.version,
                "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                "source_sha256": {key: sha256(ROOT / key) for key in source_keys},
                "adapter_source_sha256": sha256(Path(__file__)), "training_examples": len(data.train),
                "validation_examples": len(data.validation),
                "input_standardization": None if data.channel_mean is None else {
                    "source": "selected training pool before augmentation; per channel over images and pixels",
                    "mean": data.channel_mean.tolist(), "std": data.channel_std.tolist(), "validation_used": False},
                "final_only_export": True, "training_seconds_exclude_export": True,
                "raw_step_5000_state_saved": False if args.steps != 5000 else True}
    write_json(output / "manifest.json", manifest)
    model, rows = legacy.train_one(args, data, args.seeds[0], args.methods[0])
    write_json(output / "metrics.json", rows)
    write_json(output / "endpoints.json", [rows[-1]])
    export_final(model, data, args, rows[-1], output)
    print(json.dumps(rows[-1], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
