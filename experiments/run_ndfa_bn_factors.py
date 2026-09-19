"""Version 1: matched A/E/K factors in the archived paired-view BN learner.

This runner performs no hyperparameter or checkpoint selection. It reuses the
original data, augmentation, initialization, BN backward and exact conditioner.
Only hidden weight matrices are conditioned; their norms are matched once.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_ndfa_paired_views as legacy
from infogeo.dfa import Gradients
from infogeo.local_preconditioning import condition_local_update

VERSION = 1
METHODS = ("dfa", "ndfa", "endfa", "kndfa")
MODES = {"ndfa": "activity", "endfa": "error", "kndfa": "kronecker"}
STATE_LISTS = ("weights", "biases", "bn_gamma", "bn_beta", "bn_running_mean", "bn_running_var")
SOURCE_FILES = (
    "experiments/run_ndfa_bn_factors.py", "experiments/run_ndfa_paired_views.py",
    "infogeo/dfa.py", "infogeo/local_preconditioning.py", "infogeo/paired_conditioning.py",
    "infogeo/geometry.py", "infogeo/__init__.py", "experiments/__init__.py",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--data-dir", default=str(ROOT / "data/torchvision"))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 512])
    for name, default in [("n-train", 10000), ("validation-size", 2000), ("steps", 5000),
                          ("batch-size", 64), ("eval-every", 500), ("threads", 4),
                          ("data-seed", 9173), ("split-seed", 80423), ("feedback-seed", 2901),
                          ("input-dim", 32)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--feedback-scale", type=float, default=.1)
    p.add_argument("--relative-damping", type=float, default=30.)
    p.add_argument("--damping-floor", type=float, default=1e-6)
    p.add_argument("--relative-error-damping", type=float, required=True)
    p.add_argument("--error-damping-floor", type=float, default=1e-6)
    p.add_argument("--diagnostic-steps", nargs="*", type=int, default=[1, 500, 2000, 5000])
    p.add_argument("--backend", choices=["auto", "feature", "sample"], default="auto")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--config", type=Path)
    p.add_argument("--config-sha256")
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--source-manifest-sha256")
    args = p.parse_args(argv)
    # These are the fixed successful recipe, not new choices or augmentations.
    args.download = False
    args.batchnorm = True
    args.standardize_inputs = args.dataset != "synthetic"
    args.regime = "task_high_variance"
    args.task_view_noise = 0.
    validate_args(args)
    return args


def validate_args(args):
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Methods and seeds must be unique; no implicit repeat cases")
    if not args.methods or not args.seeds or any(m not in METHODS for m in args.methods):
        raise ValueError("Require declared methods and seeds")
    for name in ("n_train", "validation_size", "steps", "batch_size", "eval_every", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.batch_size < 2 or args.input_dim < 8 or not args.hidden_dims or min(args.hidden_dims) < 1:
        raise ValueError("Invalid batch or network dimensions")
    if args.relative_damping != 30. or args.damping_floor != 1e-6:
        raise ValueError("A and K retain the successful activity damping30/floor1e-6")
    for name in ("lr", "feedback_scale", "relative_error_damping", "error_damping_floor"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if any(s < 1 for s in args.diagnostic_steps):
        raise ValueError("Diagnostics use positive actual update indices")
    if not args.batchnorm or args.download:
        raise ValueError("BN is required; downloads are not supported")


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    h = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_state(model, *, cpu=True):
    # ManualMLP replaces these tensors rather than mutating them in place.
    # cpu=False therefore retains a cheap last-finite state for a failed update.
    return {name: [v.detach().cpu().clone() if cpu else v for v in getattr(model, name)]
            for name in STATE_LISTS} | {"training": model.training,
                                      "bn_eps": model.bn_eps, "bn_momentum": model.bn_momentum}


def state_hash(state):
    return hashlib.sha256(json.dumps(
        {k: [tensor_hash(v) for v in state[k]] for k in STATE_LISTS}, sort_keys=True).encode()).hexdigest()


def finite_tensor(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def finite_gradients(raw, prefix):
    if not math.isfinite(raw.loss):
        raise FloatingPointError(f"Nonfinite {prefix}.loss")
    finite_tensors([(f"{prefix}.{name}[{layer}]", value)
                    for name in ("weights", "biases", "deltas", "bn_gammas", "bn_betas")
                    for layer, value in enumerate(getattr(raw, name) or [])])


def finite_tensors(named_values):
    # One host synchronization on the successful path, retaining named failures.
    if not bool(torch.stack([torch.isfinite(v).all() for _, v in named_values]).all()):
        for name, value in named_values:
            finite_tensor(value, name)


def finite_model(model):
    finite_tensors([(f"model.{name}[{layer}]", value)
                    for name in STATE_LISTS for layer, value in enumerate(getattr(model, name))])


def new_conditioning_stats(hidden_layers):
    return {"operator_attempts": 0, "operator_successes": 0, "operator_failures": 0,
            "damping_escalations": 0, "pseudoinverse_fallbacks": 0,
            "policy": "Exact inherited solve; failures retained; no fallback or damping adaptation",
            "layers": [{factor: {"count": 0, "floor_count": 0, "ridge_sum": 0.,
                                  "ridge_min": None, "ridge_max": None}
                        for factor in ("activity", "error")} for _ in range(hidden_layers)]}


def record_ridge(stats, layer, factor, ridge, floor_active):
    row = stats["layers"][layer][factor]
    row["count"] += 1
    row["floor_count"] += int(floor_active)
    row["floor_fraction"] = row["floor_count"] / row["count"]
    row["ridge_sum"] += ridge
    row["ridge_mean"] = row["ridge_sum"] / row["count"]
    row["ridge_min"] = ridge if row["ridge_min"] is None else min(row["ridge_min"], ridge)
    row["ridge_max"] = ridge if row["ridge_max"] is None else max(row["ridge_max"], ridge)


def moment_spectrum(samples, ridge):
    """Sparse diagnostic only: smaller Gram, float64, no RNG or model forward."""
    x = samples.double()
    gram = (x @ x.T if x.shape[0] < x.shape[1] else x.T @ x) / len(x)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.)
    largest = eigenvalues[-1].item()
    # Input precision sets numerical-rank tolerance; float64 accumulation does
    # not recover precision absent from the actual float32 learner samples.
    tolerance = max(x.shape) * torch.finfo(samples.dtype).eps * largest
    trace, square_trace = eigenvalues.sum().item(), eigenvalues.square().sum().item()
    return {"n": len(x), "dimension": x.shape[1], "eigenvalues": eigenvalues.cpu().tolist(),
            "rank": int((eigenvalues > tolerance).sum().item()), "rank_tolerance": tolerance,
            "rank_tolerance_dtype": str(samples.dtype), "spectrum_dtype": str(x.dtype),
            "unrepresented_zero_eigenvalues": max(0, x.shape[1] - len(eigenvalues)),
            "participation_rank": trace * trace / square_trace if square_trace else 0.,
            "maximum_eigenvalue": largest, "ridge": ridge,
            "ridge_over_maximum_eigenvalue": ridge / largest if largest else None}


@torch.no_grad()
def condition(model, raw, x, method, args, *, stats=None, diagnostics=None):
    """Apply factors to actual post-gate, post-local-BN deltas, then match once."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    finite_gradients(raw, "raw")
    if method == "dfa" and diagnostics is None:
        return raw
    with legacy.preserve_batchnorm_state(model):
        _, activities, _ = model.forward(x)
    weights = list(raw.weights)
    for layer in range(model.n_hidden_layers):
        activity = activities[layer]
        errors = raw.deltas[layer] * len(x)
        finite_tensor(activity, f"activity[{layer}]")
        finite_tensor(errors, f"unnormalized_error[{layer}]")
        moment_a, moment_e = activity.square().mean().item(), errors.square().mean().item()
        ridge_a = max(args.relative_damping * moment_a, args.damping_floor)
        ridge_e = max(args.relative_error_damping * moment_e, args.error_damping_floor)
        if not all(math.isfinite(v) for v in (moment_a, moment_e, ridge_a, ridge_e)):
            raise FloatingPointError(f"Nonfinite layer{layer} moment/ridge")
        if method != "dfa":
            if stats is not None:
                stats["operator_attempts"] += 1
                if method in {"ndfa", "kndfa"}:
                    record_ridge(stats, layer, "activity", ridge_a,
                                 args.relative_damping * moment_a <= args.damping_floor)
                if method in {"endfa", "kndfa"}:
                    record_ridge(stats, layer, "error", ridge_e,
                                 args.relative_error_damping * moment_e <= args.error_damping_floor)
            try:
                update = condition_local_update(activity, errors, activity_damping=ridge_a,
                                                error_damping=ridge_e, mode=MODES[method],
                                                backend=args.backend)
                # Exactly the archived A implementation, after ALL factors.
                original_norm, new_norm = raw.weights[layer].norm(), update.norm()
                finite_tensor(original_norm, f"raw_norm[{layer}]")
                finite_tensor(new_norm, f"conditioned_norm[{layer}]")
                if new_norm > 0:
                    update = update * (original_norm / new_norm)
                finite_tensor(update, f"conditioned_weight[{layer}]")
            except Exception as error:
                if stats is not None:
                    stats["operator_failures"] += 1
                # The inherited operator reports this specific numeric failure
                # as RuntimeError. Keep other runtime/CUDA errors distinguishable.
                if (isinstance(error, RuntimeError)
                        and str(error) == "conditioning produced a non-finite update; damping was not changed"):
                    raise FloatingPointError(str(error)) from error
                raise
            if stats is not None:
                stats["operator_successes"] += 1
            weights[layer] = update
        if diagnostics is not None:
            diagnostics.append({"layer": layer, "method": method, "moment_a": moment_a,
                                "moment_e": moment_e, "ridge_a": ridge_a, "ridge_e": ridge_e,
                                "activity": moment_spectrum(activity, ridge_a),
                                "error": moment_spectrum(errors, ridge_e),
                                "raw_weight_norm": raw.weights[layer].norm().item(),
                                "final_weight_norm": weights[layer].norm().item()})
    if method == "dfa":
        return raw
    return Gradients(weights, raw.biases, raw.deltas, raw.loss, raw.bn_gammas, raw.bn_betas)


@torch.no_grad()
def validation_predictions(model, data, args):
    logits = []
    was_training = model.training
    try:
        model.training = False
        with legacy.preserve_batchnorm_state(model):
            for x in data.validation.split(256):
                if args.dataset != "synthetic":
                    x = legacy.standardize_images(x.float().div(255.), data).flatten(1)
                value = model.forward(x.to(args.device))[0]
                finite_tensor(value, "validation logits")
                logits.append(value.cpu())
    finally:
        model.training = was_training
    joined = torch.cat(logits)
    return {"logits": joined, "predictions": joined.argmax(1),
            "labels": data.validation_labels.clone(), "official_test_accessed": False}


def checkpoint(model, feedback, sampler, views, step):
    return {"model": model_state(model), "feedback": [v.cpu().clone() for v in feedback],
            "sampler_rng_state": sampler.get_state(), "view_rng_state": views.get_state(),
            "step": step}


def train_one(args, data, seed, method, *, output_dir):
    """One fixed case; preserve complete histories and numerical failure evidence."""
    validate_args(args)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, feedback = legacy.make_model_and_feedback(data, args, seed)
    sampler = torch.Generator().manual_seed(30000 + seed)
    views = torch.Generator().manual_seed(40000 + seed)
    initial = checkpoint(model, feedback, sampler, views, 0)
    torch.save(initial, output / "initial.pt")
    write_json(output / "args.json", vars(args) | {"seed": seed, "method": method})
    stats = new_conditioning_stats(model.n_hidden_layers)
    status = {"version": VERSION, "seed": seed, "method": method, "status": "RUNNING",
              "args": vars(args), "source_sha256": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
              "completed_steps": 0, "official_test_accessed": False,
              "initial_state_sha256": state_hash(initial["model"]),
              "feedback_sha256": [tensor_hash(v) for v in feedback],
              "conditioner": stats, "training_seconds": 0.,
              "training_seconds_scope": "sampling, augmentation, gradients, factors, sparse spectra, updates and finite checks; excludes validation and artifact writes",
              "update_logging_seconds": 0., "validation_seconds": [],
              "update_logging_seconds_scope": "per-update bookkeeping, moment JSON and update JSONL write/flush",
              "validation_seconds_scope": "each full evaluation and validation/status write, including step0",
              "derived_rng_seeds": {"model": 10000 + seed, "sampler": 30000 + seed,
                                    "views": 40000 + seed, "feedback": args.feedback_seed + 100 * seed}}
    write_json(output / "status.json", status)
    order_hash = hashlib.sha256()
    rows = []
    started = time.perf_counter()
    last_finite = model_state(model, cpu=False)
    raw = update = None
    step = 0
    failure_stage = "initial_validation"
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (output / "updates.jsonl").open("x") as updates, (output / "validation.jsonl").open("x") as validation:
        try:
            for step in range(args.steps + 1):
                if step:
                    legacy.synchronize(args)
                    update_started = time.perf_counter()
                    idx = torch.randint(len(data.train), (args.batch_size,), generator=sampler)
                    order_hash.update(idx.numpy().tobytes())
                    x, y = legacy.paired_batch(data, idx, args, views)
                    model.training = True
                    raw = update = None
                    failure_stage = "raw_gradient"
                    raw = model.dfa_gradients(x, y, feedback)
                    sparse = [] if step in args.diagnostic_steps else None
                    failure_stage = "conditioning_and_sparse_diagnostics"
                    update = condition(model, raw, x, method, args, stats=stats, diagnostics=sparse)
                    finite_gradients(update, "update")
                    failure_stage = "parameter_update"
                    model.apply_gradients(update, lr=args.lr)
                    finite_model(model)
                    legacy.synchronize(args)
                    status["training_seconds"] += time.perf_counter() - update_started
                    logging_started = time.perf_counter()
                    status["completed_steps"] = step
                    last_finite = model_state(model, cpu=False)
                    if sparse is not None:
                        write_json(output / f"moments_{step:05d}.json", sparse)
                    updates.write(json.dumps({"step": step, "raw_loss": raw.loss,
                                              "training_seconds": status["training_seconds"]}, allow_nan=False) + "\n")
                    updates.flush()
                    status["update_logging_seconds"] += time.perf_counter() - logging_started
                if step == 0 or step == args.steps or step % args.eval_every == 0:
                    validation_started = time.perf_counter()
                    failure_stage = "validation"
                    loss, accuracy = legacy.evaluate(model, data, args)
                    if not math.isfinite(loss) or not math.isfinite(accuracy):
                        raise FloatingPointError("Nonfinite validation endpoint")
                    row = {"seed": seed, "method": method, "step": step,
                           "validation_loss": loss, "validation_accuracy": accuracy,
                           "training_seconds": status["training_seconds"],
                           "parameter_count": legacy.parameter_count(model)}
                    rows.append(row)
                    validation.write(json.dumps(row, allow_nan=False) + "\n")
                    validation.flush()
                    write_json(output / "status.json", status)
                    status["validation_seconds"].append(time.perf_counter() - validation_started)
            failure_stage = "prediction_and_checkpoint_artifacts"
            predictions = validation_predictions(model, data, args)
            np.savez(output / "validation_predictions.npz",
                     logits=predictions["logits"].numpy(),
                     predictions=predictions["predictions"].numpy(),
                     labels=predictions["labels"].numpy())
            final = checkpoint(model, feedback, sampler, views, args.steps)
            torch.save(final, output / "final.pt")
            status.update(status="COMPLETE", endpoint=rows[-1], final_state_sha256=state_hash(final["model"]))
        except Exception as error:
            # Numeric and infrastructure errors stay distinguishable and visible.
            status.update(status="FAILED", failure_step=step, error_type=type(error).__name__,
                          error=str(error), failure_stage=failure_stage,
                          numerical_failure=isinstance(error, (FloatingPointError, torch.linalg.LinAlgError)),
                          last_finite_completed_step=status["completed_steps"])
            torch.save({"last_finite": last_finite, "failed_state": model_state(model),
                        "raw": asdict(raw) if raw is not None else None,
                        "update": asdict(update) if update is not None else None,
                        "sampler_rng_state": sampler.get_state(), "view_rng_state": views.get_state()},
                       output / "failure.pt")
        finally:
            status.update(wall_seconds=time.perf_counter() - started,
                          training_order_sha256=order_hash.hexdigest(),
                          final_sampler_rng_sha256=tensor_hash(sampler.get_state()),
                          final_view_rng_sha256=tensor_hash(views.get_state()),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
            write_json(output / "metrics.json", rows)
            status["artifact_sha256"] = {p.name: file_hash(p) for p in sorted(output.iterdir())
                                         if p.is_file() and p.name != "status.json"}
            write_json(output / "status.json", status)
    return model, rows, status


def verify_pins(args):
    source = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        module_name = name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        if name == "experiments/run_ndfa_bn_factors.py":
            actual = Path(__file__).resolve()
        else:
            module = sys.modules.get(module_name)
            if module is None:
                raise ValueError(f"Expected imported source module missing: {module_name}")
            actual = Path(module.__file__).resolve()
        if actual != (ROOT / name).resolve():
            raise ValueError(f"Unexpected import origin: {module_name}: {actual}")
    if bool(args.source_manifest) != bool(args.source_manifest_sha256):
        raise ValueError("Source manifest and its hash must be supplied together")
    if args.source_manifest:
        if file_hash(args.source_manifest) != args.source_manifest_sha256:
            raise ValueError("Source manifest hash mismatch")
        manifest = json.loads(args.source_manifest.read_text())
        for name, digest in manifest["files"].items():
            path = (args.source_manifest.parent / name).resolve()
            if not path.is_relative_to(args.source_manifest.parent.resolve()) or file_hash(path) != digest:
                raise ValueError(f"Archived source mismatch: {name}")
        for name, digest in source.items():
            if manifest["files"].get(name) != digest:
                raise ValueError(f"Imported source not covered by manifest: {name}")
    if bool(args.config) != bool(args.config_sha256):
        raise ValueError("Configuration and its hash must be supplied together")
    if args.config and file_hash(args.config) != args.config_sha256:
        raise ValueError("Configuration hash mismatch")
    return source


def main(argv=None):
    args = parse_args(argv)
    source = verify_pins(args)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = legacy.load_data(args)  # train=True only, download=False, exact original split.
    manifest = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "args": vars(args), "source_sha256": source, "official_test_accessed": False,
                "training_examples": len(data.train), "validation_examples": len(data.validation),
                "data_tensor_sha256": {k: tensor_hash(getattr(data, k))
                                       for k in ("train", "labels", "validation", "validation_labels")},
                "standardization": {"mean": None if data.channel_mean is None else data.channel_mean.tolist(),
                                    "std": None if data.channel_std is None else data.channel_std.tolist()},
                "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                "tf32_cudnn": torch.backends.cudnn.allow_tf32},
                "diagnostic_scope": "Sparse actual-batch spectra only; conditioning/norm matching is unchanged",
                "normalization": "D=raw.deltas*N after local BN backward; final matrix norm match once"}
    write_json(output / "manifest.json", manifest)
    inventory = []
    for seed in args.seeds:
        for method in args.methods:
            case = f"seed{seed}_{method}"
            model, _, status = train_one(args, data, seed, method, output_dir=output / case)
            inventory.append({"case": case, "seed": seed, "method": method, "status": status["status"]})
            write_json(output / "inventory.json", inventory)
            print(json.dumps(inventory[-1]), flush=True)
            del model
    return 0 if all(row["status"] == "COMPLETE" for row in inventory) else 2


if __name__ == "__main__":
    raise SystemExit(main())
