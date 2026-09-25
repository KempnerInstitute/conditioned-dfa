"""Full-pool CIFAR-10 comparison of conditioning and forward normalization.

This runner only loads the official training split. Development selects the
last epoch, never the best intermediate checkpoint. A separate frozen-selection
wrapper is responsible for authorizing any official-test loading/evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from infogeo.dfa import Gradients, ManualMLP, init_feedback
from infogeo.decorrelated_dfa import DecorrelatedManualMLP
from infogeo.forward_decorrelation import UPSTREAM_REVISION, UPSTREAM_URL
from infogeo.local_preconditioning import condition_local_update

METHODS = ("bp", "dfa", "bp_activity", "dfa_activity", "dfa_error", "dfa_kronecker",
           "bp_batchnorm", "dfa_batchnorm", "bp_decorrelation", "dfa_decorrelation")
SOURCE_FILES = ("experiments/run_ndfa_submission_benchmark.py", "infogeo/__init__.py",
                "infogeo/geometry.py", "infogeo/dfa.py", "infogeo/local_preconditioning.py",
                "infogeo/decorrelated_dfa.py", "infogeo/forward_decorrelation.py")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def tensors_hash(values):
    h = hashlib.sha256()
    for value in values:
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class Data:
    train: torch.Tensor
    labels: torch.Tensor
    validation: torch.Tensor
    validation_labels: torch.Tensor
    channel_mean: torch.Tensor
    channel_std: torch.Tensor
    provenance: dict
    classes: int = 10
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def input_dim(self):
        return self.train[0].numel()

    def on(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = {name: getattr(self, name).to(device) for name in
                                ("train", "labels", "validation", "validation_labels", "channel_mean", "channel_std")}
        return self._cache[key]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=METHODS, default="bp")
    parser.add_argument("--dataset", choices=("cifar10", "fixture"), default="cifar10")
    parser.add_argument("--data-dir", default=str(ROOT / "data/torchvision"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 512])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.3)
    parser.add_argument("--activity-rho", type=float, default=.1)
    parser.add_argument("--error-rho", type=float, default=.1)
    parser.add_argument("--damping-floor", type=float, default=1e-6)
    parser.add_argument("--decor-lr", type=float, default=1e-5)
    parser.add_argument("--feedback-scale", type=float, default=1.)
    parser.add_argument("--model-seed", type=int, default=15100)
    parser.add_argument("--feedback-seed", type=int, default=15101)
    parser.add_argument("--order-seed", type=int, default=15102)
    parser.add_argument("--augmentation-seed", type=int, default=15103)
    parser.add_argument("--split-seed", type=int, default=15104)
    parser.add_argument("--fixture-seed", type=int, default=15105)
    parser.add_argument("--fixture-train", type=int, default=512)
    parser.add_argument("--fixture-validation", type=int, default=128)
    parser.add_argument("--fixture-side", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stage", choices=("development", "confirmation", "throughput", "fixture"), default="development")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Fixture-only: use this many timed training updates, without validation")
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "threads", "fixture_train", "fixture_validation", "fixture_side"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.batch_size < 2 or args.fixture_side < 5 or min(args.hidden_dims) < 1:
        parser.error("batch size >= 2, image side >= 5, and positive hidden widths required")
    for name in ("lr", "activity_rho", "error_rho", "damping_floor", "feedback_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if not math.isfinite(args.decor_lr) or args.decor_lr < 0:
        parser.error("decor_lr must be finite and nonnegative")
    if args.benchmark_steps < 0 or args.benchmark_warmup < 0:
        parser.error("benchmark counts must be nonnegative")
    if args.benchmark_steps and (args.dataset != "fixture" or args.stage != "throughput"):
        parser.error("throughput benchmarking requires synthetic fixture data and throughput stage")
    if args.dataset == "fixture" and args.stage not in {"fixture", "throughput"}:
        parser.error("fixture data must be explicitly labeled fixture or throughput")
    if args.stage == "throughput" and not args.benchmark_steps:
        parser.error("throughput stage requires a positive benchmark_steps")
    return args


def configure(args):
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_data(args):
    """Read train=True only; fixed disjoint 45k/5k split, training-only scaling."""
    if args.dataset == "fixture":
        generator = torch.Generator().manual_seed(args.fixture_seed)
        count = args.fixture_train + args.fixture_validation
        images = torch.randint(0, 256, (count, 3, args.fixture_side, args.fixture_side),
                               generator=generator, dtype=torch.uint8)
        labels = torch.randint(10, (count,), generator=generator)
        train_idx = torch.arange(args.fixture_train)
        val_idx = torch.arange(args.fixture_train, count)
        origin = {"dataset": "synthetic uint8 throughput/test fixture", "fixture_seed": args.fixture_seed,
                  "official_test_loaded": False, "substantive_learning_evidence": False}
    else:
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=args.data_dir, train=True, download=args.download)
        images = torch.as_tensor(dataset.data).permute(0, 3, 1, 2).contiguous()
        labels = torch.as_tensor(dataset.targets, dtype=torch.long)
        if len(labels) != 50000 or images.shape != (50000, 3, 32, 32):
            raise ValueError("CIFAR-10 training pool must contain exactly 50,000 32x32 RGB identities")
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(args.split_seed))
        val_idx, train_idx = order[:5000], order[5000:]
        origin = {"dataset": "CIFAR-10", "official_split": "train=True", "official_test_loaded": False,
                  "split_seed": args.split_seed, "substantive_learning_evidence": True}
    train, validation = images[train_idx], images[val_idx]
    sums = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    for chunk in train.split(512):
        values = chunk.double() / 255.
        sums += values.sum((0, 2, 3))
        squares += values.square().sum((0, 2, 3))
    pixels = len(train) * train.shape[-2] * train.shape[-1]
    mean = sums / pixels
    std = (squares / pixels - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    origin.update({"training_examples": len(train), "validation_examples": len(validation),
                   "train_identity_sha256": tensor_hash(train_idx), "validation_identity_sha256": tensor_hash(val_idx),
                   "full_train_images_sha256": tensor_hash(images), "full_train_labels_sha256": tensor_hash(labels),
                   "training_images_sha256": tensor_hash(train), "training_labels_sha256": tensor_hash(labels[train_idx]),
                   "validation_images_sha256": tensor_hash(validation), "validation_labels_sha256": tensor_hash(labels[val_idx]),
                   "normalization": {"source": "training pool only; before augmentation", "mean": mean.tolist(),
                                     "std": std.tolist(), "definition": "population channel moments after uint8/255", "std_floor": 1e-6}})
    return Data(train, labels[train_idx], validation, labels[val_idx], mean.float(), std.float(), origin)


def normalize(images, state):
    return ((images.float() / 255. - state["channel_mean"][None, :, None, None])
            / state["channel_std"][None, :, None, None]).flatten(1)


def augmented_batch(data, indices, generator, device):
    """One reflected random crop and flip per identity; CPU-drawn paired stream."""
    state = data.on(device)
    images = state["train"][indices.to(device)]
    offsets = torch.randint(0, 9, (len(indices), 2), generator=generator)
    flip = torch.rand(len(indices), generator=generator) < .5
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    rows = offsets[:, 0, None, None].to(device) + torch.arange(images.shape[-2], device=device)[None, :, None]
    columns = offsets[:, 1, None, None].to(device) + torch.arange(images.shape[-1], device=device)[None, None, :]
    selected = padded.permute(0, 2, 3, 1)[torch.arange(len(images), device=device)[:, None, None], rows, columns]
    selected = selected.permute(0, 3, 1, 2)
    selected = torch.where(flip.to(device)[:, None, None, None], selected.flip(-1), selected)
    trace = offsets.numpy().tobytes() + flip.numpy().tobytes()
    return normalize(selected, state), state["labels"][indices.to(device)], trace


class CachedManualMLP(ManualMLP):
    """Expose the already-used activities, without a second BN/state forward."""
    def forward(self, x):
        logits, activities, preactivations = super().forward(x)
        self.last_activities = activities
        return logits, activities, preactivations


def make_model(args, data):
    kwargs = {"seed": args.model_seed, "device": args.device}
    if args.method.endswith("decorrelation"):
        model = DecorrelatedManualMLP(data.input_dim, args.hidden_dims, data.classes,
                                      decor_lr=args.decor_lr, decor_sample_fraction=.1,
                                      decor_mean_momentum=.1, decor_backend="low_rank", **kwargs)
    else:
        model = CachedManualMLP(data.input_dim, args.hidden_dims, data.classes,
                               batchnorm=args.method.endswith("batchnorm"), **kwargs)
    return model, init_feedback(model, seed=args.feedback_seed, scale=args.feedback_scale)


@torch.no_grad()
def gradients(model, feedback, x, y, args):
    try:
        raw = model.bp_gradients(x, y) if args.method.startswith("bp") else model.dfa_gradients(x, y, feedback)
    except ValueError as error:
        translate_decorrelation_numerical_failure(args, error)
        raise
    tensors = [*raw.weights, *raw.biases, *raw.deltas, *(raw.bn_gammas or []), *(raw.bn_betas or [])]
    if not math.isfinite(raw.loss) or not all(torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("nonfinite raw loss or gradient")
    mode = next((name for name in ("activity", "error", "kronecker") if args.method.endswith(name)), None)
    if mode is None:
        return raw
    weights = list(raw.weights)
    for layer in range(model.n_hidden_layers):
        activity = model.last_activities[layer]
        per_example_error = raw.deltas[layer] * len(x)  # Undo CE's batch mean exactly once.
        damping_a = (max(args.activity_rho * float(activity.square().mean()), args.damping_floor)
                     if mode in {"activity", "kronecker"} else args.damping_floor)
        damping_e = (max(args.error_rho * float(per_example_error.square().mean()), args.damping_floor)
                     if mode in {"error", "kronecker"} else args.damping_floor)
        if not math.isfinite(damping_a) or not math.isfinite(damping_e):
            raise FloatingPointError("nonfinite local second moment or relative damping")
        try:
            update = condition_local_update(activity, per_example_error, activity_damping=damping_a,
                                            error_damping=damping_e, mode=mode, backend="auto")
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError(f"conditioner linear solve: {error}") from error
        except RuntimeError as error:
            if "non-finite update" in str(error):
                raise FloatingPointError(str(error)) from error
            raise
        old_norm = torch.linalg.vector_norm(raw.weights[layer].double())
        new_norm = torch.linalg.vector_norm(update.double())
        if new_norm == 0 and old_norm != 0:
            raise FloatingPointError("conditioner destroyed a nonzero update")
        if new_norm > 0:
            update = update * (old_norm / new_norm).to(update.dtype)
        if not torch.isfinite(update).all():
            raise FloatingPointError("nonfinite norm-matched update")
        weights[layer] = update
    return Gradients(weights, raw.biases, raw.deltas, raw.loss, raw.bn_gammas, raw.bn_betas)


def translate_decorrelation_numerical_failure(args, error):
    """Recognize only the primitive's two explicit nonfinite-value guards.

    Shape, dtype, device, feedback and cache errors remain structural. The
    primitive's public input-validation exception types are left unchanged.
    """
    if args.method.endswith("decorrelation") and str(error) in {
            "activity must be finite", "signal must be finite"}:
        raise FloatingPointError(f"forward decorrelation: {error}") from error


def model_state(model):
    def copies(values):
        return [value.detach().cpu().clone() for value in values]
    state = {"weights": copies(model.weights), "biases": copies(model.biases), "batchnorm": model.batchnorm,
             "bn_gamma": copies(model.bn_gamma), "bn_beta": copies(model.bn_beta),
             "bn_running_mean": copies(model.bn_running_mean), "bn_running_var": copies(model.bn_running_var),
             "bn_eps": model.bn_eps, "bn_momentum": model.bn_momentum, "training": model.training}
    state["decorators"] = [
        {"decor_weight": d.decor_weight.detach().cpu().clone(), "running_mean": d.running_mean.detach().cpu().clone(),
         "lr": d.lr, "mean_momentum": d.mean_momentum, "sample_fraction": d.sample_fraction, "backend": d.backend}
        for d in getattr(model, "decorators", [])]
    return state


def restore_model(args, data, state):
    model, _ = make_model(args, data)
    for name in ("weights", "biases", "bn_gamma", "bn_beta", "bn_running_mean", "bn_running_var"):
        setattr(model, name, [value.to(args.device) for value in state[name]])
    model.bn_eps, model.bn_momentum, model.training = state["bn_eps"], state["bn_momentum"], state["training"]
    for decorator, saved in zip(getattr(model, "decorators", []), state["decorators"], strict=True):
        decorator.decor_weight = saved["decor_weight"].to(args.device)
        decorator.running_mean = saved["running_mean"].to(args.device)
        for name in ("lr", "mean_momentum", "sample_fraction", "backend"):
            setattr(decorator, name, saved[name])
    return model


def state_tensors(model):
    return [*model.weights, *model.biases, *model.bn_gamma, *model.bn_beta,
            *model.bn_running_mean, *model.bn_running_var,
            *(value for d in getattr(model, "decorators", []) for value in d.buffers())]


def sync(args):
    if args.device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, images, labels, data, args):
    """No state updates or augmentation; images supplied explicitly by caller."""
    state = data.on(args.device)
    was_training = model.training
    cache = list(model._bn_cache)
    model.training = False
    total, correct = 0., 0
    try:
        for start in range(0, len(labels), args.eval_batch_size):
            x = normalize(images[start:start + args.eval_batch_size].to(args.device), state)
            y = labels[start:start + args.eval_batch_size].to(args.device)
            try:
                logits = model.forward(x)[0]
            except ValueError as error:
                translate_decorrelation_numerical_failure(args, error)
                raise
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite evaluation logits")
            total += F.cross_entropy(logits.double(), y, reduction="sum").item()
            if not math.isfinite(total):
                raise FloatingPointError("nonfinite evaluation cross entropy")
            correct += (logits.argmax(1) == y).sum().item()
    finally:
        model.training = was_training
        model._bn_cache = cache
    return {"loss": total / len(labels), "accuracy": correct / len(labels), "correct": correct, "examples": len(labels)}


def manifest(args, data, model, feedback):
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "args": vars(args), "stage": args.stage, "test_evaluated": False, "data": data.provenance,
            "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
            "environment": {"torch": torch.__version__, "python": sys.version, "cuda": torch.version.cuda,
                            "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                            "host": socket.gethostname(),
                            "slurm": {key: os.environ.get(key) for key in
                                      ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")},
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "threads": torch.get_num_threads()},
            "initial_parameter_sha256": tensors_hash([*model.weights, *model.biases]),
            "initial_feedback_sha256": tensors_hash(feedback),
            "initial_full_state_sha256": tensors_hash(state_tensors(model)),
            "supervised_parameter_count": sum(value.numel() for value in [*model.weights, *model.biases, *model.bn_gamma, *model.bn_beta]),
            "decorrelation_state_numel": getattr(model, "decorrelation_state_numel", 0),
            "normalization_buffer_bytes": sum(value.numel() * value.element_size() for value in
                                              [*model.bn_running_mean, *model.bn_running_var,
                                               *(value for d in getattr(model, "decorators", []) for value in d.buffers())]),
            "optimizer": {"name": "SGD", "momentum": 0, "weight_decay": 0, "schedule": "constant", "lr": args.lr},
            "update_scope": "conditioned hidden weight gradients norm-matched to own raw gradient; raw classifier and bias gradients; BN affine parameters learned",
            "damping": "max(rho * mean(square(local samples)), damping_floor), separate activity/error scales; raw error undoes batch mean",
            "decorrelation": {"upstream_revision": UPSTREAM_REVISION, "upstream_url": UPSTREAM_URL,
                              "adaptation": "fixed gated DFA before outgoing decorator; upstream FA experiments are not reproduced",
                              "scope": "every linear input including classifier", "sample_fraction": .1, "mean_momentum": .1},
            "selection": "final-epoch validation only; intermediate epochs descriptive",
            "augmentation": "one reflected-padding-4 crop plus random horizontal flip; CPU streams paired across methods",
            "batching": "without replacement each epoch; final partial batch retained"}


@torch.no_grad()
def train_one(args, data, reference_dir=None):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, feedback = make_model(args, data)
    run_manifest = manifest(args, data, model, feedback)
    write_json(output / "manifest.json", run_manifest)
    order_rng = torch.Generator().manual_seed(args.order_seed)
    augmentation_rng = torch.Generator().manual_seed(args.augmentation_seed)
    if reference_dir is not None:
        reference = Path(reference_dir)
        reference.mkdir(parents=True, exist_ok=False)
        torch.save({"schema_version": 1, "model": model_state(model),
                    "feedback": [value.detach().cpu().clone() for value in feedback],
                    "order_rng_state": order_rng.get_state(), "augmentation_rng_state": augmentation_rng.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                    "data_provenance": data.provenance,
                    "input_normalization": {"channel_mean": data.channel_mean.clone(), "channel_std": data.channel_std.clone()},
                    "args": vars(args)}, reference / "initial.pt")
        write_json(reference / "manifest.json", {"schema_version": 1, "initial_checkpoint_sha256": sha256(reference / "initial.pt"),
                                                "case_manifest_sha256": sha256(output / "manifest.json"),
                                                "initial_parameter_sha256": run_manifest["initial_parameter_sha256"],
                                                "initial_feedback_sha256": run_manifest["initial_feedback_sha256"],
                                                "initial_full_state_sha256": run_manifest["initial_full_state_sha256"],
                                                "source_sha256": run_manifest["source_sha256"], "data": data.provenance})
    order_trace, augmentation_trace = hashlib.sha256(), hashlib.sha256()
    history = []
    step, epoch, batch_position = 0, 0, 0
    training_seconds, evaluation_seconds = 0., 0.
    model.training = True
    state = data.on(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    current_order = torch.empty(0, dtype=torch.long)
    benchmark_times = []

    def checkpoint(status):
        return {"schema_version": 1, "status": status, "args": vars(args), "model": model_state(model),
                "feedback": [value.detach().cpu().clone() for value in feedback], "completed_updates": step,
                "epoch": epoch, "batch_position": batch_position, "current_order": current_order.clone(),
                "order_rng_state": order_rng.get_state(), "augmentation_rng_state": augmentation_rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                "input_normalization": {"channel_mean": data.channel_mean.clone(), "channel_std": data.channel_std.clone()},
                "data_provenance": data.provenance,
                "order_sha256": order_trace.hexdigest(), "augmentation_sha256": augmentation_trace.hexdigest(),
                "manifest_sha256": sha256(output / "manifest.json"), "optimizer": run_manifest["optimizer"]}

    try:
        target_updates = args.benchmark_steps + args.benchmark_warmup if args.benchmark_steps else None
        done = False
        actual_epochs = math.ceil(target_updates * args.batch_size / len(data.train)) + 1 if target_updates else args.epochs
        for epoch in range(1, actual_epochs + 1):
            current_order = torch.randperm(len(data.train), generator=order_rng)
            for batch_position in range(0, len(current_order), args.batch_size):
                indices = current_order[batch_position:batch_position + args.batch_size]
                order_trace.update(indices.numpy().tobytes())
                sync(args)
                start = time.perf_counter()
                x, y, trace = augmented_batch(data, indices, augmentation_rng, args.device)
                augmentation_trace.update(trace)
                model.training = True
                update = gradients(model, feedback, x, y, args)
                model.apply_gradients(update, lr=args.lr)
                if not all(torch.isfinite(value).all() for value in state_tensors(model)):
                    raise FloatingPointError("nonfinite supervised parameter or normalization state")
                step += 1
                sync(args)
                duration = time.perf_counter() - start
                training_seconds += duration
                if target_updates and step > args.benchmark_warmup:
                    benchmark_times.append(duration)
                if target_updates and step >= target_updates:
                    done = True
                    break
            if target_updates:
                if done:
                    break
                continue
            sync(args)
            start = time.perf_counter()
            validation = evaluate(model, state["validation"], state["validation_labels"], data, args)
            sync(args)
            evaluation_seconds += time.perf_counter() - start
            row = {"epoch": epoch, "step": step, "method": args.method, "validation_loss": validation["loss"],
                   "validation_accuracy": validation["accuracy"], "validation_correct": validation["correct"],
                   "validation_examples": validation["examples"], "training_seconds": training_seconds,
                   "evaluation_seconds": evaluation_seconds, "order_sha256": order_trace.hexdigest(),
                   "augmentation_sha256": augmentation_trace.hexdigest()}
            history.append(row)
            write_json(output / "history.json", history)
        status = "complete"
        torch.save(checkpoint(status), output / "final.pt")
        endpoint = {"schema_version": 1, "status": status, "method": args.method,
                    "completed_updates": step, "completed_epochs": epoch,
                    "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                    "order_sha256": order_trace.hexdigest(), "augmentation_sha256": augmentation_trace.hexdigest(),
                    "final_checkpoint_sha256": sha256(output / "final.pt"), "manifest_sha256": sha256(output / "manifest.json"),
                    "test_evaluated": False, "final_validation": history[-1] if history else None,
                    "benchmark_step_seconds": benchmark_times,
                    "benchmark_mean_step_seconds": sum(benchmark_times) / len(benchmark_times) if benchmark_times else None}
        write_json(output / "history.json", history)
        write_json(output / "endpoint.json", endpoint)
        return model, endpoint
    except FloatingPointError as error:
        torch.save(checkpoint("numerical_failure"), output / "failure.pt")
        failure = {"schema_version": 1, "status": "numerical_failure", "method": args.method,
                   "error": str(error), "completed_updates": step, "epoch": epoch, "batch_position": batch_position,
                   "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                   "order_sha256": order_trace.hexdigest(), "augmentation_sha256": augmentation_trace.hexdigest(),
                   "failure_checkpoint_sha256": sha256(output / "failure.pt"), "manifest_sha256": sha256(output / "manifest.json"),
                   "test_evaluated": False}
        write_json(output / "history.json", history)
        write_json(output / "failure.json", failure)
        return model, failure


def run(args, data=None, reference_dir=None):
    configure(args)
    return train_one(args, load_data(args) if data is None else data, reference_dir=reference_dir)


if __name__ == "__main__":
    _, result = run(parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
