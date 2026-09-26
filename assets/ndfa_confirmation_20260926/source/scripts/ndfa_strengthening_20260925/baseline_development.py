"""Long-horizon validation development for stable BP/DFA baselines.

No official test data are loaded. This first stage compares the current-batch
activity operator; a full EMA/amortized FOOF baseline is a separate work item.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import run_ndfa_submission_benchmark as base
from infogeo.dfa import Gradients
from infogeo.local_preconditioning import condition_local_update


def parameters(model):
    return [*model.weights, *model.biases, *model.bn_gamma, *model.bn_beta]


@torch.no_grad()
def gradient(model, feedback, x, y, args, case):
    # One forward only: BN statistics and FD transforms advance once per step.
    credit = "bp" if case["credit"] == "bp" else "dfa"
    raw_args = argparse.Namespace(**vars(args))
    raw_args.method = credit + ("_decorrelation" if case["normalization"] == "fd" else "")
    raw = base.gradients(model, feedback, x, y, raw_args)
    if case["operator"] == "none":
        return raw
    weights = list(raw.weights)
    for layer in range(model.n_hidden_layers):
        activity = model.last_activities[layer]
        error = raw.deltas[layer] * len(x)
        damping = max(case["rho"] * float(activity.square().mean()), args.damping_floor)
        if not math.isfinite(damping):
            raise FloatingPointError("nonfinite activity moment")
        try:
            value = condition_local_update(activity, error, activity_damping=damping,
                                           error_damping=args.damping_floor,
                                           mode="activity", backend="auto")
        except torch.linalg.LinAlgError as exc:
            raise FloatingPointError(f"conditioner solve: {exc}") from exc
        except RuntimeError as exc:
            if "non-finite update" in str(exc):
                raise FloatingPointError(str(exc)) from exc
            raise
        before = torch.linalg.vector_norm(raw.weights[layer].double())
        after = torch.linalg.vector_norm(value.double())
        if after == 0 and before != 0:
            raise FloatingPointError("conditioning destroyed a nonzero update")
        if after > 0:
            value = value * (before / after).to(value.dtype)
        weights[layer] = value
    return Gradients(weights, raw.biases, raw.deltas, raw.loss, raw.bn_gammas, raw.bn_betas)


def optimizer_for(model, case, weight_decay):
    groups = [{"params": model.weights, "weight_decay": weight_decay},
              {"params": [*model.biases, *model.bn_gamma, *model.bn_beta], "weight_decay": 0.}]
    if case["optimizer"] == "sgd_momentum":
        return torch.optim.SGD(groups, lr=case["lr"], momentum=.9, foreach=False, fused=False)
    if case["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=case["lr"], betas=(.9, .999), eps=1e-8,
                                 foreach=False, fused=False)
    raise ValueError(case["optimizer"])


def step_optimizer(model, update, optimizer, lr):
    grads = [*update.weights, *update.biases, *(update.bn_gammas or []), *(update.bn_betas or [])]
    params = parameters(model)
    assert len(params) == len(grads)
    for p, g in zip(params, grads):
        assert p.shape == g.shape
        p.grad = g
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def learning_rate(peak, step, total, warmup):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    fraction = (step - warmup) / max(total - warmup - 1, 1)
    return peak * (.01 + .99 * .5 * (1 + math.cos(math.pi * fraction)))


@torch.no_grad()
def train(config, case, seed, output, *, device="cuda", fixture=False):
    # The fixture path is only used by CPU protocol tests.
    args = base.parse_args([
        "--output-dir", str(output), "--dataset", "fixture" if fixture else "cifar10",
        "--stage", "fixture" if fixture else "development", "--data-dir", config["data_dir"],
        "--device", device, "--threads", str(config["threads"]),
        "--hidden-dims", *map(str, config["hidden_dims"]),
        "--batch-size", str(config["batch_size"]), "--epochs", str(config["epochs"]),
        "--lr", str(case["lr"]), "--feedback-scale", str(config["feedback_scale"]),
        "--model-seed", str(seed), "--feedback-seed", str(seed + 1000),
        "--order-seed", str(seed + 2000), "--augmentation-seed", str(seed + 3000),
        "--split-seed", str(config["split_seed"]),
        "--fixture-side", "8", "--fixture-train", "32", "--fixture-validation", "16",
    ])
    args.method = case["credit"] + {"none": "", "bn": "_batchnorm", "fd": "_decorrelation"}[case["normalization"]]
    args.decor_lr = case.get("decor_lr", 1e-5)
    base.configure(args)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = base.load_data(args)
    state = data.on(device)
    model, feedback = base.make_model(args, data)
    optimizer = optimizer_for(model, case, config["weight_decay"])
    manifest = base.manifest(args, data, model, feedback)
    manifest.update(case=case, protocol=config, official_test_loaded=False,
                    optimizer={"name": case["optimizer"], "matrix_weight_decay": config["weight_decay"],
                               "schedule": "linear warmup then cosine to 1%", "peak_lr": case["lr"],
                               "momentum": .9 if case["optimizer"] == "sgd_momentum" else None,
                               "betas": [.9,.999] if case["optimizer"] == "adamw" else None},
                    selection="mean final validation CE over two development seeds; accuracy then ID breaks ties",
                    operator=("current-batch activity inverse, hidden weights only, norm matched to own raw update"
                              if case["operator"] == "activity" else "none"),
                    work_scope="sampling, augmentation, gradient, conditioning, optimizer and finite checks; excludes validation, hashes and export",
                    runner_sha256=base.sha256(Path(__file__)))
    base.write_json(output / "manifest.json", manifest)
    order_rng = torch.Generator().manual_seed(args.order_seed)
    aug_rng = torch.Generator().manual_seed(args.augmentation_seed)
    order_hash, aug_hash = hashlib.sha256(), hashlib.sha256()
    steps_per_epoch = math.ceil(len(data.train) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warmup = steps_per_epoch * config["warmup_epochs"]
    work, evaluation_seconds, step = 0., 0., 0
    history = []
    model.training = True
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def save_state(status):
        return {"status": status, "model": base.model_state(model), "feedback": feedback,
                "optimizer_state": optimizer.state_dict(), "args": vars(args), "case": case,
                "completed_updates": step, "order_rng": order_rng.get_state(),
                "augmentation_rng": aug_rng.get_state(), "data": data.provenance,
                "normalization": {"mean": data.channel_mean, "std": data.channel_std}}

    def record(epoch, train_loss=None):
        nonlocal evaluation_seconds
        base.sync(args); began = time.perf_counter()
        metrics = base.evaluate(model, state["validation"], state["validation_labels"], data, args)
        base.sync(args); evaluation_seconds += time.perf_counter() - began
        history.append(dict(epoch=epoch, step=step, training_seconds=work,
                            evaluation_seconds=evaluation_seconds, training_loss=train_loss,
                            validation=metrics, order_sha256=order_hash.hexdigest(),
                            augmentation_sha256=aug_hash.hexdigest()))
        base.write_json(output / "history.json", history)

    status, failure = "complete", None
    try:
        record(0)
        for epoch in range(1, args.epochs + 1):
            base.sync(args); began = time.perf_counter()
            order = torch.randperm(len(data.train), generator=order_rng)
            base.sync(args); work += time.perf_counter() - began
            total_loss = 0.
            for start in range(0, len(order), args.batch_size):
                base.sync(args); began = time.perf_counter()
                idx = order[start:start+args.batch_size]
                x, y, trace = base.augmented_batch(data, idx, aug_rng, device)
                model.training = True
                update = gradient(model, feedback, x, y, args, case)
                lr = learning_rate(case["lr"], step, total, warmup)
                step_optimizer(model, update, optimizer, lr)
                if not all(torch.isfinite(t).all() for t in base.state_tensors(model)):
                    raise FloatingPointError("nonfinite parameter or normalization state")
                base.sync(args); work += time.perf_counter() - began
                order_hash.update(idx.numpy().tobytes()); aug_hash.update(trace)
                total_loss += update.loss * len(idx)
                step += 1
            if epoch == 1 or epoch % config["evaluate_every_epochs"] == 0 or epoch == args.epochs:
                record(epoch, total_loss / len(data.train))
    except FloatingPointError as exc:
        status, failure = "numerical_failure", str(exc)
    torch.save(save_state(status), output / "final.pt")
    result = dict(status=status, error=failure, case=case, seed=seed, completed_updates=step,
                  training_seconds=work, evaluation_seconds=evaluation_seconds,
                  final_validation=history[-1]["validation"] if status == "complete" else None,
                  initial_parameter_sha256=manifest["initial_parameter_sha256"],
                  initial_feedback_sha256=manifest["initial_feedback_sha256"],
                  order_sha256=order_hash.hexdigest(), augmentation_sha256=aug_hash.hexdigest(),
                  official_test_loaded=False, manifest_sha256=base.sha256(output/"manifest.json"),
                  final_checkpoint_sha256=base.sha256(output/"final.pt"))
    base.write_json(output / "endpoint.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--case-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    assert base.sha256(args.config) == args.config_sha256, "configuration changed"
    config = json.loads(args.config.read_text())
    for name, digest in config["source_sha256"].items():
        assert base.sha256(ROOT/name) == digest, name
    case = config["cases"][args.case_index]
    for seed in config["development_seeds"]:
        output = Path(config["output_root"]) / case["id"] / f"seed_{seed}"
        result = train(config, case, seed, output)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
