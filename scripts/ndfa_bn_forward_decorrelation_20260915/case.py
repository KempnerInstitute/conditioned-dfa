"""One measured-work SGD case, or CPU-only checks/audit; archived gradients reused."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from common import (WorkClock, import_legacy, learning_rate, legacy_args, read_config,
                    require, resolve_case, selected_candidates, sha256, write_json)


def parameters(model):
    return [*model.weights, *model.biases, *model.bn_gamma, *model.bn_beta]


def make_optimizer(model, config, peak_lr):
    import torch
    opt = config["optimizer"]
    return torch.optim.SGD([
        {"params": model.weights, "weight_decay": opt["weight_decay"]},
        {"params": [*model.biases, *model.bn_gamma, *model.bn_beta], "weight_decay": 0.},
    ], lr=peak_lr, momentum=opt["momentum"], dampening=0., nesterov=False,
        foreach=False, fused=False, differentiable=False)


def optimizer_step(model, update, optimizer, lr):
    # condition() has already norm-matched each current hidden DFA weight update.
    # SGD then accumulates that direction (plus matrix-only L2) in momentum.
    grads = [*update.weights, *update.biases, *(update.bn_gammas or []), *(update.bn_betas or [])]
    tensors = parameters(model)
    require(len(tensors) == len(grads), "parameter/gradient mapping mismatch")
    for parameter, gradient in zip(tensors, grads):
        require(parameter.shape == gradient.shape, "gradient shape mismatch")
        parameter.grad = gradient
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train(config, case, data, args, legacy, exports, *, timer=time.perf_counter, progress=None):
    import torch
    model, feedback = legacy.make_model_and_feedback(data, args, case["seed"])
    if case["method"] == "fd_dfa":
        from model import DecorrelatedMLP
        model = DecorrelatedMLP.from_base(model, case["decorrelation_lr"])
    optimizer = make_optimizer(model, config, case["peak_lr"])
    from experiments.run_ndfa_bn_confirmation_case import model_fingerprint as supervised_fingerprint
    initial = supervised_fingerprint(model)
    sampler = torch.Generator().manual_seed(30000 + case["seed"])
    views = torch.Generator().manual_seed(40000 + case["seed"])
    clock = WorkClock(config["update_budget_seconds"])
    rows, evaluation_seconds, step = [], 0., 0
    if progress is not None:
        progress.update(model=model, optimizer=optimizer, rows=rows, sampler=sampler, views=views,
                        clock=clock, initial_model_fingerprint=initial)
    next_observation = config["observe_every_work_seconds"]
    cap = config["benchmark_steps"] if case["stage"] == "benchmark" else args.steps
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def record(reason, lr):
        nonlocal evaluation_seconds
        began = time.perf_counter()
        # Common diagnostics are outside the update-work clock. Include BN state.
        from model import decorator_state, learned_state_count
        tensors = [*parameters(model), *model.bn_running_mean, *model.bn_running_var, *decorator_state(model)]
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise FloatingPointError("nonfinite parameter or BN running state at observation")
        loss, accuracy = legacy.evaluate(model, data, args)
        legacy.synchronize(args)
        evaluation_seconds += time.perf_counter() - began
        if not torch.isfinite(torch.tensor([loss, accuracy])).all().item():
            raise FloatingPointError("nonfinite validation")
        rows.append({"method": case["method"], "seed": case["seed"], "step": step,
                     "validation_loss": loss, "validation_accuracy": accuracy,
                     "training_seconds": clock.used, "last_update_seconds": clock.last,
                     "learning_rate": lr, "observation_reason": reason,
                     "parameter_count": legacy.parameter_count(model),
                     "batchnorm_affine_parameter_count": 2 * sum(args.hidden_dims),
                     "decorrelation_learned_state_count": learned_state_count(model),
                     "decorrelation_updates": list(getattr(model, "decorrelation_updates", [])),
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None})

    record("initial", learning_rate(case["peak_lr"], 0., clock.budget, config["schedule"]))
    # Carry the read between updates so LR/loop/clock bookkeeping is charged.
    # Reset it only after an untimed observation (diagnostics + validation).
    last_clock_read = timer()
    while not clock.done and step < cap:
        lr = learning_rate(case["peak_lr"], clock.used, clock.budget, config["schedule"])
        indices = torch.randint(len(data.train), (args.batch_size,), generator=sampler)
        x, labels = legacy.paired_batch(data, indices, args, views)
        model.training = True
        try:
            raw = model.bp_gradients(x, labels) if case["method"] == "bp" else model.dfa_gradients(x, labels, feedback)
        except ValueError as error:
            if case["method"] == "fd_dfa" and str(error) in ("activity must be finite", "signal must be finite"):
                raise FloatingPointError(str(error)) from error
            raise
        try:
            update = raw if case["method"] == "fd_dfa" else legacy.condition(model, raw, x, case["method"], args)
        except RuntimeError as error:
            if str(error) == "conditioning produced a non-finite update; damping was not changed":
                raise FloatingPointError(str(error)) from error
            raise
        optimizer_step(model, update, optimizer, lr)
        legacy.synchronize(args)
        clock_read = timer()
        clock.add(clock_read - last_clock_read)
        last_clock_read = clock_read
        step += 1
        reasons = []
        if clock.used >= next_observation:
            reasons.append("work_checkpoint")
            while next_observation <= clock.used:
                next_observation += config["observe_every_work_seconds"]
        if step == config["secondary_step_endpoint"]:
            reasons.append("same_update_secondary")
        if clock.done or step == cap:
            reasons.append("final")
        if reasons:
            record("+".join(reasons), lr)
            last_clock_read = timer()
    require(case["stage"] == "benchmark" or clock.done, "safety step cap reached before work budget")
    require(case["stage"] != "benchmark" or step == config["benchmark_steps"], "benchmark hit work cap before fixed steps")
    require(not clock.done or clock.used - clock.budget < clock.last + 1e-9, "work overshoot exceeds last update")
    auxiliary = {"initial_model_fingerprint": initial,
                 "evaluation_seconds": evaluation_seconds, "validation_count": len(rows),
                 "stop_reason": "benchmark_step_cap" if case["stage"] == "benchmark" else "work_budget",
                 "work_budget_seconds": clock.budget, "work_overshoot_seconds": max(0., clock.used-clock.budget),
                 "final_sampler_state": sampler.get_state(), "final_views_state": views.get_state()}
    return model, optimizer, rows, auxiliary


def audit_case(config, entry, legacy, exports):
    directory = Path(config["output_root"]) / entry["case_id"]
    receipt = json.loads((directory / "receipt.json").read_text())
    require(receipt["case"] == {k: entry[k] for k in receipt["case"]}, "case identity changed")
    require(entry["artifacts_sha256"] == {p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}, "missing/extra/changed case artifact")
    require(set(entry["artifacts_sha256"]) == {"receipt.json","metrics.json","final_state.pt","final_validation.pt","final_export.json","optimizer_sampling_state.pt"}, "unexpected successful artifact inventory")
    expected_decor = None if entry["method"] != "fd_dfa" else {"learning_rate":entry["decorrelation_lr"],"sample_fraction":.1,"mean_momentum":.1,"backend":"low_rank","injection":"before outgoing decorator","upstream_revision":"00cf47050bbd20e6a153e10bd86e1651524f9779"}
    require(receipt["decorrelation"] == expected_decor, "decorrelation configuration changed")
    rows = json.loads((directory / "metrics.json").read_text())
    require(rows[-1] == entry["endpoint"] == receipt["endpoint"], "endpoint mismatch")
    require(rows[0]["step"] == 0 and rows[0]["training_seconds"] == 0., "missing initialization")
    require(all(a["step"] < b["step"] and a["training_seconds"] < b["training_seconds"] for a,b in zip(rows,rows[1:])), "nonmonotone history")
    require(len(rows) == receipt["validation_count"], "wrong history count")
    require(vars(legacy_args(config,receipt["case"],legacy)) == receipt["args"], "saved scientific arguments changed")
    for row in rows:
        require(row["method"] == entry["method"] and row["seed"] == entry["seed"], "wrong history identity")
        require(all(math.isfinite(row[key]) for key in ("validation_loss","validation_accuracy","training_seconds","last_update_seconds","learning_rate")), "nonfinite history")
        require(row["validation_loss"] >= 0 and 0 <= row["validation_accuracy"] <= 1, "invalid metric range")
        require(row["parameter_count"] == 3679754 and row["batchnorm_affine_parameter_count"] == 3072, "wrong architecture")
        expected_decor_count = sum(d*d+d for d in (3072,1024,512)) if entry["method"] == "fd_dfa" else 0
        require(row["decorrelation_learned_state_count"] == expected_decor_count, "wrong decorator state size")
        require(row["decorrelation_updates"] == ([row["step"]]*3 if entry["method"] == "fd_dfa" else []), "decorator updated more/less than once per step")
        prior_work = 0. if row["step"] == 0 else row["training_seconds"]-row["last_update_seconds"]
        expected_lr = learning_rate(entry["peak_lr"],prior_work,config["update_budget_seconds"],config["schedule"])
        require(math.isclose(row["learning_rate"],expected_lr,rel_tol=1e-10,abs_tol=1e-12), "recorded LR differs from schedule")
    # Reconstruct all observation triggers from the emitted increasing work/step sequence.
    next_work = config["observe_every_work_seconds"]
    for i,row in enumerate(rows[1:],1):
        expected=[]
        if row["training_seconds"] >= next_work:
            require(row["training_seconds"]-next_work < row["last_update_seconds"]+1e-9, "missing work checkpoint")
            expected.append("work_checkpoint")
            while next_work <= row["training_seconds"]:
                next_work += config["observe_every_work_seconds"]
        if row["step"] == config["secondary_step_endpoint"]:
            expected.append("same_update_secondary")
        if i == len(rows)-1:
            expected.append("final")
        require(row["observation_reason"] == "+".join(expected), "unexpected/missing observation trigger")
    require(rows[-1]["step"] < config["secondary_step_endpoint"] or any(r["step"] == config["secondary_step_endpoint"] for r in rows), "missing secondary step observation")
    require(receipt["official_test_evaluation"] is False, "unexpected test access")
    require(receipt["device"] == config["execution"]["device"] and receipt["training_examples"] == 48000 and receipt["validation_examples"] == 2000, "device/data size mismatch")
    if entry["stage"] != "benchmark":
        require(receipt["stop_reason"] == "work_budget" and rows[-1]["training_seconds"] >= config["update_budget_seconds"], "short work budget")
        require(receipt["work_overshoot_seconds"] < rows[-1]["last_update_seconds"] + 1e-9, "invalid overshoot")
    else:
        require(rows[-1]["step"] == config["benchmark_steps"], "wrong benchmark endpoint")
    command = {"method": entry["method"], "seed": entry["seed"], "steps": rows[-1]["step"],
               "args": receipt["args"], "parameter_count": 3679754, "hidden_dims": [1024,512], "decorrelation":receipt["decorrelation"]}
    state = exports.audit_saved_case(directory, command, rows[-1], receipt["input_standardization"])
    return {"case_id": entry["case_id"], "checkpoint_audit": state, "observations": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--selection")
    parser.add_argument("--selection-sha256")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--audit-all", action="store_true")
    command = parser.parse_args()
    config = read_config(command.config, command.config_sha256)
    legacy, exports = import_legacy(config)
    import torch
    torch.set_num_threads(config["shared_args"]["threads"])
    if command.check_only:
        fake = {"development_complete": True, "development_artifact_hashes": dict.fromkeys(map(str,range(72)), "check only"),
                "selected": {method: config["candidates"][method][4] for method in config["methods"]}}
        count = 0
        for stage in ("benchmark", "development", "confirmation"):
            for planned in config[stage]:
                resolved = resolve_case(config, planned["case_id"], fake)
                legacy_args(config, resolved, legacy)
                count += 1
        require(not torch.cuda.is_initialized(), "check-only initialized CUDA")
        print(json.dumps({"accepted": True, "cli_round_trips": count, "cuda_initialized": False,
                          "torch": str(torch.__version__)}))
        return
    if command.audit_all:
        require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
        workflow = json.loads((Path(config["output_root"]) / "workflow.json").read_text())
        require(workflow["config_sha256"] == command.config_sha256 and workflow["source_sha256"] == config["source_sha256"], "workflow source/config mismatch")
        require(workflow["official_test_evaluation"] is False and workflow["parent_cuda_initialized"] is False, "workflow scope changed")
        expected=[dict(case,stage=stage) for stage in ("benchmark","development","confirmation") for case in config[stage]]
        require(len(workflow["cases"]) == 108 and all(all(e[k] == value for k,value in plan.items()) for e,plan in zip(workflow["cases"],expected)), "fixed 108-case inventory changed")
        out=Path(config["output_root"])
        selection=json.loads((out/"selection.json").read_text())
        require(sha256(out/"selection.json") == workflow["selection_sha256"] and selection["config_sha256"] == command.config_sha256, "selection binding changed")
        development=[e for e in workflow["cases"] if e["stage"] == "development"]
        selected,scores=selected_candidates(config,development)
        require(selected == selection["selected"] and scores == selection["all_candidate_scores"], "selection rule changed")
        require(selection["development_artifact_hashes"] == {e["case_id"]:e["artifacts_sha256"] for e in development}, "selection evidence changed")
        for entry in workflow["cases"]:
            resolved=resolve_case(config,entry["case_id"],selection)
            require(all(entry[key] == value for key,value in resolved.items()), "case differs from selected configuration")
            directory=out/entry["case_id"]
            require(entry["artifacts_sha256"] == {p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}, "changed/failing case evidence")
            receipt=json.loads((directory/"receipt.json").read_text())
            require(entry["status"] == receipt["status"] and entry["status"] in ("complete","numerical_failed"), "nonterminal/mismatched case status")
            require(receipt["config_sha256"] == command.config_sha256, "case config mismatch")
            require(receipt["selection_sha256"] == (workflow["selection_sha256"] if entry["stage"] == "confirmation" else None), "case selection mismatch")
            if entry["stage"] == "confirmation":
                require(selection["created_utc"] < entry["started_utc"], "confirmation preceded selection lock")
            if entry["status"] == "numerical_failed":
                prefix=json.loads((directory/"metrics.json").read_text())
                require(receipt.get("finite_prefix_observations",0) == len(prefix), "failed finite-prefix count mismatch")
                for row in prefix:
                    require(row["method"] == entry["method"] and row["seed"] == entry["seed"], "failed prefix identity")
                    require(all(math.isfinite(row[k]) for k in ("validation_loss","validation_accuracy","training_seconds","learning_rate")), "nonfinite failed prefix")
                    require(row["validation_loss"] >= 0 and 0 <= row["validation_accuracy"] <= 1, "invalid failed prefix metric")
        audited = [audit_case(config, entry, legacy, exports) for entry in workflow["cases"] if entry["status"] == "complete"]
        groups = {}
        preprocessing = None
        validation_reference = None
        for entry in workflow["cases"]:
            if entry["status"] == "complete":
                receipt = json.loads((Path(config["output_root"])/entry["case_id"]/"receipt.json").read_text())
                signature = receipt["initial_model_fingerprint"]
                require(entry["seed"] not in groups or groups[entry["seed"]] == signature, "paired initialization differs")
                groups[entry["seed"]] = signature
                require(preprocessing is None or preprocessing == receipt["input_standardization"], "preprocessing differs between cases")
                preprocessing = receipt["input_standardization"]
                predictions=torch.load(out/entry["case_id"]/"final_validation.pt",map_location="cpu",weights_only=True)
                identity=(predictions["indices"],predictions["labels"])
                require(validation_reference is None or all(torch.equal(a,b) for a,b in zip(identity,validation_reference)), "validation identities/labels differ")
                validation_reference=identity
        complete_confirmation = sum(e["status"] == "complete" and e["stage"] == "confirmation" for e in workflow["cases"])
        accepted = complete_confirmation == 32 and workflow["status"] == "TRAINING_COMPLETE"
        value = {"accepted": accepted, "complete_confirmation": complete_confirmation, "expected_confirmation": 32,
                 "cases": audited, "official_test_evaluation": False,
                 "config_sha256":command.config_sha256,"source_sha256":config["source_sha256"],
                 "selection_sha256":workflow["selection_sha256"],"bound_case_inventory":workflow["cases"],
                 "scope": "CPU artifact/state/logit audit; not an independent GPU checkpoint replay"}
        write_json(Path(config["output_root"])/"audit.json", value)
        print(json.dumps({"accepted": accepted, "audited_cases": len(audited)}))
        return
    require(command.case_id, "case ID required")
    selection = None
    if command.selection:
        require(sha256(command.selection) == command.selection_sha256, "selection hash mismatch")
        selection = json.loads(Path(command.selection).read_text())
        require(selection["config_sha256"] == command.config_sha256, "selection config mismatch")
    case = resolve_case(config, command.case_id, selection)
    args = legacy_args(config, case, legacy)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {"case": case, "status": "running", "args": vars(args),
               "config_sha256": command.config_sha256, "selection_sha256": command.selection_sha256,
               "official_test_evaluation": False}
    write_json(output/"receipt.json", receipt)
    progress = {}
    try:
        require(torch.cuda.is_available(), "CUDA unavailable; no fallback")
        require(torch.cuda.get_device_name() == config["execution"]["device"], "wrong GPU")
        # This is the isolated training child; the orchestration parent never uses CUDA.
        probe = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize()
        require(probe.item() == 2., "CUDA allocation probe failed")
        del probe
        data = legacy.load_data(args)
        model, optimizer, rows, auxiliary = train(config, case, data, args, legacy, exports, progress=progress)
        write_json(output/"metrics.json", rows)
        exports.export_final(model, data, args, rows[-1], output)
        torch.save({"optimizer": optimizer.state_dict(),
                    "sampler": auxiliary.pop("final_sampler_state"), "views": auxiliary.pop("final_views_state"),
                    "clock_seconds": rows[-1]["training_seconds"],
                    "scope": "additional optimizer/sampling artifacts; exact training resumption not claimed"}, output/"optimizer_sampling_state.pt")
        receipt.update(auxiliary, decorrelation=model.decorrelation_spec() if hasattr(model,"decorators") else None, status="complete", endpoint=rows[-1], child_wall_seconds=time.perf_counter()-started,
                       training_examples=len(data.train),validation_examples=len(data.validation),
                       device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                       input_standardization={"mean":data.channel_mean.tolist(), "std":data.channel_std.tolist(), "validation_used":False})
        write_json(output/"receipt.json", receipt)
        print(json.dumps({"case_id":case["case_id"], "status":"complete", "steps":rows[-1]["step"],
                          "update_seconds":rows[-1]["training_seconds"]}), flush=True)
    except Exception as error:
        receipt.update(status="numerical_failed" if isinstance(error, FloatingPointError) else "infrastructure_failed",
                       error=repr(error), child_wall_seconds=time.perf_counter()-started)
        # Preserve finite observed history and diagnostic state; neither is a completed endpoint.
        write_json(output/"metrics.json",progress.get("rows",[]))
        if progress:
            receipt.update(finite_prefix_observations=len(progress["rows"]),
                           initial_model_fingerprint=progress["initial_model_fingerprint"],
                           observed_update_seconds=progress["clock"].used)
            try:
                torch.save({"model":exports.state_tensors(progress["model"]),
                            "optimizer":progress["optimizer"].state_dict(),
                            "sampler":progress["sampler"].get_state(),"views":progress["views"].get_state(),
                            "is_completed_endpoint":False},output/"failure_diagnostic_state.pt")
            except Exception as diagnostic_error:
                receipt["failure_diagnostic_error"] = repr(diagnostic_error)
        write_json(output/"receipt.json", receipt)
        raise


if __name__ == "__main__":
    main()
