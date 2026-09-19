"""Small shared, CPU-only helpers for the declared BN forward-decorrelation comparison."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path, expected):
    require(sha256(path) == expected, "config hash mismatch")
    config = json.loads(Path(path).read_text())
    require(config["status"] == "frozen_prospective", "unfrozen configuration")
    require(config["official_test_evaluation"] is False, "test access prohibited")
    source = Path(config["source_root"]).resolve()
    for relative, digest in config["source_sha256"].items():
        require(sha256(source / relative) == digest, "changed source: " + relative)
    require(config["methods"] == ["bp", "dfa", "ndfa", "fd_dfa"], "unexpected method inventory")
    require(len(config["benchmark"]) == 4 and len(config["development"]) == 72 and
            len(config["confirmation"]) == 32, "wrong fixed inventory")
    require(len({c["case_id"] for stage in ("benchmark", "development", "confirmation")
                 for c in config[stage]}) == 108, "duplicate case ID")
    return config


def import_legacy(config):
    source = Path(config["source_root"]).resolve()
    sys.path.insert(0, str(source))
    from experiments import run_ndfa_paired_views as legacy
    import exports
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("experiments", "infogeo") and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(source), "project import escaped snapshot")
    return legacy, exports


def selected_candidates(config, development):
    require(len(development) == 72, "selection requires every declared development outcome")
    require({r["case_id"] for r in development} == {r["case_id"] for r in config["development"]},
            "development inventory mismatch")
    selected, scores = {}, []
    for method in config["methods"]:
        available = []
        for candidate in config["candidates"][method]:
            rows = [r for r in development if r["method"] == method and r["candidate_id"] == candidate["candidate_id"]]
            require(len(rows) == 2 and {r["seed"] for r in rows} == set(config["development_seeds"]), "unpaired development")
            valid = all(r["status"] == "complete" for r in rows)
            row = dict(method=method, **candidate, valid=valid,
                       mean_ce=statistics.mean(r["endpoint"]["validation_loss"] for r in rows) if valid else None,
                       mean_accuracy=statistics.mean(r["endpoint"]["validation_accuracy"] for r in rows) if valid else None)
            scores.append(row)
            if valid:
                available.append(row)
        require(available, "no finite complete candidate for " + method)
        winner = min(available, key=lambda r: (r["mean_ce"], -r["mean_accuracy"], r["peak_lr"], r.get("relative_damping", 0.), r.get("decorrelation_lr", 0.)))
        grids = {key: sorted({c[key] for c in config["candidates"][method]}) for key in ("peak_lr", "relative_damping", "decorrelation_lr") if key in winner}
        boundaries = {key: winner[key] in (values[0], values[-1]) for key, values in grids.items()}
        selected[method] = dict(winner, boundary_selected=any(boundaries.values()), boundary_dimensions=boundaries)
    return selected, scores


def resolve_case(config, case_id, selection=None):
    matches = [(stage, c) for stage in ("benchmark", "development", "confirmation")
               for c in config[stage] if c["case_id"] == case_id]
    require(len(matches) == 1, "undeclared case")
    stage, case = matches[0]
    case = dict(case, stage=stage)
    if stage == "confirmation":
        require(selection is not None, "confirmation requires a locked selection artifact")
        require(selection["development_complete"] and len(selection["development_artifact_hashes"]) == 72,
                "incomplete selection evidence")
        case["candidate_id"] = selection["selected"][case["method"]]["candidate_id"]
    candidate = next(c for c in config["candidates"][case["method"]] if c["candidate_id"] == case["candidate_id"])
    return dict(case, **{key:value for key,value in candidate.items() if key != "candidate_id"})


def legacy_args(config, case, legacy):
    values = dict(config["shared_args"], methods=["dfa" if case["method"] == "fd_dfa" else case["method"]], seeds=[case["seed"]],
                  lr=case["peak_lr"], relative_damping=case.get("relative_damping", .3),
                  output_dir=str(Path(config["output_root"]) / case["case_id"]))
    argv = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, *(map(str, value) if isinstance(value, list) else [str(value)])])
    args = legacy.parse_args(argv)
    require(vars(args) == values, "legacy CLI round trip mismatch")
    return args


def learning_rate(peak, used, budget, schedule):
    progress = min(1., max(0., used / budget))
    warmup = schedule["warmup_fraction"]
    if progress < warmup:
        fraction = schedule["warmup_start_fraction"] + (1. - schedule["warmup_start_fraction"]) * progress / warmup
    else:
        q = (progress - warmup) / (1. - warmup)
        fraction = schedule["minimum_fraction"] + (1. - schedule["minimum_fraction"]) * (1. + math.cos(math.pi*q)) / 2
    return peak * fraction


class WorkClock:
    """Only explicitly submitted synchronized update durations advance work."""
    def __init__(self, budget):
        require(math.isfinite(budget) and budget > 0, "invalid work budget")
        self.budget, self.used, self.last = budget, 0., 0.

    def add(self, duration):
        require(math.isfinite(duration) and duration > 0, "invalid update duration")
        self.last = duration
        self.used += duration

    @property
    def done(self):
        return self.used >= self.budget


def runtime_projection(config, benchmark, elapsed, maximum_seconds=None):
    require(len(benchmark) == 4 and all(r["status"] == "complete" for r in benchmark), "benchmark incomplete")
    require(all(r["endpoint"]["step"] == config["benchmark_steps"] for r in benchmark), "short benchmark")
    # Add missing validation observations explicitly; benchmark has only start/end.
    worst_extra = max(r["invocation_wall_seconds"] - r["endpoint"]["training_seconds"] +
                      max(0, config["maximum_validation_observations"] - r["validation_count"]) *
                      r["evaluation_seconds"] / r["validation_count"] for r in benchmark)
    per_case = config["update_budget_seconds"] + config["runtime_gate"]["overhead_multiplier"] * worst_extra
    projected = elapsed + 104 * per_case + config["runtime_gate"]["audit_and_shutdown_reserve_seconds"]
    maximum = min(config["execution"]["maximum_seconds"], maximum_seconds if maximum_seconds is not None else float("inf"))
    return {"elapsed_seconds": elapsed, "projected_case_seconds": per_case,
            "projected_total_seconds": projected, "maximum_seconds": maximum,
            "accepted": projected <= maximum}
