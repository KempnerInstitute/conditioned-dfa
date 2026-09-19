"""An exact finite-batch A/E/K diagnostic with independently assigned moments.

This is an engineered, BP-informed diagnostic, not a proposed learning algorithm.
It realizes the prescribed teaching signal with one trainable linear layer and a
fixed readout and fixed DFA feedback. It does not train, select, or tune a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("raw", "activity", "error", "kronecker")
TRAJECTORY_METHODS = (*METHODS, "bp")
ORIENTATIONS = ("isotropic", "signal_low", "signal_high")
SOURCE_FILES = (
    "experiments/run_ndfa_factor_mechanism.py",
    "analysis/analyze_ndfa_factor_mechanism.py",
    "tests/test_ndfa_factor_mechanism.py",
    "docs/research/ndfa_factor_mechanism_protocol_20260915.md",
)
DEFAULT_CONFIG = {
    "schema_version": 1,
    "study_id": "ndfa_factor_mechanism_20260915",
    "purpose": "engineered_fixed_readout_DFA_directional_diagnostic",
    "dimension": 8,
    "batch_sizes": [32, 64, 128],
    "spectrum_low": 1.0,
    "spectrum_high": 16.0,
    "feedback_readout_cosine": 0.5,
    "rotation_seeds": list(range(9151500, 9151508)),
    "activity_damping": [0.1, 1.0, 16.0],
    "error_damping": [0.1, 1.0, 16.0],
    "step_sizes": [0.001, 0.01, 0.1],
    "orientations": list(ORIENTATIONS),
    "methods": list(METHODS),
    "cone_anchors": ["activity", "error", "kronecker"],
    "cone_margins": ["below", "boundary", "above"],
    "cone_sides": ["adverse", "favorable"],
    "absolute_tolerance": 2e-10,
    "trajectory_seed": 9151500,
    "trajectory_batch_size": 64,
    "trajectory_steps": 100,
    "trajectory_lr": 0.03,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_config(config: dict) -> None:
    required = set(DEFAULT_CONFIG) | {"source_sha256", "frozen_utc"}
    if set(config) != required:
        raise ValueError("configuration keys differ from the prospective schema")
    d = config["dimension"]
    ns = config["batch_sizes"]
    if (not isinstance(d, int) or d < 2 or d % 2 or not ns or len(set(ns)) != len(ns)
            or any(not isinstance(n, int) or n < 2 * d for n in ns)):
        raise ValueError("require an even dimension >=2 and batch size >=2*dimension")
    if not 0 < config["spectrum_low"] < config["spectrum_high"]:
        raise ValueError("invalid covariance spectrum")
    if config["spectrum_low"] ** 2 <= 1 / d:
        raise ValueError("joint covariance is not strictly positive for the prescribed cross moment")
    if not 0 < config["feedback_readout_cosine"] < 1:
        raise ValueError("feedback/readout cosine must be strictly between zero and one")
    for name in ("activity_damping", "error_damping", "step_sizes"):
        values = config[name]
        if not values or len(values) != len(set(values)) or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"invalid {name}")
    if not config["rotation_seeds"] or len(set(config["rotation_seeds"])) != len(config["rotation_seeds"]):
        raise ValueError("rotation seeds must be nonempty and unique")
    for name in ("orientations", "methods", "cone_anchors", "cone_margins", "cone_sides"):
        if config[name] != DEFAULT_CONFIG[name]:
            raise ValueError(f"prospective {name} cannot be omitted or reordered")
    if config["schema_version"] != 1 or not 0 < config["absolute_tolerance"] < 1e-7:
        raise ValueError("unsupported schema or audit tolerance")
    if config["source_sha256"] != source_hashes():
        raise ValueError("source hashes differ from the frozen configuration")
    if (config["trajectory_seed"] not in config["rotation_seeds"]
            or config["trajectory_batch_size"] not in ns
            or not isinstance(config["trajectory_steps"], int) or config["trajectory_steps"] < 1
            or not 0 < config["trajectory_lr"] < 1):
        raise ValueError("invalid prospective trajectory settings")


def orthogonal(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    return q * np.where(np.diag(r) >= 0, 1.0, -1.0)[None, :]


def moment_eigenvalues(orientation: str, dimension: int, low: float, high: float) -> np.ndarray:
    half = dimension // 2
    if orientation == "isotropic":
        return np.full(dimension, (low + high) / 2)
    if orientation == "signal_low":
        return np.array([low] * half + [high] * half)
    if orientation == "signal_high":
        return np.array([high] * half + [low] * half)
    raise ValueError(f"unknown orientation {orientation}")


def make_base(config: dict, seed: int, activity_orientation: str, error_orientation: str,
              batch_size: int | None = None) -> dict[str, np.ndarray]:
    """Realize CA, CE and U exactly using disjoint sample subspaces."""
    d, n = config["dimension"], batch_size or config["trajectory_batch_size"]
    rng = np.random.default_rng(seed)
    q_left = orthogonal(rng, d, d)
    q_right = orthogonal(rng, d, d)
    q_sample = orthogonal(rng, n, 2 * d)
    a = moment_eigenvalues(activity_orientation, d, config["spectrum_low"], config["spectrum_high"])
    e = moment_eigenvalues(error_orientation, d, config["spectrum_low"], config["spectrum_high"])
    ca = (q_right * a[None, :]) @ q_right.T
    ce = (q_left * e[None, :]) @ q_left.T
    u = q_left @ q_right.T / np.sqrt(d)
    activity = np.sqrt(n) * (q_sample[:, :d] * np.sqrt(a)[None, :]) @ q_right.T
    explained = activity @ np.linalg.solve(ca, u.T)
    residual_moment = ce - u @ np.linalg.solve(ca, u.T)
    residual_factor = np.linalg.cholesky(residual_moment)
    teaching = explained + np.sqrt(n) * q_sample[:, d:] @ residual_factor.T
    signal = np.array([1.0] * (d // 2) + [0.0] * (d // 2)) / np.sqrt(d // 2)
    g = (q_left * signal[None, :]) @ q_right.T
    return {
        "activity": activity, "teaching": teaching, "ca_expected": ca,
        "ce_expected": ce, "raw_expected": u, "signal_gradient": g,
        "q_left": q_left, "q_right": q_right,
        "activity_eigenvalues": a, "error_eigenvalues": e,
    }


def conditioned_update(u: np.ndarray, ca: np.ndarray, ce: np.ndarray, method: str,
                       damping_a: float, damping_e: float) -> np.ndarray:
    """Apply the paper's A/E/K factors and match the whole matrix Frobenius norm."""
    if method not in METHODS or damping_a <= 0 or damping_e <= 0:
        raise ValueError("invalid method or damping")
    out = u.copy()
    if method in ("activity", "kronecker"):
        out = np.linalg.solve(ca + damping_a * np.eye(ca.shape[0]), out.T).T
    if method in ("error", "kronecker"):
        out = np.linalg.solve(ce + damping_e * np.eye(ce.shape[0]), out)
    norm = np.linalg.norm(out)
    if norm == 0 or not np.isfinite(norm):
        raise FloatingPointError("invalid conditioned update norm")
    return out * (np.linalg.norm(u) / norm)


def rotation_components(u: np.ndarray, v: np.ndarray) -> tuple[float, float, np.ndarray]:
    un, vn = u / np.linalg.norm(u), v / np.linalg.norm(v)
    cosine = float(np.clip(np.vdot(un, vn), -1.0, 1.0))
    remainder = vn - cosine * un
    sine = float(np.linalg.norm(remainder))
    if sine < 1e-12:
        raise ValueError("cone anchor must produce a nonzero rotation")
    return cosine, sine, remainder / sine


def cone_gradient(u: np.ndarray, v: np.ndarray, margin: str, side: str) -> tuple[np.ndarray, float, float]:
    """Choose a cone-boundary teacher; this is explicitly BP-informed."""
    cosine, sine, z = rotation_components(u, v)
    if margin == "below":
        c = sine / 2
    elif margin == "boundary":
        c = sine
    elif margin == "above":
        c = (1 + sine) / 2
    else:
        raise ValueError("unknown cone margin")
    if side not in ("adverse", "favorable"):
        raise ValueError("unknown cone side")
    sign = -1 if side == "adverse" else 1
    g = c * u / np.linalg.norm(u) + sign * np.sqrt(1 - c * c) * z
    predicted_anchor_cosine = c * cosine + sign * np.sqrt(1 - c * c) * sine
    return g, float(c), float(predicted_anchor_cosine)


def realize_fixed_feedback(base: dict[str, np.ndarray], gradient: np.ndarray, cosine: float) -> dict[str, np.ndarray]:
    """Return an exact linear network at W=0 with both desired BP and DFA updates.

    The readout V and feedback B are constant across all conditions. All 2d
    output coordinates have a nonzero readout. A fixed readout cannot fit the
    output residual's component orthogonal to V; absolute task loss is therefore
    not an inter-condition performance measure.
    """
    a, dfa = base["activity"], base["teaching"]
    d = a.shape[1]
    ident = np.eye(d)
    readout = np.concatenate([ident, ident], axis=0) / np.sqrt(2)
    complement = np.concatenate([ident, -ident], axis=0) / np.sqrt(2)
    feedback = cosine * readout + np.sqrt(1 - cosine**2) * complement
    true_teaching = a @ np.linalg.solve(base["ca_expected"], gradient.T)
    output_residual = true_teaching @ readout.T + (dfa - cosine * true_teaching) @ complement.T / np.sqrt(1 - cosine**2)
    return {"readout": readout, "feedback": feedback, "complement": complement, "output_residual": output_residual,
            "true_teaching": true_teaching, "gradient_expected": gradient}


def loss_after_step(a: np.ndarray, output_residual: np.ndarray, readout: np.ndarray,
                    update: np.ndarray, step: float) -> float:
    residual = output_residual - step * a @ update.T @ readout.T
    return float(np.sum(residual * residual) / (2 * len(a)))


def orientation_prediction(a: np.ndarray, e: np.ndarray, method: str,
                           damping_a: float, damping_e: float) -> dict[str, float | str]:
    """Closed form from assigned eigenvalues, independent of sampled matrices."""
    weights = np.ones_like(a)
    if method in ("activity", "kronecker"):
        weights /= a + damping_a
    if method in ("error", "kronecker"):
        weights /= e + damping_e
    predicted = float(weights[:len(a)//2].sum() / np.sqrt((len(a)//2) * np.sum(weights**2)))
    difference = predicted - 1 / np.sqrt(2)
    sign = "beneficial" if difference > 1e-12 else "harmful" if difference < -1e-12 else "neutral"
    return {"predicted_cosine": predicted, "predicted_difference": difference, "prediction": sign}


def planned_conditions(config: dict) -> list[dict]:
    conditions = []
    for seed, batch_size in ((s, n) for s in config["rotation_seeds"] for n in config["batch_sizes"]):
        for ao in config["orientations"]:
            for eo in config["orientations"]:
                for da in config["activity_damping"]:
                    for de in config["error_damping"]:
                        conditions.append({"panel": "orientation", "rotation_seed": seed, "batch_size": batch_size,
                                           "activity_orientation": ao, "error_orientation": eo,
                                           "damping_a": da, "damping_e": de,
                                           "anchor": "none", "margin": "none", "side": "none"})
        for da in config["activity_damping"]:
            for de in config["error_damping"]:
                for anchor in config["cone_anchors"]:
                    for margin in config["cone_margins"]:
                        for side in config["cone_sides"]:
                            conditions.append({"panel": "cone", "rotation_seed": seed, "batch_size": batch_size,
                                               "activity_orientation": "signal_low", "error_orientation": "signal_low",
                                               "damping_a": da, "damping_e": de,
                                               "anchor": anchor, "margin": margin, "side": side})
    for index, condition in enumerate(conditions):
        condition["condition_id"] = f"condition_{index:05d}"
    return conditions


def evaluate_condition(config: dict, condition: dict, base: dict[str, np.ndarray] | None = None) -> tuple[list[dict], dict[str, np.ndarray]]:
    if base is None:
        base = make_base(config, condition["rotation_seed"], condition["activity_orientation"], condition["error_orientation"], condition["batch_size"])
    a, d = base["activity"], base["teaching"]
    ca, ce, u = a.T @ a / len(a), d.T @ d / len(a), d.T @ a / len(a)
    da, de = condition["damping_a"], condition["damping_e"]
    if condition["panel"] == "orientation":
        gradient = base["signal_gradient"]
        raw_cosine, anchor_prediction = 1 / np.sqrt(2), None
    else:
        anchor = conditioned_update(u, ca, ce, condition["anchor"], da, de)
        gradient, raw_cosine, anchor_prediction = cone_gradient(u, anchor, condition["margin"], condition["side"])
    realization = realize_fixed_feedback(base, gradient, config["feedback_readout_cosine"])
    output_residual, readout = realization["output_residual"], realization["readout"]
    actual_true = (output_residual @ readout).T @ a / len(a)
    initial_loss = loss_after_step(a, output_residual, readout, u, 0)
    initial_reducible_loss = float(np.sum((output_residual @ readout)**2) / (2 * len(a)))
    irreducible_loss = float(np.sum((output_residual @ realization["complement"])**2) / (2 * len(a)))
    rows = []
    updates = []
    for method in METHODS:
        update = conditioned_update(u, ca, ce, method, da, de)
        updates.append(update)
        cosine = float(np.vdot(actual_true, update) / (np.linalg.norm(actual_true) * np.linalg.norm(update)))
        curvature = float(np.trace(update @ ca @ update.T))
        dot = float(np.vdot(actual_true, update))
        rotation_cosine = float(np.vdot(u, update) / (np.linalg.norm(u) * np.linalg.norm(update)))
        rotation_sine = float(np.linalg.norm(update/np.linalg.norm(update) - rotation_cosine*u/np.linalg.norm(u)))
        row = {**condition, "method": method, "raw_cosine": raw_cosine, "actual_cosine": cosine,
               "difference_from_raw_cosine": cosine - raw_cosine,
               "true_gradient_dot_update": dot, "update_norm": float(np.linalg.norm(update)),
               "curvature": curvature, "rotation_cosine": rotation_cosine,
               "rotation_sine": rotation_sine,
               "initial_loss": initial_loss, "initial_reducible_loss": initial_reducible_loss,
               "irreducible_loss": irreducible_loss, "anchor_predicted_cosine": anchor_prediction,
               "predicted_cosine": None, "predicted_difference": None, "prediction": "not_preassigned",
               "finite_steps": []}
        if condition["panel"] == "orientation":
            row.update(orientation_prediction(base["activity_eigenvalues"], base["error_eigenvalues"], method, da, de))
        elif method == condition["anchor"]:
            row["predicted_cosine"] = anchor_prediction
            row["prediction"] = "descent" if anchor_prediction > 1e-12 else "ascent" if anchor_prediction < -1e-12 else "zero_derivative"
        for step in config["step_sizes"]:
            observed = loss_after_step(a, output_residual, readout, update, step) - initial_loss
            predicted = -step * dot + 0.5 * step**2 * curvature
            row["finite_steps"].append({"step_size": step, "observed_loss_change": observed,
                                        "quadratic_predicted_loss_change": predicted})
        rows.append(row)
    arrays = {**base, **realization, "updates": np.stack(updates)}
    return rows, arrays


def run_trajectory(config: dict, condition: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Short actual DFA trajectory, with current teaching moments at every step."""
    if condition["panel"] != "orientation":
        raise ValueError("only orientation cells have prospective trajectories")
    base = make_base(config, condition["rotation_seed"], condition["activity_orientation"], condition["error_orientation"], condition["batch_size"])
    realized = realize_fixed_feedback(base, base["signal_gradient"], config["feedback_readout_cosine"])
    a, e0, v, b = base["activity"], realized["output_residual"], realized["readout"], realized["feedback"]
    n, d = a.shape
    ca = a.T @ a / n
    traces, weights = [], []
    for method in TRAJECTORY_METHODS:
        w = np.zeros((d, d))
        weight_trace = []
        failed = False
        for step in range(config["trajectory_steps"] + 1):
            weight_trace.append(w.copy())
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    e = e0 + a @ w.T @ v.T
                    true = e @ v
                    teaching = e @ b
                    raw = teaching.T @ a / n
                    gradient = true.T @ a / n
                    ce = teaching.T @ teaching / n
                    # Zero raw update is a fixed point and needs no norm division.
                    update = (gradient.copy() if method == "bp" else np.zeros_like(raw) if np.linalg.norm(raw) == 0 else
                              conditioned_update(raw, ca, ce, method, condition["damping_a"], condition["damping_e"]))
                    values = [e, teaching, raw, gradient, ce, update, w]
                    if not all(np.all(np.isfinite(value)) for value in values):
                        raise FloatingPointError("nonfinite trajectory state")
                    metrics = {"loss": float(np.sum(e**2) / (2*n)),
                               "reducible_loss": float(np.sum(true**2) / (2*n)),
                               "irreducible_loss": float(np.sum((e @ realized["complement"])**2) / (2*n)),
                               "raw_norm": float(np.linalg.norm(raw)), "gradient_norm": float(np.linalg.norm(gradient)),
                               "update_norm": float(np.linalg.norm(update)),
                               "true_gradient_dot_update": float(np.vdot(gradient, update))}
                    eig = np.linalg.eigvalsh(ce)
                    if not all(np.isfinite(value) for value in metrics.values()) or not np.all(np.isfinite(eig)):
                        raise FloatingPointError("nonfinite trajectory statistic")
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                failed = True
                traces.append({**condition, "method": method, "step": step, "status": "numerical_failure",
                               "failure_type": type(exc).__name__, "failure_message": str(exc),
                               "weights_finite": bool(np.all(np.isfinite(w)))})
                break
            traces.append({**condition, "method": method, "step": step, "status": "valid",
                           **metrics, "ce_eigenvalues": eig.tolist()})
            if step < config["trajectory_steps"]:
                with np.errstate(over="ignore", invalid="ignore"):
                    w = w - config["trajectory_lr"] * update
        if failed:
            # Keep planned step coverage explicit without selecting an earlier endpoint.
            while len(weight_trace) <= config["trajectory_steps"]:
                weight_trace.append(np.zeros((d, d)))
        weights.append(np.stack(weight_trace))
    return traces, {"trajectory_weights": np.stack(weights)}


def run(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_hashes = source_hashes()
    manifest = {"study_id": config["study_id"], "status": "running", "config_sha256": sha256(config_path),
                "source_sha256": frozen_hashes, "started_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                                "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}}
    json_write(output_dir / "manifest.running.json", manifest)
    json_write(output_dir / "config.json", config)
    conditions = planned_conditions(config)
    json_write(output_dir / "planned_conditions.json", conditions)
    rows, stacks = [], {}
    base_cache = {}
    for index, condition in enumerate(conditions):
        key = (condition["rotation_seed"], condition["activity_orientation"], condition["error_orientation"], condition["batch_size"])
        if key not in base_cache:
            base_cache[key] = make_base(config, *key)
        new_rows, arrays = evaluate_condition(config, condition, base_cache[key])
        rows.extend(new_rows)
        for name, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"nonfinite {name} in {condition['condition_id']}")
            if name in ("activity", "teaching", "true_teaching", "output_residual"):
                array = np.pad(array, ((0, max(config["batch_sizes"]) - len(array)), (0, 0)))
            stacks.setdefault(name, []).append(array)
        if (index + 1) % 100 == 0:
            print(f"completed {index + 1}/{len(conditions)} conditions", flush=True)
    trajectory_conditions = [c for c in conditions if c["panel"] == "orientation"
                             and c["rotation_seed"] == config["trajectory_seed"]
                             and c["batch_size"] == config["trajectory_batch_size"]]
    trajectory_rows, trajectory_weights = [], []
    for condition in trajectory_conditions:
        history, weights = run_trajectory(config, condition)
        trajectory_rows.extend(history)
        trajectory_weights.append(weights["trajectory_weights"])
    if source_hashes() != frozen_hashes:
        raise RuntimeError("source changed during the diagnostic")
    np.savez_compressed(output_dir / "arrays.npz", **{name: np.stack(values) for name, values in stacks.items()})
    json_write(output_dir / "rows.json", rows)
    json_write(output_dir / "trajectory_rows.json", trajectory_rows)
    json_write(output_dir / "trajectory_conditions.json", trajectory_conditions)
    np.savez_compressed(output_dir / "trajectory_arrays.npz", weights=np.stack(trajectory_weights))
    artifacts = {name: sha256(output_dir / name) for name in ("config.json", "planned_conditions.json", "arrays.npz", "rows.json", "trajectory_rows.json", "trajectory_conditions.json", "trajectory_arrays.npz")}
    manifest.update({"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "condition_count": len(conditions), "row_count": len(rows),
                     "trajectory_count": len(trajectory_conditions) * len(TRAJECTORY_METHODS), "artifact_sha256": artifacts})
    json_write(output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "conditions": len(conditions), "rows": len(rows), "output_dir": str(output_dir)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-config", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.freeze_config is not None:
        if args.config is not None or args.output_dir is not None:
            parser.error("freezing and execution are separate actions")
        config = {**DEFAULT_CONFIG, "source_sha256": source_hashes(), "frozen_utc": datetime.now(timezone.utc).isoformat()}
        args.freeze_config.parent.mkdir(parents=True, exist_ok=True)
        json_write(args.freeze_config, config)
        print(json.dumps({"frozen_config": str(args.freeze_config), "sha256": sha256(args.freeze_config)}))
    elif args.config is not None and args.output_dir is not None:
        run(args.config, args.output_dir)
    else:
        parser.error("use --freeze-config, or --config with --output-dir")


if __name__ == "__main__":
    main()
