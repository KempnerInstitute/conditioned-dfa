"""Complete-only audit of the frozen factor mechanism diagnostic.

All checks use the archived finite batch, actual readout/feedback and full
trajectory weights. No model is selected and no new task is generated here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_ndfa_factor_mechanism import (  # noqa: E402
    METHODS, TRAJECTORY_METHODS, json_write, planned_conditions, sha256, validate_config,
)


class AuditError(ValueError):
    """An incomplete or invalid artifact must not produce scientific summaries."""


class Checks:
    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.count = 0
        self.max_error = 0.0

    def close(self, actual: object, expected: object, label: str, tolerance: float | None = None) -> None:
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AuditError(f"{label}: invalid shape or nonfinite values")
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.max_error = max(self.max_error, error)
        self.count += 1
        tol = self.tolerance if tolerance is None else tolerance
        if error > tol * max(1.0, float(np.max(np.abs(b))) if b.size else 1.0):
            raise AuditError(f"{label}: numerical discrepancy {error:.3g}")

    def require(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            raise AuditError(label)


def independent_update(raw: np.ndarray, ca: np.ndarray, ce: np.ndarray, method: str,
                       damping_a: float, damping_e: float) -> np.ndarray:
    """Direct dense inverse audit, independent of the runner's solve path."""
    update = raw.copy()
    if method in ("activity", "kronecker"):
        update = update @ np.linalg.inv(ca + damping_a * np.eye(ca.shape[0]))
    if method in ("error", "kronecker"):
        update = np.linalg.inv(ce + damping_e * np.eye(ce.shape[0])) @ update
    if np.linalg.norm(raw) == 0:
        return np.zeros_like(raw)
    return update * np.linalg.norm(raw) / np.linalg.norm(update)


def read_complete(directory: Path) -> tuple[dict, dict, list, list]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        validate_config(config)
    except (OSError, ValueError, KeyError) as exc:
        raise AuditError(f"incomplete or invalid manifest/configuration: {exc}") from exc
    if manifest.get("status") != "complete":
        raise AuditError("run is not complete")
    expected_files = {"config.json", "planned_conditions.json", "arrays.npz", "rows.json",
                      "trajectory_rows.json", "trajectory_conditions.json", "trajectory_arrays.npz"}
    if set(manifest.get("artifact_sha256", {})) != expected_files:
        raise AuditError("artifact closure is incomplete")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256(directory / name) != digest:
            raise AuditError(f"artifact hash mismatch: {name}")
    if manifest["source_sha256"] != config["source_sha256"]:
        raise AuditError("manifest source closure differs from configuration")
    if manifest["config_sha256"] != sha256(directory / "config.json"):
        raise AuditError("configuration differs from the original frozen bytes")
    conditions = json.loads((directory / "planned_conditions.json").read_text())
    if conditions != planned_conditions(config):
        raise AuditError("planned condition coverage differs from frozen configuration")
    rows = json.loads((directory / "rows.json").read_text())
    expected_keys = [(c["condition_id"], m) for c in conditions for m in METHODS]
    actual_keys = [(row.get("condition_id"), row.get("method")) for row in rows]
    if actual_keys != expected_keys or manifest["row_count"] != len(expected_keys) or manifest["condition_count"] != len(conditions):
        raise AuditError("missing, duplicate, reordered or unplanned diagnostic rows")
    return manifest, config, conditions, rows


def audit(directory: Path) -> dict:
    manifest, config, conditions, rows = read_complete(directory)
    checks = Checks(config["absolute_tolerance"])
    d = config["dimension"]
    half = d // 2
    comparisons = defaultdict(list)
    cone_signs = Counter()
    maxima = defaultdict(float)
    batches = {}
    arrays = np.load(directory / "arrays.npz", allow_pickle=False)
    required_arrays = {"activity", "teaching", "ca_expected", "ce_expected", "raw_expected", "signal_gradient",
                       "q_left", "q_right", "activity_eigenvalues", "error_eigenvalues", "readout", "feedback",
                       "complement", "output_residual", "true_teaching", "gradient_expected", "updates"}
    checks.require(set(arrays.files) == required_arrays, "array closure differs from declared schema")
    for name in arrays.files:
        checks.require(arrays[name].shape[0] == len(conditions), f"{name}: wrong condition count")
    # Materialize once: repeated compressed-array access would decompress the full study per row.
    data = {name: arrays[name] for name in arrays.files}
    arrays.close()
    for index, condition in enumerate(conditions):
        label = condition["condition_id"]
        n = condition["batch_size"]
        a, teaching = data["activity"][index, :n], data["teaching"][index, :n]
        e0 = data["output_residual"][index, :n]
        v, b, complement = (data[name][index] for name in ("readout", "feedback", "complement"))
        ql, qr = data["q_left"][index], data["q_right"][index]
        ca, ce, raw = a.T @ a / n, teaching.T @ teaching / n, teaching.T @ a / n
        gradient = (e0 @ v).T @ a / n
        for name in ("activity", "teaching", "output_residual", "true_teaching"):
            checks.close(data[name][index, n:], np.zeros_like(data[name][index, n:]), f"{label} padding {name}")
        checks.close(v.T @ v, np.eye(d), f"{label} readout isometry")
        fixed_v = np.concatenate([np.eye(d), np.eye(d)], axis=0)/np.sqrt(2)
        fixed_n = np.concatenate([np.eye(d), -np.eye(d)], axis=0)/np.sqrt(2)
        fixed_b = config["feedback_readout_cosine"]*fixed_v+np.sqrt(1-config["feedback_readout_cosine"]**2)*fixed_n
        checks.close(v, fixed_v, f"{label} common fixed readout")
        checks.close(b, fixed_b, f"{label} common fixed feedback")
        checks.close(complement, fixed_n, f"{label} common fixed complement")
        checks.close(ql.T @ ql, np.eye(d), f"{label} orthogonal hidden basis")
        checks.close(qr.T @ qr, np.eye(d), f"{label} orthogonal input basis")
        checks.close(b.T @ b, np.eye(d), f"{label} feedback norm/rank")
        checks.close(v.T @ b, config["feedback_readout_cosine"] * np.eye(d), f"{label} fixed feedback/readout angle")
        checks.close(v.T @ complement, np.zeros((d, d)), f"{label} complement orthogonality")
        checks.close(complement.T @ complement, np.eye(d), f"{label} complement isometry")
        checks.close(e0 @ b, teaching, f"{label} realized DFA teaching")
        checks.close(e0 @ v, data["true_teaching"][index, :n], f"{label} realized BP teaching")
        checks.close(ca, data["ca_expected"][index], f"{label} sample activity covariance")
        checks.close(ce, data["ce_expected"][index], f"{label} sample error covariance")
        checks.close(raw, data["raw_expected"][index], f"{label} sample cross moment")
        checks.close(raw, ql @ qr.T / np.sqrt(d), f"{label} prescribed raw matrix")
        checks.close(np.linalg.svd(raw, compute_uv=False), np.full(d, 1 / np.sqrt(d)), f"{label} raw rank and spectrum")
        checks.close(np.linalg.norm(raw), 1.0, f"{label} raw norm")
        checks.close(np.linalg.norm(gradient), 1.0, f"{label} true gradient norm")
        checks.close(gradient, data["gradient_expected"][index], f"{label} prescribed true gradient")
        for name, actual, orientation in (("activity", ca, condition["activity_orientation"]), ("error", ce, condition["error_orientation"])):
            lo, hi = config["spectrum_low"], config["spectrum_high"]
            eig = np.full(d, (lo + hi)/2) if orientation == "isotropic" else np.array([lo]*half + [hi]*half)
            checks.close(np.linalg.eigvalsh(actual), eig, f"{label} fixed {name} spectrum")
            prescribed_diagonal = (np.full(d, (lo+hi)/2) if orientation == "isotropic" else
                                   np.array(([lo]*half + [hi]*half) if orientation == "signal_low" else ([hi]*half + [lo]*half)))
            checks.close(data[f"{name}_eigenvalues"][index], prescribed_diagonal, f"{label} assigned {name} orientation")
            q = qr if name == "activity" else ql
            checks.close(q.T @ actual @ q, np.diag(prescribed_diagonal), f"{label} measured {name} orientation")
        key = (condition["rotation_seed"], n, condition["activity_orientation"], condition["error_orientation"])
        if key in batches:
            prior_a, prior_d, prior_raw = batches[key]
            checks.close(a, prior_a, f"{label} unchanged activity under credit/metric interventions", tolerance=0)
            checks.close(teaching, prior_d, f"{label} unchanged teaching under credit/metric interventions", tolerance=0)
            checks.close(raw, prior_raw, f"{label} unchanged raw update under credit/metric interventions", tolerance=0)
        else:
            batches[key] = (a, teaching, raw)
        if condition["panel"] == "orientation":
            prescribed_g = (ql * (np.array([1.0]*half+[0.0]*half)/np.sqrt(half))[None, :]) @ qr.T
            checks.close(gradient, prescribed_g, f"{label} task orientation")
            checks.close(np.vdot(gradient, raw), 1/np.sqrt(2), f"{label} credit quality held fixed")
        else:
            anchor = independent_update(raw, ca, ce, condition["anchor"], condition["damping_a"], condition["damping_e"])
            cosine = float(np.vdot(raw, anchor))
            remainder = anchor - cosine*raw
            sine = float(np.linalg.norm(remainder))
            c = sine/2 if condition["margin"] == "below" else sine if condition["margin"] == "boundary" else (1+sine)/2
            sign = -1 if condition["side"] == "adverse" else 1
            prescribed_g = c*raw + sign*np.sqrt(1-c*c)*remainder/sine
            checks.close(gradient, prescribed_g, f"{label} prescribed cone gradient")
            checks.close(np.vdot(gradient, raw), c, f"{label} measured cone margin")
        initial_loss = float(np.sum(e0**2)/(2*n))
        reducible = float(np.sum((e0 @ v)**2)/(2*n))
        irreducible = float(np.sum((e0 @ complement)**2)/(2*n))
        checks.close(initial_loss, reducible+irreducible, f"{label} orthogonal loss decomposition")
        for method_index, method in enumerate(METHODS):
            row = rows[len(METHODS)*index + method_index]
            for name, value in condition.items():
                checks.require(row.get(name) == value, f"{label} metadata mismatch {name}")
            update = data["updates"][index, method_index]
            expected_update = independent_update(raw, ca, ce, method, condition["damping_a"], condition["damping_e"])
            checks.close(update, expected_update, f"{label}/{method} actual A/E/K rule")
            checks.close(np.linalg.norm(update), np.linalg.norm(raw), f"{label}/{method} norm matching")
            dot = float(np.vdot(gradient, update))
            curvature = float(np.sum((a @ update.T @ v.T)**2)/n)
            checks.close(curvature, np.trace(update @ ca @ update.T), f"{label}/{method} exact quadratic Hessian")
            expected_values = {"initial_loss": initial_loss, "initial_reducible_loss": reducible,
                               "irreducible_loss": irreducible, "actual_cosine": dot,
                               "raw_cosine": float(np.vdot(gradient, raw)), "curvature": curvature,
                               "true_gradient_dot_update": dot, "update_norm": float(np.linalg.norm(update)),
                               "difference_from_raw_cosine": dot - float(np.vdot(gradient, raw)),
                               "rotation_cosine": float(np.vdot(raw, update)/(np.linalg.norm(raw)*np.linalg.norm(update))),
                               "rotation_sine": float(np.linalg.norm(update/np.linalg.norm(update)-float(np.vdot(raw,update)/(np.linalg.norm(raw)*np.linalg.norm(update)))*raw/np.linalg.norm(raw)))}
            for name, value in expected_values.items():
                checks.close(row[name], value, f"{label}/{method} {name}")
            if condition["panel"] == "orientation":
                diagonal = np.ones(d)
                if method in ("activity", "kronecker"):
                    diagonal /= data["activity_eigenvalues"][index] + condition["damping_a"]
                if method in ("error", "kronecker"):
                    diagonal /= data["error_eigenvalues"][index] + condition["damping_e"]
                predicted = float(diagonal[:half].sum()/np.sqrt(half*np.sum(diagonal**2)))
                checks.close(dot, predicted, f"{label}/{method} prospective closed form")
                checks.close(row["predicted_cosine"], predicted, f"{label}/{method} stored prediction")
                delta = predicted - 1/np.sqrt(2)
                prediction = "beneficial" if delta > 1e-12 else "harmful" if delta < -1e-12 else "neutral"
                checks.require(row["prediction"] == prediction, f"{label}/{method} prospective sign")
                comparisons[(condition["activity_orientation"], condition["error_orientation"], condition["damping_a"], condition["damping_e"], method)].append(dot)
            elif method == condition["anchor"]:
                sign = -1 if condition["side"] == "adverse" else 1
                prediction = c*cosine + sign*np.sqrt(1-c*c)*sine
                checks.close(dot, prediction, f"{label}/{method} exact cone bound attained")
                checks.close(row["predicted_cosine"], prediction, f"{label}/{method} stored cone prediction")
                cone_signs[(condition["anchor"], condition["margin"], condition["side"], "descent" if dot > 1e-10 else "ascent" if dot < -1e-10 else "zero_derivative")] += 1
            checks.require(len(row["finite_steps"]) == len(config["step_sizes"]), f"{label}/{method} finite-step coverage")
            for finite, step in zip(row["finite_steps"], config["step_sizes"]):
                checks.require(finite["step_size"] == step, f"{label}/{method} finite-step identity")
                observed = float(np.sum((e0-step*a @ update.T @ v.T)**2)/(2*n)-initial_loss)
                predicted_change = -step*dot + step**2*curvature/2
                checks.close(finite["observed_loss_change"], observed, f"{label}/{method} observed finite loss")
                checks.close(finite["quadratic_predicted_loss_change"], predicted_change, f"{label}/{method} quadratic formula")
                checks.close(observed, predicted_change, f"{label}/{method} direct finite-step verification")
                maxima["finite_step_formula_error"] = max(maxima["finite_step_formula_error"], abs(observed-predicted_change))
    trajectory_summary = audit_trajectories(directory, config, conditions, data, checks)
    orientation_summary = []
    for key, values in comparisons.items():
        checks.close(values, np.full(len(values), values[0]), f"basis/N equivariance {key}")
        orientation_summary.append(dict(zip(("activity_orientation", "error_orientation", "damping_a", "damping_e", "method"), key))
                                   | {"cosine": float(np.mean(values)), "basis_and_batch_checks": len(values),
                                      "max_basis_and_batch_spread": float(np.ptp(values))})
    return {"status": "complete", "study_id": config["study_id"], "run_manifest_sha256": sha256(directory/"manifest.json"),
            "source_sha256": manifest["source_sha256"], "artifact_sha256": manifest["artifact_sha256"],
            "condition_count": len(conditions), "row_count": len(rows), "numerical_check_count": checks.count,
            "max_absolute_checked_error": checks.max_error, **maxima,
            "orientation_summary": orientation_summary,
            "cone_summary": [{"anchor": k[0], "margin": k[1], "side": k[2], "outcome": k[3], "count": v} for k,v in sorted(cone_signs.items())],
            "trajectory_summary": trajectory_summary,
            "scope": "Engineered diagnostic; basis/batch copies are equivariance checks, not independent statistical replicates. No model selection or generalization claim."}


def audit_trajectories(directory: Path, config: dict, conditions: list, data: dict, checks: Checks) -> list:
    tc = json.loads((directory/"trajectory_conditions.json").read_text())
    expected = [c for c in conditions if c["panel"] == "orientation" and c["rotation_seed"] == config["trajectory_seed"] and c["batch_size"] == config["trajectory_batch_size"]]
    checks.require(tc == expected, "trajectory condition coverage differs from protocol")
    histories = json.loads((directory/"trajectory_rows.json").read_text())
    groups = defaultdict(list)
    for row in histories:
        groups[(row["condition_id"], row["method"])].append(row)
    checks.require(set(groups) == {(c["condition_id"], m) for c in tc for m in TRAJECTORY_METHODS}, "trajectory method coverage")
    with np.load(directory/"trajectory_arrays.npz", allow_pickle=False) as archive:
        weights = archive["weights"]
    checks.require(weights.shape == (len(tc), len(TRAJECTORY_METHODS), config["trajectory_steps"]+1, config["dimension"], config["dimension"]), "trajectory checkpoint coverage")
    condition_index = {c["condition_id"]: i for i,c in enumerate(conditions)}
    summaries = []
    for ci, condition in enumerate(tc):
        idx = condition_index[condition["condition_id"]]
        n, d = condition["batch_size"], config["dimension"]
        a = data["activity"][idx,:n]
        e0 = data["output_residual"][idx,:n]
        v, b, complement = (data[name][idx] for name in ("readout", "feedback", "complement"))
        ca = a.T @ a/n
        g0 = (e0 @ v).T @ a/n
        raw0 = (e0 @ b).T @ a/n
        stationary = -raw0 @ np.linalg.inv(ca)/config["feedback_readout_cosine"]
        bp_optimum = -g0 @ np.linalg.inv(ca)
        checks.close((e0+a @ stationary.T @ v.T) @ b, (e0 @ b)-a @ np.linalg.inv(ca) @ raw0.T, "DFA stationary residual decomposition")
        checks.close(((e0+a @ stationary.T @ v.T) @ b).T @ a/n, np.zeros((d,d)), "DFA stationary point")
        checks.close(((e0+a @ bp_optimum.T @ v.T) @ v).T @ a/n, np.zeros((d,d)), "BP optimum")
        stationary_loss = float(np.sum(((e0+a @ stationary.T @ v.T) @ v)**2)/(2*n))
        initial_irreducible = float(np.sum((e0 @ complement)**2)/(2*n))
        for mi, method in enumerate(TRAJECTORY_METHODS):
            history = groups[(condition["condition_id"], method)]
            valid = [row for row in history if row["status"] == "valid"]
            failures = [row for row in history if row["status"] == "numerical_failure"]
            checks.require(len(history) == len(valid)+len(failures), "unrecognized trajectory status")
            checks.require([r["step"] for r in valid] == list(range(len(valid))), "trajectory valid prefix is not contiguous")
            if failures:
                checks.require(len(failures) == 1 and failures[0]["step"] == len(valid), "invalid failure endpoint")
            else:
                checks.require(len(valid) == config["trajectory_steps"]+1, "incomplete trajectory horizon")
            for step, row in enumerate(valid):
                w = weights[ci,mi,step]
                checks.close(w, np.zeros_like(w), "shared initial weights") if step == 0 else None
                residual = e0+a @ w.T @ v.T
                teaching = residual @ b
                raw = teaching.T @ a/n
                gradient = (residual @ v).T @ a/n
                ce = teaching.T @ teaching/n
                update = gradient if method == "bp" else independent_update(raw, ca, ce, method, condition["damping_a"], condition["damping_e"])
                checks.close(raw, raw0+config["feedback_readout_cosine"]*w @ ca, "actual fixed-feedback dynamics")
                checks.close(gradient, g0+w @ ca, "actual true-gradient dynamics")
                expected_values = {"loss": np.sum(residual**2)/(2*n), "reducible_loss": np.sum((residual @ v)**2)/(2*n),
                                   "irreducible_loss": np.sum((residual @ complement)**2)/(2*n),
                                   "raw_norm": np.linalg.norm(raw), "gradient_norm": np.linalg.norm(gradient),
                                   "update_norm": np.linalg.norm(update), "true_gradient_dot_update": np.vdot(gradient,update)}
                for name,value in expected_values.items():
                    checks.close(row[name], value, f"trajectory {condition['condition_id']}/{method}/{step} {name}")
                checks.close(row["ce_eigenvalues"], np.linalg.eigvalsh(ce), "recomputed error moment each step")
                checks.close(row["irreducible_loss"], initial_irreducible, "constant complement loss")
                if step < len(valid)-1:
                    checks.close(weights[ci,mi,step+1], w-config["trajectory_lr"]*update, "actual trajectory update recurrence")
                elif failures:
                    with np.errstate(over="ignore", invalid="ignore"):
                        expected_failed_w = w-config["trajectory_lr"]*update
                    checks.require(np.array_equal(weights[ci,mi,step+1], expected_failed_w, equal_nan=True)
                                   or np.allclose(weights[ci,mi,step+1], expected_failed_w, rtol=2e-10, atol=2e-10),
                                   "failed trajectory checkpoint does not follow the valid prefix")
            if failures:
                failed_w = weights[ci,mi,len(valid)]
                checks.require(failures[0]["weights_finite"] == bool(np.all(np.isfinite(failed_w))), "incorrect failure weight-finiteness record")
                numerically_invalid = not np.all(np.isfinite(failed_w))
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        er = e0+a @ failed_w.T @ v.T
                        dt = er @ b
                        ur = dt.T @ a/n
                        gg = (er @ v).T @ a/n
                        cc = dt.T @ dt/n
                        uu = gg if method == "bp" else independent_update(ur, ca, cc, method, condition["damping_a"], condition["damping_e"])
                        numeric = [np.sum(er**2), np.linalg.norm(ur), np.linalg.norm(gg), np.linalg.norm(uu), np.vdot(gg,uu)]
                        numerically_invalid |= not all(np.isfinite(value) for value in numeric)
                        numerically_invalid |= not np.all(np.isfinite(np.linalg.eigvalsh(cc)))
                except (FloatingPointError, np.linalg.LinAlgError):
                    numerically_invalid = True
                checks.require(numerically_invalid, "claimed numerical failure has a finite independently evaluable state")
                checks.close(weights[ci,mi,len(valid)+1:], np.zeros_like(weights[ci,mi,len(valid)+1:]), "failed suffix padding must not masquerade as checkpoints")
            summaries.append({**condition, "method": method, "status": "numerical_failure" if failures else "valid",
                              "completed_updates": len(valid)-1, "initial_reducible_loss": valid[0]["reducible_loss"] if valid else None,
                              "final_reducible_loss": valid[-1]["reducible_loss"] if valid and not failures else None,
                              "irreducible_loss": initial_irreducible, "dfa_stationary_reducible_loss": stationary_loss,
                              "bp_optimum_reducible_loss": 0.0,
                              "stationary_weight_gap_norm": float(np.linalg.norm(stationary-bp_optimum))})
    return summaries


def render_report(result: dict) -> str:
    lines = ["# Controlled activity/error-factor diagnostic", "",
             f"Complete: {result['row_count']:,} one-step rows, {len(result['trajectory_summary'])} full prescribed trajectories; "
             f"{result['numerical_check_count']:,} numerical/provenance checks. No setting was selected.", "",
             result["scope"], "",
             "The task and teaching residuals were constructed from desired moments and desired true gradients. This identifies a causal operator effect inside the engineered system; it does not discover useful credit or establish a new learning guarantee.", "",
             "At equal damping 1/1, all updates have the same Frobenius norm. Entries are true-gradient cosines; the raw cosine is 1/sqrt(2).", "",
             "| Activity orientation | Error orientation | Raw | A | E | K |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    lookup = {(r["activity_orientation"],r["error_orientation"],r["method"]):r for r in result["orientation_summary"] if r["damping_a"] == 1 and r["damping_e"] == 1}
    for ao in ("isotropic", "signal_low", "signal_high"):
        for eo in ("isotropic", "signal_low", "signal_high"):
            lines.append(f"| {ao} | {eo} | " + " | ".join(f"{lookup[(ao,eo,m)]['cosine']:.6f}" for m in METHODS) + " |")
    lines += ["", "Signal-low means the task gradient occupies the low-variance half of an unchanged anisotropic spectrum; signal-high swaps the task/variance relationship. Isotropic controls preserve trace, but necessarily change the spectrum. Swapped anisotropic cells preserve the entire spectrum. The error moment changes through teaching variation orthogonal to activity, leaving the raw update unchanged.", "",
              "The adverse cone construction reaches the exact rotation bound: below the margin threshold the conditioned update ascends although the raw update descends; at the threshold its directional derivative is zero; above it the anchor descends. A positive finite step at the zero-derivative boundary increases this quadratic loss through curvature. Favorable cone teachers demonstrate that a failed worst-case certificate is not a necessary condition for a particular update to descend.", "",
              "## Short actual fixed-feedback trajectories", "",
              "All 81 orientation/damping cells receive 100 full-batch updates at learning rate .03, using seed 9151500 and N=64. A/E/K recompute the current error second moment and match their current raw-DFA norm at every step. BP uses its actual gradient at the same learning rate. The comparison is a diagnostic at one prescribed step size, not a tuned optimizer benchmark.", "",
              "The following table shows damping 1/1 only for readability; the JSON retains all settings and all endpoints. Lower reducible loss is better. Irreducible readout-complement loss is audited separately and remains constant. No best intermediate checkpoint is selected.", "",
              "| Activity orientation | Error orientation | Initial | Raw final | A final | E final | K final | BP final | DFA stationary |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tl = {(r["activity_orientation"],r["error_orientation"],r["method"]):r for r in result["trajectory_summary"] if r["damping_a"] == 1 and r["damping_e"] == 1}
    for ao in ("isotropic", "signal_low", "signal_high"):
        for eo in ("isotropic", "signal_low", "signal_high"):
            ref = tl[(ao,eo,"raw")]
            values = [ref["initial_reducible_loss"]] + [tl[(ao,eo,m)]["final_reducible_loss"] for m in TRAJECTORY_METHODS] + [ref["dfa_stationary_reducible_loss"]]
            lines.append(f"| {ao} | {eo} | " + " | ".join("failure" if v is None else f"{v:.6f}" for v in values) + " |")
    failures = sum(r["status"] != "valid" for r in result["trajectory_summary"])
    lines += ["", f"Numerical failures: {failures}/{len(result['trajectory_summary'])}. Every planned outcome is retained.", "",
              "The fixed-feedback stationary weight is −U(0) CA⁻¹/0.5; the BP optimum is −g(0) CA⁻¹. Invertible conditioning preserves zero raw updates, so these metrics cannot remove the mismatch. Initial directional predictions do not automatically predict later loss ordering as residuals, credit alignment and CE change. The data support the listed operator identities and observed trajectories, not a convergence or generalization theorem.", "",
              f"Maximum absolute finite-step formula discrepancy: {result['finite_step_formula_error']:.3g}. Run manifest SHA-256: `{result['run_manifest_sha256']}`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    json_path, md_path = args.output_stem.with_suffix(".json"), args.output_stem.with_suffix(".md")
    if json_path.exists() or md_path.exists():
        parser.error("refusing to overwrite an existing report")
    result = audit(args.run_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_write(json_path, result)
    with md_path.open("x") as handle:
        handle.write(render_report(result))
    print(json.dumps({"status": result["status"], "rows": result["row_count"], "checks": result["numerical_check_count"]}))


if __name__ == "__main__":
    main()
