"""Plot the complete frozen factor diagnostic without selecting outcomes.

The displayed damping 1/1 is the protocol's predeclared illustrative slice.
The main trajectories are the three diagonal orientation pairs; the supplement
shows all nine pairs. Every one of the 405 prescribed endpoints is exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ORIENTATIONS = ("isotropic", "signal_low", "signal_high")
TICKS = ("isotropic", "signal low", "signal high")
METHODS = ("raw", "activity", "error", "kronecker", "bp")
LABELS = {"raw": "Raw DFA", "activity": "Activity (A)", "error": "Error (E)", "kronecker": "Both (K)", "bp": "BP"}
COLORS = {"raw": "#777777", "activity": "#009E73", "error": "#D55E00", "kronecker": "#7850A0", "bp": "#0072B2"}
STYLES = {"raw": (0, (6, 2)), "activity": (0, (3, 2)), "error": ":", "kronecker": "-", "bp": "-"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(audit_path: Path, run_dir: Path) -> tuple[dict, dict, dict, dict]:
    audit_hash = sha256(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete" or audit.get("row_count") != 23328:
        raise ValueError("The complete frozen one-step audit is required")
    manifest_path = run_dir / "manifest.json"
    actual_hash = sha256(manifest_path)
    if actual_hash != audit["run_manifest_sha256"]:
        # Anonymous exports relocate execution paths. Bind that transformation
        # through both original and export hashes without rewriting the audit.
        package_root = audit_path.resolve().parents[2]
        package_manifest = package_root / "PACKAGE_MANIFEST.json"
        if not package_manifest.is_file():
            raise ValueError("Run manifest changed after the complete audit")
        relative = str(manifest_path.resolve().relative_to(package_root))
        record = json.loads(package_manifest.read_text())["files"].get(relative, {})
        if (record.get("original_sha256") != audit["run_manifest_sha256"]
                or record.get("export_sha256") != actual_hash):
            raise ValueError("Anonymous manifest transformation lacks matching provenance")
    rows_path = run_dir / "trajectory_rows.json"
    if sha256(rows_path) != audit["artifact_sha256"][rows_path.name]:
        raise ValueError("Trajectory rows changed after the complete audit")
    endpoints = audit["trajectory_summary"]
    if len(endpoints) != 405 or any(row["status"] != "valid" or row["completed_updates"] != 100 for row in endpoints):
        raise ValueError("All 405 prescribed full trajectories must be valid")
    endpoint_keys = {(r["condition_id"], r["method"]) for r in endpoints}
    if len(endpoint_keys) != 405:
        raise ValueError("Duplicated trajectory endpoints")
    rows = json.loads(rows_path.read_text())
    if len(rows) != 405 * 101:
        raise ValueError("Incomplete full trajectory records")
    groups = {}
    for row in rows:
        key = (row["condition_id"], row["method"])
        groups.setdefault(key, []).append(row)
    if set(groups) != endpoint_keys:
        raise ValueError("Curve and endpoint cohorts differ")
    for endpoint in endpoints:
        curve = sorted(groups[(endpoint["condition_id"], endpoint["method"])], key=lambda row: row["step"])
        if [row["step"] for row in curve] != list(range(101)):
            raise ValueError("Missing or duplicated trajectory update")
        losses = np.array([row["reducible_loss"] for row in curve])
        if not np.isfinite(losses).all() or min(losses) < -1e-12:
            raise ValueError("Invalid loss trajectory")
        if not np.isclose(losses[-1], endpoint["final_reducible_loss"], rtol=1e-12, atol=1e-12):
            raise ValueError("Audited endpoint disagrees with its full trajectory")
    curves = {}
    selected_endpoints = {}
    for endpoint in endpoints:
        if endpoint["damping_a"] == 1 and endpoint["damping_e"] == 1:
            key = (endpoint["activity_orientation"], endpoint["error_orientation"], endpoint["method"])
            if key in curves:
                raise ValueError("Duplicated predeclared display condition")
            curves[key] = sorted(groups[(endpoint["condition_id"], endpoint["method"])], key=lambda row: row["step"])
            selected_endpoints[key] = endpoint
    expected = {(a, e, method) for a in ORIENTATIONS for e in ORIENTATIONS for method in METHODS}
    if set(curves) != expected:
        raise ValueError(f"Missing displayed curves or unrecognized methods: {set(curves) ^ expected}")
    cosine = {}
    for row in audit["orientation_summary"]:
        if row["damping_a"] == 1 and row["damping_e"] == 1:
            key = (row["activity_orientation"], row["error_orientation"], row["method"])
            if key in cosine:
                raise ValueError("Duplicated one-step display cell")
            cosine[key] = row["cosine"]
    if len(cosine) != 36:
        raise ValueError("Incomplete initial-cosine display grid")
    if sha256(audit_path) != audit_hash:
        raise ValueError("Audit changed during figure loading")
    return audit, curves, selected_endpoints, cosine


def trajectory(ax, curves, endpoints, activity: str, error: str, *, title: str, ylabel: bool = True):
    for method in METHODS:
        key = (activity, error, method)
        curve = curves[key]
        ax.plot([row["step"] for row in curve], [row["reducible_loss"] for row in curve], color=COLORS[method], linestyle=STYLES[method], lw=1.5 if method != "raw" else 2.2, label=LABELS[method], zorder=3)
    stationary = endpoints[(activity, error, "raw")]["dfa_stationary_reducible_loss"]
    ax.axhline(stationary, color="#222222", lw=.9, ls=(0, (1, 2)), label="DFA stationary loss", zorder=2)
    ax.set_title(title, loc="left", fontsize=8.4)
    ax.set_xlabel("training update")
    if ylabel:
        ax.set_ylabel("reducible loss")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.15, lw=.5)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def write_figure(fig, stem: Path) -> dict:
    outputs = {}
    for suffix in ("pdf", "png", "svg"):
        path = Path(str(stem) + "." + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Creator": "Matplotlib"} if suffix == "pdf" else None)
        outputs[path.name] = sha256(path)
    plt.close(fig)
    return outputs


def main(args):
    output = args.output_stem
    suffixes = [".pdf", ".png", ".svg", "_all_orientations.pdf", "_all_orientations.png", "_all_orientations.svg", ".manifest.json", ".endpoints.csv", ".caption.md"]
    if any(Path(str(output) + suffix).exists() for suffix in suffixes):
        raise FileExistsError("Refusing to overwrite existing mechanism figure artifacts")
    audit, curves, endpoints, cosine = load(args.audit_json, args.run_dir)
    source_hashes = {"audit_json": sha256(args.audit_json), "run_manifest": sha256(args.run_dir / "manifest.json"), "trajectory_rows": sha256(args.run_dir / "trajectory_rows.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.1, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.2, 4.7), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=.04)
    top = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, .055], wspace=.07)
    norm = TwoSlopeNorm(vmin=0, vcenter=1 / np.sqrt(2), vmax=1)
    for index, method in enumerate(("activity", "error", "kronecker")):
        ax = fig.add_subplot(top[index])
        values = np.array([[cosine[(a, e, method)] for e in ORIENTATIONS] for a in ORIENTATIONS])
        display = ax.imshow(values, cmap="RdBu", norm=norm)
        for i in range(3):
            for j in range(3):
                color = "white" if values[i, j] < .30 or values[i, j] > .95 else "#202020"
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8.4, fontweight="bold")
        ax.set_xticks(range(3), TICKS, rotation=30, ha="right")
        ax.set_yticks(range(3), TICKS if index == 0 else [])
        ax.set_xlabel("error orientation")
        if index == 0:
            ax.set_ylabel("activity orientation")
        ax.set_title(f"{'ABC'[index]}  {LABELS[method]}", loc="left", fontweight="bold")
    colorbar = fig.colorbar(display, cax=fig.add_subplot(top[3]), ticks=[0, 1 / np.sqrt(2), 1])
    colorbar.ax.set_yticklabels(["0", "raw: 0.707", "1"])
    colorbar.set_label("initial true-gradient cosine")
    bottom = outer[1].subgridspec(1, 3, wspace=.1)
    titles = ("D  Both isotropic", "E  Both signal low", "F  Both signal high")
    axes = []
    for index, orientation in enumerate(ORIENTATIONS):
        ax = fig.add_subplot(bottom[index])
        trajectory(ax, curves, endpoints, orientation, orientation, title=titles[index])
        axes.append(ax)
    axes[0].text(.97, .43, "Four local curves coincide;\nfinal loss exceeds initial", transform=axes[0].transAxes, ha="right", va="top", fontsize=7, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 2})
    low_raw = endpoints[("signal_low", "signal_low", "raw")]["final_reducible_loss"]
    low_k = endpoints[("signal_low", "signal_low", "kronecker")]["final_reducible_loss"]
    low_cosine = cosine[("signal_low", "signal_low", "kronecker")]
    axes[1].text(.97, .91, f"K starts at cosine {low_cosine:.4f},\nends at {low_k:.3f} vs raw {low_raw:.3f}", transform=axes[1].transAxes, ha="right", va="top", fontsize=7, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 2})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, handlelength=2.4, columnspacing=.9)
    generated = write_figure(fig, output)
    supplement, grid = plt.subplots(3, 3, figsize=(7.2, 6.1), layout="constrained")
    for i, activity in enumerate(ORIENTATIONS):
        for j, error in enumerate(ORIENTATIONS):
            title = f"A: {TICKS[i]}; E: {TICKS[j]}"
            trajectory(grid[i, j], curves, endpoints, activity, error, title=title)
    supplement.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, handlelength=2.4, columnspacing=.9)
    generated.update(write_figure(supplement, Path(str(output) + "_all_orientations")))
    endpoint_path = Path(str(output) + ".endpoints.csv")
    with endpoint_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(audit["trajectory_summary"][0]))
        writer.writeheader()
        writer.writerows(audit["trajectory_summary"])
    generated[endpoint_path.name] = sha256(endpoint_path)
    caption = """Controlled factor geometry separates initial alignment from later learning.

A–C: true-gradient cosines after activity (A), error (E), or two-sided (K)
conditioning at the protocol's predeclared display damping λA = λE = 1.
Every update is matched to the same raw-DFA norm; raw cosine is 1/√2 in all
nine cells. Heatmap entries are rounded to three decimals; low/low K is
.999904, not exactly aligned with BP. Signal low/high swaps whether the true-gradient direction occupies
the low/high-variance half of an unchanged anisotropic spectrum. The isotropic
control preserves trace and changes the spectrum. The teaching residual and
desired true gradient are engineered; these are operator checks, not learned
credit recovery or independent statistical replicates.

D–F: the three diagonal orientation pairs, with all five methods shown for the
full prescribed 100 updates at learning rate .03. The dotted line is the
fixed-feedback DFA stationary reducible loss. BP uses its actual gradient;
A/E/K recompute current factors and match the current raw-DFA norm. In the
isotropic case all four local rules initially descend but finish above their
initial loss. In the low/low case K begins at cosine .999904 yet finishes above
raw DFA, near the same unfavorable stationary point. Initial alignment gains
therefore do not establish a better finite-horizon endpoint. Reducible loss
excludes a separately audited constant readout-complement loss. No setting or
intermediate checkpoint was selected. The supplemental figure shows all nine
orientation pairs at the same damping; the companion CSV retains all 405
orientation/damping/method endpoints. These are untuned engineered trajectories,
not a generalization or convergence guarantee.
"""
    caption_path = Path(str(output) + ".caption.md")
    caption_path.write_text(caption)
    generated[caption_path.name] = sha256(caption_path)
    if source_hashes != {"audit_json": sha256(args.audit_json), "run_manifest": sha256(args.run_dir / "manifest.json"), "trajectory_rows": sha256(args.run_dir / "trajectory_rows.json")}:
        raise ValueError("Figure inputs changed during generation")
    manifest = {"status": "complete", "source_sha256": source_hashes, "generator_sha256": sha256(Path(__file__)), "display_damping_a": 1, "display_damping_e": 1, "main_trajectory_pairs": [[value, value] for value in ORIENTATIONS], "supplement_all_nine_orientations": True, "all_endpoints_exported": len(audit["trajectory_summary"]), "model_evaluations": 0, "scope": audit["scope"], "outputs_sha256": generated}
    Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    main(parser.parse_args())
