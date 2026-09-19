"""Build a private, inspectable review package without changing source history.

The package deliberately retains the repository's legally required LICENSE.
Its identifying copyright notice is reported as an unresolved release blocker.
No archive is uploaded, and no redaction is represented as relicensing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


TEXT_SUFFIXES = {".py", ".json", ".csv", ".md", ".txt", ".tex", ".sty", ".bst", ".bib"}
SCANNABLE_TEXT_SUFFIXES = TEXT_SUFFIXES | {".svg"}
DENIED = ("hsafaai", "houman_safaai", "houman1359", "varun04reddy", "KempnerInstitute/conditioned-dfa", "Houman Safaai", "Varun Reddy", "Bernardo L. Sabatini")
PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9:/])/(?:n|home|Users|scratch)/[^\s\"'<>`)\]}]+")
SITE_TOKENS = ("Kempner", "Harvard")
COHORTS = (
    "dfa_stall_threefactor_dev_v1", "dfa_stall_threefactor_confirmation_v1", "dfa_stall_threefactor_analysis_v1",
    "dfa_stall_fashion_threefactor_dev_v1", "dfa_stall_fashion_threefactor_confirmation_v1", "dfa_stall_fashion_threefactor_analysis_v1",
    "dfa_relu_mnist_dev_v1", "dfa_relu_mnist_threefactor_confirmation_v1", "dfa_relu_mnist_threefactor_analysis_v1",
    "dfa_relu_fashion_dev_v1", "error_kndfa_replication_figure_v1",
    "dfa_stall_fashion_bpsource_retune_dev_v1", "dfa_stall_fashion_bpsource_retune_confirmation_v1", "dfa_stall_fashion_bpsource_retune_analysis_v1",
    "dfa_stall_fashion_bpsource_scale_audit_v1", "dfa_stall_fashion_bpsource_scale_audit_d3_v1",
    "benchmark_overhead_v1", "imagenet100_strongform_v1", "imagenet100_strongform_lrcheck_v1", "imagenet100_noisy_deconfound_v1",
    "infodfa_actwhiten_synthetic_v1", "infodfa_adam_diagk_aggregate_v1", "infodfa_adam_diagk_approx_v1",
    "infodfa_alignment_dynamics_v1", "infodfa_amortized_refresh_v1", "infodfa_bn_baseline_v1", "infodfa_bn_headtohead_v1",
    "infodfa_bn_ndfa_synthetic_v1", "infodfa_bn_ndfa_vision_v1", "infodfa_bp_tuning_synthetic_v1", "infodfa_bpwhiten_synthetic_v1",
    "infodfa_capable_cifar10_v1", "infodfa_capable_normmatch_v2", "infodfa_damping_theory_v1",
    "infodfa_factor_ablation_aggregate_v1", "infodfa_factor_ablation_synthetic_v1", "infodfa_factor_ablation_vision_v1",
    "infodfa_feedback_variance_v1", "infodfa_honest_selection_reanalysis", "infodfa_kfac_control_v1", "infodfa_localrule_baselines_v1",
    "infodfa_mode_timing_v1", "infodfa_normmatch_factor_controls_aggregate_v1", "infodfa_normmatch_factor_controls_v1",
    "infodfa_paper_tables_20260527", "infodfa_prospective_diagnostic_v1", "infodfa_retrospective_diagnostic_v1",
    "infodfa_seedlevel_stats_v1", "infodfa_spatialkron_controls", "infodfa_spatialkron_nuisance", "infodfa_spatialkron_nuisance_cifar100",
    "infodfa_vision_noise_sweep_val_agg_v1", "infodfa_vision_noise_sweep_val_v1", "infodfa_wallclock_curves_v1",
    "ndfa_cifar10_confirmation_20260914", "ndfa_confirmation_snapshot_20260914", "ndfa_covariance_power_audit_20260914",
    "ndfa_exact_cuda_20260914", "ndfa_exact_cuda_precision_20260914", "ndfa_exact_scaling_pilot_20260914",
    "ndfa_task_noise_development_20260914", "ndfa_task_noise_boundary_development_20260914",
    "ndfa_task_noise_confirmation_20260914", "ndfa_task_noise_snapshot_v2_20260914", "ndfa_task_noise_confirmation_snapshot_20260914",
    "ndfa_mixer_task_development_20260914", "ndfa_mixer_task_development_snapshot_v2_20260914",
    "ndfa_mixer_noise_confirmation_20260914", "ndfa_mixer_noise_confirmation_snapshot_20260914",
    "ndfa_teaching_regression_development_20260914", "ndfa_teaching_regression_development_snapshot_20260914",
)
LEGACY = ("infodfa_multioutput_noise_sweep_aggregate_v2", "infodfa_vision_noise_sweep_aggregate_v2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anonymous_tex(text: str) -> str:
    """Resolve only standalone ARXIV conditionals to their review branch.

    Unexpected nesting is rejected rather than guessed. Ordinary bibliography
    entries and third-party style comments are not anonymized.
    """
    result = []
    branch = None
    for line in text.splitlines(keepends=True):
        token = line.strip()
        if token == r"\ifdefined\ARXIV":
            if branch is not None:
                raise ValueError("Nested ARXIV conditional requires explicit review")
            branch = False
        elif token == r"\else" and branch is not None:
            if branch:
                raise ValueError("Repeated else in ARXIV conditional")
            branch = True
        elif token == r"\fi" and branch is not None:
            branch = None
        elif branch is not None and token.startswith(r"\if"):
            raise ValueError("Nested TeX conditional requires explicit review")
        elif branch is not False:
            result.append(line)
    if branch is not None:
        raise ValueError("Unclosed ARXIV conditional")
    return "".join(result)


def normalize_text(text: str, repo: Path, legacy: Path) -> str:
    # Paths remain usable when they refer to files in the portable package.
    for prefix, target in [(str(repo) + "/", ""), (str(legacy) + "/", "results/")]:
        text = text.replace(prefix, target)
    text = text.replace(str(repo), ".").replace(str(legacy), "results")
    text = PRIVATE_PATH.sub(lambda match: "EXTERNAL_PATH/" + Path(match.group()).name, text)
    # No author names or bibliographic citations are replaced by this function.
    return text


def redact_execution_hosts(text: str) -> str:
    """Remove execution-host strings without reserializing scientific JSON."""
    keys = r"hostname|host_name|host|node_name|SLURM_JOB_NODELIST|SLURM_NODELIST"
    return re.sub(r'("(?:' + keys + r')"\s*:\s*)"(?:[^"\\]|\\.)*"',
                  r'\1"HOST_REDACTED"', text)


def redact_scheduler_site(text: str) -> str:
    """Export scheduler locations, preserving GPU names and scientific splits."""
    for kind in ("ACCOUNT", "PARTITION"):
        keys = rf"SLURM_JOB_{kind}|SLURM_{kind}|slurm_{kind.lower()}|scheduler_{kind.lower()}"
        text = re.sub(r'("(?:' + keys + r')"\s*:\s*)"(?:[^"\\]|\\.)*"',
                      rf'\1"SITE_{kind}"', text)
        # A generic "partition" can denote a scientific data split. Only
        # recognized site-prefixed scheduler values are removed in that form.
        text = re.sub(r'("' + kind.lower() + r'"\s*:\s*)"kempner[^"\\]*"',
                      rf'\1"SITE_{kind}"', text, flags=re.IGNORECASE)
    # Scheduler command arrays and recorded command/reason strings can carry
    # the same site names without an account/partition JSON key.
    text = re.sub(r"\bkempner_dev\b", "SITE_ACCOUNT", text, flags=re.IGNORECASE)
    text = re.sub(r"\bkempner_[A-Za-z0-9_-]+\b", "SITE_PARTITION", text, flags=re.IGNORECASE)
    return text


def checked_relative(value: str) -> Path:
    result = Path(value)
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"Unsafe package-relative path: {value}")
    return result


def export_text(source: Path, repo: Path, legacy: Path, *, paper=False, redact_firstparty_remote=False):
    """Transform identifying text while preserving original newline bytes."""
    with source.open("r", newline="") as handle:
        raw = handle.read()
    changes = []
    value = anonymous_tex(raw) if paper and source.suffix == ".tex" else raw
    if value != raw:
        changes.append("resolved_inactive_arxiv_branch")
    if redact_firstparty_remote:
        value = value.replace("git@github.com:varun04reddy/DFA-Stall.git", "identifying upstream URI withheld in this review export")
        changes.append("identifying_firstparty_provenance_uri_removed")
    portable = normalize_text(value, repo, legacy)
    if portable != value:
        changes.append("portable_path_export")
    if source.suffix == ".json":
        metadata = redact_execution_hosts(portable)
        if metadata != portable:
            changes.append("execution_hostname_removed")
        portable = redact_scheduler_site(metadata)
        if portable != metadata:
            changes.append("execution_scheduler_site_removed")
    return portable, changes


def verify_archive(archive: Path, expected: dict[str, str]) -> dict:
    """Read compressed bytes back and reject changed payloads or owner metadata."""
    seen = set()
    total = 0

    def verify(name, source):
        nonlocal total
        if name in seen or name not in expected:
            raise ValueError(f"Unexpected or duplicated archive member: {name}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
        if digest.hexdigest() != expected[name]:
            raise ValueError(f"Archive payload differs from exported file: {name}")
        seen.add(name)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if info.create_system != 0 or info.comment or info.date_time != (1980, 1, 1, 0, 0, 0):
                    raise ValueError("Unnormalized ZIP metadata")
                with bundle.open(info) as source:
                    verify(info.filename, source)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            for info in bundle:
                if info.uid or info.gid or info.uname or info.gname or info.mtime:
                    raise ValueError("Unnormalized tar owner metadata")
                if info.isdir():
                    continue
                if not info.isfile():
                    raise ValueError("Unexpected non-file tar member")
                with bundle.extractfile(info) as source:
                    verify(info.name, source)
    if seen != set(expected):
        raise ValueError("Archive is missing exported files")
    return {"status": "passed", "all_payload_hashes_verified": len(seen), "uncompressed_bytes": total, "owner_metadata_cleared": True}


def run(command: list[str], cwd: Path, *, timeout: int = 600) -> str:
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(cwd), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    # User-installed dependencies belong to the prepared Python environment;
    # disabling them can make an otherwise valid environment appear empty.
    # Project imports are checked explicitly against the isolated source root.
    env.pop("PYTHONNOUSERSITE", None)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Command failed: {command}\n{completed.stdout[-5000:]}\n{completed.stderr[-5000:]}")
    return completed.stdout


VERIFY_SCRIPT = '''"""Verify the exact distributed file inventory; no repository is required."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
expected_paths = set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}
actual_paths = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
extra = sorted(actual_paths - expected_paths)
if extra:
    raise SystemExit("Unlisted package files (verify an untouched extraction): " + ", ".join(extra[:10]))
for relative, expected in manifest["files"].items():
    path = root / relative
    if path.is_symlink():
        raise SystemExit("Unexpected symlink: " + relative)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["export_sha256"]:
        raise SystemExit("Changed package file: " + relative)
print("Verified", len(manifest["files"]), "distributed files")
'''

TUNED_BP_SCRIPT = '''"""Reproduce the separately test-selected historical BP reference."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

root = Path(__file__).resolve().parent / "results/infodfa_bp_tuning_synthetic_v1"
frames, sources = [], []
for path in sorted(root.rglob("dfa_multioutput_results.csv")):
    frame = pd.read_csv(path, usecols=["condition", "method", "seed", "epoch", "test_acc", "n_train", "input_noise", "train_label_noise"])
    if not frame.method.eq("bp").all():
        raise ValueError("Non-BP source in the historical BP sweep")
    frame = frame.loc[frame.epoch.eq(frame.epoch.max())].copy()
    if not frame.test_acc.between(0, 1).all():
        raise ValueError("Missing, nonfinite, or invalid BP test accuracy")
    frame["lr"] = float(re.search(r"/lr_([0-9p]+)/", str(path)).group(1).replace("p", "."))
    frames.append(frame)
    sources.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
rows = pd.concat(frames, ignore_index=True)
if len(sources) != 640 or len(rows) != 3200 or set(rows.seed) != set(range(5)):
    raise ValueError("Incomplete historical BP file/seed cohort")
if sorted(rows.lr.unique()) != [.02, .04, .08, .16, .32]:
    raise ValueError("Unexpected historical BP learning-rate grid")
keys = ["condition", "n_train", "input_noise", "train_label_noise", "seed", "lr"]
if rows.duplicated(keys).any() or not rows.groupby(keys[:-1]).lr.nunique().eq(5).all():
    raise ValueError("Incomplete or duplicated five-rate BP cohort")
grid = rows.groupby(["condition", "lr"]).test_acc.agg(["mean", "count"]).reset_index()
if not grid["count"].eq(160).all():
    raise ValueError("Unbalanced historical BP cell/seed coverage")
selected = grid.loc[grid.groupby("condition")["mean"].idxmax()].copy()
selected["percent"] = 100 * selected["mean"]
expected = {"nuisance_dominant": 27.9, "mixed_context": 43.8, "low_sample_noisy": 53.3, "task_aligned": 92.0}
if {r["condition"]: round(r["percent"], 1) for r in selected.to_dict(orient="records")} != expected:
    raise ValueError("Recomputed BP percentages disagree with the retained manuscript table")
print(json.dumps({"status": "recomputed_from_included_raw_csvs", "files": len(sources), "final_rows": len(rows), "learning_rates": sorted(rows.lr.unique()), "seeds": [int(x) for x in sorted(rows.seed.unique())], "final_epochs": [int(x) for x in sorted(rows.epoch.unique())], "selection": "maximum mean final test accuracy over five rates per regime, pooling matched cells and five seeds; not LOSO or validation selection", "grid": grid.to_dict(orient="records"), "selected": selected.to_dict(orient="records"), "source_sha256": sources}))
'''


README = '''# Conditioned direct feedback alignment: private review artifact

This artifact contains source code, tests, portable configurations, retained
run tables/reports, and the complete local manuscript source/figure closure.
It is a **private review candidate**, not a published anonymous release.
The unchanged MIT copyright notice in LICENSE identifies its authors and must
be resolved by the copyright holders before anonymous distribution. Third-party
citations and attribution are retained. No external dataset, checkpoint, or private path is
required for the offline paper build and bounded checks below.

## Environment and bounded offline verification

Use Python 3.10 and `pip install -r requirements.txt` in a prepared environment.
The validation environment is recorded in VALIDATION.json; the dependency
manifest specifies compatible versions, not an exact lockfile. CUDA training
requires an appropriate PyTorch build. A standard TeX Live installation with
pdflatex and the packages requested by paper/paper_preamble.tex builds the PDF.
No package download is performed by these commands once dependencies exist.

```bash
python verify_package.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_local_preconditioning.py tests/test_dfa_smoke.py tests/test_forward_decorrelation.py tests/test_decorrelated_dfa.py
cd paper
mkdir -p build
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory=build conditioned_dfa_iclr.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory=build conditioned_dfa_iclr.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory=build conditioned_dfa_iclr.tex
```

The builder checks the complete extracted PDF text against the retained audited
manuscript. It removes only inactive identifying arXiv branches from TeX sources.
The same figures and scientific text are retained. PDF metadata, embedded-file
count, annotations/links, and decoded PDF strings are inspected separately.

## Recorded cohorts and reproduction boundaries

ARTIFACT_INVENTORY.json lists every included cohort, missing requested root,
and deliberately omitted non-text artifact. Large final checkpoints and public
datasets are excluded. Their hashes and paths remain in retained run/audit
manifests when originally recorded. Consequently, tensor/checkpoint audits that
need those objects cannot run from this compact export alone. CSV-based
statistics and the original A/E/K main replication figure can be regenerated.
Paper figure PDFs and the recovered historical composition script are included;
do not interpret an included figure as proof of a complete from-training
regeneration. CLAIM_PROVENANCE.md gives the explicit mapping.

Original A/E/K confirmation cohorts are MNIST tanh model seeds 50--54,
Fashion-MNIST tanh 70--74, and ReLU MNIST 100--107, each crossed with feedback
seeds 0, 1, 2. Later appendix extensions are 55--59, 75--79, and 108--112;
they must not silently expand the main comparisons. The cohort-aware figure
script preserves these original sets:

```bash
python analysis/make_error_kndfa_replication_figure.py --output-dir reproduced/replication --no-copy-to-paper
python verify_tuned_bp.py
```

The latter independently reconstructs the historical five-rate BP reference
from all 640 run CSVs. It reports the full grid and checks the displayed
27.9/43.8/53.3/92.0 percentages. This per-regime test selection is distinct from
the fixed-rate BP trajectory in the original synthetic aggregate, and from the
separate LOSO comparator. The paper identifies it as exploratory.

The original composition script is paper/scripts/make_iclr_figures.py. Set
INFODFA_RESULTS and INFODFA_LEGACY_RESULTS to the absolute path of this
package's results/ directory, and INFODFA_FIGURES to a new output directory,
before using it. Its source lookups and generation scope should be inspected
for the requested panel; rebuilding every historical supplementary panel is
broader than the bounded checks recorded here. Existing paper figures remain
the authoritative assets for the audited source snapshot.

The recent studies are 72-model wide CIFAR-10 confirmation; 312-cell synthetic
development; 126-cell separate boundary diagnostic; 320-model frozen eight-draw
confirmation; 52-cell four-epoch CIFAR-100 development; ten 30-epoch CIFAR-100
models from one new seed; and 496 continuous-teaching regression development
outcomes (including retained numerical failures). They have different scopes.
The recorded reports and frozen configurations provide their actual methods,
hyperparameters, horizons, and outcome counts. New work added to this package
is listed separately in ARTIFACT_INVENTORY.json.

The full-pool CIFAR-10 benchmark, when included, has a separate development
history: the first attempt was interrupted by a numerical-failure reporting
error; the unchanged 120-configuration replay resolved every case but all
12 DFA forward-decorrelation settings diverged. A prospectively frozen recovery
added 12 distinct settings to every method, yielding 24 per method and 240
unique development configurations in total. The combined audit retains 199
completed models and 41 numerical failures. The interrupted attempt, replay,
recovery and all source snapshots remain separate. Development selection and
five fresh paired training replicates precede the optional official-test
prediction audit. Read cross entropy together with accuracy: finite endpoints
can have extreme loss. The package makes no claim of superiority for A, E or K;
the complete accepted prediction report and manuscript state the comparisons.

When EVIDENCE_REQUEST.json exists, run `python verify_new_evidence.py` from
the package root. The requested checks also run automatically during builder
validation. Mechanism verification reruns its frozen analyzer on both complete
NPZ archives and compares every scientific report field. Its original source,
configuration and numerical data retain their original bytes; any changed
manifest digest is explained by execution-host redaction alone.

Prediction verification recomputes all 50 final-test metrics and the six
declared paired comparisons from saved logits, labels and identity indices.
It checks binary metadata against a strict schema and verifies original/export
digests. It does not restore the omitted training weights or independently
reestablish the original training, timing or one-use evaluation audit. Reported
training time and memory are retained measurements, with arithmetic summaries
recomputed. EVIDENCE_PROVENANCE.json explicitly maps the evidence files before
and after portable path/host redaction; embedded original hashes are retained.

## Portable configurations and original provenance

Every copied file has an original SHA-256 and exported SHA-256 in
PACKAGE_MANIFEST.json. Paths in configs, manifests, and reports are normalized
to this package. These are **reproduction exports**, not newly prospective
registrations or byte-identical copies of the original audit evidence. Existing
embedded hashes refer to original artifacts unless explicitly labeled export
hashes. Source files with no transformations retain their original bytes.
Pinned source snapshots are included independently of current development code;
use the snapshot belonging to the desired frozen configuration. Source changes
needed for anonymization would require distinct export source pins and are
reported rather than silently accepted.

The full historical synthetic and noisy-vision aggregate CSVs are relocated to
results/ under the names expected by the analysis scripts. No sibling repository
is required for those inputs. Do not overwrite retained results when reanalyzing:
use a fresh output directory, or a separate copy of the package for older scripts
with fixed output paths. Submitting the site's original SLURM launchers is not
necessary; invoke the archived Python runner with its portable config on your
own machine/allocation.

## Public data and expensive training

MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 use torchvision download helpers.
They download only when the corresponding training command is explicitly run
with download enabled; the bounded offline checks above use synthetic fixtures.
Official dataset sources are https://yann.lecun.com/exdb/mnist/,
https://github.com/zalandoresearch/fashion-mnist, and
https://www.cs.toronto.edu/~kriz/cifar.html. ImageNet must be obtained separately
from https://www.image-net.org/ and supplied in ImageFolder train/val layout;
the retained commands use the explicitly seeded 100-class subset. Dataset
licenses/terms are those of the original providers, not this code's MIT license.
No ImageNet download, network training, or full experiment rerun is part of the
package validation.
'''


def export_evidence(args, repo: Path, package: Path, records: dict, add) -> dict:
    """Include complete optional numerical evidence and its original/export pins."""
    request, paths = {}, set()

    def include(relative, source=None):
        relative = str(checked_relative(str(relative)))
        source = repo / relative if source is None else source
        if relative in records:
            if records[relative]["original_sha256"] != sha256(source):
                raise ValueError(f"Conflicting evidence export: {relative}")
        else:
            add(source, relative)
        paths.add(relative)

    mechanism = [getattr(args, name, None) for name in ("mechanism_root", "mechanism_source", "mechanism_report")]
    if any(mechanism):
        if not all(mechanism):
            raise ValueError("Mechanism verification requires --mechanism-root, --mechanism-source and --mechanism-report")
        root, source, report = map(checked_relative, mechanism)
        for name in ("manifest.json", "config.json", "planned_conditions.json", "arrays.npz", "rows.json", "trajectory_rows.json", "trajectory_conditions.json", "trajectory_arrays.npz"):
            include(root / name)
        config = json.loads((repo / root / "config.json").read_text())
        for name in config["source_sha256"]:
            include(source / checked_relative(name))
        include(report)
        request["mechanism"] = {"run_dir": str(root), "source_root": str(source), "report": str(report)}
    prediction = [getattr(args, name, None) for name in ("prediction_root", "prediction_audit")]
    if any(prediction):
        if not all(prediction):
            raise ValueError("Prediction verification requires --prediction-root and --prediction-audit")
        root, report = map(checked_relative, prediction)
        include(root / "evaluation_manifest.json")
        include(root / "plan.json")
        manifest = json.loads((repo / root / "evaluation_manifest.json").read_text())
        case_ids = [entry["case_id"] for entry in manifest["cases"]]
        if len(case_ids) != 50 or len(set(case_ids)) != 50:
            raise ValueError("Prediction export requires exactly fifty distinct cases")
        for case_id in case_ids:
            if Path(case_id).name != case_id or case_id in {".", ".."}:
                raise ValueError("Unsafe prediction case identifier")
            for name in ("evaluation.json", "predictions.pt"):
                include(root / "cases" / case_id / name)
        include(report)
        include("analysis/analyze_ndfa_submission_confirmation.py")
        include("analysis/analyze_ndfa_submission_benchmark.py")
        request["prediction"] = {"root": str(root), "report": str(report)}
    if request:
        include("verify_new_evidence.py", Path(__file__).with_name("verify_ndfa_review_evidence.py"))
        request["provenance_paths"] = sorted(paths)
    return request


def build(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    legacy = args.legacy_results.resolve()
    paper_source = args.paper_source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), output / "private_builder_snapshot.py")
    package = output / "review_package"
    package.mkdir()
    records = {}
    inventory = {"included_roots": [], "missing_roots": [], "omitted": [], "extra_roots": args.artifact_root}

    def add(source: Path, relative: str, *, paper: bool = False, redact_firstparty_remote: bool = False) -> None:
        destination = package / checked_relative(relative)
        if destination.exists():
            raise ValueError(f"Duplicate export path: {relative}")
        if source.is_symlink():
            raise ValueError(f"Symlink requires explicit review: {source}")
        original_hash = sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        transformations = []
        if source.suffix in SCANNABLE_TEXT_SUFFIXES:
            portable, transformations = export_text(source, repo, legacy, paper=paper, redact_firstparty_remote=redact_firstparty_remote)
            with destination.open("w", newline="") as handle:
                handle.write(portable)
        else:
            shutil.copyfile(source, destination)
        if sha256(source) != original_hash:
            raise RuntimeError(f"Source changed during export: {source}")
        records[relative] = {"original_sha256": original_hash, "export_sha256": sha256(destination), "bytes": destination.stat().st_size, "transformations": transformations}

    for directory in ["infogeo", "experiments", "analysis", "tests"]:
        for source in sorted((repo / directory).glob("*.py")):
            if source.name == "test_ndfa_anonymous_package.py":
                continue  # This builder's tests contain private-path fixtures.
            add(source, str(source.relative_to(repo)))
    add(repo / "requirements.txt", "requirements.txt")
    add(repo / "LICENSE", "LICENSE")
    add(repo / "external/DFA-Stall/train.py", "external/DFA-Stall/train.py")
    add(repo / "external/DFA-Stall/VENDORED_INFO.md", "external/DFA-Stall/VENDORED_INFO.md", redact_firstparty_remote=True)
    add(repo / "drafts/Info-DFA/scripts/make_iclr_figures.py", "paper/scripts/make_iclr_figures.py")
    for source in sorted((repo / "configs").glob("*.json")):
        if "followthrough" not in source.name:
            add(source, str(source.relative_to(repo)))
    closure = json.loads((paper_source / "manifest.json").read_text())
    for relative, expected in closure["file_sha256"].items():
        source = paper_source / checked_relative(relative)
        if sha256(source) != expected:
            raise ValueError(f"Changed audited manuscript dependency: {relative}")
        add(source, "paper/" + relative, paper=True)
    # The snapshot manifest contains private source-root metadata; its relevant
    # dependency hashes are already represented in our export manifest.
    for source in sorted((repo / "docs/research").glob("*.json")):
        # The project completion audit binds the finished archive hash and
        # therefore belongs outside the archive it audits.
        if "runs_" not in source.name and "registry_" not in source.name and "resource_amendment" not in source.name and "completion_audit" not in source.name:
            add(source, str(source.relative_to(repo)))
    for relative in ["docs/research/ndfa_teaching_regression_development_20260915/audit.json", "docs/research/ndfa_cohort_reconciliation_2026-09-14.md"]:
        if (repo / relative).exists():
            add(repo / relative, relative)
    roots = [(repo / "results" / name, name) for name in COHORTS]
    roots += [(legacy / name, name) for name in LEGACY]
    roots += [(repo / checked_relative(relative), checked_relative(relative).name) for relative in args.artifact_root]
    for source_root, name in roots:
        if source_root.is_symlink():
            raise ValueError(f"Symlink cohort root requires explicit review: {source_root}")
        if not source_root.is_dir():
            inventory["missing_roots"].append(name)
            continue
        inventory["included_roots"].append(name)
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = str(Path("results") / name / source.relative_to(source_root))
            if source.suffix in TEXT_SUFFIXES:
                add(source, relative)
            else:
                reason = ("site-specific scheduler launcher omitted; original launch/source pins remain historical evidence"
                          if source.suffix == ".sbatch" else "binary dataset/checkpoint/duplicate figure omitted; inspect retained audit for recorded hashes")
                inventory["omitted"].append({"path": relative, "bytes": source.stat().st_size, "reason": reason})
    for relative in args.extra_file:
        add(repo / checked_relative(relative), relative)
    evidence = export_evidence(args, repo, package, records, add)
    inventory["optional_evidence"] = {key: value for key, value in evidence.items() if key != "provenance_paths"}
    inventory["omitted"] = [row for row in inventory["omitted"] if row["path"] not in records]
    (package / "README.md").write_text(README)
    (package / "verify_package.py").write_text(VERIFY_SCRIPT)
    (package / "verify_tuned_bp.py").write_text(TUNED_BP_SCRIPT)
    (package / "ARTIFACT_INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (package / "CLAIM_PROVENANCE.md").write_text(claim_map())
    if evidence:
        evidence_paths = evidence.pop("provenance_paths")
        (package / "EVIDENCE_REQUEST.json").write_text(json.dumps(evidence, indent=2) + "\n")
        # This subset is available before validation; the final package manifest
        # independently binds these generated records and all exported files.
        (package / "EVIDENCE_PROVENANCE.json").write_text(json.dumps({name: records[name] for name in evidence_paths}, indent=2) + "\n")
    (output / "copy_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Copied {len(records)} source/artifact files; validating in an isolated directory", flush=True)
    validation = validate(package, paper_source.parent / "conditioned_dfa_iclr.pdf", args.skip_validation)
    (package / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    scan = scan_package(package)
    (output / "anonymity_review.json").write_text(json.dumps(scan, indent=2) + "\n")
    for source in package.rglob("*"):
        if source.is_file() and str(source.relative_to(package)) not in records:
            relative = str(source.relative_to(package))
            records[relative] = {"original_sha256": None, "export_sha256": sha256(source), "bytes": source.stat().st_size, "transformations": ["generated_export_document"]}
    manifest = {"schema_version": 1, "release_status": "private_review_only", "license_notice_resolution_required": True, "files": records}
    (package / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run([sys.executable, "verify_package.py"], package)
    archive = output / ("ndfa_private_review." + args.archive_format)
    if args.archive_format == "zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(str(Path("review_package") / path.relative_to(package)), date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 0
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with path.open("rb") as source, bundle.open(info, "w", force_zip64=True) as destination:
                        shutil.copyfileobj(source, destination)
    else:
        with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
            for path in sorted(package.rglob("*")):
                info = bundle.gettarinfo(str(path), str(Path("review_package") / path.relative_to(package)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as handle:
                        bundle.addfile(info, handle)
                else:
                    bundle.addfile(info)
    expected_archive = {"review_package/" + name: row["export_sha256"] for name, row in records.items()}
    expected_archive["review_package/PACKAGE_MANIFEST.json"] = sha256(package / "PACKAGE_MANIFEST.json")
    archive_check = verify_archive(archive, expected_archive)
    result = {"file_count": len(records) + 1, "builder_sha256": sha256(output / "private_builder_snapshot.py"), "archive": archive.name, "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive), "archive_integrity": archive_check, "validation": validation, "anonymity": scan, "missing_roots": inventory["missing_roots"]}
    (output / "build_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def claim_map() -> str:
    return '''# Claim and figure provenance

| Claim or figure | Included evidence | Reproduction boundary |
| --- | --- | --- |
| Original fixed-full synthetic comparison | results/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv; analysis/reanalyze_synthetic_honest_selection.py | Full retained aggregate, including historical excluded/error-scaled variants; apply the manuscript's exact claim exclusions. Its fixed-rate BP column differs from the separately tuned reference. |
| Separately tuned BP reference in main synthetic table/figure | results/infodfa_bp_tuning_synthetic_v1; verify_tuned_bp.py; paper/scripts/make_iclr_figures.py | Derive per-regime maximum mean final test accuracy from five rates, 32 cells and five seeds; reproduce display values 27.9/43.8/53.3/92.0 without mistaking this for validation selection or LOSO. |
| Historical noisy-vision table | results/infodfa_vision_noise_sweep_aggregate_v2; results/infodfa_vision_noise_sweep_val_agg_v1; paper table snapshots | Test-selected and validation-selected summaries have different scopes. |
| Main A/E/K replication figure and original confirmation tables | results/dfa_{stall,stall_fashion,relu_mnist}* threefactor roots; results/error_kndfa_replication_figure_v1; cohort-aware figure script | Original cohorts are explicit in README. Pooled appendix figures are distinct. |
| BatchNorm, decorrelation-power, BP conditioning, norm match and optimizer controls | Retained infodfa_bn*, actwhiten, bpwhiten, normmatch, kfac and adam roots in ARTIFACT_INVENTORY.json | Read covariance-power cohort audit for duplicate/excluded cohort handling. |
| ImageNet-100 spatial credit boundary | results/imagenet100_strongform_v1 and companion lrcheck/noisy roots | Separate inverse-square-root pooled-block operator, not the central activity inverse update. |
| Wide CIFAR-10 confirmation | results/ndfa_cifar10_confirmation_20260914; frozen config/source snapshot and docs/research audit JSON | Checkpoints omitted; recorded endpoints and source hashes included. |
| Synthetic noise/task development and independent confirmation | 312/126/320 run metadata, source snapshots, configs and audited JSONs | Search-boundary diagnostic is separate; original restricted-search confirmation retained. |
| CIFAR-100 Mixer development and final result | 52/10 run metadata, source snapshots, configs and audited JSONs | Final result is one seed; omitted checkpoint tensors prevent a fresh full checkpoint audit. |
| Continuous-teaching regression development | 496 run outcomes and audit JSON with retained numerical failures | Development only; no independent confirmation claim. |
| Optional engineered factor mechanism | EVIDENCE_REQUEST.json mechanism entry; complete NPZ arrays, frozen source, original report and verify_new_evidence.py | Independently reconstruct all finite-batch/trajectory quantities. Source and numerical evidence must be byte-identical; execution-host manifest redaction is explicitly reconciled. |
| Optional full-pool CIFAR-10 confirmation | EVIDENCE_REQUEST.json prediction entry; all 50 prediction tensors, accepted audit and verify_new_evidence.py | Independently recompute 50 test metrics and six paired contrasts; omitted checkpoints prevent redoing the original training/restoration audit. |
| All paper layout/figure assets | paper/ contains the entire audited local TeX/style/PDF closure and recovered composition script | The offline PDF build is complete; the bounded validation does not rerun every historical panel's complete analysis pipeline. |

The package inventory and optional EVIDENCE_REQUEST.json are the authoritative
lists of included cohorts and numerical reproduction checks. An optional row
above claims no completed reproduction unless its entry appears in VALIDATION.json
with successful checks. Frozen historical results are not rewritten by this export.
'''


def validate(package: Path, original_pdf: Path, skip: bool) -> dict:
    if skip:
        return {"status": "not_run", "reason": "explicit --skip-validation"}
    with tempfile.TemporaryDirectory(prefix="ndfa-review-") as temporary:
        isolated = Path(temporary) / "review_package"
        shutil.copytree(package, isolated, copy_function=shutil.copyfile)
        run([sys.executable, "-c", "from pathlib import Path; import infogeo,infogeo.dfa,infogeo.local_preconditioning; assert all(Path(m.__file__).resolve().is_relative_to(Path.cwd()) for m in [infogeo,infogeo.dfa,infogeo.local_preconditioning]); print('Project modules resolve inside isolated export')"], isolated)
        # Explicit imports test package closure and one-step learning, not just
        # compilation. The selected tests need no network, GPU, or real dataset.
        test_output = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_local_preconditioning.py", "tests/test_dfa_smoke.py", "tests/test_forward_decorrelation.py", "tests/test_decorrelated_dfa.py"], isolated)
        run([sys.executable, "-c", "from experiments.run_dfa_stall_comparison import load_dfa_stall_module; m=load_dfa_stall_module(); assert hasattr(m,'TanhMLP'); print('Vendored tanh learner imports without data access')"], isolated)
        statistical = validate_statistics(isolated)
        paper = isolated / "paper"
        (paper / "build").mkdir()
        for _ in range(3):
            run(["pdflatex", "-no-shell-escape", "-halt-on-error", "-interaction=nonstopmode", "-output-directory=build", "conditioned_dfa_iclr.tex"], paper)
        built = paper / "build/conditioned_dfa_iclr.pdf"
        old_text = run(["pdftotext", "-layout", str(original_pdf), "-"], isolated)
        new_text = run(["pdftotext", "-layout", str(built), "-"], isolated)
        if old_text != new_text:
            raise ValueError("Anonymous source export changed extracted manuscript text")
        log = (paper / "build/conditioned_dfa_iclr.log").read_text()
        if "Overfull \\hbox" in log or "undefined references" in log or "undefined citations" in log:
            raise ValueError("Manuscript has unresolved or overfull build diagnostics")
        shutil.copyfile(built, package / "paper/conditioned_dfa_iclr.pdf")
        versions = run([sys.executable, "-c", "import json,platform,torch,numpy,scipy,matplotlib,pytest; print(json.dumps({m.__name__:m.__version__ for m in [torch,numpy,scipy,matplotlib,pytest]} | {'python':platform.python_version()}))"], isolated)
        return {"status": "passed", "isolated_from_repository": True, "offline_synthetic_tests": test_output.strip().splitlines()[-1], "statistical_reproduction": statistical, "paper_passes": 3, "full_pdf_text_equal": True, "pdf_text_sha256": hashlib.sha256(new_text.encode()).hexdigest(), "pdf_pages": new_text.count("\f"), "paper_pdf_sha256": sha256(built), "versions": json.loads(versions), "scope": "Selected meaningful CPU tests, main A/E/K contrast regeneration, historical fixed-full/tuned-BP summaries, requested additional numerical evidence, and full offline paper build; not every training entrypoint or every statistical claim was rerun."}


def validate_statistics(isolated: Path) -> dict:
    run([sys.executable, "analysis/make_error_kndfa_replication_figure.py", "--output-dir", "reproduced/replication", "--no-copy-to-paper"], isolated)
    comparison = run([sys.executable, "-c", "import json,pandas as pd; a=pd.read_csv('results/error_kndfa_replication_figure_v1/replication_contrasts.csv'); b=pd.read_csv('reproduced/replication/replication_contrasts.csv'); pd.testing.assert_frame_equal(a,b,rtol=1e-12,atol=1e-12); print(json.dumps({'rows':len(a),'cohorts':[5,5,8],'recorded_contrasts_reproduced':True}))"], isolated)
    synthetic = run([sys.executable, "-c", "import json; import analysis.reanalyze_synthetic_honest_selection as a; d=a.load_final(); v=a.per_cell_method_value(d,'fixed_full'); s=a.summarize(v); print(json.dumps({'aggregate_resolved_inside_package':a.SRC.is_relative_to(a.ROOT),'fixed_full_summary':s[['condition','BP','DFA','nDFA']].to_dict(orient='records'),'seed_ids':[int(x) for x in sorted(d.seed.unique())]}))"], isolated)
    tuned = run([sys.executable, "verify_tuned_bp.py"], isolated)
    result = {"main_replication": json.loads(comparison), "historical_synthetic": json.loads(synthetic), "separately_test_selected_bp": json.loads(tuned)}
    if (isolated / "EVIDENCE_REQUEST.json").exists():
        result["additional_evidence"] = json.loads(run([sys.executable, "verify_new_evidence.py"], isolated))
    return result


def scan_package(package: Path) -> dict:
    import fitz

    problems = []
    citations = []
    pdf_count = 0
    raster_count = 0
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(package))
        if path.name == "LICENSE":
            problems.append({"path": relative, "kind": "required_identifying_copyright_notice", "resolution": "Preserve until copyright holders authorize an anonymous review notice; no permission inferred."})
            continue
        if path.suffix in SCANNABLE_TEXT_SUFFIXES:
            text = path.read_text()
            for token in DENIED:
                if token.casefold() in text.casefold():
                    problems.append({"path": relative, "kind": "identifying_token", "token": token})
            if PRIVATE_PATH.search(text):
                problems.append({"path": relative, "kind": "personal_absolute_path"})
            outside_bibliography = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", text, flags=re.DOTALL) if path.suffix == ".tex" else text
            for token in SITE_TOKENS:
                if token.casefold() in outside_bibliography.casefold():
                    problems.append({"path": relative, "kind": "identifying_institution_token", "token": token})
                elif token.casefold() in text.casefold():
                    citations.append({"path": relative, "kind": "institution_in_retained_bibliography_requires_context_review", "token": token})
            if path.suffix == ".tex" and re.search(r"Sabatini", text, re.I):
                citations.append({"path": relative, "kind": "author_name_in_retained_bibliography_requires_context_review"})
        elif path.suffix == ".png":
            from PIL import Image

            raster_count += 1
            with Image.open(path) as raster:
                metadata = json.dumps({"info": raster.info, "exif": dict(raster.getexif())}, default=str)
            for token in (*DENIED, *SITE_TOKENS):
                if token.casefold() in metadata.casefold():
                    problems.append({"path": relative, "kind": "identifying_raster_metadata", "token": token})
            if PRIVATE_PATH.search(metadata):
                problems.append({"path": relative, "kind": "personal_raster_metadata_path"})
        elif path.suffix == ".pdf":
            pdf_count += 1
            with fitz.open(path) as pdf:
                metadata = json.dumps(pdf.metadata)
                strings = "\n".join(pdf.xref_object(index, compressed=False) for index in range(1, pdf.xref_length()))
                streams = "\n".join(pdf.xref_stream(index).decode("utf-8", errors="ignore") for index in range(1, pdf.xref_length()) if pdf.xref_is_stream(index))
                searchable = metadata + strings + streams + pdf.get_xml_metadata() + "\n".join(page.get_text() for page in pdf)
                for token in DENIED:
                    if token.casefold() in searchable.casefold():
                        problems.append({"path": relative, "kind": "identifying_pdf_token", "token": token})
                for token in SITE_TOKENS:
                    if token.casefold() in searchable.casefold():
                        problems.append({"path": relative, "kind": "institution_in_pdf_requires_context_review", "token": token})
                if pdf.metadata.get("author"):
                    problems.append({"path": relative, "kind": "nonempty_pdf_author"})
                if PRIVATE_PATH.search(searchable):
                    problems.append({"path": relative, "kind": "personal_pdf_absolute_path"})
                if pdf.embfile_count():
                    problems.append({"path": relative, "kind": "embedded_pdf_files"})
    return {"ready_for_anonymous_distribution": not problems, "findings": problems, "retained_citation_context": citations, "pdf_files_inspected": pdf_count, "raster_metadata_files_inspected": raster_count, "scope": "Known identity tokens and absolute paths across text/SVG; PDF metadata, decoded objects, text, embedded files; PNG metadata. Citations are retained; human content review of figures and bibliography contexts is still required."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--legacy-results", type=Path, default=Path(__file__).resolve().parents[2] / "Info-Man/results")
    parser.add_argument("--paper-source", type=Path, required=True, help="Audited source snapshot containing manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Exclusively new directory")
    parser.add_argument("--artifact-root", action="append", default=[], help="Additional repository-relative cohort root")
    parser.add_argument("--extra-file", action="append", default=[], help="Additional repository-relative file")
    parser.add_argument("--mechanism-root", help="Optional repository-relative complete mechanism output; includes both NPZ archives")
    parser.add_argument("--mechanism-source", help="Matching repository-relative frozen mechanism source_snapshot")
    parser.add_argument("--mechanism-report", help="Matching repository-relative authoritative mechanism JSON report")
    parser.add_argument("--prediction-root", help="Optional repository-relative complete 50-case official prediction output; includes predictions.pt")
    parser.add_argument("--prediction-audit", help="Matching repository-relative accepted prediction audit JSON; checkpoints remain omitted")
    parser.add_argument("--archive-format", choices=("zip", "tar.gz"), default="zip")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    existed_before = arguments.output.exists()
    try:
        print(json.dumps(build(arguments), indent=2))
    except Exception as error:
        if not existed_before and arguments.output.is_dir():
            with (arguments.output / "failure.json").open("x") as failure:
                json.dump({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, failure, indent=2)
                failure.write("\n")
        raise
