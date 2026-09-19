"""Create a NEW anonymous review export; never modify canonical or frozen inputs.

The private build receipt records the authorized first-party notice substitution.
No deployment or upload is performed. Final paper/late-study binding is explicit.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import shutil

import build_ndfa_anonymous_package as previous


ANONYMOUS_NOTICE = "Copyright (c) 2026 the authors of the accompanying anonymous submission"
TEXT = previous.SCANNABLE_TEXT_SUFFIXES | {".jsonl", ".yaml", ".yml", ".toml", ".sh", ".sbatch"}
PREDICTIONS = {"final_validation.pt", "predictions.pt"}
ROOTS = (
    "ndfa_bn_confirmation_20260915", "ndfa_bn_confirmation_snapshot_20260915",
    "ndfa_bn_factors_h100_234min_20260915", "ndfa_bn_factors_h100_234min_snapshot_20260915",
    "ndfa_bn_optimizer_60min_20260915", "ndfa_bn_optimizer_60min_snapshot_20260915",
    "ndfa_bn_length_bridge_20260915", "ndfa_bn_length_bridge_analysis_snapshot_20260915",
    "ndfa_bn_length_bridge_snapshot_20260915", "ndfa_bn_scale_h100_20260915",
    "ndfa_bn_scale_h100_snapshot_20260915", "ndfa_bn_scale_analysis_snapshot_20260915",
    "ndfa_factor_mechanism_20260915", "ndfa_factor_mechanism_snapshot_20260915",
    "ndfa_submission_analysis_snapshot_20260915", "ndfa_submission_confirmation_20260915",
    "ndfa_submission_confirmation_evaluation_20260915", "ndfa_submission_confirmation_evaluation_snapshot_20260915",
    "ndfa_submission_confirmation_snapshot_20260915", "ndfa_submission_development_20260915",
    "ndfa_submission_development_recovery_20260915", "ndfa_submission_development_recovery_snapshot_20260915",
    "ndfa_submission_development_snapshot_20260915", "ndfa_submission_development_v2_20260915",
    "ndfa_submission_development_v2_snapshot_20260915", "ndfa_submission_recovery_analysis_snapshot_20260915",
    "ndfa_teaching_channel_rank3_pilot_h100_20260915", "ndfa_teaching_channel_rank3_snapshot_20260915",
    "ndfa_teaching_channel_rank3_execution_h100_75min_20260915",
)
DOCS = (
    "ndfa_bn_confirmation_20260915", "ndfa_bn_length_bridge_20260915",
    "ndfa_bn_scale_h100_20260915", "ndfa_submission_confirmation_evaluation_20260915",
    "ndfa_factor_mechanism_20260915.json", "ndfa_bn_factors_completed_review_20260915.md",
    "ndfa_bn_factors_completed_review_20260915.json",
)
CONFIGS = ("ndfa_bn_confirmation_20260915.json", "ndfa_bn_factors_h100_234min_20260915.json",
           "ndfa_bn_optimizer_60min_20260915.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def anonymous_license(raw):
    lines = raw.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith("Copyright (c) 2026 ")]
    require(len(hits) == 1, "Unexpected first-party license notice")
    index = hits[0]
    original = lines[index].rstrip("\r\n")
    require(original == "Copyright (c) 2026 Houman Safaai, Varun Reddy, and Bernardo L. Sabatini", "Unexpected copyright holder scope")
    lines[index] = ANONYMOUS_NOTICE + ("\r\n" if lines[index].endswith("\r\n") else "\n")
    return "".join(lines), original


def transform_details(raw, repo, *, paper=False):
    value, changes = raw, []
    functions = [
        ("resolved_inactive_arxiv_branch", previous.anonymous_tex if paper else lambda text: text),
        ("portable_path_export", lambda text: previous.normalize_text(text, repo, repo.parent / "Info-Man" / "results")),
        ("execution_hostname_removed", previous.redact_execution_hosts),
        ("execution_scheduler_site_removed", previous.redact_scheduler_site),
        ("execution_scheduler_qos_removed", lambda text: re.sub(r"\bkemp_gpu\w*\b", "SITE_QOS", text, flags=re.I)),
        ("execution_compute_node_removed", lambda text: re.sub(r"\bholygpu\w*\b", "COMPUTE_NODE", text, flags=re.I)),
        ("execution_scheduler_identity_removed", redact_scheduler_identity),
    ]
    for name, function in functions:
        changed = function(value)
        if changed != value:
            changes.append(name)
        value = changed
    return value, changes


def transform(raw, repo, *, paper=False):
    return transform_details(raw, repo, paper=paper)[0]


def redact_scheduler_identity(text):
    text = re.sub(r"\b(?:UserId|GroupId)=[^\s\\\"]+", lambda match: match.group().split("=", 1)[0] + "=ANONYMOUS", text)
    text = re.sub(r"\bslurm_group_kempner_\w+\b", "SITE_GROUP", text, flags=re.I)
    text = re.sub(r"\bbsabatini_lab\b", "SITE_GROUP", text, flags=re.I)
    text = re.sub(r"\bholylogin\w*(?::\d+)?\b", "LOGIN_NODE", text, flags=re.I)
    # Execution host suffixes only; bibliographic institution names are retained.
    text = re.sub(r"(?:[\w.-]+\.)?rc\.fas\.harvard\.edu\b", "COMPUTE_HOST", text, flags=re.I)
    return text


def numeric_leaves(value):
    if isinstance(value, (int, float, bool)) or value is None:
        return [value]
    if isinstance(value, list):
        return [x for child in value for x in numeric_leaves(child)]
    if isinstance(value, dict):
        return [x for child in value.values() for x in numeric_leaves(child)]
    return []


@lru_cache(maxsize=4096)
def recorded_file_hashes(path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    # These keys specifically denote file hashes. Do not confuse the factor
    # runner's initial_state_sha256/final_state_sha256 tensor digests with them.
    result = dict(value.get("artifact_sha256", value.get("artifacts_sha256", {})))
    if path.name == "final_export.json" and "final_state_sha256" in value:
        result["final_state.pt"] = value["final_state_sha256"]
    return result


def omitted_provenance(source, repo):
    for name in ("status.json", "final_export.json", "receipt.json", "manifest.json"):
        record = source.parent / name
        digest = recorded_file_hashes(record).get(source.name)
        if digest is not None:
            require(isinstance(digest, str) and re.fullmatch("[a-f0-9]{64}", digest), "Malformed producer file hash")
            return {"original_file_sha256": digest, "hash_basis": "producer-recorded file SHA-256, not rehashed during this export",
                    "record_path": str(record.relative_to(repo)), "record_original_sha256": previous.sha256(record)}
    return {"original_file_sha256": None, "hash_basis": "No unambiguous adjacent producer file hash; no hash claim is made"}


def build(args):
    repo = Path(args.repo).resolve()
    base, paper = Path(args.base).resolve(), Path(args.paper_source).resolve()
    output = Path(args.output).resolve()
    require(not output.exists(), "Use a NEW output directory")
    output.mkdir(parents=True)
    package = output / "review_package"
    package.mkdir()
    records, omissions = {}, []
    base_manifest = json.loads((base / "PACKAGE_MANIFEST.json").read_text())
    skip = {"README.md", "LICENSE", "VALIDATION.json", "ARTIFACT_INVENTORY.json", "CLAIM_PROVENANCE.md", "verify_package.py"}
    for name, old in base_manifest["files"].items():
        if name.startswith("paper/") or name in skip:
            continue
        source = base / previous.checked_relative(name)
        require(previous.sha256(source) == old["export_sha256"], "Changed historical base file: " + name)
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        records[name] = dict(old)
        if source.suffix == ".json":
            raw = source.read_text()
            text, changes = transform_details(raw, repo)
            require(numeric_leaves(json.loads(raw)) == numeric_leaves(json.loads(text)), "Historical numbers changed: " + name)
            target.write_text(text)
            records[name].update(export_sha256=previous.sha256(target), bytes=target.stat().st_size,
                                 transformations=old["transformations"] + changes)
        else:
            shutil.copyfile(source, target)

    def add(source, name=None):
        source = Path(source)
        name = str(previous.checked_relative(str(name or source.relative_to(repo))))
        require(source.is_file() and not source.is_symlink(), "Missing/nonregular input: " + str(source))
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        changes = []
        if source.suffix in TEXT:
            with source.open(newline="") as stream:
                raw = stream.read()
            text, changes = transform_details(raw, repo, paper=name.startswith("paper/") and source.suffix == ".tex")
            if source.suffix == ".json":
                require(numeric_leaves(json.loads(raw)) == numeric_leaves(json.loads(text)), "Numerical metadata changed: " + name)
            with target.open("w", newline="") as stream:
                stream.write(text)
        else:
            shutil.copyfile(source, target)
        records[name] = {"original_sha256": previous.sha256(source), "export_sha256": previous.sha256(target),
                         "bytes": target.stat().st_size, "transformations": changes}

    def generated(name, value):
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
        records[name] = {"original_sha256": None, "export_sha256": previous.sha256(target),
                         "bytes": target.stat().st_size, "transformations": ["generated_export_document"]}

    def tree(directory):
        require(directory.is_dir(), "Missing declared evidence directory: " + str(directory))
        for source in sorted(directory.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.name.endswith(".tmp"):
                continue
            if source.suffix in TEXT or source.suffix == ".npz" or source.name in PREDICTIONS:
                add(source)
            elif source.suffix == ".pt":
                omissions.append({"path": str(source.relative_to(repo)), "bytes": source.stat().st_size,
                                  "reason": "Large checkpoint/optimizer/anchor state omitted", "hash_recomputed_for_export": False,
                                  **omitted_provenance(source, repo)})

    for name in ROOTS:
        tree(repo / "results" / name)
    if getattr(args,'final',False):
        for name in ('ndfa_bn_forward_decorrelation_20260915','ndfa_bn_forward_decorrelation_snapshot_20260915',
                     'ndfa_final_test_20260915','ndfa_final_test_snapshot_20260915'):
            tree(repo/'results'/name)
        tree(repo/'docs/research/ndfa_final_test_20260915')
        tree(repo/'scripts/ndfa_final_test_20260915')
        add(repo/'configs/ndfa_bn_forward_decorrelation_20260915.json')
        add(repo/'docs/research/ndfa_final_test_protocol_20260915.md')
        for name in ('prepare_data.py','export_assets.py','verify_assets.py','late_assets.py'):
            add(repo/'scripts/iclr_completion_20260915'/name)
        for name in ('verified_data.json','data_verification.json','asset_manifest.json','asset_verification.json'):
            add(repo/'scripts/iclr_completion_20260915/artifacts'/name)
        for name in ('forward_manifest.json','test_manifest.json'):
            add(repo/'scripts/iclr_completion_20260915/late_artifacts'/name)
        add(repo/'results/ndfa_bn_optimizer_completed_review_20260915.json')
        add(repo/'results/ndfa_bn_forward_decorrelation_completed_review_v2_20260915.json')

    if getattr(args, 'visual_source', None):
        visual = Path(args.visual_source).resolve()
        require(visual.is_relative_to(repo / 'scripts'), 'Visual sources must be within scripts/')
        for source in sorted(visual.rglob('*')):
            if source.is_file() and source.suffix in {'.py', '.json', '.md', '.pdf'}:
                add(source)

    for name in DOCS:
        path = repo / "docs/research" / name
        if path.is_dir():
            tree(path)
        elif path.is_file():
            add(path)
    for name in CONFIGS:
        add(repo / "configs" / name)
    # Refresh only explicitly needed top-level implementation closures. Unrelated
    # private submission metadata is never globbed into the export.
    for directory in ("infogeo", "experiments", "analysis"):
        for source in sorted((repo / directory).glob("*.py")):
            add(source)
    for pattern in ("ndfa_bn_*protocol_20260915.md", "ndfa_bn_*amendment_20260915.md"):
        for source in sorted((repo / "docs/research").glob(pattern)):
            add(source)
    paper_manifest = json.loads((paper / "manifest.json").read_text())
    for name, expected in paper_manifest["file_sha256"].items():
        source = paper / previous.checked_relative(name)
        require(previous.sha256(source) == expected, "Changed paper closure: " + name)
        add(source, "paper/" + name)
    add(paper / "manifest.json", "paper/SOURCE_MANIFEST.json")
    add(repo / "drafts/Info-DFA/scripts/make_iclr_figures.py", "paper/scripts/make_iclr_figures.py")
    add(paper.parent / "conditioned_dfa_iclr.pdf", "paper/conditioned_dfa_iclr.pdf")
    require(previous.sha256(paper.parent / "conditioned_dfa_iclr.pdf") == paper_manifest["checked_pdf_sha256"], "Wrong reviewed paper PDF")
    request_args = argparse.Namespace(mechanism_root="results/ndfa_factor_mechanism_20260915",
        mechanism_source="results/ndfa_factor_mechanism_snapshot_20260915/source_snapshot",
        mechanism_report="docs/research/ndfa_factor_mechanism_20260915.json",
        prediction_root="results/ndfa_submission_confirmation_evaluation_20260915",
        prediction_audit="docs/research/ndfa_submission_confirmation_evaluation_20260915/audit.json")
    request = previous.export_evidence(request_args, repo, package, records, add)
    provenance_paths = request.pop("provenance_paths")
    generated("EVIDENCE_REQUEST.json", json.dumps(request, indent=2) + "\n")
    generated("EVIDENCE_PROVENANCE.json", json.dumps({name: records[name] for name in provenance_paths}, indent=2) + "\n")
    add(repo / "scripts/reproduce_ndfa_completion.py", "reproduce.py")
    add(repo / "scripts/verify_ndfa_latest_results.py", "verify_latest_results.py")
    generated("verify_tuned_bp.py", previous.TUNED_BP_SCRIPT)
    notice, original_notice = anonymous_license((repo / "LICENSE").read_text())
    generated("LICENSE", notice)
    records["LICENSE"]["original_sha256"] = previous.sha256(repo / "LICENSE")
    records["LICENSE"]["transformations"] = ["authorized first-party review-copy copyright identifier blinding; all remaining MIT text verbatim"]
    generated("OMITTED_LARGE_ARTIFACTS.json", json.dumps(omissions, indent=2) + "\n")
    add(base / "ARTIFACT_INVENTORY.json", "HISTORICAL_ARTIFACT_INVENTORY.json")
    evidence = {"baseline": {"root": "results/ndfa_bn_confirmation_20260915", "summary": "docs/research/ndfa_bn_confirmation_20260915/summary.json"},
                "factors": {"root": "results/ndfa_bn_factors_h100_234min_20260915"},
                "optimizer": {"root": "results/ndfa_bn_optimizer_60min_20260915"},
                "status": "preparation_candidate", "pending_final_bindings": ["final completion manuscript", "prospective FD comparison", "locked final test inference"]}
    if getattr(args,'final',False):
        final_summary=json.loads((repo/'docs/research/ndfa_final_test_20260915/summary.json').read_text())
        require(final_summary['accepted'] and final_summary['models']==280,'Final package requires the whole completed test population')
        evidence.update(status='complete_review_candidate',pending_final_bindings=[],
                        forward={'root':'results/ndfa_bn_forward_decorrelation_20260915'},
                        final_test={'root':'results/ndfa_final_test_20260915','summary':'docs/research/ndfa_final_test_20260915/summary.json','plan':'results/ndfa_final_test_snapshot_20260915/plan.json'})
    generated("COMPLETION_EVIDENCE.json", json.dumps(evidence, indent=2) + "\n")
    readme=README
    if getattr(args,'final',False):
        readme=readme.replace('This preparation copy contains the manuscript dependency closure, historical evidence, and the completed BN seed-confirmation, factor-transfer and optimized-work cohorts. The final manuscript, prospective forward-decorrelation comparison and locked final test evaluation are still pending in COMPLETION_EVIDENCE.json. It is not yet a submission-ready final archive.',
                             'This complete review copy contains the final manuscript dependency closure, historical evidence, completed BN scale/factor/optimizer/forward-decorrelation cohorts, and the locked final 280-model CIFAR-10 test evaluation. COMPLETION_EVIDENCE.json identifies every final cohort. No external upload is implied.')
        readme=readme.replace('The default statistics use only training-pool validation predictions.',
                             'The default statistics also reconstruct all 108 forward-decorrelation study outcomes and all 280 final test outcomes, 45 pairwise comparisons and both prespecified Holm-adjusted accuracy tests from raw saved logits. The official CIFAR-10 test was used in earlier project studies; this is not a previously unseen dataset.')
    generated("requirements-review.txt", "# Tested offline review environment; CPU execution is sufficient.\n" +
              "numpy==2.2.6\nscipy==1.15.3\ntorch==2.9.1\npandas==2.3.3\npytest==9.0.2\nmatplotlib==3.10.6\n")
    generated("README.md", readme)
    manifest = {"schema_version": 2, "release_status": "anonymous_review_candidate" if getattr(args,'final',False) else "anonymous_review_preparation_candidate",
                "files": records, "original_hashes_are_not_export_hashes": True,
                "numerical_artifact_policy": "All included predictions/arrays remain byte-identical; text paths/host metadata may be blinded."}
    write(package / "PACKAGE_MANIFEST.json", manifest)
    # This private receipt intentionally remains OUTSIDE the review directory.
    write(output / "private_build_receipt.json", {"builder_sha256": previous.sha256(Path(__file__)),
        "historical_base_manifest_sha256": previous.sha256(base / "PACKAGE_MANIFEST.json"),
        "paper_manifest_sha256": previous.sha256(paper / "manifest.json"),
        "license": {"canonical_sha256": previous.sha256(repo / "LICENSE"), "original_line": original_notice,
                    "anonymous_line": ANONYMOUS_NOTICE, "authorization": "Explicit user authorization to finish anonymous review export, with root's specific first-party MIT identifier-blinding instruction; restore full attribution for public release", "canonical_unchanged": previous.sha256(repo / "LICENSE") == records["LICENSE"]["original_sha256"]},
        "omitted_large_artifacts": len(omissions), "files": len(records), "status": "awaiting verification and final evidence bindings"})
    print(json.dumps({"package": str(package), "files": len(records), "omitted_large_artifacts": len(omissions)}), flush=True)
    return package


README = """# Anonymous review reproduction bundle

This preparation copy contains the manuscript dependency closure, historical evidence, and the completed BN seed-confirmation, factor-transfer and optimized-work cohorts. The final manuscript, prospective forward-decorrelation comparison and locked final test evaluation are still pending in COMPLETION_EVIDENCE.json. It is not yet a submission-ready final archive.

From the extracted review_package directory, use Python 3.10 or later with numpy, scipy and torch, then run:

    python -B reproduce.py

The default CPU-only command checks every included file and independently reconstructs all 96 baseline, 144 factor and 45 optimizer case endpoints from saved validation logits, histories, paired means, sample SDs and confidence intervals. It also checks the recorded selection rules. It loads no dataset, trains no model, and writes no output into the package. Optional checks are:

    python -B reproduce.py --checks integrity,statistics,legacy,mechanism,paper,smoke --output ../new_reproduction_result

The mechanism option also checks the earlier 50-model saved official-test prediction cohort. These are previously disclosed test results, not new test access. Full mechanism reconstruction uses complete included arrays and may take several minutes. Paper reconstruction needs pdflatex and pdftotext. Legacy checks additionally need pandas; smoke checks need pytest. The tested Python package versions are in requirements-review.txt. The default statistics use only training-pool validation predictions.

PACKAGE_MANIFEST.json separates original SHA-256 hashes from exported hashes. Source text may contain portable-path and execution-site substitutions; numeric arrays and prediction tensors are unchanged. Embedded historical hash strings continue to denote ORIGINAL artifacts. Use the export manifest to check distributed bytes. Large checkpoints and optimizer states are omitted and listed in OMITTED_LARGE_ARTIFACTS.json; their producer-recorded hashes remain in the included receipts. The offline verifier does not claim to reconstruct predictions by forwarding omitted checkpoints. This is not an exact training-resume package.

Public datasets are not redistributed. CIFAR-10/CIFAR-100 are available from https://www.cs.toronto.edu/~kriz/cifar.html; MNIST from https://yann.lecun.com/exdb/mnist/ and Fashion-MNIST from https://github.com/zalandoresearch/fashion-mnist. Dataset loader and frozen configuration files specify training-file checksums, splits and preprocessing. Training scripts and source snapshots are included for inspection/reproduction; running them on another machine requires setting local dataset/output/interpreter paths and producing new configuration/source pins. Historical cluster launchers are records, not portable execution promises. Figure-generation sources and scalar provenance are included under scripts/iclr_completion_20260915; their archived paths and original-hash guards need adaptation to an exported working copy. The portable paper check rebuilds the reviewed PDF from its included vector figures. ImageNet requires its own official access and license.

The first-party MIT notice is blinded for anonymous review with the copyright holders' authorization; its permission, conditions and warranty are unchanged. Third-party references and notices are retained. This does not change licensing or scientific authorship. Public release must restore full first-party attribution.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base", required=True)
    parser.add_argument("--paper-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument('--final',action='store_true',help='Require and bind completed FD and all 280 locked final-test models')
    parser.add_argument('--visual-source', help='Optional explicit visual-revision source and original-vector closure')
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
