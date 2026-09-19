"""One bounded sequential allocation; this stdlib parent never imports CUDA/torch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from common import read_config, require, resolve_case, runtime_projection, selected_candidates, sha256, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config, args.config_sha256)
    source = Path(config["source_root"])
    child = source/"scripts/ndfa_bn_forward_decorrelation_20260915/case.py"
    command = [config["execution"]["python"], "-u", str(child), "--config", args.config,
               "--config-sha256", args.config_sha256]
    if args.check_only:
        for path, digest in config["training_files_sha256"].items():
            require(sha256(path) == digest, "changed training file")
        result = subprocess.run([*command,"--check-only"], env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),
                                check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        return
    require(os.environ.get("SLURM_JOB_ID"), "training workflow requires the declared Slurm allocation")
    require(os.environ.get("SLURM_JOB_PARTITION") == config["execution"]["partition"], "wrong partition")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "4", "wrong CPU allocation")
    scheduler = subprocess.run(["scontrol","show","job","-o",os.environ["SLURM_JOB_ID"]],
                               env=dict(os.environ,TZ="UTC"),capture_output=True,text=True,check=True).stdout
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)",scheduler))
    require(fields.get("Account") == config["execution"]["account"] and fields.get("QOS") == config["execution"]["qos"], "wrong account/QOS")
    require(fields.get("TimeLimit") == "01:50:00", "wrong Slurm time limit")
    end = datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=timezone.utc)
    maximum_runtime = min(config["execution"]["maximum_seconds"], (end-datetime.now(timezone.utc)).total_seconds())
    require(maximum_runtime > 0, "Slurm allocation has no remaining time")
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    workflow = {"config_sha256":args.config_sha256, "source_sha256":config["source_sha256"],
                "job_id":os.environ["SLURM_JOB_ID"], "started_utc":now(), "status":"BENCHMARK_RUNNING",
                "scheduler_receipt":scheduler,"remaining_allocation_seconds_at_start":maximum_runtime,
                "official_test_evaluation":False, "parent_cuda_initialized":False,
                "cases":[dict(case,stage=stage,status="planned") for stage in ("benchmark","development","confirmation")
                         for case in config[stage]]}

    def save():
        workflow["updated_utc"] = now()
        workflow["elapsed_wall_seconds"] = time.monotonic()-started
        write_json(output/"workflow.json",workflow)

    def run(entry, selection_path=None):
        selection = json.loads(selection_path.read_text()) if selection_path else None
        resolved = resolve_case(config,entry["case_id"],selection)
        entry.update(resolved,status="running",started_utc=now())
        save()
        argv = [*command,"--case-id",entry["case_id"]]
        if selection_path:
            argv += ["--selection",str(selection_path),"--selection-sha256",sha256(selection_path)]
        entry["command"] = argv
        remaining = maximum_runtime-(time.monotonic()-started)-config["runtime_gate"]["audit_and_shutdown_reserve_seconds"]
        require(remaining > 0,"allocation time exhausted")
        entry["timeout_seconds"] = remaining
        save()
        began = time.monotonic()
        directory = output/entry["case_id"]
        try:
            with (output/(entry["case_id"]+".stdout.txt")).open("x") as stdout, (output/(entry["case_id"]+".stderr.txt")).open("x") as stderr:
                result = subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=remaining,
                                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
            entry["returncode"] = result.returncode
            receipt = json.loads((directory/"receipt.json").read_text())
        except Exception as error:
            entry.update(status="infrastructure_failed",error=repr(error),invocation_wall_seconds=time.monotonic()-began,
                         artifacts_sha256={p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file()} if directory.exists() else {})
            save()
            raise
        entry["invocation_wall_seconds"] = time.monotonic()-began
        entry["status"] = receipt["status"]
        if result.returncode == 0:
            require(entry["status"] == "complete", "successful child without complete receipt")
            entry.update({key:receipt[key] for key in ("endpoint","evaluation_seconds","validation_count","work_overshoot_seconds")})
        else:
            entry["error"] = receipt.get("error", "child failed")
        entry["artifacts_sha256"] = {str(p.relative_to(directory)):sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
        save()
        require(entry["status"] in ("complete","numerical_failed"), "infrastructure failure; no retry")

    save()
    try:
        for path,digest in config["training_files_sha256"].items():
            require(sha256(path) == digest, "changed training file")
        for entry in workflow["cases"][:4]:
            run(entry)
        gate = runtime_projection(config,workflow["cases"][:4],time.monotonic()-started,maximum_runtime)
        write_json(output/"runtime_gate.json",gate)
        workflow["runtime_gate"] = gate
        require(gate["accepted"], "RESOURCE_GATE_STOPPED: fixed full cohort does not fit allocation")
        projection = gate["projected_case_seconds"]
        for stage in ("development","confirmation"):
            if stage == "confirmation":
                development = [e for e in workflow["cases"] if e["stage"] == "development"]
                selected,scores = selected_candidates(config,development)
                selection = {"config_sha256":args.config_sha256,"created_utc":now(),"development_complete":True,
                             "criterion":config["selection"]["criterion"],
                             "selected":selected,"all_candidate_scores":scores,
                             "development_artifact_hashes":{e["case_id"]:e["artifacts_sha256"] for e in development},
                             "confirmation_started":False,"test_evaluated":False}
                selection_path = output/"selection.json"
                require(not selection_path.exists(),"selection already exists")
                write_json(selection_path,selection)
                selection_path.chmod(0o444)
                workflow["selection_sha256"] = sha256(selection_path)
            workflow["status"] = stage.upper()+"_RUNNING"
            save()
            for entry in [e for e in workflow["cases"] if e["stage"] == stage]:
                remaining = sum(e["status"] == "planned" and e["stage"] != "benchmark" for e in workflow["cases"])
                projected_finish = time.monotonic()-started + remaining*projection + config["runtime_gate"]["audit_and_shutdown_reserve_seconds"]
                require(projected_finish <= maximum_runtime, "RESOURCE_GATE_STOPPED: remaining full cohort no longer fits")
                run(entry,selection_path if stage == "confirmation" else None)
        workflow["status"] = "TRAINING_COMPLETE"
        save()
        cpu_env = dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",MPLBACKEND="Agg")
        with (output/"audit.stdout.txt").open("x") as stdout, (output/"audit.stderr.txt").open("x") as stderr:
            subprocess.run([*command,"--audit-all"],env=cpu_env,stdout=stdout,stderr=stderr,check=True,timeout=240)
        workflow["status"] = "AUDITED"
        save()
        analyzer = source/"scripts/ndfa_bn_forward_decorrelation_20260915/report.py"
        with (output/"analysis.stdout.txt").open("x") as stdout, (output/"analysis.stderr.txt").open("x") as stderr:
            subprocess.run([config["execution"]["python"],str(analyzer),"--config",args.config,"--config-sha256",args.config_sha256],
                           env=cpu_env,stdout=stdout,stderr=stderr,check=True,timeout=240)
        workflow["status"] = "COMPLETE" if json.loads((output/"audit.json").read_text())["accepted"] else "INCOMPLETE_CONFIRMATION"
        workflow["finished_utc"] = now()
        save()
    except Exception as error:
        workflow["status"] = "INCOMPLETE"
        workflow["error"] = repr(error)
        workflow["finished_utc"] = now()
        save()
        raise


if __name__ == "__main__":
    main()
