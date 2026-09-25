"""Run an unchanged short development case safely after Slurm requeue.

Completed seeds are verified and skipped. An interrupted seed without an
endpoint is preserved as a separate attempt and restarted from its fixed
initialization. This wrapper does not claim to resume within a seed.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, case, seed):
    output=Path(output)
    endpoint=output/"endpoint.json"
    if endpoint.exists():
        saved=json.loads(endpoint.read_text())
        assert saved["case"]==case and saved["seed"]==seed
        assert saved["status"] in {"complete","numerical_failure"}
        assert sha(output/"manifest.json")==saved["manifest_sha256"]
        assert sha(output/"final.pt")==saved["final_checkpoint_sha256"]
        return saved
    if output.exists():
        manifest=output/"manifest.json"
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            assert saved["case"]==case and saved["args"]["model_seed"]==seed
        archive=output.parent/"interrupted_attempts"
        archive.mkdir(exist_ok=True)
        attempt=sum(p.name.startswith(output.name+".") for p in archive.iterdir())
        if attempt>=5:
            raise RuntimeError("Five interrupted attempts retained; inspect infrastructure before retrying")
        dest=archive/f"{output.name}.{attempt:02d}"
        output.rename(dest)
        (dest/"interruption_record.json").write_text(json.dumps({
            "action":"restart this seed from its fixed initialization",
            "recorded_utc":datetime.now(timezone.utc).isoformat(),
            "resuming_job":os.environ.get("SLURM_JOB_ID"),
            "reason":"incomplete prior attempt; no scientific endpoint",
        },indent=2)+"\n")
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--config-sha256",required=True)
    p.add_argument("--case-index",type=int,default=int(os.environ.get("SLURM_ARRAY_TASK_ID","0")))
    args=p.parse_args(); assert sha(args.config)==args.config_sha256
    config=json.loads(args.config.read_text())
    for name,digest in config["source_sha256"].items():
        assert sha(args.source/name)==digest,name
    sys.path.insert(0,str(args.source))
    path=args.source/"scripts/ndfa_strengthening_20260925/baseline_development.py"
    spec=importlib.util.spec_from_file_location("frozen_baseline_development",path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    case=config["cases"][args.case_index]
    parent=Path(config["output_root"])/case["id"];parent.mkdir(parents=True,exist_ok=True)
    with (parent/".execution.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in config["development_seeds"]:
            output=parent/f"seed_{seed}"
            result=prepare(output,case,seed)
            if result is None:
                result=runner.train(config,case,seed,output)
            print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
