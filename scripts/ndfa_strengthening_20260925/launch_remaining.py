"""Queue a frozen development grid behind complete, validated GPU probes.

The gate checks execution and provenance only, never an accuracy threshold.
It does not select hyperparameters or evaluate a test set.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def verify(config_path):
    config_path=Path(config_path)
    config=json.loads(config_path.read_text())
    receipt=json.loads((config_path.parent/"submission.json").read_text())
    assert sha(config_path)==receipt["config_sha256"],"Changed configuration"
    for rel,expected in config["source_sha256"].items():
        assert sha(config_path.parent/"source"/rel)==expected,rel
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    verified=[]
    for index in receipt["verification_indices"]:
        case=config["cases"][index]
        for seed in config["development_seeds"]:
            run=Path(config["output_root"])/case["id"]/f"seed_{seed}"
            endpoint=json.loads((run/"endpoint.json").read_text())
            manifest=json.loads((run/"manifest.json").read_text())
            history=json.loads((run/"history.json").read_text())
            assert endpoint["case"]==case and endpoint["seed"]==seed
            assert manifest["case"]==case and manifest["seed"]==seed
            assert manifest["protocol"]==config
            assert endpoint["config_hash"]==manifest["config_hash"]==config_hash
            assert endpoint["status"]=="complete","GPU verification did not finish normally"
            assert endpoint["official_test_loaded"] is False
            assert manifest["official_test_loaded"] is False
            examples=config["cells"][case["cell"]]["n_train"] if config["dataset"]=="synthetic" else 45000
            assert endpoint["completed_updates"]==math.ceil(examples/config["batch_size"])*config["epochs"]
            assert endpoint["epoch"]==history[-1]["epoch"]==config["epochs"]
            assert endpoint["final_validation"]==history[-1]["validation"]
            assert math.isfinite(endpoint["final_validation"]["loss"])
            assert sha(run/"final.pt")==endpoint["checkpoint_sha256"]
            verified.append(dict(case=case["id"],seed=seed,training_seconds=endpoint["training_seconds"]))
    return dict(config_sha256=sha(config_path),verified=verified)


def submit(config_path,partition,constraint,cap):
    config_path=Path(config_path).resolve();out=config_path.parent
    receipt_path=out/"submission.json"
    receipt=json.loads(receipt_path.read_text())
    if "remaining_job" in receipt:
        print(json.dumps(receipt),flush=True)
        return
    assert sha(config_path)==receipt["config_sha256"]
    config=json.loads(config_path.read_text())
    indices=[i for i in range(len(config["cases"])) if i not in receipt["verification_indices"]]
    dispatch=out/"remaining_dispatch_v1";dispatch.mkdir(exist_ok=False)
    gate=dispatch/"launch_remaining.py";shutil.copy2(__file__,gate)
    wrapper=dispatch/"dispatch.sbatch"
    wrapper.write_text('''#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=ndfa_strength_next
#SBATCH --output=logs/ndfa_strength_next_%A_%a.out
#SBATCH --error=logs/ndfa_strength_next_%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NDFA_NEXT_SOURCE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/n/sw/Mambaforge-23.11.0-0/bin/python "$NDFA_NEXT_GATE" --config "$NDFA_NEXT_CONFIG" --check
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u "$NDFA_NEXT_SOURCE/scripts/ndfa_strengthening_20260925/next_round.py" --config "$NDFA_NEXT_CONFIG" --config-sha256 "$NDFA_NEXT_CONFIG_SHA256"
''')
    env=os.environ.copy()
    env.update(NDFA_NEXT_SOURCE=str(out/"source"),NDFA_NEXT_CONFIG=str(config_path),
               NDFA_NEXT_CONFIG_SHA256=sha(config_path),NDFA_NEXT_GATE=str(gate))
    command=["sbatch","--parsable",f"--partition={partition}",f"--constraint={constraint}",
             f"--dependency=afterok:{receipt['verification_job']}",
             "--array="+','.join(map(str,indices))+f"%{cap}",str(wrapper)]
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr)
    receipt.update(remaining_job=result.stdout.strip(),remaining_array=indices,
                   remaining_partition=partition,remaining_hardware=constraint,remaining_concurrency=cap,
                   remaining_submitted_utc=datetime.now(timezone.utc).isoformat(),
                   dispatch_gate_sha256=sha(gate),dispatch_wrapper_sha256=sha(wrapper),
                   dependency=f"afterok:{receipt['verification_job']}",
                   gate="Full expected GPU-probe updates, paired seeds, finite endpoint, configuration/source/checkpoint hashes; no performance threshold",
                   remaining_command=command)
    receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in {"remaining_array","remaining_command"}}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--partition",choices=["kempner_eng","kempner_requeue"])
    parser.add_argument("--constraint",choices=["h100","h200"])
    parser.add_argument("--cap",type=int,default=4)
    args=parser.parse_args()
    if args.check:print(json.dumps(verify(args.config)),flush=True)
    else:
        assert args.partition and args.constraint and args.cap>0
        submit(args.config,args.partition,args.constraint,args.cap)
