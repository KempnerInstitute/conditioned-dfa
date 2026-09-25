"""Freeze and submit CIFAR boundary and horizon follow-ups on H100."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.ndfa_strengthening_20260925.freeze_round2 import extension_cases,horizon_cases,sha


def freeze(study,kind):
    parent=ROOT/f"results/ndfa_strengthening_20260925/{study}_development_v1"
    config_path=parent/"config.json";summary_path=parent/"development_summary.json"
    cfg=json.loads(config_path.read_text());summary=json.loads(summary_path.read_text())
    assert summary["complete"] and summary["config_sha256"]==sha(config_path)
    for rel,digest in cfg["source_sha256"].items():assert sha(parent/"source"/rel)==digest
    label="baseline" if study=="baseline" else "foof"
    out=ROOT/f"results/ndfa_strengthening_20260925/cifar_{label}_{kind}_v1"
    if (out/"submission.json").exists():return json.loads((out/"submission.json").read_text())
    out.mkdir(exist_ok=False);shutil.copytree(parent/"source",out/"source")
    shutil.copy2(summary_path,out/"parent_summary.json")
    shutil.copy2(__file__,out/"freeze_cifar_followups.py")
    if kind=="boundaries":
        cases,ledger=extension_cases(cfg,summary)
        scope="Validation-only boundary extension pooled with parent; identical outward 3x/10x and neighboring-coordinate rule for every family"
    else:
        cases=horizon_cases(cfg,summary);ledger=[]
        scope="Fixed first-grid winners restarted at twice the epoch horizon with stretched schedule; provisional horizon sensitivity, not a convergence claim"
    frozen=copy.deepcopy(cfg)
    frozen.update(cases=cases,epochs=cfg["epochs"]*(2 if kind=="horizon" else 1),
                  frozen_utc=datetime.now(timezone.utc).isoformat(),output_root=str(out/"runs"),
                  parent_config=str(config_path),parent_config_sha256=sha(config_path),parent_summary_sha256=sha(summary_path),
                  followup_kind=kind,extension_ledger=ledger,scope=scope,
                  selection="Pool original and new candidates by mean final validation CE; accuracy then case ID tie break" if kind=="boundaries" else "Fixed first-grid winner in every family; paired descriptive horizon contrasts")
    path=out/"config.json";path.write_text(json.dumps(frozen,indent=2)+"\n")
    dispatch=out/"dispatch";dispatch.mkdir()
    env=os.environ.copy()
    if study=="baseline":
        wrapper=dispatch/"restart_case.py";shutil.copy2(ROOT/"scripts/ndfa_strengthening_20260925/restart_case.py",wrapper)
        batch=dispatch/"run.sbatch";shutil.copy2(ROOT/"slurm/ndfa_strengthening_restart_20260925.sbatch",batch)
        env.update(NDFA_STRENGTH_SOURCE=str(out/"source"),NDFA_STRENGTH_CONFIG=str(path),NDFA_STRENGTH_CONFIG_SHA256=sha(path),NDFA_RESTART_WRAPPER=str(wrapper))
    else:
        batch=out/"source/slurm/ndfa_strengthening_next_20260925.sbatch"
        env.update(NDFA_NEXT_SOURCE=str(out/"source"),NDFA_NEXT_CONFIG=str(path),NDFA_NEXT_CONFIG_SHA256=sha(path))
    command=["sbatch","--parsable","--partition=kempner_requeue","--constraint=h100","--time=01:00:00",f"--array=0-{len(cases)-1}%60",str(batch)]
    r=subprocess.run(command,env=env,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr)
    receipt=dict(job=r.stdout.strip(),study=study,kind=kind,cases=len(cases),training_runs=2*len(cases),config_sha256=sha(path),partition="kempner_requeue",hardware="H100",concurrency=60,command=command)
    (out/"submission.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--study",choices=["baseline","foof_cifar"],required=True);p.add_argument("--kind",choices=["boundaries","horizon"],required=True)
    args=p.parse_args();print(json.dumps(freeze(args.study,args.kind)),flush=True)
