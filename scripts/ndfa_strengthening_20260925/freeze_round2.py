"""Freeze bounded follow-ups to a completed validation-development grid."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neighbors(values,value):
    values=sorted(set(values));i=values.index(value)
    return values[max(0,i-1):i+2]


def extension_cases(config,summary):
    """Expand each selected boundary 3x and 10x, crossing nearby old values.

    Same trigger and expansion rule for every cell/family. Existing points are
    excluded. A family with two boundary coordinates adds at most 12 candidates.
    """
    assert summary["complete"]
    cases=[];ledger=[]
    for key,result in sorted(summary["selections"].items()):
        flags=result.get("boundary_flags",{})
        if not flags:continue
        winner=result["selected"]
        previous=[c for c in config["cases"] if c["cell"]==winner["cell"] and c["family"]==winner["family"] and c["optimizer"]==winner["optimizer"]]
        rates=neighbors([c["lr"] for c in previous],winner["lr"])
        dampings=neighbors([c["damping"] for c in previous],winner["damping"]) if winner["damping"] is not None else [None]
        for name,values in [("lr",rates),("damping",dampings)]:
            if name in flags:
                edge=flags[name]
                factors=[1/3,1/10] if edge["selected"]==edge["range"][0] else [3,10]
                values.extend(float(f'{edge["selected"]*factor:.12g}') for factor in factors)
        old={(c["lr"],c["damping"]) for c in previous}
        added=0
        for lr,damping in itertools.product(sorted(set(rates)),sorted(set(dampings)) if dampings!=[None] else [None]):
            if (lr,damping) in old:continue
            case=copy.deepcopy(winner)
            case.update(id=f"case_{len(config['cases'])+len(cases):03d}",lr=lr,damping=damping)
            cases.append(case);added+=1
        ledger.append(dict(group=key,parent_case=winner["id"],boundaries=flags,new_candidates=added))
    return cases,ledger


def horizon_cases(config,summary):
    assert summary["complete"]
    cases=[]
    for _,result in sorted(summary["selections"].items()):
        assert result["selected"] is not None
        case=copy.deepcopy(result["selected"]);case["id"]=f"case_{len(cases):03d}"
        cases.append(case)
    return cases


def snapshot_config(kind,parent,summary_path):
    parent=Path(parent).resolve();summary_path=Path(summary_path).resolve()
    config=json.loads(parent.read_text());summary=json.loads(summary_path.read_text())
    assert summary["complete"] and summary["config_sha256"]==sha(parent)
    for name,expected in config["source_sha256"].items():assert sha(parent.parent/"source"/name)==expected,name
    out=ROOT/f"results/ndfa_strengthening_20260925/synthetic_{kind}_v1"
    assert not out.exists(),f"Already frozen: {out}"
    out.mkdir();shutil.copytree(parent.parent/"source",out/"source")
    shutil.copy2(summary_path,out/"parent_summary.json")
    shutil.copy2(__file__,out/"freeze_round2.py")
    result=copy.deepcopy(config)
    result.update(frozen_utc=datetime.now(timezone.utc).isoformat(),output_root=str(out/"runs"),
                  parent_config_sha256=sha(parent),parent_summary_sha256=sha(summary_path),
                  parent_config=str(parent),followup_kind=kind,official_test_loaded=False)
    if kind=="boundaries":
        cases,ledger=extension_cases(config,summary)
        result.update(cases=cases,extension_ledger=ledger,
                      extension_rule="Winner's optimizer; extend each boundary 3x and 10x; cross adjacent old values; exclude all already-run points; identical policy across families",
                      selection="Pool with the original grid, lowest mean final validation CE over the same two development seeds; accuracy then unique case ID break ties",
                      scope="Validation-only boundary extension; no confirmation claim")
    elif kind=="horizon":
        result.update(cases=horizon_cases(config,summary),epochs=2*config["epochs"],
                      selection="Fixed first-grid winner in every cell/family; report paired 500-vs-1000 epoch outcomes, not a new hyperparameter search",
                      scope="Validation-only horizon sensitivity at provisional first-grid settings; restart from paired initialization with stretched cosine schedule, not checkpoint continuation or evidence of convergence")
    else:raise ValueError(kind)
    path=out/"config.json";path.write_text(json.dumps(result,indent=2)+"\n")
    return path


def submit(config_path,partition,cap):
    config_path=Path(config_path).resolve();out=config_path.parent
    receipt_path=out/"submission.json"
    if receipt_path.exists():return json.loads(receipt_path.read_text())
    cfg=json.loads(config_path.read_text());count=len(cfg["cases"])
    assert count>0
    env=os.environ.copy();env.update(NDFA_NEXT_SOURCE=str(out/"source"),NDFA_NEXT_CONFIG=str(config_path),NDFA_NEXT_CONFIG_SHA256=sha(config_path))
    command=["sbatch","--parsable",f"--partition={partition}","--constraint=h200",f"--array=0-{count-1}%{cap}",
             str(out/"source/slurm/ndfa_strengthening_next_20260925.sbatch")]
    response=subprocess.run(command,env=env,text=True,capture_output=True)
    if response.returncode:raise RuntimeError(response.stderr)
    receipt=dict(job=response.stdout.strip(),kind=cfg["followup_kind"],cases=count,
                 training_runs=count*len(cfg["development_seeds"]),config_sha256=sha(config_path),
                 partition=partition,hardware="H200",concurrency=cap,command=command,
                 parent_summary_sha256=cfg["parent_summary_sha256"],submitted_utc=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=["boundaries","horizon"],required=True)
    p.add_argument("--parent",type=Path,required=True);p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--partition",choices=["kempner_eng","kempner_requeue"],default="kempner_eng")
    p.add_argument("--cap",type=int,default=12);p.add_argument("--submit",action="store_true")
    args=p.parse_args();assert args.cap>0
    config=snapshot_config(args.kind,args.parent,args.summary)
    if args.submit:print(json.dumps(submit(config,args.partition,args.cap)),flush=True)
    else:print(config)
