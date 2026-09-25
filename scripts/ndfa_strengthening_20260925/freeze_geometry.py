"""Freeze an exploratory synthetic mean/covariance control study."""
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
from scripts.ndfa_strengthening_20260925.freeze_round2 import sha

STATISTICS=("full","diagonal","centered","diagonal_covariance_plus_mean","isotropic_covariance_plus_mean")


def design(summary):
    assert summary["complete"]
    cases=[];anchors=[];verification=[]
    for cell in ["nuisance","task_aligned"]:
        for norm in ["none","bn"]:
            parent=summary["selections"][f"{cell}/dfa_{norm}_activity"]["selected"]
            assert parent is not None
            anchors.append(dict(cell=cell,normalization=norm,parent_case=parent))
            for statistic in STATISTICS:
                for multiplier in [1/3,1.,3.]:
                    c=copy.deepcopy(parent)
                    c.update(id=f"case_{len(cases):03d}",family=f"dfa_{norm}_{statistic}",operator="activity_geometry",statistic=statistic,
                             damping=parent["damping"]*multiplier,anchor_case=parent["id"])
                    if cell=="nuisance" and norm=="bn" and multiplier==1.:verification.append(len(cases))
                    cases.append(c)
            for operator,label in [("none","raw_matched"),("activity","batchspace_bridge")]:
                c=copy.deepcopy(parent)
                c.update(id=f"case_{len(cases):03d}",family=f"dfa_{norm}_{label}",operator=operator,
                         damping=None if operator=="none" else parent["damping"],anchor_case=parent["id"])
                cases.append(c)
    return cases,anchors,verification


def freeze(parent,summary_path):
    parent=Path(parent).resolve();summary_path=Path(summary_path).resolve()
    cfg=json.loads(parent.read_text());summary=json.loads(summary_path.read_text())
    assert summary["complete"] and summary["config_sha256"]==sha(parent)
    for rel,digest in cfg["source_sha256"].items():assert sha(parent.parent/"source"/rel)==digest
    out=ROOT/"results/ndfa_strengthening_20260925/synthetic_geometry_v1"
    out.mkdir(exist_ok=False);shutil.copytree(parent.parent/"source",out/"source")
    shutil.copy2(summary_path,out/"parent_summary.json")
    files=["infogeo/activity_geometry.py","scripts/ndfa_strengthening_20260925/next_round.py",
           "scripts/ndfa_strengthening_20260925/freeze_geometry.py","scripts/ndfa_strengthening_20260925/freeze_round2.py",
           "tests/test_activity_geometry.py","tests/test_round2_design.py","slurm/ndfa_strengthening_next_20260925.sbatch"]
    for rel in files:
        dest=out/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        cfg["source_sha256"][rel]=sha(dest)
    cases,anchors,indices=design(summary)
    cfg.update(frozen_utc=datetime.now(timezone.utc).isoformat(),output_root=str(out/"runs"),cases=cases,
               parent_config_sha256=sha(parent),parent_summary_sha256=sha(summary_path),anchors=anchors,
               stage="exploratory_mechanism_development",geometry_probe=True,
               selection="Within each condition/statistic, lowest mean final validation CE over both seeds; accuracy then case ID break ties; matched-damping contrasts also retained",
               scope="Provisional same-recipe controls anchored to first-grid activity winners; not a confirmation or best-recipe comparison. Baseline boundary tuning proceeds independently; final mechanism settings require that review.",
               activity_scope="Five dense current-minibatch moment variants, same raw numerator, hidden weights only, absolute damping, own-raw norm matching; raw biases and BN-affine updates",
               control_matrices=dict(full="E[hh^T]",diagonal="diag(E[hh^T])",centered="E[(h-mu)(h-mu)^T]",
                                     diagonal_covariance_plus_mean="diag(Cov(h)) + mu mu^T",isotropic_covariance_plus_mean="tr(Cov(h))/d I + mu mu^T"),
               reference_scope="raw_matched shares the activity anchor's optimizer/rate; batchspace_bridge uses the original full-moment implementation. Neither is a newly tuned raw-DFA comparator.",
               geometry_probe_scope="Fixed training-mode minibatch from an independent RNG, no labels; restore BN running statistics and forward cache. Mean energy, covariance participation rank, and off-diagonal covariance energy are descriptive.",
               timing="All five geometry variants use dense feature-space solves; diagnostic probes excluded with validation. This prototype is for mechanism, not optimized throughput.")
    path=out/"config.json";path.write_text(json.dumps(cfg,indent=2)+"\n")
    env=os.environ.copy();env.update(NDFA_NEXT_SOURCE=str(out/"source"),NDFA_NEXT_CONFIG=str(path),NDFA_NEXT_CONFIG_SHA256=sha(path))
    command=["sbatch","--parsable","--partition=kempner_requeue","--constraint=h200","--array="+','.join(map(str,indices))+"%60",
             str(out/"source/slurm/ndfa_strengthening_next_20260925.sbatch")]
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr)
    receipt=dict(kind="synthetic_geometry",verification_job=result.stdout.strip(),verification_indices=indices,
                 config_sha256=sha(path),cases=len(cases),training_runs=2*len(cases),remaining_array="pending GPU verification",
                 command=command,partitions=["kempner_requeue"],hardware="H200",concurrency=60)
    (out/"submission.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",type=Path,required=True);p.add_argument("--summary",type=Path,required=True)
    args=p.parse_args();freeze(args.parent,args.summary)
