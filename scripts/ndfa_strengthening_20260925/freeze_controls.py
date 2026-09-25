"""Freeze independent digit baselines and causal conditioning schedules."""
import argparse
import copy
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.ndfa_strengthening_20260925.freeze_round2 import sha


def freeze(kind):
    anchor_kind="baseline" if kind in {"alignment_cifar","geometry_cifar"} else "synthetic"
    parent=ROOT/f"results/ndfa_strengthening_20260925/{anchor_kind}_development_v1"
    cfg=json.loads((parent/"config.json").read_text());summary=json.loads((parent/"development_summary.json").read_text())
    assert summary["complete"] and summary["config_sha256"]==sha(parent/"config.json")
    out=ROOT/f"results/ndfa_strengthening_20260925/{kind}_v1"
    if (out/"submission.json").exists():return json.loads((out/"submission.json").read_text())
    out.mkdir(exist_ok=False)
    # Start from the common resumable runner's complete dependency snapshot.
    common=ROOT/"results/ndfa_strengthening_20260925/synthetic_development_v1"
    common_cfg=json.loads((common/"config.json").read_text())
    shutil.copytree(common/"source",out/"source")
    hashes=copy.deepcopy(common_cfg["source_sha256"])
    files=["infogeo/activity_geometry.py","infogeo/tanh_dfa.py","infogeo/credit_alignment.py",
           "scripts/ndfa_strengthening_20260925/next_round.py","scripts/ndfa_strengthening_20260925/freeze_controls.py",
           "scripts/ndfa_strengthening_20260925/freeze_round2.py","tests/test_strengthening_controls.py",
           "slurm/ndfa_strengthening_next_20260925.sbatch"]
    for rel in files:
        dest=out/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest);hashes[rel]=sha(dest)
    cases=[];anchors=[]
    if kind=="digits":
        settings=[("mnist","tanh"), ("fashion_mnist","tanh"),("mnist","relu")]
        for dataset,activation in settings:
            for credit,operator in [("bp","none"),("dfa","none"),("dfa","activity"),("bp","foof")]:
                specs=[dict(optimizer=optimizer,lr=lr,damping=.3 if operator=="activity" else None)
                       for optimizer,rates in [("sgd_momentum",[.001,.01,.1]),("adamw",[.0001,.001,.01])] for lr in rates]
                if operator=="foof":specs=[dict(optimizer="sgd",lr=lr,damping=d) for lr,d in itertools.product([.03,.3,3.],[.3,3.])]
                for spec in specs:
                    cases.append(dict(id=f"case_{len(cases):03d}",cell=f"{dataset}_{activation}",dataset=dataset,activation=activation,
                                      hidden_dims=[300,300,300] if activation=="tanh" else [256,128],family=f"{credit}_{operator}",
                                      credit=credit,operator=operator,normalization="none",**spec))
        cfg.update(dataset="digits",cells={},epochs=60,development_seeds=[928101,928102],split_seed=928005,
                   evaluate_every_epochs=2,validation_examples=5000,
                   scope="Validation-only stronger optimizer/horizon controls for the factor studies; tanh uses stable summed sigmoid BCE, ReLU uses softmax CE; not a historical rerun",
                   digit_protocol="55k/5k split of train=True only; pixel/255; no augmentation; 60 epochs (25,800 updates); six configurations per method per setting; tanh 300x3, ReLU 256x128")
        indices=[next(i for i,c in enumerate(cases) if c['cell']==cell and c['operator']==operator and c['lr']==lr and (operator!='foof' or c['damping']==.3))
                 for cell,operator,lr in [('mnist_tanh','foof',.03),('fashion_mnist_tanh','activity',.001),('mnist_relu','none',.001)]]
    elif kind=="geometry_cifar":
        from infogeo.activity_geometry import STATISTICS
        for norm in ["none","bn"]:
            anchor=summary["selections"][f"cifar10/dfa_{norm}_activity"]["selected"]
            assert anchor is not None
            anchors.append(dict(cell="cifar10",normalization=norm,case=anchor))
            for statistic,factor in itertools.product(STATISTICS,[1/3,1.,3.]):
                case=copy.deepcopy(anchor)
                case.update(id=f"case_{len(cases):03d}",cell="cifar10",family=f"dfa_{norm}_{statistic}",
                            operator="activity_geometry",statistic=statistic,rho=anchor["rho"]*factor,damping=None)
                cases.append(case)
            for operator in ["none","activity"]:
                case=copy.deepcopy(anchor)
                case.update(id=f"case_{len(cases):03d}",cell="cifar10",family=f"dfa_{norm}_{operator}_bridge",operator=operator,damping=None)
                cases.append(case)
        cfg.update(geometry_probe=True,scope="Exploratory CIFAR mean/covariance controls at first-grid activity optimizer/rate anchors; two development seeds; no test evaluation",
                   geometry_protocol="Five equally searched moments, three relative damping values each, identical raw numerator and dense solves; matched raw and original sample-space activity bridges. Fixed training probe with independent augmentation RNG and restored BN state. These dense-solve runs are mechanism controls, not efficiency benchmarks.")
        indices=[next(i for i,c in enumerate(cases) if c['normalization']=='bn' and c.get('statistic')==statistic and c['rho']==anchors[1]['case']['rho']) for statistic in ['full','diagonal_covariance_plus_mean']]
    else:
        conditions=["cifar10"] if kind=="alignment_cifar" else ["nuisance","task_aligned"]
        for cell,norm in itertools.product(conditions,["none","bn"]):
            anchor=summary["selections"][f"{cell}/dfa_{norm}_activity"]["selected"]
            assert anchor is not None
            anchors.append(dict(cell=cell,normalization=norm,case=anchor))
            for schedule in ["always","never","early","late"]:
                case=copy.deepcopy(anchor)
                case.update(id=f"case_{len(cases):03d}",cell=cell,family=f"dfa_{norm}_{schedule}",conditioning_schedule=schedule,damping=anchor.get("damping"))
                cases.append(case)
        cfg.update(development_seeds=[927101,927102,927103,927104],alignment_probe=True,
                   alignment_probe_steps=[1,2,4,8,16,32,64,128,256,512,1024],
                   checkpoint_every_epochs=5,foof_inverse_period=100,foof_calibration_batches=50,
                   scope="Exploratory causal schedule controls anchored to first-grid activity recipes; four paired global seeds; not final confirmation or isolated feedback-matrix variance",
                   alignment_protocol="always, never, first quarter, last quarter; same optimizer/rate/norm matching. Raw and conditioned gradient alignment measured at the same model and fixed training probe, before optimizer. Restore BN state after offline BP diagnostics; exclude probes from learning-work timer.",
                   alignment_analysis="First three consecutive strictly positive raw cosines define sustained onset (never-onset is censored); trapezoidal negative raw descent projection over updates 0..256 per hidden layer. Four-seed contrasts descriptive; no optional seed extension based on p-values.")
        indices=[next(i for i,c in enumerate(cases) if c['normalization']=='bn' and c['conditioning_schedule']=='early')]
    cfg.update(cases=cases,source_sha256=hashes,frozen_utc=datetime.now(timezone.utc).isoformat(),output_root=str(out/"runs"),
               data_dir=str(ROOT/"data/torchvision"),stage="validation_development" if kind=="digits" else ("exploratory_geometry" if kind=="geometry_cifar" else "exploratory_alignment"),
               followup_kind=kind,anchors=anchors,parent_config_sha256=sha(parent/"config.json"),parent_summary_sha256=sha(parent/"development_summary.json"))
    # A CIFAR parent predates resumable-runner metadata; supply its required knobs.
    cfg.setdefault('checkpoint_every_epochs',5);cfg.setdefault('foof_inverse_period',100);cfg.setdefault('foof_calibration_batches',50)
    cfg.setdefault('validation_examples',5000)
    path=out/"config.json";path.write_text(json.dumps(cfg,indent=2)+"\n")
    shutil.copy2(parent/"development_summary.json",out/"parent_summary.json")
    hardware="h100" if kind in {"alignment_cifar","geometry_cifar"} else "h200"
    env=os.environ.copy();env.update(NDFA_NEXT_SOURCE=str(out/"source"),NDFA_NEXT_CONFIG=str(path),NDFA_NEXT_CONFIG_SHA256=sha(path))
    cmd=["sbatch","--parsable","--partition=kempner_requeue",f"--constraint={hardware}","--time="+("03:00:00" if kind=="geometry_cifar" else "01:00:00"),"--array="+','.join(map(str,indices))+"%60",str(out/"source/slurm/ndfa_strengthening_next_20260925.sbatch")]
    r=subprocess.run(cmd,env=env,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr)
    receipt=dict(kind=kind,verification_job=r.stdout.strip(),verification_indices=indices,config_sha256=sha(path),cases=len(cases),training_runs=len(cases)*len(cfg['development_seeds']),remaining_array="pending verification",command=cmd,hardware=hardware,partitions=['kempner_requeue'],concurrency=60)
    (out/"submission.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['digits','alignment_synthetic','alignment_cifar','geometry_cifar'],required=True)
    args=p.parse_args();print(json.dumps(freeze(args.kind)),flush=True)
