"""Freeze independent synthetic and FOOF-BP validation grids before execution."""
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def settings(credit,normalization,operator):
    if operator=="foof":
        return [dict(optimizer="sgd",lr=lr,damping=d) for lr,d in itertools.product([.003,.03,.3,3.],[.03,.3,3.])]
    values=[]
    for optimizer in ["sgd_momentum","adamw"]:
        if operator=="none":
            if optimizer=="adamw":rates=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3]
            elif credit=="dfa" and normalization=="none":rates=[1e-5,1e-4,1e-3,1e-2,.03,.1]
            else:rates=[1e-4,1e-3,1e-2,.03,.1,.3]
            values.extend(dict(optimizer=optimizer,lr=lr,damping=None) for lr in rates)
        else:
            rates=[.001,.01,.1] if optimizer=="sgd_momentum" else [.0001,.001,.01]
            values.extend(dict(optimizer=optimizer,lr=lr,damping=d) for lr,d in itertools.product(rates,[.03,.3]))
    assert len(values)==12
    return values


def freeze(kind):
    out=ROOT/f"results/ndfa_strengthening_20260925/{kind}_development_v1"
    if (out/"submission.json").exists():
        print((out/"submission.json").read_text(),flush=True)
        return
    if (out/"config.json").exists():
        config=json.loads((out/"config.json").read_text())
        for rel,expected in config["source_sha256"].items():
            assert sha(out/"source"/rel)==expected,rel
        submit_verification(kind,out,config)
        return
    src=out/"source";src.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/"infogeo").glob("*.py"))
    files += [ROOT/f"experiments/{n}.py" for n in ["run_ndfa_submission_benchmark","run_dfa_multioutput_synthetic","run_dfa_synthetic","run_dfa_vision_baselines"]]
    files += [ROOT/f"scripts/ndfa_strengthening_20260925/{n}.py" for n in ["next_round","baseline_development","freeze_next"]]
    files += [ROOT/"tests/test_foof.py",ROOT/"tests/test_ndfa_next_round.py",ROOT/"slurm/ndfa_strengthening_next_20260925.sbatch"]
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);target=src/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=sha(target)
    cells={
        "nuisance":dict(n_train=512,task_scale=.45,nuisance_scale=2.,interaction=False,input_noise=.15,label_noise=.2),
        "low_sample":dict(n_train=512,task_scale=.7,nuisance_scale=1.5,interaction=False,input_noise=.15,label_noise=.4),
        "mixed":dict(n_train=1024,task_scale=.75,nuisance_scale=1.2,interaction=True,input_noise=.15,label_noise=.2),
        "task_aligned":dict(n_train=4096,task_scale=1.3,nuisance_scale=.25,interaction=False,input_noise=.05,label_noise=0.),
    }
    families=[("bp","none","none"),("dfa","none","none"),("dfa","none","activity"),
              ("dfa","bn","none"),("dfa","bn","activity"),("bp","bn","none"),
              ("bp","none","activity"),("bp","none","foof")]
    if kind=="foof_cifar":families=[("bp","none","foof"),("bp","bn","foof")]
    cases=[]
    for cell in cells if kind=="synthetic" else ["cifar10"]:
        for credit,norm,operator in families:
            for spec in settings(credit,norm,operator):
                cases.append(dict(id=f"case_{len(cases):03d}",cell=cell,
                    family=f"{credit}_{norm}_{operator}",credit=credit,normalization=norm,operator=operator,**spec))
    config=dict(schema_version=1,stage="validation_development",dataset="synthetic" if kind=="synthetic" else "cifar10",
                frozen_utc=datetime.now(timezone.utc).isoformat(),output_root=str(out/"runs"),
                data_dir=str(ROOT/"data/torchvision"),split_seed=925005,
                source_sha256=hashes,cells=cells if kind=="synthetic" else {},
                development_seeds=[926101,926102] if kind=="synthetic" else [925101,925102],
                hidden_dims=[256,128] if kind=="synthetic" else [1024,512],
                epochs=500 if kind=="synthetic" else 200,
                batch_size=128 if kind=="synthetic" else 256,threads=4,
                feedback_scale=1. if kind=="synthetic" else .1,
                weight_decay=0. if kind=="synthetic" else 1e-4,
                warmup_epochs=5,validation_examples=4096,evaluate_every_epochs=25 if kind=="synthetic" else 5,
                checkpoint_every_epochs=5,foof_inverse_period=100,foof_calibration_batches=50,
                selection="lowest mean final validation CE across both seeds; mean accuracy then ID breaks ties; incomplete/numerically failed candidates ineligible",
                scope="development only; fixed horizon is not a claim of convergence; expand boundary optima on validation before confirmation",
                foof_scope="Algorithm 1 with normalized .95 EMA, all-weight inverse conditioning, no norm matching, raw bias/BN-affine SGD, mean minibatch moments, 50 calibration batches; not a reproduction of the original paper's architecture",
                activity_scope="current-batch inverse, absolute damping, hidden weights only, own-raw-gradient norm matching",
                timing="Includes calibration, sampling, augmentation, moment refresh, inverse, gradient, optimizer and numerical checks; excludes evaluation/checkpoint/hash work; hardware-separated development only",
                cases=cases)
    path=out/"config.json";path.write_text(json.dumps(config,indent=2)+"\n")
    submit_verification(kind,out,config)


def submit_verification(kind,out,config):
    src=out/"source";path=out/"config.json";cases=config["cases"]
    if kind=="synthetic":
        wanted=[("nuisance","dfa_none_activity",.01,.3),("task_aligned","bp_none_foof",.03,.3)]
        indices=[next(i for i,c in enumerate(cases) if (c["cell"],c["family"],c["lr"],c["damping"])==w and c["optimizer"] in {"sgd","sgd_momentum"}) for w in wanted]
    else:
        indices=[next(i for i,c in enumerate(cases) if c["family"]=="bp_bn_foof" and c["lr"]==.03 and c["damping"]==.3)]
    env=os.environ.copy();env.update(NDFA_NEXT_SOURCE=str(src),NDFA_NEXT_CONFIG=str(path),NDFA_NEXT_CONFIG_SHA256=sha(path))
    command=["sbatch","--parsable","--array="+','.join(map(str,indices))]
    if kind=="foof_cifar":command += ["--partition=kempner_requeue","--constraint=h100"]
    else:command += ["--partition=kempner_eng","--constraint=h200"]
    command += [str(src/"slurm/ndfa_strengthening_next_20260925.sbatch")]
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Slurm rejected verification submission: {result.stderr}")
    record=dict(kind=kind,verification_job=result.stdout.strip(),verification_indices=indices,
                config_sha256=sha(path),cases=len(cases),training_runs=2*len(cases),
                remaining_array="pending production verification",
                command=command,
                partitions=["kempner_eng"] if kind=="synthetic" else ["kempner_requeue"],
                hardware="H200" if kind=="synthetic" else "H100")
    (out/"submission.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    freeze("synthetic")
    freeze("foof_cifar")
