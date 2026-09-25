"""Freeze E1a development inputs and submit its first GPU verification case."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import itertools
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "results/ndfa_strengthening_20260925/baseline_development_v1"
    source = out / "source"
    source.mkdir(parents=True, exist_ok=False)
    files = list((ROOT/"infogeo").glob("*.py")) + [
        ROOT/"experiments/run_ndfa_submission_benchmark.py",
        ROOT/"scripts/ndfa_strengthening_20260925/baseline_development.py",
        Path(__file__), ROOT/"tests/test_ndfa_strengthening_20260925.py",
        ROOT/"slurm/ndfa_strengthening_development_20260925.sbatch",
    ]
    hashes={}
    for path in files:
        relative=path.relative_to(ROOT); target=source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target); hashes[str(relative)]=sha(target)
    cases=[]
    # Equal numbers of candidates per family. The search is deliberately
    # validation-adaptive: boundary optima trigger disclosed follow-up grids.
    for credit, norm, operator in itertools.product(["bp","dfa"], ["none","bn"], ["none","activity"]):
        family=f"{credit}_{norm}_{operator}"
        for optimizer in ["sgd_momentum","adamw"]:
            if operator == "none":
                if optimizer == "adamw":
                    rates=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3]
                elif credit == "dfa" and norm == "none":
                    rates=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3]
                elif credit == "dfa":
                    rates=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
                else:
                    rates=[1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]
                settings=[(lr,None) for lr in rates]
            else:
                if optimizer == "adamw":
                    rates=[1e-5,1e-4,1e-3]
                elif credit == "dfa" and norm == "none":
                    rates=[1e-4,1e-3,1e-2]
                elif credit == "dfa":
                    rates=[1e-3,1e-2,1e-1]
                else:
                    rates=[3e-3,3e-2,3e-1]
                settings=itertools.product(rates,[1.,30.])
            for lr,rho in settings:
                cases.append(dict(id=f"case_{len(cases):03d}",family=family,credit=credit,
                                  normalization=norm,operator=operator,optimizer=optimizer,
                                  lr=lr,rho=rho))
    for optimizer,rates in [("sgd_momentum",[1e-4,1e-3,1e-2]),("adamw",[1e-5,1e-4,1e-3])]:
        for lr,decor_lr in itertools.product(rates,[1e-6,1e-5]):
            cases.append(dict(id=f"case_{len(cases):03d}",family="dfa_fd_none",credit="dfa",
                              normalization="fd",operator="none",optimizer=optimizer,
                              lr=lr,rho=None,decor_lr=decor_lr))
    assert len(cases)==108
    config=dict(schema_version=1, stage="validation_development",
                frozen_utc=datetime.now(timezone.utc).isoformat(),
                dataset="cifar10", data_dir=str(ROOT/"data/torchvision"),
                output_root=str(out/"runs"), source_sha256=hashes,
                development_seeds=[925101,925102], split_seed=925005,
                epochs=200, hidden_dims=[1024,512], batch_size=256,
                threads=4, feedback_scale=.1, weight_decay=1e-4,
                warmup_epochs=5, evaluate_every_epochs=5,
                selection="lowest mean final validation CE across both development seeds; highest mean accuracy then case ID break ties; numerical or incomplete runs are ineligible",
                boundary_policy="expand boundary selections prospectively on validation before freezing confirmation; no test access",
                convergence_policy="200 epochs is a horizon, not evidence of convergence; inspect late loss and validation curves before deciding on extension",
                confirmation="not authorized by this config; freeze selected settings and seed count in a separate config",
                limits="current-batch norm-matched activity operator; full EMA FOOF and equal-work confirmation remain separate stages",
                cases=cases)
    path=out/"config.json"; path.write_text(json.dumps(config,indent=2)+"\n")
    env=os.environ.copy();env.update(NDFA_STRENGTH_SOURCE=str(source),NDFA_STRENGTH_CONFIG=str(path),NDFA_STRENGTH_CONFIG_SHA256=sha(path))
    # Verify the exact production path on an activity+BN case before the array.
    index=next(i for i,c in enumerate(cases) if c["family"]=="dfa_bn_activity" and c["optimizer"]=="sgd_momentum" and c["lr"]==.01 and c["rho"]==30.)
    job=subprocess.run(["sbatch","--parsable",f"--array={index}",str(source/"slurm/ndfa_strengthening_development_20260925.sbatch")],env=env,capture_output=True,text=True,check=True).stdout.strip()
    (out/"submission.json").write_text(json.dumps(dict(verification_job=job,verification_index=index,config_sha256=sha(path),remaining_array="pending production-path verification",partition="kempner_h100_priority"),indent=2)+"\n")
    print(json.dumps(dict(job=job,index=index,config=str(path),sha256=sha(path))))


if __name__ == "__main__":
    main()
