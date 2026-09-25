"""Resumable validation-only synthetic controls and EMA-FOOF development."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments import run_ndfa_submission_benchmark as base
from experiments.run_dfa_multioutput_synthetic import sample_multioutput_split, corrupt_labels
from infogeo.dfa import Gradients
from infogeo.foof import FOOF
from infogeo.activity_geometry import transform as geometry_transform, describe as describe_geometry
from infogeo.local_preconditioning import condition_local_update
from scripts.ndfa_strengthening_20260925.baseline_development import optimizer_for, step_optimizer, learning_rate

STOP=False


@dataclass
class SyntheticData:
    train: torch.Tensor
    labels: torch.Tensor
    validation: torch.Tensor
    validation_labels: torch.Tensor
    provenance: dict
    classes: int=8

    @property
    def input_dim(self):return self.train.shape[1]


def synthetic_data(config,case,seed):
    """Reuse the original generator formula with independent data RNG streams.

    Only a training split and a development-validation split are generated.
    Changing validation size cannot change training samples or corrupted labels.
    """
    cell=config["cells"][case["cell"]]
    rng=lambda stream:np.random.default_rng(np.random.SeedSequence([seed,stream]))
    projection=rng(0).normal(size=(28,64))/np.sqrt(28)
    kwargs=dict(n_classes=8,nuisance_dim=24,task_scale=cell["task_scale"],
                nuisance_scale=cell["nuisance_scale"],interaction=cell["interaction"],
                input_noise=cell["input_noise"],projection=projection)
    x,y,_=sample_multioutput_split(cell["n_train"],rng=rng(1),**kwargs)
    labels=corrupt_labels(y,n_classes=8,noise=cell["label_noise"],rng=rng(2))
    vx,vy,_=sample_multioutput_split(config["validation_examples"],rng=rng(3),**kwargs)
    tensors=[torch.tensor(x,dtype=torch.float32),torch.tensor(labels,dtype=torch.long),
             torch.tensor(vx,dtype=torch.float32),torch.tensor(vy,dtype=torch.long)]
    provenance=dict(dataset="synthetic latent-circle",cell=case["cell"],recipe=cell,
                    data_seed=seed,split_streams={"projection":0,"train":1,"label_noise":2,"validation":3,"reserved_test":4},
                    test_generated=False,validation_labels="clean",training_only_scaling="none",
                    hashes={name:base.tensor_hash(t) for name,t in zip(["train","labels","validation","validation_labels"],tensors)})
    return SyntheticData(*tensors,provenance)


def load_data(config,case,seed,args):
    return synthetic_data(config,case,seed) if config["dataset"]=="synthetic" else base.load_data(args)


def batch(data,idx,rng,args):
    if isinstance(data,SyntheticData):
        return data.train[idx].to(args.device),data.labels[idx].to(args.device),b""
    return base.augmented_batch(data,idx,rng,args.device)


@torch.no_grad()
def evaluate(model,data,args):
    if not isinstance(data,SyntheticData):
        return base.evaluate(model,data.validation,data.validation_labels,data,args)
    training,cache=model.training,list(model._bn_cache)
    model.training=False
    loss,correct=0.,0
    try:
        for start in range(0,len(data.validation),args.eval_batch_size):
            logits=model.forward(data.validation[start:start+args.eval_batch_size].to(args.device))[0]
            y=data.validation_labels[start:start+args.eval_batch_size].to(args.device)
            value=F.cross_entropy(logits.double(),y,reduction="sum")
            if not torch.isfinite(value):raise FloatingPointError("Nonfinite validation loss")
            loss+=float(value);correct+=int((logits.argmax(1)==y).sum())
    finally:model.training,model._bn_cache=training,cache
    return dict(loss=loss/len(data.validation),accuracy=correct/len(data.validation),examples=len(data.validation))


@torch.no_grad()
def gradient(model,feedback,x,y,args,case,foof):
    raw=base.gradients(model,feedback,x,y,args)
    activities=model.last_activities[:len(model.weights)]
    if case["operator"]=="foof":return foof.transform(raw,activities)
    if case["operator"]=="none":return raw
    if case["operator"] not in {"activity","activity_geometry"}:raise ValueError(case["operator"])
    values=list(raw.weights)
    for layer in range(model.n_hidden_layers):
        try:
            update=geometry_transform(activities[layer],raw.weights[layer],case["damping"],case["statistic"]) if case["operator"]=="activity_geometry" else condition_local_update(activities[layer],raw.deltas[layer]*len(x),
                    activity_damping=case["damping"],error_damping=1e-6,mode="activity",backend="auto")
        except torch.linalg.LinAlgError as exc:
            raise FloatingPointError("Activity solve failed at declared damping") from exc
        except RuntimeError as exc:
            if "non-finite update" in str(exc):raise FloatingPointError(str(exc)) from exc
            raise
        before,after=raw.weights[layer].double().norm(),update.double().norm()
        if after==0 and before!=0:raise FloatingPointError("Zero conditioned update")
        if after>0:update=update*(before/after).to(update.dtype)
        values[layer]=update
    return Gradients(values,raw.biases,raw.deltas,raw.loss,raw.bn_gammas,raw.bn_betas)


@torch.no_grad()
def geometry_probe(model,data,args,seed):
    """One fixed training-mode minibatch; restore every forward-mutated state."""
    assert isinstance(data,SyntheticData),"This probe currently supports synthetic data only"
    rng=torch.Generator().manual_seed(seed+7000)
    indices=torch.randperm(len(data.train),generator=rng)[:args.batch_size]
    state=(model.training,list(model.bn_running_mean),list(model.bn_running_var),list(model._bn_cache),getattr(model,"last_activities",None))
    try:
        model.training=True
        model.forward(data.train[indices].to(args.device))
        values=[describe_geometry(a) for a in model.last_activities[:model.n_hidden_layers]]
    finally:
        model.training,model.bn_running_mean,model.bn_running_var,model._bn_cache,model.last_activities=state
    return dict(indices_sha256=base.tensor_hash(indices),layers=values,
                scope="fixed training-data minibatch, training-mode BN, statistics restored; descriptive probe, not population covariance")


def atomic_save(state,path):
    tmp=path.with_suffix(".pending")
    torch.save(state,tmp)
    os.replace(tmp,path)


@torch.no_grad()
def train(config,case,seed,output,*,device="cuda",stop_after_epoch=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (output/"endpoint.json").exists():
        result=json.loads((output/"endpoint.json").read_text())
        assert result["config_hash"]==config_hash and result["case"]==case and result["seed"]==seed
        assert base.sha256(output/"final.pt")==result["checkpoint_sha256"]
        return result
    args=base.parse_args(["--output-dir",str(output),"--data-dir",config.get("data_dir","unused"),
            "--hidden-dims",*map(str,config["hidden_dims"]),"--device",device,
            "--threads",str(config["threads"]),"--batch-size",str(config["batch_size"]),
            "--model-seed",str(seed),"--feedback-seed",str(seed+1000),
            "--feedback-scale",str(config["feedback_scale"]),"--split-seed",str(config.get("split_seed",925005))])
    args.method=case["credit"]+("_batchnorm" if case["normalization"]=="bn" else "")
    base.configure(args);data=load_data(config,case,seed,args)
    model,feedback=base.make_model(args,data)
    if case["optimizer"]=="sgd":
        optimizer=torch.optim.SGD([
            {"params":model.weights,"weight_decay":config["weight_decay"]},
            {"params":[*model.biases,*model.bn_gamma,*model.bn_beta],"weight_decay":0.}],lr=case["lr"],foreach=False,fused=False)
    else:optimizer=optimizer_for(model,case,config["weight_decay"])
    foof=FOOF(case["damping"],inverse_period=config["foof_inverse_period"]) if case["operator"]=="foof" else None
    initial=base.tensors_hash([*model.weights,*model.biases])
    order_rng=torch.Generator().manual_seed(seed+2000)
    aug_rng=torch.Generator().manual_seed(seed+3000)
    manifest=dict(config_hash=config_hash,case=case,seed=seed,protocol=config,
                  data=data.provenance,initial_parameter_sha256=initial,
                  feedback_sha256=base.tensors_hash(feedback),
                  operator="FOOF Algorithm 1 normalized EMA / raw bias and BN-affine updates" if foof else case["operator"],
                  official_test_loaded=False,source_sha256=base.sha256(Path(__file__)))
    if (output/"manifest.json").exists():
        assert json.loads((output/"manifest.json").read_text())==manifest,"Resume identity changed"
    else:base.write_json(output/"manifest.json",manifest)
    attempts=output/"attempts";attempts.mkdir(exist_ok=True)
    attempt=len(list(attempts.glob("*.json")))
    if attempt>=8:raise RuntimeError("Inspect infrastructure after eight attempts")
    base.write_json(attempts/f"{attempt:02d}.json",dict(utc=datetime.now(timezone.utc).isoformat(),
                    job=os.environ.get("SLURM_JOB_ID"),partition=os.environ.get("SLURM_JOB_PARTITION"),
                    device=torch.cuda.get_device_name() if device=="cuda" else "cpu",torch=torch.__version__,
                    tf32=False,precision="float32 training, float64 validation loss"))
    history=[];step=0;start_epoch=0;work=0.;evaluation_seconds=0.;calibration_seconds=0.
    trace=hashlib.sha256(b"paired-stream-v1").hexdigest()
    checkpoint=output/"resume.pt"
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        assert saved["config_hash"]==config_hash and saved["case"]==case and saved["seed"]==seed
        model=base.restore_model(args,data,saved["model"])
        # The optimizer must reference the restored model's tensors.
        if case["optimizer"]=="sgd":
            optimizer=torch.optim.SGD([
                {"params":model.weights,"weight_decay":config["weight_decay"]},
                {"params":[*model.biases,*model.bn_gamma,*model.bn_beta],"weight_decay":0.}],lr=case["lr"],foreach=False,fused=False)
        else:optimizer=optimizer_for(model,case,config["weight_decay"])
        optimizer.load_state_dict(saved["optimizer"])
        if foof:foof.load_state_dict(saved["foof"])
        order_rng.set_state(saved["order_rng"].cpu());aug_rng.set_state(saved["aug_rng"].cpu())
        start_epoch,step,history=saved["epoch"],saved["step"],saved["history"]
        work,evaluation_seconds,calibration_seconds=saved["work"],saved["evaluation_seconds"],saved["calibration_seconds"]
        trace=saved["stream_sha256"]

    def save(epoch,path=checkpoint):
        atomic_save(dict(config_hash=config_hash,case=case,seed=seed,epoch=epoch,step=step,
                   model=base.model_state(model),optimizer=optimizer.state_dict(),
                   foof=foof.state_dict() if foof else None,order_rng=order_rng.get_state(),
                   aug_rng=aug_rng.get_state(),history=history,work=work,
                   evaluation_seconds=evaluation_seconds,calibration_seconds=calibration_seconds,
                   stream_sha256=trace),path)

    def record(epoch,training_loss=None):
        nonlocal evaluation_seconds
        base.sync(args);began=time.perf_counter();val=evaluate(model,data,args)
        geometry=geometry_probe(model,data,args,seed) if config.get("geometry_probe",False) else None
        base.sync(args);evaluation_seconds+=time.perf_counter()-began
        history.append(dict(epoch=epoch,step=step,training_seconds=work,
                            calibration_seconds=calibration_seconds,training_loss=training_loss,
                            validation=val,stream_sha256=trace))
        if geometry is not None:history[-1]["activity_geometry"]=geometry
        base.write_json(output/"history.json",history)

    status,error="complete",None
    epoch=start_epoch
    try:
        if not checkpoint.exists():
            if foof:
                base.sync(args);began=time.perf_counter()
                warm_rng=torch.Generator().manual_seed(seed+5000)
                warm_aug=torch.Generator().manual_seed(seed+6000)
                means=[m.clone() for m in model.bn_running_mean];variances=[m.clone() for m in model.bn_running_var]
                model.training=True
                for _ in range(config["foof_calibration_batches"]):
                    idx=torch.randint(len(data.train),(args.batch_size,),generator=warm_rng)
                    x,_,_=batch(data,idx,warm_aug,args)
                    model.forward(x);foof.observe(model.last_activities[:len(model.weights)])
                foof.refresh()
                model.bn_running_mean,model.bn_running_var=means,variances
                base.sync(args);calibration_seconds=time.perf_counter()-began;work+=calibration_seconds
            record(0);save(0)
        total=math.ceil(len(data.train)/args.batch_size)*config["epochs"]
        warmup=math.ceil(len(data.train)/args.batch_size)*config["warmup_epochs"]
        for epoch in range(start_epoch+1,config["epochs"]+1):
            base.sync(args);began=time.perf_counter()
            order=torch.randperm(len(data.train),generator=order_rng)
            base.sync(args);work+=time.perf_counter()-began
            sum_loss=0.
            for start in range(0,len(order),args.batch_size):
                base.sync(args);began=time.perf_counter()
                idx=order[start:start+args.batch_size];x,y,augmentation=batch(data,idx,aug_rng,args)
                model.training=True
                update=gradient(model,feedback,x,y,args,case,foof)
                step_optimizer(model,update,optimizer,learning_rate(case["lr"],step,total,warmup))
                if not all(torch.isfinite(t).all() for t in base.state_tensors(model)):
                    raise FloatingPointError("Nonfinite model or normalization state")
                base.sync(args);work+=time.perf_counter()-began
                trace=hashlib.sha256(bytes.fromhex(trace)+idx.numpy().tobytes()+augmentation).hexdigest()
                sum_loss+=update.loss*len(idx);step+=1
            if epoch==1 or epoch%config["evaluate_every_epochs"]==0 or epoch==config["epochs"]:
                record(epoch,sum_loss/len(data.train))
            if epoch%config["checkpoint_every_epochs"]==0 or STOP or epoch==stop_after_epoch:
                save(epoch)
            if STOP or epoch==stop_after_epoch:
                return dict(status="interrupted",epoch=epoch,checkpoint=str(checkpoint))
    except FloatingPointError as exc:status,error="numerical_failure",str(exc)
    save(epoch,output/"final.pt")
    result=dict(status=status,error=error,case=case,seed=seed,config_hash=config_hash,
                epoch=epoch,completed_updates=step,training_seconds=work,
                calibration_seconds=calibration_seconds,evaluation_seconds=evaluation_seconds,
                final_validation=history[-1]["validation"] if status=="complete" else None,
                stream_sha256=trace,checkpoint_sha256=base.sha256(output/"final.pt"),
                official_test_loaded=False)
    base.write_json(output/"endpoint.json",result)
    checkpoint.unlink(missing_ok=True)
    return result


def stop(signum,frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--config-sha256",required=True)
    p.add_argument("--case-index",type=int,default=int(os.environ.get("SLURM_ARRAY_TASK_ID","0")))
    args=p.parse_args();assert base.sha256(args.config)==args.config_sha256
    config=json.loads(args.config.read_text())
    for name,digest in config["source_sha256"].items():assert base.sha256(ROOT/name)==digest,name
    case=config["cases"][args.case_index]
    parent=Path(config["output_root"])/case["id"];parent.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    with (parent/".execution.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in config["development_seeds"]:
            result=train(config,case,seed,parent/f"seed_{seed}")
            print(json.dumps(result),flush=True)
            if result["status"]=="interrupted":return


if __name__=="__main__":main()
