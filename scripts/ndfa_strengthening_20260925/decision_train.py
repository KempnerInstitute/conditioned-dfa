"""Frozen decision-round training with validation checkpoints and work budgets.

No official test loader is reachable from training. Instrumentation and exports
are outside synchronized learning-work time. FOOF calibration is charged.
"""
import argparse
from datetime import datetime,timezone
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.ndfa_strengthening_20260925 import next_round as nr
from scripts.ndfa_strengthening_20260925.baseline_development import optimizer_for,step_optimizer,learning_rate
from infogeo.foof import FOOF
base=nr.base
STOP=False


@contextmanager
def execution_claim(out):
    """Atomic per-task ownership; reclaim only a demonstrably ended Slurm attempt."""
    lock=Path(out)/'.execution_claim';token=uuid.uuid4().hex
    owner=dict(token=token,job=os.environ.get('SLURM_JOB_ID'),restart=int(os.environ.get('SLURM_RESTART_COUNT','0')),
               host=socket.gethostname(),pid=os.getpid())
    try:lock.mkdir()
    except FileExistsError:
        previous=json.loads((lock/'owner.json').read_text())
        stale=bool(owner['job']) and previous['job']==owner['job'] and owner['restart']>previous['restart']
        if not stale and previous['job']:
            q=subprocess.run(['squeue','-h','-j',previous['job'],'-o','%T'],capture_output=True,text=True)
            stale=q.returncode==0 and not q.stdout.strip()
        if not stale:raise RuntimeError('Another or unverified attempt owns this run')
        lock.rename(Path(out)/f".old_execution_claim_{previous['token']}");lock.mkdir()
    (lock/'owner.json').write_text(json.dumps(owner)+'\n')
    try:yield
    finally:
        if lock.exists() and json.loads((lock/'owner.json').read_text())['token']==token:
            lock.rename(Path(out)/f'.completed_execution_claim_{token}')


def optimizer(model,case,config):
    if case['optimizer']!='sgd':return optimizer_for(model,case,config['weight_decay'])
    return torch.optim.SGD([{'params':model.weights,'weight_decay':config['weight_decay']},
        {'params':[*model.biases,*model.bn_gamma,*model.bn_beta],'weight_decay':0.}],lr=case['lr'],foreach=False,fused=False)


def active_case(case,progress):
    policy=case.get('conditioning_schedule','always')
    assert policy in {'always','never','early','late'}
    active=policy=='always' or (policy=='early' and progress<.25) or (policy=='late' and progress>=.75)
    return case if active else dict(case,operator='none')


def rate(case,config,progress,step,total,warmup):
    schedule=config.get('lr_schedule','cosine')
    if schedule=='constant':return case['lr']
    assert schedule=='cosine'
    if config['budget_kind']=='updates':return learning_rate(case['lr'],step,total,warmup)
    warm=config['work_warmup_fraction'];fraction=min(max(progress,0.),1.)
    if fraction<warm:return case['lr']*max(fraction,1e-6)/warm
    fraction=(fraction-warm)/(1-warm)
    return case['lr']*(.01+.99*.5*(1+math.cos(math.pi*fraction)))


def args_for(config,case,seed,output,device):
    args=base.parse_args(['--output-dir',str(output),'--data-dir',config.get('data_dir','unused'),
        '--hidden-dims',*map(str,config['hidden_dims']),'--device',device,'--threads',str(config['threads']),
        '--batch-size',str(config['batch_size']),'--model-seed',str(seed),'--feedback-seed',str(seed+1000),
        '--feedback-scale',str(config['feedback_scale']),'--split-seed',str(config['split_seed'])])
    args.method=case['credit']+{'none':'','bn':'_batchnorm','fd':'_decorrelation'}[case['normalization']]
    args.decor_lr=case.get('decor_lr',1e-5)
    if config['dataset']=='fixture':
        args.dataset='fixture';args.fixture_train=32;args.fixture_validation=16;args.fixture_side=8
    return args


@torch.no_grad()
def train(config,case,seed,output,device='cuda',stop_after_step=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (out/'endpoint.json').exists():
        e=json.loads((out/'endpoint.json').read_text());assert e['config_hash']==digest and e['case']==case and e['seed']==seed
        assert base.sha256(out/'final.pt')==e['checkpoint_sha256']
        if e['status']=='complete':assert base.sha256(out/'best.pt')==e['best_checkpoint_sha256']
        return e
    args=args_for(config,case,seed,out,device);base.configure(args)
    data=nr.load_data(config,case,seed,args);model,feedback=nr.make_model(config,case,args,data)
    opt=optimizer(model,case,config)
    foof=FOOF(case['damping'],inverse_period=config['foof_inverse_period']) if case['operator']=='foof' else None
    manifest=dict(protocol=config,config_hash=digest,case=case,seed=seed,data=data.provenance,
        initial_parameter_sha256=base.tensors_hash([*model.weights,*model.biases]),feedback_sha256=base.tensors_hash(feedback),official_test_loaded=False)
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:base.write_json(out/'manifest.json',manifest)
    attempts=out/'attempts';attempts.mkdir(exist_ok=True);attempt=len(list(attempts.glob('*.json')))
    assert attempt<8,'Repeated infrastructure interruption requires inspection'
    base.write_json(attempts/f'{attempt:02d}.json',dict(utc=datetime.now(timezone.utc).isoformat(),job=os.environ.get('SLURM_JOB_ID'),
        partition=os.environ.get('SLURM_JOB_PARTITION'),device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
        precision='float32 training; float64 evaluation loss; TF32 disabled',torch_version=str(torch.__version__),cuda_version=torch.version.cuda,resumed=(out/'resume.pt').exists()))
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    order_rng=torch.Generator().manual_seed(seed+2000);aug_rng=torch.Generator().manual_seed(seed+3000)
    step=epoch=cursor=conditioned_steps=0;order=None;work=evaluation_seconds=calibration_seconds=0.
    history=[];trace=hashlib.sha256(b'paired-stream-v1').hexdigest();best=None;last_checkpoint_work=0.
    checkpoint=out/'resume.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        assert saved['config_hash']==digest and saved['case']==case and saved['seed']==seed
        model=nr.restore_model(config,case,args,data,saved['model']);opt=optimizer(model,case,config);opt.load_state_dict(saved['optimizer'])
        if foof:foof.load_state_dict(saved['foof'])
        order_rng.set_state(saved['order_rng'].cpu());aug_rng.set_state(saved['aug_rng'].cpu())
        step,epoch,cursor,order=saved['step'],saved['epoch'],saved['cursor'],saved['order']
        if order is not None:order=order.cpu()
        work,evaluation_seconds,calibration_seconds=saved['work'],saved['evaluation_seconds'],saved['calibration_seconds']
        history,trace,best=saved['history'],saved['stream_sha256'],saved['best']
        conditioned_steps=saved['conditioned_steps'];last_checkpoint_work=work
        # The resume bundle owns its matching best checkpoint, even after a crash
        # following a later best.pt write but before the next resume save.
        nr.atomic_save(saved['best_payload'],out/'best.pt')
    steps_per_epoch=math.ceil(len(data.train)/args.batch_size);total=steps_per_epoch*config['epochs'];warmup=steps_per_epoch*config['warmup_epochs']
    budget=case.get('work_budget_seconds',config.get('work_budget_seconds'))
    is_work=config['budget_kind']=='work'
    if is_work:assert budget and budget>0
    def progress():return work/budget if is_work else step/total
    def model_payload():
        return dict(config_hash=digest,case=case,seed=seed,model=base.model_state(model),args=vars(args),step=step,epoch=epoch,
                    work=work,data=data.provenance,input_dim=data.input_dim,classes=data.classes,
                    input_normalization={k:getattr(data,k).cpu().clone() for k in ['channel_mean','channel_std']} if hasattr(data,'channel_mean') else None)
    def save():
        nr.atomic_save(dict(**model_payload(),optimizer=opt.state_dict(),foof=foof.state_dict() if foof else None,
            cursor=cursor,order=order,order_rng=order_rng.get_state(),aug_rng=aug_rng.get_state(),history=history,
            evaluation_seconds=evaluation_seconds,calibration_seconds=calibration_seconds,stream_sha256=trace,best=best,
            best_payload=torch.load(out/'best.pt',map_location='cpu',weights_only=False),conditioned_steps=conditioned_steps),checkpoint)
    def record():
        nonlocal best,evaluation_seconds
        if history and history[-1]['step']==step:return
        base.sync(args);began=time.perf_counter();val=nr.evaluate(model,data,args);base.sync(args)
        evaluation_seconds+=time.perf_counter()-began
        row=dict(step=step,epoch=step/steps_per_epoch,training_seconds=work,validation=val,stream_sha256=trace)
        history.append(row)
        if best is None or val['loss']<best['validation']['loss']:
            best=row.copy();nr.atomic_save(model_payload(),out/'best.pt')
        base.write_json(out/'history.json',history)
    status='complete';error=None;final_step_seconds=0.
    try:
        if not checkpoint.exists():
            if foof:
                base.sync(args);began=time.perf_counter();rng=torch.Generator().manual_seed(seed+5000);aug=torch.Generator().manual_seed(seed+6000)
                means=[v.clone() for v in model.bn_running_mean];variances=[v.clone() for v in model.bn_running_var];model.training=True
                for _ in range(config['foof_calibration_batches']):
                    idx=torch.randint(len(data.train),(args.batch_size,),generator=rng);x,_,_=nr.batch(data,idx,aug,args);model.forward(x);foof.observe(model.last_activities[:len(model.weights)])
                foof.refresh();model.bn_running_mean=means;model.bn_running_var=variances
                base.sync(args);calibration_seconds=time.perf_counter()-began;work+=calibration_seconds
            record();save()
        while (work<budget if is_work else step<total):
            if order is None or cursor>=len(order):
                base.sync(args);began=time.perf_counter();order=torch.randperm(len(data.train),generator=order_rng);cursor=0;base.sync(args);work+=time.perf_counter()-began
                if is_work and work>=budget:break
            before_progress=progress();selected=active_case(case,before_progress)
            lr=rate(case,config,before_progress,step,total,warmup)
            base.sync(args);began=time.perf_counter()
            idx=order[cursor:cursor+args.batch_size];x,y,aug=nr.batch(data,idx,aug_rng,args);model.training=True
            update=base.gradients(model,feedback,x,y,args) if selected['operator']=='none' else nr.gradient(model,feedback,x,y,args,selected,foof)
            step_optimizer(model,update,opt,lr)
            if not all(torch.isfinite(v).all() for v in base.state_tensors(model)):raise FloatingPointError('Nonfinite model or normalization state')
            base.sync(args);final_step_seconds=time.perf_counter()-began;work+=final_step_seconds
            step+=1;cursor+=len(idx);conditioned_steps+=int(selected['operator'] in {'activity','activity_geometry'})
            trace=hashlib.sha256(bytes.fromhex(trace)+idx.numpy().tobytes()+aug).hexdigest()
            if cursor>=len(order):epoch+=1
            if is_work:
                previous=history[-1]['training_seconds']/budget if history else 0
                if int(progress()*20)>int(previous*20):record()
            elif cursor>=len(order) and (epoch==1 or epoch%config['evaluate_every_epochs']==0):record()
            if work-last_checkpoint_work>=30 or STOP or step==stop_after_step:
                save();last_checkpoint_work=work
            if STOP or step==stop_after_step:return dict(status='interrupted',step=step)
        record()
    except FloatingPointError as exc:status='numerical_failure';error=str(exc)
    nr.atomic_save(model_payload(),out/'final.pt')
    result=dict(config_hash=digest,case=case,seed=seed,status=status,error=error,epoch=epoch,completed_updates=step,
        training_seconds=work,evaluation_seconds=evaluation_seconds,calibration_seconds=calibration_seconds,
        final_validation=history[-1]['validation'] if status=='complete' else None,best_validation=best['validation'] if best else None,
        best_step=best['step'] if best else None,best_training_seconds=best['training_seconds'] if best else None,
        conditioned_updates=conditioned_steps,stream_sha256=trace,official_test_loaded=False,
        checkpoint_sha256=base.sha256(out/'final.pt'),best_checkpoint_sha256=base.sha256(out/'best.pt') if (out/'best.pt').exists() else None,
        work_budget_seconds=budget if is_work else None,budget_overshoot_seconds=max(0.,work-budget) if is_work else None,
        last_step_seconds=final_step_seconds,attempts=attempt+1,peak_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    base.write_json(out/'endpoint.json',result);checkpoint.unlink(missing_ok=True)
    return result


def stop(signum,frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--task-index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=p.parse_args();assert base.sha256(args.config)==args.config_sha256
    config=json.loads(args.config.read_text())
    for rel,digest in config['source_sha256'].items():assert base.sha256(ROOT/rel)==digest,rel
    task=config['tasks'][args.task_index];case=config['cases'][task['case_index']];seed=task['seed']
    out=Path(config['output_root'])/case['id']/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    with execution_claim(out):
        result=train(config,case,seed,out);print(json.dumps(result),flush=True)
        if result['status']=='interrupted':
            job=os.environ.get('SLURM_JOB_ID')
            if job:subprocess.run(['scontrol','requeue',job],check=True)
            sys.exit(75)


if __name__=='__main__':main()
