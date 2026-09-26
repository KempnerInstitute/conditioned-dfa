"""Operators, training-only benchmark data and explicit arithmetic estimates."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from scripts.ndfa_strengthening_20260925 import next_round as nr
from infogeo.dfa import Gradients
from infogeo.local_preconditioning import condition_local_update

base=nr.base
FLOP_SCOPE=('Estimated leading-order dense linear-algebra FLOPs, FMA=2: forward, credit, weight products, '
 'moment products, solves, inverse refresh and calibration. Excludes pointwise activations, BN reductions, '
 'optimizer arithmetic, data transforms, diagnostics and exports; not measured hardware FLOPs. '
 'FD estimates include its documented matrix products separately. No FLOP-matched claim from time matching.')


def add_flops(counts,values):
    for key,value in values.items():counts[key]=counts.get(key,0.)+float(value)


def calibration_flops(model,batch,batches):
    return dict(calibration=sum(2*batch*(w.shape[0]*w.shape[1]+w.shape[1]**2)*batches+w.shape[1]**3 for w in model.weights))


def factor_flops(n,din,dout,*,error=False):
    # The implementation recomputes the numerator and uses min(din,dout) RHS
    # columns when operating in sample space.
    d=dout if error else din
    if n<d:return 2*n*n*d+2/3*n**3+2*n*n*min(din,dout)
    return 2*n*d*d+2/3*d**3+2*d*d*(din if error else dout)


def step_flops(model,case,n,*,foof_step=None,inverse_period=100):
    sizes=[(w.shape[0],w.shape[1]) for w in model.weights]
    values=dict(forward_and_weight_products=4*n*sum(o*i for o,i in sizes),credit=0.,conditioning=0.)
    values['credit']=2*n*(sum(o*model.output_dim for o,i in sizes[:-1]) if case['credit']=='dfa' else sum(sizes[k][0]*sizes[k+1][0] for k in range(len(sizes)-1)))
    if case['operator']=='foof':
        values['conditioning']=sum(2*n*i*i+2*o*i*i+(i**3 if foof_step%inverse_period==0 else 0) for o,i in sizes)
    elif case['operator']=='activity':
        for o,i in sizes[:-1]:
            values['conditioning']+=2*n*i*o+factor_flops(n,i,o)
            if case.get('error_operator')=='full':values['conditioning']+=factor_flops(n,i,o,error=True)
            elif case.get('error_operator')=='diagonal':values['conditioning']+=2*n*o+i*o
    elif case['operator']=='activity_geometry':
        values['conditioning']=sum(2*n*i*i+2/3*i**3+2*i*i*o for o,i in sizes[:-1])
    if case['normalization']=='fd':
        # A decorrelator before every weight layer; low-rank state update uses
        # two products on a fixed subsample. DFA has no decorrelator pullback.
        m=min(n,int(n*.1)+1)
        values['forward_decorrelation']=sum((2*n+4*m)*i*i for o,i in sizes)
        if case['credit']=='bp':values['forward_decorrelation']+=sum(2*n*i*i for o,i in sizes[1:])
    return values


def active_case(case,progress):
    result=case
    policy=case.get('conditioning_schedule','always')
    active=policy=='always' or (policy=='early' and progress<.25) or (policy=='late' and progress>=.75)
    if policy not in {'always','never','early','late'}:raise ValueError(policy)
    if not active:result=dict(result,operator='none')
    ep=case.get('error_schedule','always')
    if ep not in {'always','never','early','late'}:raise ValueError(ep)
    eactive=ep=='always' or (ep=='early' and progress<.25) or (ep=='late' and progress>=.75)
    if not eactive:result=dict(result,error_operator=None)
    return result


def _match(value,raw):
    a,b=raw.double().norm(),value.double().norm()
    if not torch.isfinite(b):raise FloatingPointError('Nonfinite conditioned update')
    if b==0 and a!=0:raise FloatingPointError('Zero conditioned update')
    return value*(a/b).to(value.dtype) if b>0 else value


@torch.no_grad()
def error_transform(activities,raw,args,case):
    weights=list(raw.weights)
    for layer,activity in enumerate(activities[:len(raw.weights)-1]):
        error=raw.deltas[layer]*len(activity)
        la=max(case['rho']*float(activity.square().mean()),args.damping_floor) if case.get('rho') is not None else case['damping']
        le=max(case['error_rho']*float(error.square().mean()),args.damping_floor)
        kind=case['error_operator']
        if kind=='full':value=condition_local_update(activity,error,activity_damping=la,error_damping=le,mode='kronecker')
        else:
            value=condition_local_update(activity,error,activity_damping=la,mode='activity')
            if kind=='diagonal':value=value/(error.square().mean(0)+le)[:,None]
            elif kind=='scalar':
                # Positive scalar left-conditioning cancels exactly after norm
                # matching. Use its algebraically identical A implementation.
                pass
            else:raise ValueError(kind)
        weights[layer]=_match(value,raw.weights[layer])
    return Gradients(weights,raw.biases,raw.deltas,raw.loss,raw.bn_gammas,raw.bn_betas)


@torch.no_grad()
def gradient(model,feedback,x,y,args,case,foof):
    if case.get('error_operator') and case['operator']!='none':
        raw=base.gradients(model,feedback,x,y,args)
        try:return error_transform(model.last_activities,raw,args,case)
        except torch.linalg.LinAlgError as exc:raise FloatingPointError('Error solve failed at declared damping') from exc
        except RuntimeError as exc:
            if 'non-finite update' in str(exc):raise FloatingPointError(str(exc)) from exc
            raise
    if case['operator']=='none':return base.gradients(model,feedback,x,y,args)
    return nr.gradient(model,feedback,x,y,args,case,foof)


def _mutable_state(model):
    return (model.training,list(model.bn_running_mean),list(model.bn_running_var),list(model._bn_cache),getattr(model,'last_activities',None))


def _restore_state(model,state):
    model.training,model.bn_running_mean,model.bn_running_var,model._bn_cache,model.last_activities=state


@torch.no_grad()
def direction_probe(model,feedback,data,args,case,seed):
    state=_mutable_state(model)
    try:
        indices=torch.randperm(len(data.train),generator=torch.Generator().manual_seed(seed+7300))[:args.batch_size]
        x,y,_=nr.batch(data,indices,torch.Generator().manual_seed(seed+7400),args);model.training=True
        raw=base.gradients(model,feedback,x,y,args)
        a=error_transform(model.last_activities,raw,args,dict(case,error_operator='scalar'))
        e=error_transform(model.last_activities,raw,args,case)
        rows=[]
        for u,v in zip(a.weights[:-1],e.weights[:-1]):
            denominator=float(u.double().norm()*v.double().norm())
            rows.append(float((u.double()*v.double()).sum())/denominator if denominator>0 else None)
        return dict(layer_cosines_to_activity=rows,reference='A at the same state, batch and raw update; equal hidden matrix norms',indices_sha256=base.tensor_hash(indices))
    finally:_restore_state(model,state)


def load_data(config,case,seed,args):
    if config['dataset']!='benchmark':return nr.load_data(config,case,seed,args)
    folder=Path(config['benchmark_root']);entry=config['benchmark_inventory'][case['cell']]
    path=folder/entry['file'];assert base.sha256(path)==entry['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['official_test_loaded'] is False
    return nr.SyntheticData(saved['train'],saved['labels'],saved['validation'],saved['validation_labels'],saved['provenance'],classes=10)


@torch.no_grad()
def representation_probe(model,data,args,config,case,seed):
    if config['dataset']!='synthetic':return None
    cell=config['cells'][case['cell']];rng=np.random.default_rng(np.random.SeedSequence([seed,0]));projection=rng.normal(size=(28,64))/np.sqrt(28)
    theta=np.linspace(0,2*np.pi,64,endpoint=False);task=np.column_stack([np.cos(theta),np.sin(theta),np.cos(2*theta),np.sin(2*theta)])
    nuisance=np.random.default_rng(np.random.SeedSequence([seed,7500])).normal(size=(64,8,24))
    latent=np.concatenate([np.broadcast_to(cell['task_scale']*task[:,None,:],(64,8,4)),cell['nuisance_scale']*nuisance],axis=-1)
    x=torch.tensor((latent@projection).reshape(-1,64),dtype=torch.float32,device=args.device)
    state=_mutable_state(model)
    try:
        model.training=False;_,activities,_=model.forward(x);rows=[]
        for h in activities[1:-1]:
            h=h.double().reshape(64,8,-1);within=h.var(1,unbiased=False).sum(-1).mean();between=h.mean(1).var(0,unbiased=False).sum()
            rows.append(dict(nuisance_variation=float(within),task_variation=float(between),nuisance_fraction=float(within/(within+between)) if within+between>0 else None))
        return dict(layers=rows,scope='Same 64 task angles, eight independent nuisance redraws per angle; fixed evaluation state; no input noise; diagnostic only')
    finally:_restore_state(model,state)


def initialize_branch(config,case,seed,out,digest):
    ref=config['branch_from'];path=Path(ref['checkpoint'])
    assert base.sha256(path)==ref['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['seed']==seed
    parent=saved['case']
    for key in ['credit','normalization','operator','optimizer','lr','rho','damping']:
        assert parent.get(key)==case.get(key),(key,parent.get(key),case.get(key))
    assert saved['step']==ref['step'] and saved['step']>0
    saved.update(config_hash=digest,case=case,seed=seed)
    nr.atomic_save(saved,out/'resume.pt')
    (out/'branch_provenance.json').write_text(json.dumps(ref,indent=2)+'\n')


def task_config(config,case,seed):
    if not case.get('branch_sources'):return config
    config=copy.deepcopy(config);config['branch_from']=case['branch_sources'][str(seed)];config['epochs']=case['target_epochs']
    return config
