"""Prospectively specified follow-ups; preserve every case and final prediction.

The work-budget study reuses the prior FD comparison's selected settings.
The factor study selects error damping on validation, separately by width and
operator. The stability study changes only BN and feedback scale within a fixed
recipe. None selects a checkpoint or a method using official-test outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import pickle
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FD = ROOT / 'scripts/ndfa_bn_forward_decorrelation_20260915'
sys.path.insert(0, str(FD))
import numpy as np
import torch
from experiments import run_ndfa_paired_views as legacy
from experiments import run_ndfa_bn_factors as factors
from infogeo.dfa import Gradients
from infogeo.local_preconditioning import condition_local_update
import case as work_case


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(logits, labels):
    z = logits.double()
    loss = torch.logsumexp(z, 1) - z[torch.arange(len(z)), labels]
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Nonfinite independent predictive loss')
    sorted_loss = loss.sort(descending=True).values
    total = loss.sum().item()
    return {'accuracy': (z.argmax(1) == labels).double().mean().item(),
            'loss': loss.mean().item(), 'logit_abs_max': z.abs().max().item(),
            'loss_quantiles': torch.quantile(loss, torch.tensor([.5,.9,.99,1.], dtype=torch.float64)).tolist(),
            'top_1pct_loss_fraction': sorted_loss[:max(1,math.ceil(.01*len(loss)))].sum().item()/total if total else 0.}


@torch.no_grad()
def predictions(model, images, labels, data, args):
    previous = model.training
    try:
        model.training = False
        with legacy.preserve_batchnorm_state(model):
            logits = torch.cat([model.forward(legacy.standardize_images(x.float()/255., data).flatten(1).to(args.device))[0].cpu()
                                for x in images.split(256)])
    finally:
        model.training = previous
    return {'logits': logits, 'labels': labels.cpu(), 'metrics': metrics(logits,labels.cpu())}


@torch.no_grad()
def conditioned(model, raw, x, method, args, diagnostic=False):
    if method in ('bp','dfa') and not diagnostic:
        return raw, []
    with legacy.preserve_batchnorm_state(model):
        _, activities, _ = model.forward(x)
    weights, records = list(raw.weights), []
    for i in range(model.n_hidden_layers):
        a, e, g = activities[i], raw.deltas[i]*len(x), raw.weights[i]
        ra=max(args.relative_damping*a.square().mean().item(),1e-6)
        re=max(args.relative_error_damping*e.square().mean().item(),1e-6)
        if method in ('bp','dfa'):
            u=g
        elif method=='ediag':
            u=g/(e.square().mean(0)+re)[:,None]
        elif method=='orth_dfa':
            left,_,right=torch.linalg.svd(g,full_matrices=False)
            u=left@right
        else:
            u=condition_local_update(a,e,activity_damping=ra,error_damping=re,
                    mode={'ndfa':'activity','endfa':'error','kndfa':'kronecker'}[method],backend='auto')
        if not torch.isfinite(u).all():
            raise FloatingPointError('Nonfinite conditioned update; no fallback')
        cosine=(g*u).sum().item()/max(g.norm().item()*u.norm().item(),1e-30)
        if u.norm()>0:
            u=u*(g.norm()/u.norm())
        weights[i]=u
        if diagnostic:
            spectrum=factors.moment_spectrum(e,re)
            vals=np.asarray(spectrum['eigenvalues'])
            positives=vals[vals>spectrum['rank_tolerance']]
            spectrum['damped_condition_number']=(vals.max()+re)/(re if spectrum['rank']<e.shape[1] else positives.min()+re)
            spectrum['ridge_over_mean_positive_eigenvalue']=re/positives.mean() if len(positives) else None
            records.append({'layer':i,'error_spectrum':spectrum,'raw_conditioned_cosine':cosine,
                            'raw_norm':g.norm().item(),'update_norm':u.norm().item(),'weight_norm':model.weights[i].norm().item()})
    return Gradients(weights,raw.biases,raw.deltas,raw.loss,raw.bn_gammas,raw.bn_betas),records


def fixed_train(args,data,seed,method,out):
    model,feedback=legacy.make_model_and_feedback(data,args,seed)
    sampler=torch.Generator().manual_seed(30000+seed)
    views=torch.Generator().manual_seed(40000+seed)
    rows=[]; used=0.
    for step in range(args.steps+1):
        if step:
            began=time.perf_counter()
            idx=torch.randint(len(data.train),(args.batch_size,),generator=sampler)
            x,y=legacy.paired_batch(data,idx,args,views)
            model.training=True
            raw=model.dfa_gradients(x,y,feedback)
            update,diagnostics=conditioned(model,raw,x,method,args,step in (1,500,args.steps))
            model.apply_gradients(update,lr=args.lr)
            legacy.synchronize(args)
            used+=time.perf_counter()-began
            if diagnostics:
                write(out/f'moments_{step:05d}.json',diagnostics)
        if step in (0,500,2000,args.steps):
            pred=predictions(model,data.validation,data.validation_labels,data,args)
            rows.append({'step':step,'training_seconds':used,**pred['metrics']})
            write(out/'history.json',rows)
    return model,rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--phase',choices=['work','width','stability'],required=True)
    args_cli=p.parse_args()
    plan=json.loads(args_cli.plan.read_text())
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest: raise RuntimeError('Source changed since freeze: '+name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(plan['output_root'])/args_cli.phase
    root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',plan)
    base=json.loads((ROOT/'configs/ndfa_bn_forward_decorrelation_20260915.json').read_text())
    selected=plan['work_selected']
    all_rows=[]; data_cache={}; test_data=None
    def run(entry,test):
        nonlocal test_data
        out=root/entry['id'];out.mkdir()
        a=SimpleNamespace(**base['shared_args'])
        a.steps=plan['fixed_steps'];a.output_dir=str(out)
        a.relative_damping=30.;a.relative_error_damping=entry.get('rho',30.)
        a.hidden_dims=entry.get('width',[1024,512]);a.batchnorm=entry.get('bn',True)
        a.feedback_scale=entry.get('feedback_scale',.1);a.lr=.1
        key=tuple(a.hidden_dims)
        if key not in data_cache:data_cache[key]=legacy.load_data(a)
        data=data_cache[key]
        receipt={'case':entry,'status':'running','test_evaluations':0,'plan_sha256':sha(args_cli.plan)}
        write(out/'receipt.json',receipt)
        began=time.perf_counter()
        try:
            if args_cli.phase=='work':
                cfg=dict(base,update_budget_seconds=entry['budget'],observe_every_work_seconds=entry['budget']/10)
                candidate=selected[entry['method']]
                a.lr=candidate['peak_lr'];a.relative_damping=candidate.get('relative_damping',.3);a.steps=200000
                case=dict(candidate,method=entry['method'],seed=entry['seed'],stage='confirmation')
                model,optimizer,history,aux=work_case.train(cfg,case,data,a,legacy,None)
                write(out/'history.json',history)
                torch.save(optimizer.state_dict(),out/'optimizer.pt')
                receipt['update_seconds']=history[-1]['training_seconds']
            else:
                model,history=fixed_train(a,data,entry['seed'],entry['method'],out)
                receipt['update_seconds']=history[-1]['training_seconds']
            v=predictions(model,data.validation,data.validation_labels,data,a)
            torch.save(v,out/'validation.pt');receipt['validation']=v['metrics']
            checkpoint={'args':vars(a),'state':factors.model_state(model),'method':entry['method']}
            if hasattr(model,'decorators'):
                checkpoint['decorators']=[{'weight':d.decor_weight.cpu(),'mean':d.running_mean.cpu()} for d in model.decorators]
            torch.save(checkpoint,out/'final.pt')
            if test:
                if test_data is None:
                    with open(Path(a.data_dir)/'cifar-10-batches-py/test_batch','rb') as f:z=pickle.load(f,encoding='bytes')
                    test_data=(torch.from_numpy(z[b'data'].reshape(-1,3,32,32)),torch.tensor(z[b'labels']))
                q=predictions(model,*test_data,data,a)
                torch.save(q,out/'test.pt');receipt['test']=q['metrics'];receipt['test_evaluations']=1
            receipt['status']='complete'
        except (FloatingPointError,RuntimeError) as error:
            receipt.update(status='failure',error=type(error).__name__+': '+str(error))
        receipt['wall_seconds']=time.perf_counter()-began
        receipt['artifacts_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}
        write(out/'receipt.json',receipt);all_rows.append(receipt)
        write(root/'results.json',all_rows)
        print(entry['id'],receipt['status'],receipt.get('validation'),flush=True)
        return receipt
    if args_cli.phase=='width':
        development=[run(e,False) for e in plan['width_development']]
        choices={}
        for width in (1024,2048):
            for method in ('endfa','ediag'):
                scores=[]
                for rho in plan['error_ridges']:
                    rows=[r for r in development if r['case']['width'][0]==width and r['case']['method']==method and r['case']['rho']==rho]
                    if len(rows)==2 and all(r['status']=='complete' for r in rows):
                        scores.append((sum(r['validation']['loss'] for r in rows)/2,-rho))
                if not scores:raise RuntimeError('No eligible validation-selected damping')
                choices[f'{width}_{method}']=-min(scores)[1]
        write(root/'selection.json',choices)
        for e in plan['width_confirmation']:
            e=dict(e)
            if e['method'] in ('endfa','kndfa','ediag'):
                e['rho']=choices[f"{e['width'][0]}_{'ediag' if e['method']=='ediag' else 'endfa'}"]
            run(e,True)
    else:
        for e in plan[args_cli.phase+'_cases']:run(e,True)
    write(root/'completion.json',{'complete':True,'cases':len(all_rows),'failed':sum(r['status']!='complete' for r in all_rows),'plan_sha256':sha(args_cli.plan)})


if __name__=='__main__':main()
