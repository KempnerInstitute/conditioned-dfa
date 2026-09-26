"""Bounded parallel development followed by one jointly frozen confirmation."""
import copy
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
from scripts.ndfa_strengthening_20260925.freeze_decision import read,write,sha,workflow

ROOT=Path(__file__).resolve().parents[2]
NEW_FILES=['integrated_train.py','integrated_components.py','integrated_plan.py','integrated_control.py','prepare_integrated_benchmarks.py']


def case(family,credit='dfa',operator='none',normalization='bn',optimizer='adamw',lr=.0003,**kw):
    return dict(family=family,credit=credit,operator=operator,normalization=normalization,optimizer=optimizer,lr=lr,damping=None,**kw)


def designs():
    """Exact development inventory. No outcome-dependent grid extensions."""
    root=ROOT/'results/ndfa_strengthening_20260925';old=root/'decision_round_v4'
    template=read(old/'plan.json')['template'];template.update(work_warmup_fraction=.025,budget_kind='work',lr_schedule='cosine',epochs=200)
    synthetic=read(root/'synthetic_development_v1/config.json');synthetic={k:v for k,v in synthetic.items() if k not in ['cases','tasks','source_sha256','output_root','frozen_utc','stage']}
    synthetic.update(budget_kind='work',lr_schedule='cosine',work_warmup_fraction=.025,evaluate_every_epochs=25)
    specs={}
    c=[]
    for credit in ['bp','dfa']:
        for lr in [.0003,.003,.03,.3,3.,30.]:c.append(case(f'{credit}_sgd',credit,optimizer='sgd',lr=lr))
        for lr,damping in itertools.product([.3,3.,30.],[.03,.3,3.]):
            v=case(f'foof_{credit}',credit,'foof',optimizer='sgd',lr=lr);v['damping']=damping;c.append(v)
        for lr in [.0001,.0003,.001]:c.append(case(f'{credit}_adamw',credit,lr=lr))
    for lr in [.00003,.0001,.0003]:c.append(case('fd_dfa',normalization='fd',lr=lr,decor_lr=1e-5))
    for lr,rho in itertools.product([.0003,.001,.003],[1/3,1.,3.]):c.append(case('activity','dfa','activity',lr=lr,rho=rho))
    for lr in [.0001,.0003,.001]:c.append(case('dfa_no_bn',normalization='none',lr=lr))
    specs['foof_cifar']=dict(template=dict(template,dataset='cifar10'),cases=[dict(v,cell='cifar10',work_budget_seconds=200.) for v in c],seeds=[925101,925102],hardware='h100')
    c=[]
    for norm,credit in itertools.product(['none','bn'],['bp','dfa']):
        for lr in [1e-5,1e-4,.001,.01,.1,1.]:c.append(case(f'{credit}_sgd_{norm}',credit,normalization=norm,optimizer='sgd',lr=lr))
        for lr,damping in itertools.product([.003,.03,.3],[.3,3.,30.]):
            v=case(f'foof_{credit}_{norm}',credit,'foof',normalization=norm,optimizer='sgd',lr=lr);v['damping']=damping;c.append(v)
        for lr in [3e-5,.0001,.0003]:c.append(case(f'{credit}_tuned_{norm}',credit,normalization=norm,optimizer='adamw',lr=lr))
    for norm,lr in itertools.product(['none','bn'],[.0001,.000333333333333,.001]):
        v=case(f'activity_{norm}',operator='activity',normalization=norm,optimizer='sgd_momentum',lr=lr);v['damping']=.03 if norm=='none' else .3;c.append(v)
    specs['foof_nuisance']=dict(template=synthetic,cases=[dict(v,cell='nuisance',work_budget_seconds=20.) for v in c],seeds=[926101,926102],hardware='h200')
    for name,dataset,norm,budget in [('error_cifar','cifar10','bn',200.),('error_mnist','digits','none',60.)]:
        c=[]
        for lr in [.0001,.0003,.001]:
            anchor=case('activity',operator='activity',normalization=norm,lr=lr,rho=1. if norm=='bn' else None)
            if norm=='none':anchor['damping']=.3
            c.append(anchor)
            for policy in ['early','always']:
                for erho in [1.,10.,100.]:c.append(dict(anchor,family='early_k' if policy=='early' else 'persistent_k',error_operator='full',error_rho=erho,error_schedule=policy))
            for erho in [1.,10.,100.]:c.append(dict(anchor,family='early_diagonal',error_operator='diagonal',error_rho=erho,error_schedule='early'))
        cfg=dict(template,dataset=dataset,lr_schedule='constant')
        if dataset=='digits':cfg.update(hidden_dims=[256,128],batch_size=128,feedback_scale=1.,weight_decay=0.,split_seed=928005)
        specs[name]=dict(template=cfg,cases=[dict(v,cell='cifar10' if dataset=='cifar10' else 'mnist_relu',dataset='mnist',work_budget_seconds=budget) for v in c],seeds=[935201,935202],hardware='h100' if dataset=='cifar10' else 'h200')
    inventory=read(ROOT/'data/ndfa_integrated_development_20260926/inventory.json');c=[]
    for cell in inventory:
        for family,credit,norm,lrs in [('dfa_bn','dfa','bn',[.0001,.0003,.001]),('bp_bn','bp','bn',[.0001,.0003,.001]),('dfa_no_bn','dfa','none',[.0001,.0003,.001]),('fd_dfa','dfa','fd',[.00003,.0001,.0003])]:
            for lr in lrs:c.append(dict(case(family,credit,normalization=norm,lr=lr,decor_lr=1e-5),cell=cell,work_budget_seconds=60.))
        for family,operator in [('activity','activity'),('centered','activity_geometry')]:
            for lr,rho in itertools.product([.0001,.0003,.001],[.3,3.]):c.append(dict(case(family,operator=operator,lr=lr,rho=rho,statistic='centered'),cell=cell,work_budget_seconds=60.))
        for credit,lr,damping in itertools.product(['bp','dfa'],[.03,.3,3.],[.3,3.]):
            v=case(f'foof_{credit}',credit,'foof',normalization='bn',optimizer='sgd',lr=lr);v['damping']=damping;c.append(dict(v,cell=cell,work_budget_seconds=60.))
    specs['benchmarks']=dict(template=dict(template,dataset='benchmark',hidden_dims=[256,128],batch_size=128,feedback_scale=1.,weight_decay=0.,benchmark_root=str(ROOT/'data/ndfa_integrated_development_20260926'),benchmark_inventory=inventory),cases=c,seeds=[935301,935302],hardware='h200')
    return specs


def make_study(plan,name,template,cases,seeds,*,stage='validation_development',tasks=None,extra=None):
    root=Path(plan['root']);folder=root/name;path=folder/'config.json'
    if path.exists():return path
    folder.mkdir();(folder/'source').symlink_to(root/'source',target_is_directory=True)
    cases=copy.deepcopy(cases)
    for i,c in enumerate(cases):c['id']=f'case_{i:03d}'
    cfg=copy.deepcopy(template);cfg.update(cases=cases,seeds=seeds,tasks=tasks or [dict(case_index=i,seed=s) for i in range(len(cases)) for s in seeds],stage=stage,
        source_root=str(root/'source'),source_sha256=plan['source_sha256'],plan_sha256=sha(root/'plan.json'),output_root=str(folder/'runs'),frozen_utc=datetime.now(timezone.utc).isoformat(),official_test_loaded=False)
    if extra:cfg.update(extra)
    write(path,cfg);return path


def submit(plan,study,action,dependencies=(),hardware='h200',partition='kempner_requeue'):
    root=Path(plan['root']);key=f'{study}:{action}';state=workflow(plan)
    if key in state:return state[key]['job']
    claims=root/'dispatch_claims';claims.mkdir(exist_ok=True);(claims/key).mkdir()
    gpu=action in ['train','evaluate']
    cmd=['sbatch','--parsable','--job-name=ndfa_integrated_'+action,'--qos=normal']
    if gpu:cmd += [f'--partition={partition}',f'--constraint={hardware}','--gres=gpu:1','--cpus-per-task=4','--mem=24G','--time=01:00:00']
    else:cmd+=['--partition=shared','--cpus-per-task=1','--mem=12G','--time=00:45:00']
    live=[]
    for job in dependencies:
        q=subprocess.run(['squeue','-h','-j',str(job),'-o','%i'],capture_output=True,text=True)
        if q.stdout.strip():live.append(str(job))
        else:
            a=subprocess.run(['sacct','-X','-n','-j',str(job),'--format=State','--parsable2'],capture_output=True,text=True,check=True)
            states=[s.strip().strip('|') for s in a.stdout.splitlines() if s.strip()]
            assert states and all(s=='COMPLETED' for s in states),(job,states)
    if live:cmd+=['--dependency=afterok:'+':'.join(live)]
    if action=='train':cmd += [f"--array=0-{len(read(root/study/'config.json')['tasks'])-1}%60"]
    cmd.append(str(root/'run.sbatch'))
    env=os.environ.copy();env.update(NDFA_INTEGRATED_SOURCE=str(root/'source'),NDFA_INTEGRATED_PLAN=str(root/'plan.json'),NDFA_INTEGRATED_ACTION=action,NDFA_INTEGRATED_STUDY=study)
    result=subprocess.run(cmd,env=env,capture_output=True,text=True)
    write(claims/key/'attempt.json',dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stderr)
    job=result.stdout.strip().split(';')[0];workflow(plan,lambda s:s.update({key:dict(job=job,command=cmd,submitted_utc=datetime.now(timezone.utc).isoformat())}));return job


def initial():
    root=ROOT/'results/ndfa_strengthening_20260925/integrated_round_v1'
    if (root/'plan.json').exists():return launch(read(root/'plan.json'))
    old=ROOT/'results/ndfa_strengthening_20260925/decision_round_v4';amend=read(old/'integrated_revision_amendment.json');assert not amend['test_plan_paths']
    specs=designs();root.mkdir();source=root/'source';shutil.copytree(old/'source',source)
    hashes=read(old/'plan.json')['source_sha256'].copy()
    for rel in ['scripts/ndfa_strengthening_20260925/'+f for f in NEW_FILES]+['tests/test_integrated_round.py']:
        dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest);hashes[rel]=sha(dest)
    shutil.copy2(ROOT/'docs/research/ndfa_integrated_protocol_20260926.md',root/'protocol.md')
    write(root/'designs.json',specs)
    plan=dict(root=str(root),source_sha256=hashes,designs_sha256=sha(root/'designs.json'),protocol_sha256=sha(root/'protocol.md'),prior_round=str(old),prior_amendment_sha256=sha(old/'integrated_revision_amendment.json'),
        development_studies=list(specs),confirmation_seeds=list(range(940101,940111)),branch_seeds=list(range(936101,936105)),
        created_utc=datetime.now(timezone.utc).isoformat(),expected_new_development_runs=sum(len(v['cases'])*len(v['seeds']) for v in specs.values()),
        official_test_policy='One joint freeze after all development and branch evidence; historical test exposure disclosed; no development task loads test data',
        error_gate=dict(primary='error_cifar',secondary='error_mnist',window_fraction=.25,constant_learning_rate=True,minimum_direction_change=.001,scalar_equivalent_to_A=True,no_repeat_after_failure=True),
        no_automatic_grid_expansions=True,alpha=.05,practical_minimum_gain_pp=1.)
    write(root/'plan.json',plan)
    (root/'run.sbatch').write_text('''#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=logs/ndfa_integrated_%A_%a.out
#SBATCH --error=logs/ndfa_integrated_%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NDFA_INTEGRATED_SOURCE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u "$NDFA_INTEGRATED_SOURCE/scripts/ndfa_strengthening_20260925/integrated_control.py" --plan "$NDFA_INTEGRATED_PLAN" --action "$NDFA_INTEGRATED_ACTION" --study "$NDFA_INTEGRATED_STUDY"
''')
    return launch(plan)


def launch(plan):
    root=Path(plan['root']);specs=read(root/'designs.json');smokes=[]
    choices={'foof_cifar':lambda c:c['family']=='foof_dfa' and c['lr']==3. and c['damping']==.3,
             'error_cifar':lambda c:c['family'] in ['early_k','early_diagonal'] and c['lr']==.0003 and c.get('error_rho')==10.,
             'foof_nuisance':lambda c:c['family']=='foof_dfa_bn' and c['lr']==.03 and c['damping']==3.,
             'error_mnist':lambda c:c['family']=='early_k' and c['lr']==.0001 and c.get('error_rho')==10.,
             'benchmarks':lambda c:c['cell']=='mnist_background_standard' and c['family'] in ['activity','centered'] and c['lr']==.0003 and c.get('rho')==3.}
    for name,predicate in choices.items():
        spec=specs[name];cases=[dict(c,work_budget_seconds=3.) for c in spec['cases'] if predicate(c)]
        assert cases;smoke='smoke_'+name;make_study(plan,smoke,spec['template'],cases,[939001],stage='technical_verification')
        smokes.append(submit(plan,smoke,'train',hardware=spec['hardware']))
    gate=submit(plan,'all','verify',dependencies=smokes)
    jobs=[]
    for name,spec in specs.items():
        make_study(plan,name,spec['template'],spec['cases'],spec['seeds']);jobs.append(submit(plan,name,'train',dependencies=[gate],hardware=spec['hardware']))
    prior=read(Path(plan['prior_round'])/'receipts/work_development:train.json')['job']
    # The branch dispatcher only uses already completed constant-rate validation
    # records and may run alongside the new development grids.
    branch=submit(plan,'all','branches',dependencies=[gate])
    submit(plan,'all','collect',dependencies=jobs+[prior,branch])
    return workflow(plan)


if __name__=='__main__':print(json.dumps(initial(),indent=2))
