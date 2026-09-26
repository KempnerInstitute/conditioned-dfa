"""Freeze and launch a finite decision round, with automatic dependent stages."""
import copy
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from scripts.ndfa_strengthening_20260925.freeze_round2 import ROOT,sha


def read(path):return json.loads(Path(path).read_text())


def write(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def workflow(plan,update=None):
    root=Path(plan['root'])
    folder=root/'receipts';folder.mkdir(exist_ok=True)
    state={p.stem:read(p) for p in folder.glob('*.json')}
    if update:
        before=copy.deepcopy(state);update(state)
        for key,value in state.items():
            if before.get(key)==value:continue
            path=folder/f'{key}.json'
            if path.exists():assert read(path)==value,'Conflicting scheduler receipt'
            else:write(path,value) # Unique dispatch claim; publish complete JSON atomically.
    return state


def specs(plan):return read(Path(plan['root'])/'anchors.json')


def make_study(plan,name,cases,seeds,*,budget='updates',schedule='cosine',stage='confirmation',epochs=200,metric='best_validation'):
    root=Path(plan['root']);out=root/name;path=out/'config.json'
    if path.exists():return path
    out.mkdir();(out/'source').symlink_to(root/'source',target_is_directory=True)
    config=copy.deepcopy(plan['template'])
    cases=copy.deepcopy(cases)
    for i,c in enumerate(cases):c.update(id=f'case_{i:03d}',cell='cifar10');c.setdefault('damping',None)
    config.update(cases=cases,tasks=[dict(case_index=i,seed=seed) for i in range(len(cases)) for seed in seeds],
        seeds=seeds,budget_kind=budget,lr_schedule=schedule,epochs=epochs,stage=stage,selection_metric=metric,
        test_checkpoint='best' if metric=='best_validation' else 'final',output_root=str(out/'runs'),
        source_root=str(root/'source'),source_sha256=plan['source_sha256'],plan_sha256=sha(root/'plan.json'),
        frozen_utc=datetime.now(timezone.utc).isoformat(),official_test_loaded=False,
        scope='Prospective fixed-seed decision round; CIFAR test was used historically, but no test outcome selects this round')
    write(path,config)
    return path


def submit(plan,study,action,*,dependencies=(),hardware='h100',cap=60,partition=None):
    root=Path(plan['root']);state=workflow(plan);key=f'{study}:{action}'
    if key in state:return state[key]['job']
    claims=root/'dispatch_claims';claims.mkdir(exist_ok=True)
    claim=claims/key
    # Atomic creation avoids the NFS advisory-lock stall seen before any v1 job
    # was submitted. An unresolved claim requires inspecting scheduler receipts.
    claim.mkdir()
    gpu=action in {'train','evaluate'}
    env=os.environ.copy();env.update(NDFA_DECISION_SOURCE=str(root/'source'),NDFA_DECISION_PLAN=str(root/'plan.json'),
        NDFA_DECISION_ACTION=action,NDFA_DECISION_STUDY=study)
    cmd=['sbatch','--parsable','--job-name=ndfa_decision_'+action]
    if gpu:
        cmd += ['--partition='+ (partition or 'kempner_requeue'),
                f'--constraint={hardware}','--gres=gpu:1','--cpus-per-task=4','--mem=24G','--time=01:00:00']
    else:cmd+=['--partition=shared','--cpus-per-task=1','--mem=8G','--time=00:30:00']
    if dependencies:cmd+=['--dependency=afterany:'+':'.join(map(str,dependencies))]
    if action=='train':
        cfg=read(root/study/'config.json');cmd+=[f"--array=0-{len(cfg['tasks'])-1}%{cap}"]
    cmd += [str(root/'run.sbatch')]
    r=subprocess.run(cmd,env=env,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr)
    job=r.stdout.strip().split(';')[0]
    workflow(plan,lambda state:state.update({key:dict(job=job,command=cmd,submitted_utc=datetime.now(timezone.utc).isoformat())}))
    return job


def maybe_final(plan):
    root=Path(plan['root'])
    state=workflow(plan);keys=[f'{s}:evaluate' for s in ['epochs','work_confirmation','geometry_confirmation','constant_rate']]
    if all(k in state for k in keys):
        try:return submit(plan,'all','decide',dependencies=[state[k]['job'] for k in keys])
        except FileExistsError:return None # The other selector owns dispatch.


def initial():
    base=ROOT/'results/ndfa_strengthening_20260925';root=base/'decision_round_v4'
    if (root/'plan.json').exists():return launch(read(root/'plan.json'))
    root.mkdir(exist_ok=False)
    common=base/'geometry_cifar_v1';source=root/'source';shutil.copytree(common/'source',source)
    hashes=copy.deepcopy(read(common/'config.json')['source_sha256'])
    for rel in ['scripts/ndfa_strengthening_20260925/decision_train.py','scripts/ndfa_strengthening_20260925/freeze_decision.py',
                'scripts/ndfa_strengthening_20260925/decision_control.py','tests/test_decision_round.py']:
        p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p);hashes[rel]=sha(p)
    baseline=read(base/'cifar_baseline_boundaries_v1/followup_summary.json');foof=read(base/'cifar_foof_boundaries_v1/followup_summary.json')
    first=read(base/'baseline_development_v1/development_summary.json');geometry=read(base/'geometry_cifar_v1/followup_summary.json')
    assert all(s['complete'] for s in [baseline,foof,first,geometry])
    def selected(summary,key):return copy.deepcopy(summary['selections'][f'cifar10/{key}']['selected'])
    anchors={
        'dfa_bn':selected(baseline,'dfa_bn_none'),
        'activity_bn':selected(baseline,'dfa_bn_activity'),
        'early_bn':dict(selected(baseline,'dfa_bn_activity'),conditioning_schedule='early'),
        'bp_bn':selected(baseline,'bp_bn_none'),
        'bp_activity_bn':selected(baseline,'bp_bn_activity'),
        'foof_bn':selected(foof,'bp_bn_foof'),
        'foof_no_bn':selected(foof,'bp_none_foof'),
        'fd_dfa':selected(baseline,'dfa_fd_none')}
    for tag,c in anchors.items():c['family']=tag
    anchors['first_activity']=selected(first,'dfa_bn_activity')
    anchors['geometry_summary']=geometry
    write(root/'anchors.json',anchors)
    shutil.copy2(ROOT/'docs/research/ndfa_decision_protocol_20260925.md',root/'protocol.md')
    template=read(common/'config.json')
    keep=['dataset','hidden_dims','batch_size','threads','feedback_scale','weight_decay','warmup_epochs','split_seed','data_dir',
          'foof_inverse_period','foof_calibration_batches','validation_examples']
    template={k:template[k] for k in keep};template.update(evaluate_every_epochs=5,work_warmup_fraction=.025)
    plan=dict(root=str(root),source_sha256=hashes,anchors_sha256=sha(root/'anchors.json'),protocol_sha256=sha(root/'protocol.md'),template=template,
        created_utc=datetime.now(timezone.utc).isoformat(),primary_work_budget=200.,secondary_work_budget=400.,
        confirmation_seeds=list(range(930101,930111)),geometry_seeds=list(range(932101,932111)),mechanism_seeds=list(range(931101,931109)),
        work_development_seeds=[925101,925102],geometry_development_seeds=[925101,925102],
        controls=dict(constant_rates=[.0001,.0003],primary_constant_rate=.0003,geometry_new_rhos=[1/30,.1,9.,30.]),
        criteria=dict(practical_minimum_gain_pp=1.,noninferiority_margin_pp=.5,minimum_work_saving=.2,alpha=.05,
            primary_family='Holm over early-vs-DFA and full-vs-DFA at 200s; full-vs-mean and centered-vs-mean geometry; DFA early-vs-late and its BP interaction at constant 0.0003',
            no_optional_seed_extension=True,no_automatic_additional_tuning=True),
        test_policy='Load official test only after both work and geometry confirmation configurations are frozen and all rows in the evaluated study terminate. Select no setting or seed on test. Historical CIFAR test exposure is disclosed.',
        expected_training_runs=670,technical_runs=6)
    write(root/'plan.json',plan)
    (root/'run.sbatch').write_text('''#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=logs/ndfa_decision_%A_%a.out
#SBATCH --error=logs/ndfa_decision_%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NDFA_DECISION_SOURCE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u "$NDFA_DECISION_SOURCE/scripts/ndfa_strengthening_20260925/decision_control.py" --plan "$NDFA_DECISION_PLAN" --action "$NDFA_DECISION_ACTION" --study "$NDFA_DECISION_STUDY"
''')
    return launch(plan)


def launch(plan):
    anchors=specs(plan);methods=[anchors[k] for k in ['dfa_bn','activity_bn','early_bn','bp_bn','bp_activity_bn','foof_bn','foof_no_bn','fd_dfa']]
    smoke=[dict(anchors[k],work_budget_seconds=10.) for k in ['early_bn','foof_bn','fd_dfa']]
    smoke.append(dict(anchors['activity_bn'],operator='activity_geometry',statistic='diagonal_covariance_plus_mean',rho=1.,work_budget_seconds=10.))
    make_study(plan,'smoke_work',smoke,[929001],budget='work',stage='technical_verification',epochs=2)
    smoke_job=submit(plan,'smoke_work','train')
    make_study(plan,'smoke_updates',[dict(anchors['activity_bn'],conditioning_schedule='early'),dict(anchors['bp_activity_bn'],conditioning_schedule='late')],[929002],schedule='constant',stage='technical_verification',epochs=2)
    smoke_updates=submit(plan,'smoke_updates','train',hardware='h200')
    smoke_jobs=[smoke_job,smoke_updates]
    make_study(plan,'epochs',methods,plan['confirmation_seeds'])
    epoch_job=submit(plan,'epochs','train',dependencies=smoke_jobs)
    development=[]
    for budget,method in itertools.product([200.,400.],methods):
        if method['family'] in {'activity_bn','early_bn'}:
            for anchor,scale in itertools.product([method,anchors['first_activity']],[.5,1.,2.]):
                c=copy.deepcopy(anchor);c.update(family=method['family'],conditioning_schedule=method.get('conditioning_schedule','always'),lr=anchor['lr']*scale,work_budget_seconds=budget)
                development.append(c)
        else:
            for scale in [1/3,.5,1.,1.5,2.,3.]:development.append(dict(method,lr=method['lr']*scale,work_budget_seconds=budget))
    make_study(plan,'work_development',development,plan['work_development_seeds'],budget='work',stage='validation_development')
    work_job=submit(plan,'work_development','train',dependencies=smoke_jobs)
    work_select=submit(plan,'work_development','select_work',dependencies=[work_job])
    from infogeo.activity_geometry import STATISTICS
    geometry=[dict(anchors['first_activity'],family=statistic,operator='activity_geometry',statistic=statistic,rho=rho)
              for statistic,rho in itertools.product(STATISTICS,plan['controls']['geometry_new_rhos'])]
    make_study(plan,'geometry_development',geometry,plan['geometry_development_seeds'],stage='validation_development',metric='final_validation')
    geometry_job=submit(plan,'geometry_development','train',dependencies=smoke_jobs)
    geometry_select=submit(plan,'geometry_development','select_geometry',dependencies=[geometry_job])
    constant=[]
    for credit,lr,policy in itertools.product(['dfa','bp'],plan['controls']['constant_rates'],['always','never','early','late']):
        constant.append(dict(anchors['first_activity'],credit=credit,lr=lr,conditioning_schedule=policy,
                             family=f'{credit}_{lr:g}_{policy}'))
    make_study(plan,'constant_rate',constant,plan['mechanism_seeds'],schedule='constant',metric='final_validation')
    mechanism_job=submit(plan,'constant_rate','train',dependencies=smoke_jobs,hardware='h200')
    submit(plan,'epochs','evaluate',dependencies=[epoch_job,work_select,geometry_select])
    submit(plan,'constant_rate','evaluate',dependencies=[mechanism_job,work_select,geometry_select],hardware='h200')
    maybe_final(plan)
    return workflow(plan)


if __name__=='__main__':print(json.dumps(initial(),indent=2))
