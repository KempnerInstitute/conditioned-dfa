"""Audit, select, evaluate and produce the frozen decision-round report."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev
import sys
import torch
import torch.nn.functional as F
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.ndfa_strengthening_20260925 import freeze_decision as launch
from scripts.ndfa_strengthening_20260925 import decision_train as train
read,write,sha=launch.read,launch.write,launch.sha


def source_gate(plan):
    root=Path(plan['root']);assert sha(root/'anchors.json')==plan['anchors_sha256']
    assert sha(root/'protocol.md')==plan['protocol_sha256']
    for rel,digest in plan['source_sha256'].items():assert sha(root/'source'/rel)==digest,rel


def audit(plan,study):
    root=Path(plan['root']);cp=root/study/'config.json';cfg=read(cp)
    assert cfg['plan_sha256']==sha(root/'plan.json') and cfg['source_sha256']==plan['source_sha256']
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();rows=[]
    for task in cfg['tasks']:
        case=cfg['cases'][task['case_index']];seed=task['seed'];folder=Path(cfg['output_root'])/case['id']/f'seed_{seed}'
        assert (folder/'endpoint.json').exists(),f'Missing training result: {folder}'
        e,m=read(folder/'endpoint.json'),read(folder/'manifest.json')
        assert e['case']==m['case']==case and e['seed']==m['seed']==seed
        assert e['config_hash']==m['config_hash']==digest and m['protocol']==cfg
        assert e['official_test_loaded'] is False and m['official_test_loaded'] is False
        assert e['status'] in {'complete','numerical_failure'}
        assert sha(folder/'final.pt')==e['checkpoint_sha256']
        if e['status']=='complete':
            h=read(folder/'history.json');assert h[-1]['step']==e['completed_updates'] and h[-1]['validation']==e['final_validation']
            assert e['best_validation']['loss']==min(v['validation']['loss'] for v in h)
            assert sha(folder/'best.pt')==e['best_checkpoint_sha256']
            assert math.isfinite(e['best_validation']['loss']) and math.isfinite(e['final_validation']['loss'])
            if cfg['budget_kind']=='updates':assert e['completed_updates']==math.ceil(45000/cfg['batch_size'])*cfg['epochs']
            else:
                assert e['training_seconds']>=case['work_budget_seconds']
                assert e['budget_overshoot_seconds']<max(1.,.01*case['work_budget_seconds']), 'Excessive budget overshoot'
        rows.append(dict(**e,folder=str(folder),manifest_sha256=sha(folder/'manifest.json')))
    result=dict(config_sha256=sha(cp),complete=True,rows=rows,successful=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='numerical_failure' for r in rows))
    write(root/study/'training_audit.json',result)
    return cfg,result


def select(plan,study):
    root=Path(plan['root']);cfg,a=audit(plan,study);groups=defaultdict(list)
    for c in cfg['cases']:
        rows=[r for r in a['rows'] if r['case']['id']==c['id']]
        if all(r['status']=='complete' for r in rows):
            key=(c['family'],c['work_budget_seconds']) if study=='work_development' else c['statistic']
            groups[key].append(dict(case=c,loss=mean(r[cfg['selection_metric']]['loss'] for r in rows),
                accuracy=mean(r[cfg['selection_metric']]['accuracy'] for r in rows),source_config_sha256=a['config_sha256']))
    if study=='geometry_development':
        for record in launch.specs(plan)['geometry_summary']['candidates']:
            c=record['case']
            if c['normalization']=='bn' and c.get('statistic') and record['status']=='complete':
                groups[c['statistic']].append(dict(case=c,loss=record['mean_validation_loss'],accuracy=record['mean_validation_accuracy'],source='frozen earlier H100 geometry grid'))
    expected=16 if study=='work_development' else 5
    assert len(groups)==expected,'A declared family has no eligible development configuration'
    selections={str(k):min(v,key=lambda r:(r['loss'],-r['accuracy'],r['case']['id'])) for k,v in sorted(groups.items())}
    selection_file=root/study/'selection.json';write(selection_file,dict(selections=selections,selection_metric=cfg['selection_metric'],source_config_sha256=a['config_sha256'],no_further_expansion=True))
    cases=[]
    for record in selections.values():
        c=copy.deepcopy(record['case'])
        if study=='geometry_development':c['family']=c['statistic']
        cases.append(c)
    if study=='work_development':
        name='work_confirmation';cp=launch.make_study(plan,name,cases,plan['confirmation_seeds'],budget='work')
        other='geometry_development:select_geometry'
    else:
        name='geometry_confirmation';anchor=launch.specs(plan)['first_activity']
        cases.extend([dict(anchor,family='raw_matched',operator='none'),dict(anchor,family='sample_bridge')])
        cp=launch.make_study(plan,name,cases,plan['geometry_seeds'],metric='final_validation')
        other='work_development:select_work'
    final_cfg=read(cp)
    expected_hash=sha(selection_file)
    if 'selection_artifact_sha256' in final_cfg:assert final_cfg['selection_artifact_sha256']==expected_hash
    else:final_cfg.update(selection_artifact_sha256=expected_hash,selection_artifact=str(selection_file));write(cp,final_cfg)
    job=launch.submit(plan,name,'train')
    state=launch.workflow(plan);assert other in state
    launch.submit(plan,name,'evaluate',dependencies=[job,state[other]['job']])
    launch.maybe_final(plan)
    return dict(study=study,confirmation=name,training_job=job,selected=len(cases))


def evaluation_gate(plan):
    # Test results must not be available while another study is still selecting.
    root=Path(plan['root'])
    for name in ['work_confirmation','geometry_confirmation']:
        cfg=read(root/name/'config.json')
        assert cfg['stage']=='confirmation'
        assert sha(Path(cfg['selection_artifact']))==cfg['selection_artifact_sha256']


@torch.no_grad()
def evaluate(plan,study):
    root=Path(plan['root']);cfg,a=audit(plan,study)
    assert cfg['stage']=='confirmation';evaluation_gate(plan)
    folder=root/study/'test_evaluation';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    entries=[]
    for r in a['rows']:
        entries.append(dict(case=r['case'],seed=r['seed'],status=r['status'],folder=r['folder'],
            checkpoint_sha256=r['checkpoint_sha256'],best_checkpoint_sha256=r['best_checkpoint_sha256'],
            training_seconds=r['training_seconds'],attempts=r['attempts']))
    test_plan=dict(config_sha256=a['config_sha256'],plan_sha256=sha(root/'plan.json'),entries=entries,
                   official_test_access=True,historical_test_exposure_disclosed=True,selection_complete=True)
    path=folder/'plan.json'
    if path.exists():assert read(path)==test_plan
    else:
        with path.open('x') as f:json.dump(test_plan,f,indent=2);f.write('\n')
    # All training/provenance/selection gates precede the only test loader.
    from torchvision.datasets import CIFAR10
    source=CIFAR10(cfg['data_dir'],train=False,download=False)
    images=torch.as_tensor(source.data).permute(0,3,1,2).contiguous();labels=torch.as_tensor(source.targets,dtype=torch.long)
    assert images.shape==(10000,3,32,32) and labels.shape==(10000,)
    data_hash=dict(images=train.base.tensor_hash(images),labels=train.base.tensor_hash(labels));results=[]
    for r in entries:
        row=copy.deepcopy(r);row['metrics']={};out=folder/r['case']['id']/f"seed_{r['seed']}";out.mkdir(parents=True,exist_ok=True)
        if r['status']=='complete':
            for which in ['best','final']:
                checkpoint=Path(r['folder'])/f'{which}.pt';expected=r['best_checkpoint_sha256' if which=='best' else 'checkpoint_sha256']
                assert sha(checkpoint)==expected
                result_path=out/f'{which}.json'
                if result_path.exists():
                    saved=read(result_path);assert saved['checkpoint_sha256']==expected and saved['test_data']==data_hash
                    assert sha(out/f'{which}.pt')==saved['predictions_sha256']
                else:
                    saved_model=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    args=train.args_for(cfg,r['case'],r['seed'],out,'cuda');train.base.configure(args)
                    from types import SimpleNamespace
                    norm={k:v.cuda() for k,v in saved_model['input_normalization'].items()}
                    data=SimpleNamespace(input_dim=saved_model['input_dim'],classes=saved_model['classes'])
                    model=train.base.restore_model(args,data,saved_model['model']);model.training=False
                    logits=torch.cat([model.forward(train.base.normalize(x.cuda(),norm))[0].cpu() for x in images.split(512)])
                    assert torch.isfinite(logits).all() and logits.shape==(10000,10)
                    predictions=out/f'{which}.pt';train.nr.atomic_save(dict(logits=logits,labels=labels,test_data=data_hash,checkpoint_sha256=expected),predictions)
                    saved=dict(checkpoint_sha256=expected,predictions_sha256=sha(predictions),test_data=data_hash,
                        loss=float(F.cross_entropy(logits.double(),labels)),accuracy=float((logits.argmax(1)==labels).double().mean()))
                    write(result_path,saved);del model
                row['metrics'][which]=saved
        results.append(row)
    summary=dict(study=study,config_sha256=a['config_sha256'],evaluation_plan_sha256=sha(path),official_test_loaded=True,test_data=data_hash,rows=results)
    write(folder/'summary.json',summary)
    return dict(study=study,evaluated=len(results))


def paired_summary(values):
    if len(values)<2:return dict(n=len(values),mean=None,low=None,high=None,p=1.)
    avg=mean(values);se=stdev(values)/math.sqrt(len(values));width=stats.t.ppf(.975,len(values)-1)*se
    p=float(stats.ttest_1samp(values,0.).pvalue) if se>0 else (0. if avg else 1.)
    return dict(n=len(values),values=values,mean=avg,low=avg-width,high=avg+width,p=p)


def holm(tests):
    ordered=sorted(tests,key=lambda k:tests[k]['p']);previous=0.
    for i,key in enumerate(ordered):
        previous=max(previous,min(1.,tests[key]['p']*(len(ordered)-i)));tests[key]['holm_p']=previous
    return tests


def decide(plan):
    root=Path(plan['root']);datasets={name:read(root/name/'test_evaluation/summary.json') for name in ['epochs','work_confirmation','geometry_confirmation','constant_rate']}
    def values(study,family,budget=None,metric='accuracy',which=None):
        cfg=read(root/study/'config.json');which=which or cfg['test_checkpoint']
        rows=[r for r in datasets[study]['rows'] if r['case']['family']==family and (budget is None or r['case'].get('work_budget_seconds')==budget)]
        expected=len(cfg['seeds'])
        if len(rows)!=expected or any(r['status']!='complete' for r in rows):return None
        if (study=='work_confirmation' or metric=='work') and any(r['attempts']!=1 for r in rows):return None
        return {r['seed']:(r['training_seconds'] if metric=='work' else r['metrics'][which][metric]) for r in rows}
    def delta(study,left,right,budget=None,metric='accuracy',which=None):
        a,b=values(study,left,budget,metric,which),values(study,right,budget,metric,which)
        if a is None or b is None:return None
        assert a.keys()==b.keys();scale=100 if metric=='accuracy' else 1
        return [scale*(a[s]-b[s]) for s in sorted(a)]
    tests={}
    for key,left,right in [('work_early','early_bn','dfa_bn'),('work_full','activity_bn','dfa_bn')]:tests[key]=paired_summary(delta('work_confirmation',left,right,200.) or [])
    for key,left in [('geometry_full','full'),('geometry_centered','centered')]:tests[key]=paired_summary(delta('geometry_confirmation',left,'diagonal_covariance_plus_mean') or [])
    d=delta('constant_rate','dfa_0.0003_early','dfa_0.0003_late');b=delta('constant_rate','bp_0.0003_early','bp_0.0003_late')
    tests['timing_dfa']=paired_summary(d or [])
    tests['timing_interaction']=paired_summary([x-y for x,y in zip(d,b)] if d is not None and b is not None else [])
    holm(tests)
    def passes(key):return tests[key]['mean'] is not None and tests[key]['mean']>0 and tests[key]['holm_p']<.05
    practical={key:passes(key) and tests[key]['mean']>=1. and mean(delta('work_confirmation',family,'dfa_bn',200.,'loss') or [float('inf')])<=0
               for key,family in [('work_early','early_bn'),('work_full','activity_bn')]}
    retained=paired_summary(delta('epochs','early_bn','activity_bn',which='final') or [])
    a,b=values('epochs','early_bn',metric='work'),values('epochs','activity_bn',metric='work')
    ratios=[a[s]/b[s] for s in sorted(a)] if a is not None and b is not None else []
    lower_one_sided=retained['mean']-float(stats.t.ppf(.95,9))*stdev(retained['values'])/math.sqrt(10) if retained['n']==10 else None
    temporary=lower_one_sided is not None and lower_one_sided>-.5 and bool(ratios) and mean(ratios)<=.8
    covariance=passes('geometry_full') and passes('geometry_centered');specificity=passes('timing_dfa') and passes('timing_interaction')
    incomplete=any(r['n']==0 for r in tests.values())
    recommendation='primary_evidence_unavailable_review_failures_and_interruptions' if incomplete else ('advance_mechanism_focused_paper' if any(practical.values()) and covariance and specificity else (
        'advance_geometry_efficiency_paper_with_limited_DFA_specificity' if any(practical.values()) and covariance else 'narrow_claims_before_scale_expansion'))
    result=dict(primary_tests=tests,practical=practical,covariance_confirmed=covariance,DFA_specific_timing=specificity,
        early_fixed_epoch_noninferiority=retained,early_noninferiority_one_sided_lower=lower_one_sided,
        early_work_ratios=ratios,temporary_conditioning_meets_margin_and_saving=temporary,recommendation=recommendation,
        scope='Fixed cohorts and historically used CIFAR test; model-seed uncertainty conditional on this dataset; no acceptance-probability claim; no optional extensions; primary work contrasts require uninterrupted runs')
    write(root/'decision.json',result)
    lines=['# Frozen nDFA decision round','',f"Decision: **{recommendation.replace('_',' ')}**.",'',
        'All tests and thresholds were frozen before this confirmation. Intervals describe paired model seeds on the fixed CIFAR test set. Historical test exposure remains disclosed.','',
        '| Primary contrast | Mean gain (pp) | 95% paired t interval | Holm p |','|---|---:|---|---:|']
    for key,r in tests.items():lines.append(f"| {key} | {r['mean']:.3f} | [{r['low']:.3f}, {r['high']:.3f}] | {r['holm_p']:.4g} |" if r['mean'] is not None else f'| {key} | unavailable | incomplete/failed cohort | 1 |')
    lines+=['','| Study / family / work budget | Mean test accuracy | Mean test CE |','|---|---:|---:|']
    for name,data in datasets.items():
        groups=sorted({(r['case']['family'],r['case'].get('work_budget_seconds',0)) for r in data['rows']})
        for family,budget in groups:
            acc=values(name,family,budget or None);ce=values(name,family,budget or None,'loss')
            lines.append(f'| {name} / {family} / {budget:g} | {100*mean(acc.values()):.2f}% | {mean(ce.values()):.4f} |' if acc is not None else f'| {name} / {family} / {budget:g} | incomplete/failed | unavailable |')
    lines+=['',f'Temporary conditioning meets the fixed 0.5-point noninferiority margin and 20% work-saving criterion: {temporary}.',
            '', 'The mechanism classification also requires a BP comparison. A timing benefit for DFA alone does not establish DFA specificity. Strong FOOF-BP and FD-DFA baselines remain in the tables even if they outperform activity DFA.']
    (root/'decision.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--action',required=True);p.add_argument('--study',required=True)
    args=p.parse_args();plan=read(args.plan);source_gate(plan);root=Path(plan['root'])
    if args.action=='train':
        if not args.study.startswith('smoke'):
            for name in ['smoke_work','smoke_updates']:
                _,verification=audit(plan,name);assert verification['failed']==0
        cp=root/args.study/'config.json';sys.argv=['decision_train','--config',str(cp),'--config-sha256',sha(cp)];train.main();return
    if args.action.startswith('select_'):result=select(plan,args.study)
    elif args.action=='evaluate':result=evaluate(plan,args.study)
    elif args.action=='decide':result=decide(plan)
    else:raise ValueError(args.action)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
