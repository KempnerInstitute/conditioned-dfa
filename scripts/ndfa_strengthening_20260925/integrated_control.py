"""Gated development, shared-state branches and one joint confirmation freeze."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev
import sys
import torch
import torch.nn.functional as F
from scipy import stats

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.ndfa_strengthening_20260925 import integrated_plan as launch
from scripts.ndfa_strengthening_20260925 import integrated_train as train
from scripts.ndfa_strengthening_20260925 import integrated_components as component
from scripts.ndfa_strengthening_20260925.decision_control import paired_summary,holm
read,write,sha=launch.read,launch.write,launch.sha


def source_gate(plan):
    root=Path(plan['root'])
    for name,key in [('designs.json','designs_sha256'),('protocol.md','protocol_sha256')]:assert sha(root/name)==plan[key]
    assert sha(Path(plan['prior_round'])/'integrated_revision_amendment.json')==plan['prior_amendment_sha256']
    for rel,digest in plan['source_sha256'].items():assert sha(root/'source'/rel)==digest,rel


def audit(plan,study):
    root=Path(plan['root']);cp=root/study/'config.json';cfg=read(cp)
    assert cfg['plan_sha256']==sha(root/'plan.json') and cfg['source_sha256']==plan['source_sha256']
    rows=[]
    for task in cfg['tasks']:
        case=cfg['cases'][task['case_index']];seed=task['seed'];effective=component.task_config(cfg,case,seed)
        digest=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
        folder=Path(cfg['output_root'])/case['id']/f'seed_{seed}';e=read(folder/'endpoint.json');m=read(folder/'manifest.json')
        assert e['case']==m['case']==case and e['seed']==m['seed']==seed
        assert e['config_hash']==m['config_hash']==digest and m['protocol']==effective
        assert not e['official_test_loaded'] and not m['official_test_loaded']
        assert e['status'] in ['complete','numerical_failure'] and sha(folder/'final.pt')==e['checkpoint_sha256']
        if e['status']=='complete':
            h=read(folder/'history.json');assert h[-1]['step']==e['completed_updates'] and h[-1]['validation']==e['final_validation']
            assert min(v['validation']['loss'] for v in h)==e['best_validation']['loss']
            assert sha(folder/'best.pt')==e['best_checkpoint_sha256']
            if cfg['budget_kind']=='work':
                assert e['training_seconds']>=case['work_budget_seconds'] and e['budget_overshoot_seconds']<max(1.,.01*case['work_budget_seconds'])
            else:assert e['completed_updates']==math.ceil(e['training_examples']/cfg['batch_size'])*effective['epochs']
        rows.append(dict(**e,folder=str(folder),manifest_sha256=sha(folder/'manifest.json')))
    result=dict(config_sha256=sha(cp),complete=True,rows=rows,successful=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='numerical_failure' for r in rows))
    write(root/study/'training_audit.json',result);return cfg,result


def verify(plan):
    root=Path(plan['root']);checks={}
    for name in plan['development_studies']:
        _,a=audit(plan,'smoke_'+name);assert a['failed']==0;checks[name]=dict(successful=a['successful'],failed=a['failed'])
    write(root/'technical_gate.json',dict(passed=True,checks=checks));return checks


def candidates(cfg,a):
    groups=defaultdict(list)
    for c in cfg['cases']:
        rows=[r for r in a['rows'] if r['case']['id']==c['id']]
        if len(rows)==len(cfg['seeds']) and all(r['status']=='complete' for r in rows):
            groups[(c['cell'],c['family'])].append(dict(case=c,loss=mean(r['best_validation']['loss'] for r in rows),accuracy=mean(r['best_validation']['accuracy'] for r in rows),seeds=cfg['seeds']))
    return groups


def collect(plan):
    root=Path(plan['root']);summaries={}
    for name in plan['development_studies']:
        cfg,a=audit(plan,name);groups=candidates(cfg,a);selected={}
        for key,rows in groups.items():
            if name.startswith('error_') and key[1]!='activity':continue
            selected['/'.join(key)]=min(rows,key=lambda r:(r['loss'],-r['accuracy'],r['case']['id']))
        if name.startswith('error_'):
            anchor=next(iter(selected.values()))['case'];lr=anchor['lr']
            for family in ['early_k','early_diagonal']:
                rows=[r for (cell,fam),rr in groups.items() if fam==family for r in rr if r['case']['lr']==lr]
                if rows:selected[anchor['cell']+'/'+family]=min(rows,key=lambda r:(r['loss'],-r['accuracy'],r['case']['id']))
            early=selected.get(anchor['cell']+'/early_k')
            if early:
                rows=[r for (cell,fam),rr in groups.items() if fam=='persistent_k' for r in rr if r['case']['lr']==lr and r['case']['error_rho']==early['case']['error_rho']]
                if rows:selected[anchor['cell']+'/persistent_k']=rows[0]
        expected={(c['cell'],c['family']) for c in cfg['cases']}
        missing=sorted('/'.join(k) for k in expected if '/'.join(k) not in selected)
        summaries[name]=dict(selections=selected,unavailable_families=missing,successful=a['successful'],failed=a['failed'],candidates={('/'.join(k)):v for k,v in groups.items()},config_sha256=a['config_sha256'])
    # Audit the original locked cohorts without loading any official test.
    from scripts.ndfa_strengthening_20260925.decision_control import audit as prior_audit
    prior=read(Path(plan['prior_round'])/'plan.json');prior_receipts={}
    for name in ['epochs','constant_rate','work_development','geometry_development']:
        _,a=prior_audit(prior,name);prior_receipts[name]=dict(successful=a['successful'],failed=a['failed'],config_sha256=a['config_sha256'])
    write(root/'development_summary.json',dict(studies=summaries,prior_validation_cohorts=prior_receipts,no_test_loaded=True))
    maybe_freeze(plan);return dict(studies={k:dict(successful=v['successful'],failed=v['failed'],unavailable=v['unavailable_families']) for k,v in summaries.items()})


def branches(plan):
    root=Path(plan['root']);prior=Path(plan['prior_round']);cfg=read(prior/'constant_rate/config.json');rows=[]
    for t in cfg['tasks']:
        c=cfg['cases'][t['case_index']];p=Path(cfg['output_root'])/c['id']/f"seed_{t['seed']}"/'endpoint.json';rows.append(read(p))
    def avg(family,key):
        r=[e for e in rows if e['case']['family']==family]
        return mean(e['final_validation'][key] for e in r) if len(r)==len(cfg['seeds']) and all(e['status']=='complete' for e in r) else None
    ea,la=avg('dfa_0.0003_early','accuracy'),avg('dfa_0.0003_late','accuracy');el,ll=avg('dfa_0.0003_early','loss'),avg('dfa_0.0003_late','loss')
    run=all(x is not None for x in [ea,la,el,ll]) and ea>la and el<ll
    gate=dict(run=run,rule='Positive mean DFA early-minus-late final validation accuracy and lower CE at the declared constant 0.0003 rate; no significance selection',early_accuracy=ea,late_accuracy=la,early_loss=el,late_loss=ll)
    write(root/'branch_gate.json',gate)
    if not run:
        write(root/'branch_summary.json',dict(skipped=True,reason='Declared development branch gate not met',gate=gate));maybe_freeze(plan);return gate
    specs=read(root/'designs.json');jobs=[]
    for dataset,reference in [('cifar','foof_cifar'),('nuisance','foof_nuisance')]:
        template=copy.deepcopy(specs[reference]['template']);template.update(budget_kind='updates',lr_schedule='constant',epochs=200,evaluate_every_epochs=5,branch_fractions=[.25,.75],representation_probes=dataset=='nuisance')
        cases=[launch.case(f'{credit}_{policy}',credit,'activity',lr=.0003,rho=1.,cell='cifar10' if dataset=='cifar' else 'nuisance',conditioning_schedule=policy) for credit in ['dfa','bp'] for policy in ['always','never']]
        name='branch_prefix_'+dataset;launch.make_study(plan,name,template,cases,plan['branch_seeds'],stage='development_intervention')
        jobs.append(launch.submit(plan,name,'train',hardware='h100' if dataset=='cifar' else 'h200'))
    launch.submit(plan,'all','branch_dispatch',dependencies=jobs);return gate


def branch_dispatch(plan):
    root=Path(plan['root']);jobs=[]
    for dataset in ['cifar','nuisance']:
        cfg,a=audit(plan,'branch_prefix_'+dataset);assert not a['failed'],'Prefix numerical failure; report rather than condition on successful prefixes'
        cases=[]
        for parent in cfg['cases']:
            for fraction in [.25,.75]:
                refs={}
                for seed in cfg['seeds']:
                    r=next(r for r in a['rows'] if r['case']['id']==parent['id'] and r['seed']==seed);step=int(r['completed_updates']*fraction);p=Path(r['folder'])/f'branch_{step}.pt'
                    refs[str(seed)]=dict(checkpoint=str(p),sha256=sha(p),step=step,source_config_sha256=a['config_sha256'])
                for policy in ['always','never']:
                    c=copy.deepcopy(parent);c.update(family=f"{parent['family']}_t{fraction:g}_{policy}",conditioning_schedule=policy,branch_sources=refs,target_epochs=int(cfg['epochs']*(fraction+.25)))
                    cases.append(c)
        template={k:v for k,v in cfg.items() if k not in ['cases','tasks','seeds','branch_fractions']};template['branch_fractions']=[]
        name='branches_'+dataset;launch.make_study(plan,name,template,cases,cfg['seeds'],stage='development_intervention')
        jobs.append(launch.submit(plan,name,'train',hardware='h100' if dataset=='cifar' else 'h200'))
    launch.submit(plan,'all','branch_report',dependencies=jobs);return dict(branch_jobs=jobs)


def branch_report(plan):
    root=Path(plan['root']);result={}
    for dataset in ['cifar','nuisance']:
        cfg,a=audit(plan,'branches_'+dataset);groups=defaultdict(list)
        for r in a['rows']:groups[r['case']['family']].append(r)
        result[dataset]={k:dict(successful=sum(r['status']=='complete' for r in rows),rows=[dict(seed=r['seed'],final_validation=r['final_validation'],folder=r['folder']) for r in rows]) for k,rows in groups.items()}
    write(root/'branch_summary.json',dict(skipped=False,studies=result,scope='Equal 50-epoch continuations from identical full training states at epochs 50 and 150; validation-only, not a proof of nuisance capture'))
    maybe_freeze(plan);return dict(branches='complete')


def maybe_freeze(plan):
    root=Path(plan['root'])
    if (root/'development_summary.json').exists() and (root/'branch_summary.json').exists():
        try:launch.submit(plan,'all','freeze')
        except FileExistsError:pass


def confirmation_specs(plan):
    root=Path(plan['root']);dev=read(root/'development_summary.json')['studies'];specs=read(root/'designs.json');out={}
    for name in ['foof_cifar','foof_nuisance','error_cifar','error_mnist','benchmarks']:
        selected=[copy.deepcopy(v['case']) for v in dev[name]['selections'].values()]
        if name=='benchmarks':selected=[c for c in selected if c['family'] not in ['centered','dfa_no_bn']]
        out['confirm_'+name]=dict(template=specs[name]['template'],cases=selected,hardware=specs[name]['hardware'])
    # The prior early activity candidate is selected solely on development CE.
    prior=Path(plan['prior_round']);wc=read(prior/'work_development/config.json');rows=[]
    for c in wc['cases']:
        if c['family']=='early_bn' and c['work_budget_seconds']==200.:
            rr=[read(Path(wc['output_root'])/c['id']/f'seed_{seed}'/'endpoint.json') for seed in wc['seeds']]
            if all(r['status']=='complete' for r in rr):rows.append((mean(r['best_validation']['loss'] for r in rr),-mean(r['best_validation']['accuracy'] for r in rr),c['id'],c))
    assert rows;early=copy.deepcopy(min(rows,key=lambda r:r[:3])[3]);early['family']='early_activity';out['confirm_foof_cifar']['cases'].append(early)
    for name,old,keep in [('confirm_epochs','epochs',lambda c:c['family'] in ['dfa_bn','activity_bn','early_bn']),('confirm_geometry','geometry_confirmation',lambda c:True),('confirm_timing','constant_rate',lambda c:c['lr']==.0003 and c['conditioning_schedule'] in ['early','late'])]:
        cfg=read(prior/old/'config.json');template={k:v for k,v in cfg.items() if k not in ['cases','tasks','seeds','selection_artifact','selection_artifact_sha256']}
        out[name]=dict(template=template,cases=[c for c in cfg['cases'] if keep(c)],hardware='h200' if name=='confirm_timing' else 'h100')
    return out


def freeze(plan):
    root=Path(plan['root']);assert (root/'branch_summary.json').exists()
    specs=confirmation_specs(plan);configs={}
    for name,spec in specs.items():
        cp=launch.make_study(plan,name,spec['template'],spec['cases'],plan['confirmation_seeds'],stage='joint_confirmation')
        configs[name]=dict(path=str(cp),sha256=sha(cp),runs=len(read(cp)['tasks']),hardware=spec['hardware'])
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),development_sha256=sha(root/'development_summary.json'),branch_sha256=sha(root/'branch_summary.json'),configs=configs,
        confirmation_seeds=plan['confirmation_seeds'],test_checkpoint='best_validation_CE; final secondary, except final primary for fixed-epoch geometry and timing',official_test_loaded=False,
        primary_family='All declared primary contrasts in report(); Holm across all, with unavailable tests assigned p=1; no optional seed extension')
    path=root/'joint_confirmation.json'
    if path.exists():
        previous=read(path);frozen['created_utc']=previous['created_utc'];assert previous==frozen
    else:write(path,frozen)
    jobs=[]
    for name,entry in configs.items():jobs.append(launch.submit(plan,name,'train',hardware=entry['hardware']))
    # A single global gate requires every included training cohort to terminate
    # before any test loader is reachable.
    gate=launch.submit(plan,'all','confirmation_gate',dependencies=jobs)
    evaluations=[launch.submit(plan,name,'evaluate',dependencies=[gate],hardware=e['hardware']) for name,e in configs.items()]
    launch.submit(plan,'all','report',dependencies=evaluations)
    return dict(training_runs=sum(e['runs'] for e in configs.values()),configs=configs)


def confirmation_gate(plan):
    root=Path(plan['root']);frozen=read(root/'joint_confirmation.json');inventory={}
    assert sha(root/'development_summary.json')==frozen['development_sha256'] and sha(root/'branch_summary.json')==frozen['branch_sha256']
    for name,entry in frozen['configs'].items():
        assert sha(Path(entry['path']))==entry['sha256'];_,a=audit(plan,name);inventory[name]=dict(successful=a['successful'],failed=a['failed'])
    write(root/'confirmation_training_gate.json',dict(passed=True,joint_freeze_sha256=sha(root/'joint_confirmation.json'),inventory=inventory))
    return inventory


def test_gate(plan,study):
    root=Path(plan['root']);frozen=read(root/'joint_confirmation.json');gate=read(root/'confirmation_training_gate.json')
    assert gate['passed'] and gate['joint_freeze_sha256']==sha(root/'joint_confirmation.json')
    for name,entry in frozen['configs'].items():assert sha(Path(entry['path']))==entry['sha256']
    assert study in frozen['configs']
    return frozen


def test_data(cfg,case,seed):
    """Called only after the global joint-confirmation training gate."""
    from torchvision.datasets import CIFAR10,MNIST,FashionMNIST
    import numpy as np
    if cfg['dataset']=='cifar10':
        source=CIFAR10(cfg['data_dir'],train=False,download=False)
        return torch.as_tensor(source.data).permute(0,3,1,2).contiguous(),torch.as_tensor(source.targets,dtype=torch.long)
    if cfg['dataset']=='synthetic':
        cell=cfg['cells'][case['cell']];rng=lambda stream:np.random.default_rng(np.random.SeedSequence([seed,stream]))
        projection=rng(0).normal(size=(28,64))/np.sqrt(28)
        x,y,_=train.nr.sample_multioutput_split(cfg['validation_examples'],rng=rng(4),n_classes=8,nuisance_dim=24,task_scale=cell['task_scale'],nuisance_scale=cell['nuisance_scale'],interaction=cell['interaction'],input_noise=cell['input_noise'],projection=projection)
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
    if cfg['dataset']=='benchmark' and case['cell']=='mnist_background_standard':
        import zipfile
        archive=Path(cfg['benchmark_root']).parent/'mnist_variations/mnist_background_images.zip'
        assert sha(archive)==cfg['benchmark_inventory'][case['cell']]['provenance']['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            with z.open('mnist_background_images_test.amat') as f:values=np.loadtxt(f,dtype=np.float32)
        assert values.shape==(50000,785) and np.isfinite(values).all()
        return torch.from_numpy(values[:,:784].copy()),torch.from_numpy(values[:,784].copy()).long()
    dataset=case.get('dataset','mnist') if cfg['dataset']=='digits' else case['cell'].split('_')[0]
    source=(MNIST if dataset=='mnist' else FashionMNIST)(cfg['data_dir'],train=False,download=False)
    images=source.data.float().flatten(1)/255.;labels=torch.as_tensor(source.targets,dtype=torch.long)
    if cfg['dataset']=='benchmark' and case['cell'].endswith('input_nuisance'):
        provenance=cfg['benchmark_inventory'][case['cell']]['provenance'];split=provenance['split_seed']
        bg=CIFAR10(cfg['data_dir'],train=True,download=False);raw=torch.as_tensor(bg.data)
        assert train.base.tensor_hash(raw)==provenance['background_source_sha256']
        order=torch.randperm(len(raw),generator=torch.Generator().manual_seed(split+2));pool=raw[order[45000:]].float()[:,2:30,2:30].mean(-1).flatten(1)/255.
        indices=torch.randint(len(pool),(len(images),),generator=torch.Generator().manual_seed(split+5));images=.35*images+.65*pool[indices]
    return images,labels


@torch.no_grad()
def evaluate(plan,study):
    root=Path(plan['root']);cfg,a=audit(plan,study);assert cfg['stage']=='joint_confirmation';test_gate(plan,study)
    folder=root/study/'test_evaluation';folder.mkdir(exist_ok=True)
    path=folder/'plan.json';test_plan=dict(config_sha256=a['config_sha256'],joint_freeze_sha256=sha(root/'joint_confirmation.json'),training_gate_sha256=sha(root/'confirmation_training_gate.json'),entries=[dict(case=r['case'],seed=r['seed'],status=r['status'],checkpoint_sha256=r['checkpoint_sha256'],best_checkpoint_sha256=r['best_checkpoint_sha256']) for r in a['rows']],historical_test_exposure_disclosed=True)
    if path.exists():assert read(path)==test_plan
    else:write(path,test_plan)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    cache={};rows=[]
    for r in a['rows']:
        row={k:r[k] for k in ['case','seed','status','training_seconds','completed_updates','attempts','flops','flop_scope']};row['metrics']={}
        if r['status']=='complete':
            key=(r['case']['cell'],r['seed'] if cfg['dataset']=='synthetic' else None)
            if key not in cache:cache[key]=test_data(cfg,r['case'],r['seed'])
            images,labels=cache[key];hashes=dict(images=train.base.tensor_hash(images),labels=train.base.tensor_hash(labels))
            out=folder/r['case']['id']/f"seed_{r['seed']}";out.mkdir(parents=True,exist_ok=True)
            for which in ['best','final']:
                checkpoint=Path(r['folder'])/f'{which}.pt';expected=r['best_checkpoint_sha256' if which=='best' else 'checkpoint_sha256'];assert sha(checkpoint)==expected
                result_path=out/f'{which}.json'
                if result_path.exists():
                    result=read(result_path);assert result['checkpoint_sha256']==expected and result['test_data']==hashes and sha(out/f'{which}.pt')==result['prediction_sha256']
                else:
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);args=train.args_for(cfg,r['case'],r['seed'],out,'cuda');train.base.configure(args)
                    from types import SimpleNamespace
                    data=SimpleNamespace(input_dim=saved['input_dim'],classes=saved['classes']);model=train.nr.restore_model(cfg,r['case'],args,data,saved['model']);model.training=False
                    norm=saved['input_normalization'];norm={k:v.cuda() for k,v in norm.items()} if norm else None
                    logits=torch.cat([model.forward(train.base.normalize(x.cuda(),norm) if norm else x.cuda())[0].cpu() for x in images.split(512)])
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite held-out logits; retain failure, no test-driven retry')
                    train.nr.atomic_save(dict(logits=logits,labels=labels,test_data=hashes,checkpoint_sha256=expected),out/f'{which}.pt')
                    result=dict(accuracy=float((logits.argmax(1)==labels).double().mean()),loss=float(F.cross_entropy(logits.double(),labels)),checkpoint_sha256=expected,test_data=hashes,prediction_sha256=sha(out/f'{which}.pt'))
                    write(result_path,result);del model
                row['metrics'][which]=result
            if r['case'].get('error_operator'):
                histories=read(Path(r['folder'])/'history.json');directions=[]
                for h in histories:
                    if h['step']>0 and h['training_seconds']<.25*r['case']['work_budget_seconds'] and h.get('error_direction'):
                        cos=h['error_direction']['layer_cosines_to_activity']
                        if any(v is None for v in cos):directions=[];break
                        directions.append(mean(1-v for v in cos))
                row['mean_early_direction_change']=mean(directions) if directions else None
        rows.append(row)
    result=dict(study=study,rows=rows,config_sha256=a['config_sha256'],test_plan_sha256=sha(path));write(folder/'summary.json',result);return dict(study=study,evaluated=len(rows))


def report(plan):
    root=Path(plan['root']);frozen=read(root/'joint_confirmation.json');datasets={name:read(root/name/'test_evaluation/summary.json') for name in frozen['configs']}
    def vals(study,family,cell=None,metric='accuracy',which='best'):
        rows=[r for r in datasets[study]['rows'] if r['case']['family']==family and (cell is None or r['case']['cell']==cell)]
        if len(rows)!=len(plan['confirmation_seeds']) or any(r['status']!='complete' for r in rows):return None
        cfg=read(root/study/'config.json')
        if (cfg['budget_kind']=='work' or metric=='work') and any(r['attempts']!=1 for r in rows):return None
        if metric=='direction':
            if any(r.get('mean_early_direction_change') is None for r in rows):return None
            return {r['seed']:r['mean_early_direction_change'] for r in rows}
        if metric=='work':return {r['seed']:r['training_seconds'] for r in rows}
        return {r['seed']:r['metrics'][which][metric] for r in rows}
    def delta(study,left,right,cell=None,metric='accuracy',which='best'):
        a,b=vals(study,left,cell,metric,which),vals(study,right,cell,metric,which)
        if a is None or b is None:return []
        assert a.keys()==b.keys();scale=100 if metric=='accuracy' else -1.
        return [scale*(a[s]-b[s]) for s in sorted(a)]
    primary={}
    def test(name,v):primary[name]=paired_summary(v)
    cf='confirm_foof_cifar'
    for family in ['activity','early_activity','foof_dfa']:test('cifar_'+family,delta(cf,family,'dfa_adamw'))
    d=delta(cf,'foof_dfa','dfa_sgd');b=delta(cf,'foof_bp','bp_sgd');test('FOOF_DFA_minus_BP_gain',[x-y for x,y in zip(d,b)] if d and b else [])
    for family in ['full','centered']:test('geometry_'+family,delta('confirm_geometry',family,'diagonal_covariance_plus_mean',which='final'))
    d=delta('confirm_timing','dfa_0.0003_early','dfa_0.0003_late',which='final');b=delta('confirm_timing','bp_0.0003_early','bp_0.0003_late',which='final')
    test('timing_DFA',d);test('timing_DFA_minus_BP',[x-y for x,y in zip(d,b)] if d and b else [])
    for comparator in ['activity','early_diagonal']:
        for metric in ['accuracy','loss']:test(f'error_vs_{comparator}_{metric}',delta('confirm_error_cifar','early_k',comparator,metric=metric))
    directions=vals('confirm_error_cifar','early_k',metric='direction');test('error_direction_above_threshold',[0. if abs(v-.001)<1e-12 else v-.001 for v in directions.values()] if directions else [])
    bench='confirm_benchmarks';equivalence={}
    for dataset in ['mnist','fashion']:
        noisy=delta(bench,'activity','dfa_bn',dataset+'_input_nuisance');clean=delta(bench,'activity','dfa_bn',dataset+'_clean')
        test(dataset+'_nuisance_accuracy',noisy);test(dataset+'_nuisance_loss',delta(bench,'activity','dfa_bn',dataset+'_input_nuisance',metric='loss'))
        test(dataset+'_nuisance_minus_clean_gain',[x-y for x,y in zip(noisy,clean)] if noisy and clean else [])
        for condition in ['clean','label_noise']:
            v=delta(bench,'activity','dfa_bn',dataset+'_'+condition);r=paired_summary(v)
            half=float(stats.t.ppf(.95,len(v)-1))*stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
            equivalence[dataset+'_'+condition]=dict(**r,margin_pp=1.,within_margin=half is not None and r['mean']-half>-1 and r['mean']+half<1,scope='Secondary TOST-equivalent 90% paired interval; not inferred from nonsignificance; no familywise null claim')
    for metric in ['accuracy','loss']:test('standard_background_'+metric,delta(bench,'activity','dfa_bn','mnist_background_standard',metric=metric))
    holm(primary)
    def passes(key):r=primary[key];return r['mean'] is not None and r['mean']>0 and r['holm_p']<plan['alpha']
    error_keys=[k for k in primary if k.startswith('error_')];error_pass=all(passes(k) for k in error_keys)
    covariance=passes('geometry_full') and passes('geometry_centered');specific=passes('timing_DFA') and passes('timing_DFA_minus_BP')
    practical={family:passes('cifar_'+family) and primary['cifar_'+family]['mean']>=plan.get('practical_minimum_gain_pp',1.) and mean(delta(cf,family,'dfa_adamw',metric='loss') or [-float('inf')])>=0 for family in ['activity','early_activity','foof_dfa']}
    v=delta('confirm_epochs','early_bn','activity_bn',which='final');r=paired_summary(v);a,b=vals('confirm_epochs','early_bn',metric='work'),vals('confirm_epochs','activity_bn',metric='work')
    ratio=mean(a[s]/b[s] for s in sorted(a)) if a and b else None;lower=r['mean']-float(stats.t.ppf(.95,len(v)-1))*stdev(v)/math.sqrt(len(v)) if v else None
    temporary=lower is not None and lower>-.5 and ratio is not None and ratio<=.8
    result=dict(primary_tests=primary,error_factor_pass=error_pass,error_disposition='eligible for main-text conditional claim' if error_pass else 'short appendix note; no further error-factor rescue round',practical=practical,covariance=covariance,DFA_specific_timing=specific,temporary_activity_noninferiority=dict(summary=r,lower_one_sided=lower,mean_work_ratio=ratio,passes=temporary),secondary_small_effect_predictions=equivalence,
        no_optional_extensions=True,scope='One frozen joint confirmation; historically used official test sets; model-seed uncertainty on fixed real-data benchmarks; synthetic seeds also regenerate data; availability requires every declared pair and uninterrupted work-budget runs')
    write(root/'decision.json',result)
    lines=['# Integrated nDFA decision','',f"Error factor: **{result['error_disposition']}**.",'',f'Practical gates: {practical}. Covariance: {covariance}. DFA-specific timing: {specific}. Temporary-activity efficiency: {temporary}.','',
           '| Prespecified primary contrast | Mean | 95% paired t interval | Holm p |','|---|---:|---|---:|']
    for key,value in primary.items():lines.append(f"| {key} | {value['mean']:.5g} | [{value['low']:.5g}, {value['high']:.5g}] | {value['holm_p']:.4g} |" if value['mean'] is not None else f'| {key} | unavailable | incomplete/failed/interrupted cohort | 1 |')
    lines+=['','Positive loss contrasts mean lower CE. Accuracy contrasts are percentage points; the directional contrast is dimensionless. Intervals are individual, while tests use the declared global Holm family.','',
            '| Study / cell / family | Test accuracy | Test CE | Mean learning work (s) | Estimated linear-algebra FLOPs |','|---|---:|---:|---:|---:|']
    for name,data in datasets.items():
        groups=defaultdict(list)
        for row in data['rows']:groups[(row['case']['cell'],row['case']['family'])].append(row)
        which='final' if name in ['confirm_geometry','confirm_timing'] else 'best'
        for (cell,family),rows in sorted(groups.items()):
            if all(r['status']=='complete' for r in rows):
                lines.append(f"| {name} / {cell} / {family} | {100*mean(r['metrics'][which]['accuracy'] for r in rows):.2f}% | {mean(r['metrics'][which]['loss'] for r in rows):.4f} | {mean(r['training_seconds'] for r in rows):.1f} | {mean(sum(r['flops'].values()) for r in rows):.4g} |")
            else:lines.append(f'| {name} / {cell} / {family} | incomplete/failed | — | — | — |')
    lines+=['',component.FLOP_SCOPE,'','Standard and constructed background benchmarks are labeled separately. Failure of a preregistered prediction remains visible. Positive scalar error conditioning is algebraically A under norm matching, not an independent replication. Venue selection requires scientific review of these results; no acceptance guarantee follows from a gate.']
    (root/'decision.md').write_text('\n'.join(lines)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--action',required=True);p.add_argument('--study',required=True);args=p.parse_args()
    plan=read(args.plan);source_gate(plan);root=Path(plan['root'])
    if args.action=='train':
        cfg=read(root/args.study/'config.json')
        if cfg['stage']!='technical_verification':assert read(root/'technical_gate.json')['passed']
        if cfg['stage']=='joint_confirmation':
            frozen=read(root/'joint_confirmation.json');assert frozen['configs'][args.study]['sha256']==sha(root/args.study/'config.json')
        cp=root/args.study/'config.json';sys.argv=['integrated_train','--config',str(cp),'--config-sha256',sha(cp)];train.main();return
    actions=dict(verify=verify,collect=collect,branches=branches,branch_dispatch=branch_dispatch,branch_report=branch_report,freeze=freeze,confirmation_gate=confirmation_gate)
    if args.action=='evaluate':result=evaluate(plan,args.study)
    elif args.action=='report':result=report(plan)
    else:result=actions[args.action](plan)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
