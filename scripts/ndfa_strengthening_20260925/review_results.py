"""Read audited endpoints and write the development-results review.

Run the existing per-study summarizers first, preserving their outputs under
job_results_check_v1. This script checks pairing and derives descriptive tables;
it neither runs training nor opens held-out test data.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from scripts.ndfa_strengthening_20260925.freeze_round2 import ROOT
from scripts.ndfa_strengthening_20260925.baseline_development import learning_rate

BASE=ROOT/'results/ndfa_strengthening_20260925'
AUDIT=BASE/'job_results_check_v1'


def read(path):return json.loads(path.read_text())


def pairing(inventory):
    reports=[]
    for item in inventory:
        cfg=read(BASE/item['study']/'config.json');groups=defaultdict(list);errors=Counter()
        for case in cfg['cases']:
            for seed in cfg['development_seeds']:
                run=Path(cfg['output_root'])/case['id']/f'seed_{seed}'
                m,e=read(run/'manifest.json'),read(run/'endpoint.json')
                key=(case.get('cell','cifar10'),case.get('activation','relu'),tuple(case.get('hidden_dims',cfg['hidden_dims'])),seed)
                groups[key].append((m,e))
                if e['status']=='numerical_failure':errors[e['error']]+=1
        for key,values in groups.items():
            for field in ['data','initial_parameter_sha256']:
                assert len({json.dumps(m[field],sort_keys=True) for m,e in values})==1,(item['study'],key,field)
            feedback={m.get('feedback_sha256',m.get('initial_feedback_sha256')) for m,e in values}
            assert len(feedback)==1 and None not in feedback,(item['study'],key,'feedback')
            streams={(e['stream_sha256'],) if 'stream_sha256' in e else (e['order_sha256'],e['augmentation_sha256']) for m,e in values if e['status']=='complete'}
            assert len(streams)<=1,(item['study'],key,'stream')
        reports.append(dict(study=item['study'],paired_groups=len(groups),pairing_verified=True,numerical_failure_causes=dict(errors)))
    return reports


def main():
    inventory=read(AUDIT/'inventory.json');audits=read(AUDIT/'audit_summary.json')
    assert len(inventory)==len(audits)==15
    assert all(all(a['saved_summary_matches'].values()) for a in audits)
    paired=pairing(inventory)
    (AUDIT/'pairing_and_failures.json').write_text(json.dumps(paired,indent=2)+'\n')
    summaries={item['study']:read(AUDIT/f"{item['study']}.json") for item in inventory}
    totals=Counter()
    for item in inventory:totals.update(item['counts'])
    assert totals==Counter(complete=1998,numerical_failure=40)
    new_names=['cifar_baseline_boundaries_v1','cifar_baseline_horizon_v1','cifar_foof_boundaries_v1','cifar_foof_horizon_v1','digits_v1','alignment_synthetic_v1','alignment_cifar_v1','geometry_cifar_v1','synthetic_boundaries_v2']
    new_totals=Counter()
    for item in inventory:
        if item['study'] in new_names:new_totals.update(item['counts'])
    assert new_totals==Counter(complete=451,numerical_failure=15)
    findings=dict(created_utc=datetime.now(timezone.utc).isoformat(),total=dict(totals),latest_round=dict(new_totals),official_test_loaded=False,
                  audit_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in AUDIT.glob('*.json') if p.name!='findings.json'},comparisons={},alignment={})
    def selected(study,key):return summaries[study]['selections'][key]
    def contrast(study,a,b):
        x={v['seed']:v['final_validation'] for v in selected(study,a)['seed_values']}
        y={v['seed']:v['final_validation'] for v in selected(study,b)['seed_values']}
        assert x.keys()==y.keys()
        return [dict(seed=s,accuracy_delta_pp=100*(y[s]['accuracy']-x[s]['accuracy']),loss_delta=y[s]['loss']-x[s]['loss']) for s in sorted(x)]
    for norm in ['none','bn']:
        findings['comparisons'][f'cifar_{norm}_activity_minus_raw']=contrast('cifar_baseline_boundaries_v1',f'cifar10/dfa_{norm}_none',f'cifar10/dfa_{norm}_activity')
    for setting in ['mnist_tanh','fashion_mnist_tanh','mnist_relu']:
        findings['comparisons'][f'{setting}_activity_minus_raw']=contrast('digits_v1',f'{setting}/dfa_none',f'{setting}/dfa_activity')
    for name in ['alignment_synthetic_v1','alignment_cifar_v1']:
        summary=summaries[name];cfg=read(BASE/name/'config.json');schedules=defaultdict(list)
        for row in summary['paired_schedule_contrasts']:schedules[(row['cell'],row['normalization'],row['contrast'])].append(row)
        # Ratio of nominal scalar learning-rate sums, not actual optimizer steps.
        examples=next(iter(cfg['cells'].values()))['n_train'] if cfg['dataset']=='synthetic' else 45000
        steps=math.ceil(examples/cfg['batch_size']);total=steps*cfg['epochs'];warmup=steps*cfg['warmup_epochs'];q=total//4
        early=sum(learning_rate(1,s,total,warmup) for s in range(q));late=sum(learning_rate(1,s,total,warmup) for s in range(total-q,total))
        findings['alignment'][name]=dict(nominal_lr_sum_early_over_late=early/late,contrasts=[dict(cell=k[0],normalization=k[1],contrast=k[2],
            accuracy_delta_pp=100*mean(r['accuracy_delta'] for r in rows),per_seed_accuracy_delta_pp=[100*r['accuracy_delta'] for r in rows],
            loss_delta=mean(r['loss_delta'] for r in rows)) for k,rows in schedules.items()])
    (AUDIT/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    text=[]
    def add(s=''):text.append(s)
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows:add('| '+' | '.join(map(str,row))+' |')
        add()
    add('# nDFA development results: completed-job review')
    add();add('Checked 25 September 2026, evening Eastern time (26 September UTC). All 15 submitted strengthening studies have finished. The scientific result is encouraging for activity geometry and temporary conditioning, while stronger baselines limit broader performance claims. These are validation-development results, not held-out confirmation.')
    add();add('## Execution and reproducibility')
    add();add('Across the full strengthening campaign, 1,998 runs finished normally and 40 ended in recorded numerical failure, accounting for all 2,038 declared runs. In the latest 466-run round, 451 finished normally and 15 failed numerically. There are no missing runs. All 15 scheduled analysis jobs completed. The 32 canceled entries in scheduler history are earlier pending tasks migrated between partitions; their replacements produced the expected results. No recovery run is needed.')
    add();table(['Latest study','Declared runs','Successful','Numerical failures'],[[n,str(next(i['expected'] for i in inventory if i['study']==n)),summaries[n]['endpoint_counts'].get('complete',0),summaries[n]['endpoint_counts'].get('numerical_failure',0)] for n in new_names])
    add('The latest failures are four forward-decorrelation runs and eleven FOOF runs at aggressive extension settings with large learning rates or inadequate damping. They remain in the development record and do not enter selection as successful trials. None of the digit, alignment, geometry or longer-horizon runs failed numerically.')
    add();add('Recomputed all 15 summaries from the saved endpoints and histories after checking frozen source hashes, configurations, manifests, completed update counts and final-checkpoint hashes. Every candidate, selected configuration and endpoint count agrees exactly with the scheduled analysis output. Within each study, paired conditions share data, initial weights, feedback matrices and complete minibatch/augmentation streams. This is a saved-artifact audit; it does not rerun training or independently regenerate every validation prediction. No official test data were opened.')
    add();add('## CIFAR: activity conditioning survives stronger baselines')
    add();add('The following means use two paired development seeds at 200 epochs. Select each method by mean final validation cross-entropy over its pooled original and extension grids. Accuracy is displayed at that selected configuration; it is not a separate accuracy-based selection. All eleven CIFAR families now select interior points in the tested coordinates.')
    add();rows=[]
    for study in ['cifar_baseline_boundaries_v1','cifar_foof_boundaries_v1']:
        for key,s in summaries[study]['selections'].items():rows.append([key.split('/')[1],f"{100*s['mean_validation_accuracy']:.2f}%",f"{s['mean_validation_loss']:.3f}"])
    table(['Method family','Validation accuracy','Validation CE'],rows)
    add('Activity DFA improves on raw DFA by 2.67 percentage points without BN and 3.15 points with BN; both paired seeds favor it in each comparison, and CE also improves. Its BN result slightly exceeds ordinary BP+BN under the tested recipes. The strongest baseline is EMA FOOF-BP, which reaches about 72% accuracy and remains substantially ahead. These are equal-epoch development comparisons, not evidence that nDFA beats BP at matched compute.')
    add();add('At 400 epochs, first-grid activity-DFA+BN reaches 68.46%, compared with 65.51% for raw DFA+BN; FD-DFA reaches 66.00%. Thus, the within-BN activity benefit does not disappear merely by doubling this horizon. Several methods gain accuracy while validation CE worsens, and FD-DFA continues to improve. A fixed horizon is not proof of convergence. The 400-epoch runs use fixed first-grid winners with stretched schedules, not the newly pooled winners, so differences between the 200- and 400-epoch tables are not always pure horizon effects.')
    add();add('## Geometry: covariance matters beyond the tested mean controls')
    add();add('The CIFAR mechanism study holds optimizer and learning rate at first-grid activity anchors, changes only the moment operator, and searches the same three damping values for each operator. Its two seeds are development seeds. All five operators use the same dense-solve implementation; separately selected endpoint means follow.')
    add();rows=[]
    for statistic in ['full','centered','diagonal','diagonal_covariance_plus_mean','isotropic_covariance_plus_mean']:
        rows.append([statistic,*[f"{100*selected('geometry_cifar_v1',f'cifar10/dfa_{norm}_{statistic}')['mean_validation_accuracy']:.2f}%" for norm in ['none','bn']]])
    table(['Moment operator','Without BN','With BN'],rows)
    add('Full and centered covariance outperform the tested diagonal and mean-preserving alternatives. At each matched damping, full moments beat every diagonal/mean-only control in both seeds and both normalization conditions. Those six comparisons per normalization reuse two seeds and are not six independent replications. Centered covariance is close to full moments on CIFAR; with BN it has slightly lower selected CE despite lower accuracy. On nuisance-dominant synthetic data without BN, centered covariance exceeds full moments at every tested damping/seed pair. On task-aligned synthetic data, gains disappear and BN can favor the simpler operators.')
    add();add('This is evidence against explaining the benefit solely by suppressing the mean direction under these recipes. It supports a role for centered covariance structure, with scope limited by provisional optimizer/rate anchors, two seeds, and remaining damping boundaries in several control families. The dense full rule and its mathematically equivalent sample-space bridge differ by about 0.6 percentage points in the CIFAR BN trajectory at the anchor; implementation sensitivity should be retained in confirmation rather than treating those trajectories as interchangeable.')
    add();add('## Temporary conditioning: the most useful new intervention')
    add();add('Four paired global seeds compare always, never, first-quarter and last-quarter conditioning at the same activity-selected optimizer/rate. These are fixed-recipe interventions; the never-conditioned arm is not the independently tuned raw-DFA baseline.')
    add();rows=[]
    for name,cell,norm in [('alignment_cifar_v1','cifar10','none'),('alignment_cifar_v1','cifar10','bn'),('alignment_synthetic_v1','nuisance','none'),('alignment_synthetic_v1','nuisance','bn')]:
        rows.append([f'{cell}, {norm}',*[f"{100*selected(name,f'{cell}/dfa_{norm}_{s}')['mean_validation_accuracy']:.2f}%" for s in ['never','always','early','late']]])
    table(['Condition','Never','Always','First quarter','Last quarter'],rows)
    add('First-quarter conditioning retains nearly all of the full-training accuracy benefit. Early minus late is positive in all four CIFAR seeds with and without BN, and in all four nuisance seeds in both normalization conditions. On CIFAR with BN, first-quarter conditioning averages 67.93% versus 68.13% for always conditioning, with roughly one-third less measured learning work in this instrumented study. It reduces the number of conditioned updates by 75%; that is not a 75% reduction in total training cost. This gives a concrete candidate for fresh-seed matched-work evaluation.')
    add();add('The mechanistic explanation is not settled. Raw alignment onset is often unchanged or later with conditioning; harmful projected-gradient area decreases in some layers and conditions but not all. Do not claim that conditioning universally removes an anti-alignment phase. Moreover, the first and last quarters have equal update counts but different learning rates: the nominal summed learning rate is about 15 times larger in the first quarter. This is a diagnostic of the schedule, not a measure of actual Adam or momentum displacement. Equal-learning-rate windows or checkpoint-branch interventions are needed to distinguish an early representation effect from schedule timing. A matched BP activity-conditioning intervention is also needed before calling this DFA-specific.')
    add();add('## Strong digit baselines narrow the older claim')
    add();rows=[]
    for setting in ['mnist_tanh','fashion_mnist_tanh','mnist_relu']:
        rows.append([setting,*[f"{100*selected('digits_v1',f'{setting}/{m}')['mean_validation_accuracy']:.2f}%" for m in ['bp_none','dfa_none','dfa_activity','bp_foof']]])
    table(['Setting','BP','DFA','Activity DFA','FOOF-BP'],rows)
    add('With stronger optimizers and 60 epochs, the old large activity gains on digits largely disappear. Activity DFA is below raw DFA on MNIST-tanh, modestly above it on Fashion-tanh, and close on MNIST-ReLU. The Fashion accuracy increase accompanies worse validation BCE in both seeds. Nine of twelve families retain a tuning boundary and activity damping was fixed, so this is a baseline-adequacy screen rather than a final optimized ranking. It does establish that the historical short-budget improvements should not be presented as general superiority on these datasets. E/K were not tested in this stronger-baseline grid.')
    add();add('## Synthetic regimes and the next scientific decisions')
    add();rows=[]
    for cell in ['nuisance','low_sample','mixed','task_aligned']:
        rows.append([cell,*[f"{100*selected('synthetic_boundaries_v2',f'{cell}/{m}')['mean_validation_accuracy']:.2f}%" for m in ['dfa_none_none','dfa_none_activity','dfa_bn_none','dfa_bn_activity']]])
    table(['Regime','DFA','Activity DFA','DFA+BN','Activity DFA+BN'],rows)
    add('The pooled synthetic search preserves the strong nuisance-regime benefit, with smaller gains in low-sample and mixed regimes and essentially no gain in the task-aligned regime. Selected raw DFA now has ordinary finite training losses: roughly 1.4 in the nuisance cell, rather than the extreme losses in the historical shared-rate sweep. The benefit therefore survives removing that earlier catastrophic instability. Conditioning also lowers training loss substantially, so this comparison still includes an optimization benefit and does not isolate generalization at matched training fit. The two remaining synthetic tuning boundaries are task-aligned activity families selecting progressively weaker conditioning. This is not a reason to keep launching unbounded searches.')
    add();add('Prioritize the covariance mechanism and temporary conditioning. Next, freeze a small fresh-seed CIFAR comparison of tuned DFA+BN, full-time and early-only activity DFA+BN, BP+BN, and FOOF-BP at matched measured work, with validation-selected horizons/checkpoints. Resolve the relevant geometry-control damping boundaries, then confirm full/centered/mean-preserving contrasts. Use an equal-rate early/late or shared-checkpoint intervention, including a BP control, to test the mechanism. Keep error-factor expansion and the larger application conditional on that result. The results justify a focused stronger paper; they do not establish a general advantage over strong BP baselines.')
    add();add('No training jobs were resubmitted during this check: every declared run already has an audited terminal record. Raw evidence, recomputed summaries, paired values and failure causes are in `results/ndfa_strengthening_20260925/job_results_check_v1/`.')
    destination=ROOT/'docs/research/ndfa_results_review_20260925.md';destination.write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(report=str(destination),total=dict(totals),latest_round=dict(new_totals),paired_groups=sum(p['paired_groups'] for p in paired))))


if __name__=='__main__':main()
