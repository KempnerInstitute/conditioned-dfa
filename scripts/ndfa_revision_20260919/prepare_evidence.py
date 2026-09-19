from pathlib import Path
import json,hashlib,os,shutil
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'assets/ndfa_revision_20260919';OUT.mkdir(parents=True,exist_ok=True)
LEGACY=Path(os.environ.get('INFODFA_LEGACY_RESULTS',ROOT.parent/'Info-Man/results'))
result={'scope':'Seed-level reaggregation of saved training endpoints; no retraining.'}
cell=['condition','input_noise','n_train','train_label_noise']
best=pd.read_csv(LEGACY/'infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_best_by_method.csv')
p=best.pivot_table(index=cell+['nuisance_energy_ratio'],columns='method',values='test_mean').reset_index();p['gain']=100*(p.ndfa_random-p.dfa_random);p=p.dropna(subset=['gain'])
pros=pd.read_csv(ROOT/'results/infodfa_prospective_diagnostic_v1/prospective_diagnostic_cells.csv')
p[cell+['nuisance_energy_ratio','gain']].to_csv(OUT/'nuisance_correlation_cells.csv',index=False)
result['correlations']={'current_best_by_method':float(spearmanr(p.nuisance_energy_ratio,p.gain).statistic),'cells':len(p),'prospective_columns':list(pros)}
if 'designed_nuisance_ratio' in pros and 'gain_ndfa_pp' in pros:
 result['correlations']['saved_prospective']=float(spearmanr(pros.designed_nuisance_ratio,pros.gain_ndfa_pp).statistic)
 common=p.merge(pros,on=cell);result['correlations']['maximum_gain_difference_pp']=float(abs(common.gain-common.gain_ndfa_pp).max())
# Endpoints grouped by the actual crossed training keys.
f=LEGACY/'infodfa_hard_cifar100_confirm_aggregate_v2/dfa_convnet_all.csv';raw=pd.read_csv(f);raw=raw.sort_values('epoch').groupby(['method','seed','feedback_seed'],dropna=False).tail(1)
rows=[]
for method,g in raw.groupby('method'):
 seeds=g.groupby('seed').test_acc.mean();rows.append({'method':method,'training_runs':len(g),'global_seeds':len(seeds),'mean_accuracy_pct':100*g.test_acc.mean(),'crossed_run_sem_pp':100*g.test_acc.sem(),'seed_mean_sem_pp':100*seeds.sem()})
result['hard_cifar100']=rows
raw.groupby(['method','seed']).test_acc.agg(['mean','count']).reset_index().to_csv(OUT/'hard_cifar100_seed_means.csv',index=False)
# Raw synthetic endpoints; preserve feedback matching and average the entire grid per seed.
keys=cell+['method','seed','feedback_seed','feedback_rank'];cols=keys+['epoch','test_acc']
legacy_all=pd.read_csv(LEGACY/'infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv',usecols=cols)
legacy_all=legacy_all[(legacy_all.feedback_rank==0)&legacy_all.method.isin(['ndfa_random','dfa_random'])].sort_values('epoch').groupby(keys).tail(1)
legacy=legacy_all[legacy_all.method.eq('ndfa_random')]
frames=[pd.read_csv(f,usecols=cols) for f in (ROOT/'results/infodfa_bn_baseline_v1/synthetic').glob('*/ntrain_*/label_*/input_*/dfa_multioutput_results.csv')]
bn=pd.concat(frames);bn=bn[(bn.feedback_rank==0)&bn.method.eq('dfa_random')].sort_values('epoch').groupby(keys).tail(1)
join=cell+['seed','feedback_seed'];both=legacy[join+['test_acc']].merge(bn[join+['test_acc']],on=join,validate='one_to_one',suffixes=('_a','_bn'));assert len(both)==128*5*3
both['delta']=100*(both.test_acc_a-both.test_acc_bn)
seed=both.groupby(['condition','seed']).delta.mean().reset_index();grid=both.groupby(cell).delta.mean().reset_index();rows=[]
for c,g in seed.groupby('condition'):
 rows.append({'regime':c,'paired_mean_pp':float(g.delta.mean()),'seed_sem_pp':float(g.delta.sem()),'seeds':len(g),'old_cell_sem_pp':float(grid.loc[grid.condition==c,'delta'].sem())})
result['bn_control']=rows;seed.to_csv(OUT/'bn_control_seed_differences.csv',index=False)
# Full-rank mechanism correlation in this original 128-cell cohort.
allcols=cell+['seed','feedback_seed','feedback_rank','epoch','method','test_acc']
# Reuse compact cell-ratio mapping; full-rank DFA read is restricted to required columns.
dfa=legacy_all[legacy_all.method.eq('dfa_random')]
a=legacy.groupby(cell).test_acc.mean();b=dfa.groupby(cell).test_acc.mean();full=(100*(a-b)).reset_index(name='gain');full=full.merge(p[cell+['nuisance_energy_ratio']],on=cell,validate='one_to_one');result['correlations']['fixed_full_rank']=float(spearmanr(full.nuisance_energy_ratio,full.gain).statistic)
(OUT/'saved_results_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
for name in ['endpoints','contrasts']:
 shutil.copyfile(ROOT/f'results/ndfa_revision_followups_audit_20260918/{name}.csv',OUT/f'followup_{name}.csv')
boundary=ROOT/'results/imagenet100_strongform_v1/strongform_multiseed_summary.csv'
shutil.copyfile(boundary,OUT/'imagenet_clean_summary.csv')
noise=[]
for method in ['dfa','ndfaDiag','ndfaFull']:
 for depth in ['layer4','all']:
  for seed in range(3):
   path=ROOT/f'results/imagenet100_noisy_deconfound_v1/{method}_{depth}_seed{seed}/imagenet_credit_assignment.csv'
   frame=pd.read_csv(path);last=frame.dropna(subset=['val_top1']).iloc[-1]
   noise.append({'method':method,'depth':depth,'seed':seed,'val_top1':float(last.val_top1),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
pd.DataFrame(noise).to_csv(OUT/'imagenet_noisy_endpoints.csv',index=False)
configs=['ndfa_bn_confirmation_20260915','ndfa_bn_factors_h100_234min_20260915',
         'ndfa_bn_optimizer_60min_20260915','ndfa_bn_forward_decorrelation_20260915',
         'ndfa_revision_followups_20260918','ndfa_revision_followups_20260918',
         'ndfa_revision_followups_20260918','ndfa_submission_confirmation_20260915']
ledger={}
for i,name in enumerate(configs,1):
 path=ROOT/'configs'/f'{name}.json';a=json.loads(path.read_text())
 ledger[f'S{i}']={'configuration':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'source_sha256':a.get('source_sha256',a.get('training_files_sha256',{})),
                  'phase':{5:'work',6:'width',7:'stability'}.get(i),'hardware':'H200' if i==8 else 'H100 80GB HBM3'}
(OUT/'study_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
