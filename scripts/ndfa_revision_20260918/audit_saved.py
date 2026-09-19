"""Recompute revision summaries from saved records, without training/inference."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.ndfa_revision_math import paired_seed_summary, whole_seed_bootstrap


def main():
    out=ROOT/'results/ndfa_revision_saved_audit_20260918'
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def bind(path):
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    seeds=pd.read_csv(bind(ROOT/'results/infodfa_seedlevel_stats_v1/seedlevel_seed_deltas.csv'))
    synthetic=[]
    for condition,g in seeds.groupby('condition',sort=False):
        assert len(g)==5 and not g.seed.duplicated().any()
        row={'condition':condition,**paired_seed_summary(g.mean_delta_pp)}
        row['seed_percentile_interval']=whole_seed_bootstrap(g.mean_delta_pp)
        synthetic.append(row)
    audit=json.loads(bind(ROOT/'docs/research/ndfa_submission_confirmation_evaluation_20260915/audit.json').read_text())
    tails=[]
    torch.set_num_threads(4)
    for row in audit['case_results']:
        file=bind(Path(row['case_directory'])/'predictions.pt')
        assert hashes[str(file.relative_to(ROOT))]==row['predictions_sha256']
        saved=torch.load(file,map_location='cpu',weights_only=True)
        z=saved['logits'].double();y=saved['labels'];n=len(y)
        ce=torch.logsumexp(z,1)-z[torch.arange(n),y]
        assert abs(ce.mean().item()-row['metrics']['loss'])<=2e-12*max(1.,row['metrics']['loss'])
        ordered=ce.sort(descending=True).values;total=ce.sum().item()
        quantiles=torch.quantile(ce,torch.tensor([.5,.9,.99,1.],dtype=torch.float64)).tolist()
        tails.append({'method':row['method'],'replicate':row['replicate'],'mean_ce':ce.mean().item(),
                      'accuracy_pct':100*(z.argmax(1)==y).double().mean().item(),
                      **dict(zip(['median_ce','p90_ce','p99_ce','max_ce'],quantiles)),
                      'max_abs_logit':z.abs().max().item(),'top1pct_loss_share':ordered[:n//100].sum().item()/total,
                      'top10pct_loss_share':ordered[:n//10].sum().item()/total})
    tails=pd.DataFrame(tails);tails.to_csv(out/'loss_tails_by_seed.csv',index=False)
    tails.groupby('method').mean(numeric_only=True).drop(columns='replicate').to_csv(out/'loss_tails_summary.csv')
    metadata=json.loads(bind(ROOT/'docs/research/figures/ndfa_cifar10_confirmation_20260914.json').read_text())
    directory=(ROOT/metadata['input_dir']).resolve()
    endpoints=pd.DataFrame(json.loads(bind(directory/'endpoints.json').read_text()))
    pivot=endpoints.pivot(index='seed',columns='name',values='test_accuracy')
    assert len(pivot)==8 and not pivot.isna().any().any()
    diagonal=paired_seed_summary(100*(pivot.ndfa-pivot.diagonal))
    diagonal['seed_differences_pp']=(100*(pivot.ndfa-pivot.diagonal)).tolist()
    diagonal['selection']='Additional direct contrast of previously frozen final models; not in original five-comparison family.'
    endpoints.to_csv(out/'metric_control_endpoints.csv',index=False)
    # Complete crossed spatial populations; no independent-run tests or SEM.
    spatial=[]
    for dataset in ['cifar10','cifar100']:
        for amplitude in [0.,.5,1.,2.]:
            directory=(ROOT/'results/infodfa_capable_cifar10_v1' if dataset=='cifar10' and amplitude==0 else
                       ROOT/'results'/('infodfa_spatialkron_nuisance' if dataset=='cifar10' else 'infodfa_spatialkron_nuisance_cifar100')/f'alpha{amplitude}')
            frames={}
            for method in ['bp','dfa_random','ndfa_random','ndfa_spatial_kron']:
                files=sorted((directory/method).glob('*dfa_convnet_results.csv'))
                assert files,(dataset,amplitude,method)
                d=pd.concat([pd.read_csv(bind(f)) for f in files],ignore_index=True)
                d=d[d.epoch==d.epoch.max()]
                assert not d.duplicated(['seed','feedback_seed']).any()
                frames[method]=d.set_index(['seed','feedback_seed']).test_acc.sort_index()
            a,b=frames['ndfa_spatial_kron'],frames['ndfa_random']
            assert a.index.equals(b.index) and len(a)==25
            differences=100*(a-b)
            spatial.append({'dataset':dataset,'amplitude':amplitude,'mean_difference_pp':float(differences.mean()),
                            'initialization_seed_means_pp':differences.groupby('seed').mean().tolist(),
                            'method_means_pct':{m:float(v.mean()*100) for m,v in frames.items()}})
    result={'synthetic':synthetic,'diagonal_control':diagonal,'spatial':spatial,
            'input_sha256':hashes,'scope':'Saved observations only; no new inference or training; crossed spatial results descriptive.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Synthetic seed intervals:',[(r['condition'],r['seed_percentile_interval']) for r in synthetic])
    print('Additional full-minus-diagonal control:',diagonal)
    print(tails.groupby('method')[['mean_ce','median_ce','p99_ce','top1pct_loss_share']].mean().to_string())


if __name__=='__main__':main()
