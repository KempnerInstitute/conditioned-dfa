"""Offline, independent reaggregation of the second-review evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,t

ROOT=Path(__file__).resolve().parents[2]


def main():
    data=ROOT/'assets/ndfa_revision_20260919'
    audit=json.loads((data/'saved_results_audit.json').read_text())
    bn=pd.read_csv(data/'bn_control_seed_differences.csv')
    for row in audit['bn_control']:
        x=bn[bn.condition==row['regime']].delta
        assert len(x)==5
        assert np.isclose(x.mean(),row['paired_mean_pp'],rtol=0,atol=1e-12)
        assert np.isclose(x.sem(),row['seed_sem_pp'],rtol=0,atol=1e-12)
    hard=pd.read_csv(data/'hard_cifar100_seed_means.csv')
    for row in audit['hard_cifar100']:
        x=hard[hard.method==row['method']]
        assert len(x)==5 and x['count'].sum()==row['training_runs']
        assert np.isclose(100*x['mean'].mean(),row['mean_accuracy_pct'],atol=1e-12)
        assert np.isclose(100*x['mean'].sem(),row['seed_mean_sem_pp'],atol=1e-12)
    cells=pd.read_csv(data/'nuisance_correlation_cells.csv')
    assert len(cells)==128 and not cells.duplicated(['condition','input_noise','n_train','train_label_noise']).any()
    rho=float(spearmanr(cells.nuisance_energy_ratio,cells.gain).statistic)
    assert np.isclose(rho,audit['correlations']['saved_prospective'],atol=1e-12)
    endpoints=pd.read_csv(data/'followup_endpoints.csv');assert len(endpoints)==202
    endpoints=endpoints[~endpoints.development];assert len(endpoints)==170
    contrasts=pd.read_csv(data/'followup_contrasts.csv');n=0
    for _,r in contrasts[contrasts.phase.isin(['work','width'])].iterrows():
        field='budget' if r.phase=='work' else 'width'
        g=endpoints[(endpoints.phase==r.phase)&(endpoints[field]==r[field])]
        a,b=r.contrast.split(' - ');p=g.pivot(index='seed',columns='method',values='test_accuracy')
        d=100*(p[a]-p[b]);assert len(d)==5
        half=t.ppf(.975,4)*d.sem()
        assert np.allclose([d.mean(),d.mean()-half,d.mean()+half],[r['mean'],r.low,r.high],atol=1e-10)
        n+=1
    # A decreasing absolute ridge is not sufficient for compression if lambda_T
    # decreases faster. This independently verifies the corrected qualification.
    eps=np.array([1e-2,1e-4,1e-6])
    ratio=(eps+np.sqrt(eps))/(eps*(1+np.sqrt(eps)))
    assert np.allclose(ratio,1/np.sqrt(eps))
    compressed=(eps+eps**2)/(eps*(1+eps**2))
    assert np.all(np.diff(compressed)<0) and abs(compressed[-1]-1)<2e-6
    # The mean outer product alone gives off-diagonal uncentered structure.
    mu=np.array([1.,2.]);moment=np.diag([2.,3.])+np.outer(mu,mu)
    assert moment[0,1]==2 and (moment-np.outer(mu,mu))[0,1]==0
    print(json.dumps({'accepted':True,'bn_regimes':4,'hard_cifar_rules':len(audit['hard_cifar100']),
                      'correlation_cells':128,'spearman':rho,'followup_intervals':n,
                      'joint_limit_counterexample':True,'uncentered_moment_identity':True,
                      'scope':'Saved seed summaries and endpoint records; no training or new test access.'},indent=2))


if __name__=='__main__':main()
