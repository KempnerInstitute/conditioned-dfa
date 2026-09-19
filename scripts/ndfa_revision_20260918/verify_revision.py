"""Offline checks of corrected summaries and compact prediction exports.

The compact exports retain per-example losses and predicted/true classes.
They validate aggregation, not a fresh logit calculation or checkpoint inference.
The original independent float64-logit audit is included separately.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from analysis.ndfa_revision_math import two_block_minimum,whole_seed_bootstrap

def main():
    base=ROOT/'results/ndfa_revision_followups_20260918'
    assert np.isclose(two_block_minimum(1,1,.1,np.inf)-two_block_minimum(1,1,.1,1),10/21)
    n=0;confirmed=0
    for phase in ['work','width','stability']:
        completion=json.loads((base/phase/'completion.json').read_text());rows=json.loads((base/phase/'results.json').read_text())
        assert completion['complete'] and len(rows)==completion['cases']
        for r in rows:
            assert r['status']=='complete';case=base/phase/r['case']['id'];n+=1
            for split in ['validation']+(['test'] if r['test_evaluations']==1 else []):
                with np.load(case/(split+'_compact.npz'),allow_pickle=False) as a:
                    ce=a['loss'].astype('float64');acc=np.mean(a['prediction']==a['label'])
                assert np.isclose(acc,r[split]['accuracy'],rtol=0,atol=1e-14)
                assert np.isclose(ce.mean(),r[split]['loss'],rtol=2e-7,atol=2e-7)
                assert np.allclose(np.quantile(ce,[.5,.9,.99,1]),r[split]['loss_quantiles'],rtol=2e-7,atol=2e-7)
                share=np.sort(ce)[-max(1,int(np.ceil(.01*len(ce)))):].sum()/ce.sum()
                assert np.isclose(share,r[split]['top_1pct_loss_fraction'],rtol=2e-7,atol=2e-7)
            confirmed+=r['test_evaluations']
    assert n==202 and confirmed==170
    audit=json.loads((ROOT/'results/ndfa_revision_saved_audit_20260918/audit.json').read_text())
    seeds=pd.read_csv(ROOT/'results/infodfa_seedlevel_stats_corrected_20260918/seedlevel_seed_deltas.csv')
    for r in audit['synthetic']:
        v=seeds[seeds.condition==r['condition']].mean_delta_pp
        assert len(v)==5 and np.allclose(whole_seed_bootstrap(v),r['seed_percentile_interval'])
    ep=pd.read_csv(ROOT/'results/ndfa_revision_saved_audit_20260918/metric_control_endpoints.csv')
    q=ep.pivot(index='seed',columns='name',values='test_accuracy');delta=100*(q.ndfa-q.diagonal)
    assert np.allclose(delta,audit['diagonal_control']['seed_differences_pp'])
    raw=pd.read_csv(ROOT/'results/ndfa_revision_saved_audit_20260918/loss_tails_by_seed.csv')
    expected=pd.read_csv(ROOT/'results/ndfa_revision_saved_audit_20260918/loss_tails_summary.csv').set_index('method')
    actual=raw.groupby('method').mean(numeric_only=True).drop(columns='replicate')
    assert np.allclose(actual[expected.columns],expected,rtol=1e-12)
    print(json.dumps({'accepted':True,'followup_cases':n,'followup_final_tests':confirmed,'corrected_seed_intervals':4,'additional_diagonal_seeds':8,'loss_tail_endpoints':len(raw),'counterexample_gain':'10/21','scope':'Compact per-example loss aggregation; no new dataset access or model inference.'},indent=2))
if __name__=='__main__':main()
