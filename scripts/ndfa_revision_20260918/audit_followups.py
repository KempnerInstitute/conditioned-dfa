"""Independently audit the complete frozen follow-up inventory and predictions."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.ndfa_revision_math import paired_seed_summary

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    torch.set_num_threads(2)
    planpath=ROOT/'configs/ndfa_revision_followups_20260918.json';plan=read(planpath)
    out=ROOT/'results/ndfa_revision_followups_audit_20260918';out.mkdir(exist_ok=True)
    base=ROOT/'results/ndfa_revision_followups_20260918'
    rows=[];moments=[];hashes={};selection=None
    for phase in ['width','stability','work']:
        root=base/phase
        if not (root/'completion.json').exists():print('Awaiting complete phase',phase,flush=True);continue
        done=read(root/'completion.json');assert done['complete'] and done['plan_sha256']==sha(planpath)
        receipts=read(root/'results.json')
        expected=plan['width_development']+plan['width_confirmation'] if phase=='width' else plan[phase+'_cases']
        assert sorted(r['case']['id'] for r in receipts)==sorted(e['id'] for e in expected)
        assert done['cases']==len(expected)
        if phase=='width':
            selection=read(root/'selection.json')
            for width in [1024,2048]:
                for method in ['endfa','ediag']:
                    choices=[]
                    for rho in plan['error_ridges']:
                        dev=[r for r in receipts if r['case']['id'].startswith('dev_') and r['case']['width'][0]==width and r['case']['method']==method and r['case']['rho']==rho]
                        assert len(dev)==2
                        if all(r['status']=='complete' for r in dev):choices.append((np.mean([r['validation']['loss'] for r in dev]),-rho))
                    assert selection[f'{width}_{method}']==-min(choices)[1]
        for receipt,entry in zip(receipts,expected):
            case=receipt['case'];assert case['id']==entry['id'];path=root/case['id'];development=case['id'].startswith('dev_')
            for key,value in entry.items():assert case[key]==value
            if phase=='width' and not development and case['method'] in ['endfa','ediag','kndfa']:
                assert case['rho']==selection[f"{case['width'][0]}_{'ediag' if case['method']=='ediag' else 'endfa'}"]
            assert receipt['test_evaluations']==(0 if development or receipt['status']!='complete' else 1)
            for name,digest in receipt['artifacts_sha256'].items():
                p=path/name;assert sha(p)==digest,str(p);hashes[str(p.relative_to(ROOT))]=digest
            row={'phase':phase,'development':development,**case,'status':receipt['status']}
            if receipt['status']=='complete':
                for split in ['validation']+([] if development else ['test']):
                    saved=torch.load(path/(split+'.pt'),weights_only=True,map_location='cpu');z=saved['logits'].numpy().astype('float64');y=saved['labels'].numpy()
                    assert len(y)==(2000 if split=='validation' else 10000)
                    losses=logsumexp(z,axis=1)-z[np.arange(len(y)),y]
                    values={'accuracy':np.mean(z.argmax(1)==y),'loss':losses.mean(),'logit_abs_max':np.abs(z).max(),'top_1pct_loss_fraction':np.sort(losses)[-max(1,int(np.ceil(.01*len(y)))):].sum()/losses.sum()}
                    for k,v in values.items():assert np.isclose(v,receipt[split][k],rtol=2e-12,atol=2e-12),(case['id'],split,k)
                    assert np.allclose(np.quantile(losses,[.5,.9,.99,1]),receipt[split]['loss_quantiles'],rtol=2e-12,atol=2e-12)
                    row.update({split+'_'+k:float(v) for k,v in values.items()})
                row['update_seconds']=receipt['update_seconds']
            if 'width' in row:row['width']=row['width'][0]
            row.setdefault('width',1024)
            rows.append(row)
            for file in path.glob('moments_*.json'):
                for v in read(file):
                    q=v['error_spectrum'];ev=np.array(q['eigenvalues']);positive=ev[ev>q['rank_tolerance']]
                    assert len(positive)==q['rank'];assert np.isclose(v['raw_norm'],v['update_norm'],rtol=2e-6)
                    expected_cond=(ev.max()+q['ridge'])/(q['ridge'] if q['rank']<q['dimension'] else positive.min()+q['ridge'])
                    assert np.isclose(q['damped_condition_number'],expected_cond)
                    moments.append({'phase':phase,'development':development,**case,'width':case.get('width',[1024])[0],'step':int(file.stem.split('_')[1]),'layer':v['layer'],'raw_conditioned_cosine':v['raw_conditioned_cosine'],**{k:q[k] for k in ['rank','participation_rank','damped_condition_number','ridge_over_mean_positive_eigenvalue']}})
        print('Audited',phase,len(receipts),'cases',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'endpoints.csv',index=False)
    pd.DataFrame(moments).to_csv(out/'moments.csv',index=False)
    contrasts=[];summary=[]
    for phase,sub in frame[(~frame.development)&(frame.status=='complete')].groupby('phase'):
        keys={'width':['width'],'stability':['bn','feedback_scale'],'work':['budget']}[phase]
        for group,g in sub.groupby(keys):
            cond=dict(zip(keys,group if isinstance(group,tuple) else [group]))
            for method,m in g.groupby('method'):
                summary.append({'phase':phase,**cond,'method':method,'n':len(m),'accuracy_pct':100*m.test_accuracy.mean(),'accuracy_sd_pct':100*m.test_accuracy.std(ddof=1),'ce':m.test_loss.mean(),'work_seconds':m.update_seconds.mean()})
            pivot=g.pivot(index='seed',columns='method',values='test_accuracy');assert len(pivot)==5 and not pivot.isna().any().any()
            pairs={'work':[('ndfa','dfa'),('ndfa','bp'),('ndfa','fd_dfa')],'width':[('ndfa','dfa'),('endfa','dfa'),('ediag','dfa'),('endfa','ediag'),('kndfa','ndfa')],'stability':[('ndfa','dfa'),('endfa','dfa'),('kndfa','ndfa')]}[phase]
            for a,b in pairs:contrasts.append({'phase':phase,**cond,'contrast':a+' - '+b,**paired_seed_summary(100*(pivot[a]-pivot[b]))})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    audit={'complete':len(rows)==202,'cases':len(rows),'failures':int((frame.status!='complete').sum()),'test_models':int((~frame.development).sum()),'selection':selection,'input_sha256':hashes,'plan_sha256':sha(planpath),'scope':'All planned cases retained; independent saved-logit replay; individual seed intervals for follow-up contrasts, without claims of simultaneous coverage.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(contrasts).drop(columns='assumption').to_string(index=False))
if __name__=='__main__':main()
