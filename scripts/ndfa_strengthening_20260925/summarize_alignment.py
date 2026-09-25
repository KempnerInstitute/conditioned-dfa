"""Predeclared descriptive alignment outcomes; no significance-driven extension."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_development import summarize


def layer_outcomes(history,layer,window=256):
    """Sparse-probe onset is interval limited; do not impute censored onsets."""
    rows=[(h['step'],h['layers'][layer]) for h in history]
    assert all(b[0]>a[0] for a,b in zip(rows,rows[1:])), 'Duplicate/out-of-order probes'
    onset=None
    for i in range(len(rows)-2):
        values=[r['raw_cosine'] for _,r in rows[i:i+3]]
        if all(v is not None and math.isfinite(v) and v>0 for v in values):
            onset=rows[i][0];break
    early=[(t,r['raw_descent_projection']) for t,r in rows if t<=window]
    covered=bool(early) and early[0][0]==0 and early[-1][0]==window and all(v is not None and math.isfinite(v) for _,v in early)
    area=sum((b-a)*(max(-x,0)+max(-y,0))/2 for (a,x),(b,y) in zip(early,early[1:])) if covered else None
    return dict(layer=layer,sustained_raw_alignment_onset=onset,onset_censored=onset is None,
                last_observed_update=rows[-1][0] if rows else None,
                negative_projection_area_0_256=area,early_window_complete=covered)


def report(config_path):
    config_path=Path(config_path);cfg=json.loads(config_path.read_text())
    result=summarize(config_path)
    result['interpretation']='Four paired global seeds, fixed optimizer/rate and conditioning schedules; descriptive outcomes, not isolated feedback variance or confirmation'
    result['outcome_definitions']=cfg['alignment_analysis']
    result['projection_scope']='Gradient-like hidden-weight matrices before optimizer; fixed training probe; no learning-rate or momentum weighting. Positive projection predicts descent.'
    result['integration']='Trapezoidal integral of max(-raw projection,0) at saved probes from update 0 to 256. Missing window yields null, not zero.'
    result['onset_scope']='First of three consecutive strictly positive saved observations, limited by probe spacing. Censored/missing values are not averaged or replaced by the final update.'
    rows=[];groups=defaultdict(dict)
    for record in result['candidates']:
        case=record['case']
        for run in record['runs']:
            path=Path(cfg['output_root'])/case['id']/f"seed_{run['seed']}"/'alignment.json'
            history=json.loads(path.read_text()) if path.exists() else []
            entry=dict(case=case,seed=run['seed'],status=run['status'],
                       final_validation=run.get('final_validation'),
                       layer_outcomes=[layer_outcomes(history,i) for i in range(len(history[0]['layers']))] if history else [])
            rows.append(entry);groups[(case['cell'],case['normalization'],run['seed'])][case['conditioning_schedule']]=entry
    contrasts=[]
    for (cell,norm,seed),schedules in sorted(groups.items()):
        for reference,comparison in [('never','always'),('late','early')]:
            a,b=schedules[reference],schedules[comparison]
            row=dict(cell=cell,normalization=norm,seed=seed,contrast=f'{comparison} minus {reference}',reference_status=a['status'],comparison_status=b['status'])
            if a['status']==b['status']=='complete':
                row.update(accuracy_delta=b['final_validation']['accuracy']-a['final_validation']['accuracy'],loss_delta=b['final_validation']['loss']-a['final_validation']['loss'])
            row['layers']=[]
            for x,y in zip(a['layer_outcomes'],b['layer_outcomes']):
                values=[x['negative_projection_area_0_256'],y['negative_projection_area_0_256']]
                row['layers'].append(dict(layer=x['layer'],reference_onset=x['sustained_raw_alignment_onset'],comparison_onset=y['sustained_raw_alignment_onset'],
                                          reference_censored=x['onset_censored'],comparison_censored=y['onset_censored'],
                                          negative_projection_area_delta=values[1]-values[0] if all(v is not None for v in values) else None))
            contrasts.append(row)
    result['alignment_runs']=rows;result['paired_schedule_contrasts']=contrasts
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=report(args.config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.pending');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.output)
    print(json.dumps({k:result[k] for k in ['endpoint_counts','complete','confirmation_ready']}))
