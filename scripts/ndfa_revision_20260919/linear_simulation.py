"""Reproduce Figure 1D's population simulation, including rate selection.

This is the small analytical linear model, not a new network experiment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def simulate(rule, seed, task_high, kappa=50.):
    d,h,o=12,10,4
    rng=np.random.default_rng(seed)
    eigenvalues=np.geomspace(1.,kappa,d)
    q=np.linalg.qr(rng.standard_normal((d,d)))[0]
    moment=q@np.diag(eigenvalues)@q.T
    readout=rng.standard_normal((o,h))/np.sqrt(h)
    order=np.argsort(eigenvalues)[::-1]
    indices=order[:o] if task_high else order[-o:]
    target=(rng.standard_normal((o,o))@q[:,indices].T)/np.sqrt(d)
    initial=rng.standard_normal((h,d))*.01
    inverse=np.linalg.inv(moment+.001*np.eye(d))
    residual=readout@initial-target
    initial_loss=.5*np.trace(residual@moment@residual.T)
    best=4001; selected=None
    for rate in np.geomspace(1e-4,5.,24):
        weights=initial.copy()
        for step in range(4000):
            residual=readout@weights-target
            gradient=readout.T@(residual@moment)
            if rule=='activity':gradient=gradient@inverse
            weights-=rate*gradient
            residual=readout@weights-target
            loss=.5*np.trace(residual@moment@residual.T)
            if not np.isfinite(loss) or loss>1000*initial_loss:break
            if loss/initial_loss<1e-6:
                if step+1<best:best=step+1;selected=float(rate)
                break
    assert selected is not None
    return {'rule':rule,'seed':seed,'task_high':task_high,'steps':best,
            'selected_learning_rate':selected,'initial_loss':float(initial_loss)}


def main():
    out=ROOT/'assets/ndfa_revision_20260919';out.mkdir(parents=True,exist_ok=True)
    rows=[simulate(rule,seed,high) for rule in ['dfa','activity'] for high in [False,True] for seed in range(3)]
    pd.DataFrame(rows).to_csv(out/'linear_simulation.csv',index=False)
    protocol={'dimensions':[12,10,4],'condition_number':50,'population_moments':True,
              'activity_damping':.001,'initial_weight_sd':.01,'loss_ratio_threshold':1e-6,
              'learning_rates':np.geomspace(1e-4,5.,24).tolist(),'max_updates':4000,
              'feedback':'fixed readout transpose','readout':'iid Gaussian / sqrt(hidden width)',
              'target':'Gaussian output map onto four lowest/highest variance eigenvectors / sqrt(input dimension)',
              'selection':'fewest updates to threshold per rule and initialization; lower grid rate breaks ties',
              'divergence_threshold':'nonfinite loss or >1000 * initial loss','seeds':[0,1,2]}
    (out/'linear_simulation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(pd.DataFrame(rows).groupby(['rule','task_high']).steps.mean().to_string())


if __name__=='__main__':main()
