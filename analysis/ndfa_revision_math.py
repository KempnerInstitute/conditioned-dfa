"""Complete two-block risk and uncertainty over whole global-seed replicates."""
import itertools
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t


def two_block_minimum(signal, nuisance_noise, task_noise, rho):
    if min(signal,nuisance_noise,task_noise)<0 or rho<=0:
        raise ValueError('Nonnegative powers and positive rate ratio required')
    if math.isinf(rho):
        fitted=signal*task_noise/(signal+task_noise) if signal+task_noise else 0.
        return min(signal,nuisance_noise+fitted)
    def risk(u):
        return signal*u*u+task_noise*(1-u)**2+nuisance_noise*(1-u**rho)**2
    # Include endpoints and every sampled local basin; do not assume convexity.
    grid=np.unique(np.r_[0.,np.geomspace(1e-14,1.,1024),np.linspace(0.,1.,4097)])
    values=risk(grid)
    candidates=[float(values[0]),float(values[-1])]
    for i in np.flatnonzero((values[1:-1]<=values[:-2]) & (values[1:-1]<=values[2:]))+1:
        candidates.append(minimize_scalar(risk,bounds=(grid[i-1],grid[i+1]),method='bounded',
                                          options={'xatol':1e-14}).fun)
    return min(candidates)


def paired_seed_summary(values):
    """Mean and individual t interval, conditional on fixed cells/feedback draws."""
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('At least two finite, independent seed-level values required')
    mean=float(x.mean()); sd=float(x.std(ddof=1))
    half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return {'n':len(x),'mean':mean,'sd':sd,'low':mean-half,'high':mean+half,
            'assumption':'Independent approximately normal global-seed differences; fixed designed grid and feedback draws'}


def whole_seed_bootstrap(values):
    """Exact empirical percentile bootstrap for small n; no cell resampling."""
    x=np.asarray(values,dtype=float)
    if len(x)>7:raise ValueError('Enumeration is intended for small seed counts')
    draws=np.asarray([np.mean(v) for v in itertools.product(x,repeat=len(x))])
    return np.quantile(draws,[.025,.975]).tolist()
