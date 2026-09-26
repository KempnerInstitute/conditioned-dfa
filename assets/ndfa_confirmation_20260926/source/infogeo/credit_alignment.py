"""Offline raw-versus-conditioned credit diagnostics at a shared model state."""
import torch
from .local_preconditioning import condition_local_update


def scheduled_operator(case,step,total):
    schedule=case.get("conditioning_schedule","always")
    if schedule not in {"always","never","early","late"}:raise ValueError(schedule)
    active=schedule=="always" or (schedule=="early" and step<total//4) or (schedule=="late" and step>=total-total//4)
    return case if active else dict(case,operator="none")


@torch.no_grad()
def measure(model,feedback,x,y,case):
    state=(model.training,list(model.bn_running_mean),list(model.bn_running_var),list(model._bn_cache),getattr(model,"last_activities",None))
    def restore():
        model.training,means,variances,cache,activities=state
        model.bn_running_mean=list(means);model.bn_running_var=list(variances);model._bn_cache=list(cache);model.last_activities=activities
    def cosine(a,b):
        a,b=a.double(),b.double();den=a.norm()*b.norm()
        return float((a*b).sum()/den) if den>0 else None
    try:
        model.training=True;raw=model.dfa_gradients(x,y,feedback)
        activities=list(model.last_activities)
        restore();model.training=True;bp=model.bp_gradients(x,y)
        values=[]
        for layer in range(model.n_hidden_layers):
            a=activities[layer];g=raw.weights[layer];b=bp.weights[layer]
            damping=max(case["rho"]*float(a.square().mean()),1e-6) if case.get("rho") is not None else case["damping"]
            conditioned=condition_local_update(a,raw.deltas[layer]*len(x),activity_damping=damping)
            before,after=g.double().norm(),conditioned.double().norm()
            if after>0:conditioned=conditioned*(before/after).to(conditioned.dtype)
            denom=b.double().square().sum()
            values.append(dict(layer=layer,raw_cosine=cosine(g,b),conditioned_cosine=cosine(conditioned,b),
                               raw_descent_projection=float((g.double()*b.double()).sum()/denom) if denom>0 else None,
                               conditioned_descent_projection=float((conditioned.double()*b.double()).sum()/denom) if denom>0 else None,
                               raw_norm=float(before),conditioned_norm=float(conditioned.double().norm()),damping=damping))
        return dict(loss=raw.loss,layers=values,scope="raw gradient-like matrices before optimizer; positive projection predicts descent on the fixed training probe; BP is offline diagnostic only")
    finally:restore()
