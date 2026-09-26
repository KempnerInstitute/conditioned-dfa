"""Matched-numerator controls for uncentered activity-moment conditioning."""
import math
import torch

STATISTICS=("full","diagonal","centered","diagonal_covariance_plus_mean","isotropic_covariance_plus_mean")


@torch.no_grad()
def moment(activity,statistic):
    if statistic not in STATISTICS:raise ValueError(statistic)
    if activity.ndim!=2 or min(activity.shape)==0:raise ValueError("Expected a nonempty activity matrix")
    if statistic in {"full","diagonal"}:
        full=activity.T @ activity/len(activity)
        return full if statistic=="full" else torch.diag(full.diagonal())
    mean=activity.mean(0);centered=activity-mean
    covariance=centered.T @ centered/len(activity)
    if statistic=="centered":return covariance
    if statistic=="diagonal_covariance_plus_mean":return torch.diag(covariance.diagonal())+torch.outer(mean,mean)
    return torch.eye(activity.shape[1],device=activity.device,dtype=activity.dtype)*covariance.trace()/activity.shape[1]+torch.outer(mean,mean)


@torch.no_grad()
def transform(activity,raw_weight,damping,statistic):
    """Change only the right-hand moment operator; leave the numerator intact.

    All five controls use the same dense solve and absolute damping. The caller
    norm-matches the result and passes bias/BN-affine gradients through unchanged.
    """
    if not math.isfinite(damping) or damping<=0:raise ValueError("Positive finite damping required")
    matrix=moment(activity,statistic)
    if not torch.isfinite(matrix).all():raise FloatingPointError("Nonfinite activity moment")
    matrix.diagonal().add_(damping)
    try:result=torch.linalg.solve(matrix,raw_weight.T).T
    except torch.linalg.LinAlgError as exc:raise FloatingPointError("Geometry solve failed at declared damping") from exc
    if not torch.isfinite(result).all():raise FloatingPointError("Nonfinite geometry-conditioned update")
    return result


@torch.no_grad()
def describe(activity):
    """Training-minibatch diagnostics; covariance uses population divisor n."""
    a=activity.double();mean=a.mean(0);centered=a-mean
    covariance=centered.T@centered/len(a)
    trace=covariance.trace();energy=covariance.square().sum();mean_energy=mean.square().sum()
    total=trace+mean_energy
    off_diagonal=energy-covariance.diagonal().square().sum()
    return dict(mean_energy_fraction=float(mean_energy/total) if total>0 else 0.,
                centered_participation_rank=float(trace.square()/energy) if energy>0 else 0.,
                centered_off_diagonal_energy_fraction=float(torch.clamp(off_diagonal/energy,min=0)) if energy>0 else 0.,
                uncentered_trace=float(total),centered_trace=float(trace))
