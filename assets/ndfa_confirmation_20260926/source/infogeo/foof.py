"""FOOF weight preconditioning following Benzing (ICML 2022), Algorithm 1.

https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf, Appendix K.
Moments use A.T @ A / batch_size in row-batch notation. Damping therefore
uses the mean-moment convention. The normalized EMA is initialized from
50 minibatches by the experiment runner. S=T: update moments every step,
and amortize the inverse over T updates. Every weight matrix, including
the classifier, is conditioned without norm matching. Bias and BN affine
gradients are passed through unchanged; the paper's MLPs omitted biases.
"""
from dataclasses import dataclass
import math
import torch
from .dfa import Gradients


@dataclass
class FOOF:
    damping: float
    decay: float = .95
    inverse_period: int = 100

    def __post_init__(self):
        if not math.isfinite(self.damping) or self.damping<=0:
            raise ValueError("FOOF damping must be positive and finite")
        if not 0<=self.decay<1 or self.inverse_period<1:
            raise ValueError("Invalid EMA decay or inverse refresh period")
        self.moments=[]
        self.inverses=[]
        self.mass=0.
        self.step=0

    @torch.no_grad()
    def observe(self, activities):
        samples=[a.T @ a / len(a) for a in activities]
        if not self.moments:
            self.moments=[torch.zeros_like(m) for m in samples]
        if len(samples)!=len(self.moments):
            raise ValueError("FOOF layer inventory changed")
        for old,new in zip(self.moments,samples):
            if old.shape!=new.shape or not torch.isfinite(new).all():
                raise FloatingPointError("Invalid FOOF activity moment")
            old.mul_(self.decay).add_(new,alpha=1-self.decay)
        self.mass=self.decay*self.mass+1-self.decay

    @torch.no_grad()
    def refresh(self):
        if not self.mass>0:
            raise RuntimeError("Calibrate FOOF before computing an inverse")
        values=[]
        for moment in self.moments:
            covariance=moment/self.mass
            damped=(covariance+covariance.T)*.5
            damped.diagonal().add_(self.damping)
            try:
                factor=torch.linalg.cholesky(damped)
                value=torch.cholesky_inverse(factor)
            except torch.linalg.LinAlgError as exc:
                raise FloatingPointError("FOOF Cholesky failed at the declared damping") from exc
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite FOOF inverse")
            values.append(value)
        self.inverses=values

    @torch.no_grad()
    def transform(self, raw, activities):
        if len(self.inverses)!=len(raw.weights):
            raise RuntimeError("FOOF requires calibrated inverses for every weight layer")
        weights=[g @ inverse for g,inverse in zip(raw.weights,self.inverses)]
        # Algorithm 1 refreshes from the previous EMA after using the old P,
        # then observes the pre-update activities. Preserve this ordering.
        if self.step % self.inverse_period==0:
            self.refresh()
        self.observe(activities)
        self.step+=1
        if not all(torch.isfinite(w).all() for w in weights):
            raise FloatingPointError("Nonfinite FOOF update")
        return Gradients(weights,raw.biases,raw.deltas,raw.loss,raw.bn_gammas,raw.bn_betas)

    def state_dict(self):
        return dict(damping=self.damping,decay=self.decay,inverse_period=self.inverse_period,
                    mass=self.mass,step=self.step,moments=self.moments,inverses=self.inverses)

    def load_state_dict(self,state):
        assert (self.damping,self.decay,self.inverse_period)==(state["damping"],state["decay"],state["inverse_period"])
        self.mass=state["mass"];self.step=state["step"]
        self.moments=[m.clone() for m in state["moments"]]
        self.inverses=[m.clone() for m in state["inverses"]]
