"""Check the published FOOF recurrence against independent dense arithmetic."""
import copy
import pytest
import torch
from infogeo.dfa import Gradients
from infogeo.foof import FOOF


@pytest.mark.parametrize("period",[1,3])
def test_dense_algorithm1_recurrence_all_layers(period):
    rng=torch.Generator().manual_seed(177)
    opt=FOOF(.3,decay=.95,inverse_period=period)
    moments=[torch.zeros(4,4,dtype=torch.double),torch.zeros(3,3,dtype=torch.double)]
    mass=0.
    for _ in range(50):
        acts=[torch.randn(7,d,generator=rng,dtype=torch.double) for d in [4,3]]
        opt.observe(acts)
        moments=[.95*m+.05*(a.T@a/len(a)) for m,a in zip(moments,acts)];mass=.95*mass+.05
    opt.refresh()
    inv=[torch.linalg.inv(m/mass+.3*torch.eye(len(m),dtype=torch.double)) for m in moments]
    for step in range(8):
        acts=[torch.randn(7,d,generator=rng,dtype=torch.double) for d in [4,3]]
        grads=[torch.randn(o,d,generator=rng,dtype=torch.double) for o,d in [(3,4),(2,3)]]
        raw=Gradients(grads,[torch.ones(3),torch.ones(2)],[],1.)
        expected=[g@p for g,p in zip(grads,inv)]
        actual=opt.transform(raw,acts)
        for a,b in zip(actual.weights,expected):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        assert actual.biases is raw.biases
        if step % period==0:
            inv=[torch.linalg.inv(m/mass+.3*torch.eye(len(m),dtype=torch.double)) for m in moments]
        moments=[.95*m+.05*(a.T@a/len(a)) for m,a in zip(moments,acts)];mass=.95*mass+.05
        for a,b in zip(opt.moments,moments):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_state_restore_preserves_ema_and_stale_inverse():
    torch.set_num_threads(1)
    rng=torch.Generator().manual_seed(8);opt=FOOF(.7,inverse_period=4)
    a=torch.randn(13,5,generator=rng);opt.observe([a]);opt.refresh()
    raw=Gradients([torch.randn(3,5,generator=rng)],[torch.zeros(3)],[],1.)
    opt.transform(raw,[a*2]);saved=copy.deepcopy(opt.state_dict())
    resumed=FOOF(.7,inverse_period=4);resumed.load_state_dict(saved)
    for i in range(5):
        x=torch.randn(13,5,generator=rng)
        torch.testing.assert_close(opt.transform(raw,[x]).weights[0],resumed.transform(raw,[x]).weights[0],atol=0,rtol=0)


def test_mean_moment_batch_duplication_invariant():
    a=torch.tensor([[1.,2.],[-3.,4.]],dtype=torch.double)
    left,right=FOOF(.2),FOOF(.2)
    left.observe([a]);right.observe([a.repeat(3,1)])
    left.refresh();right.refresh()
    torch.testing.assert_close(left.inverses[0],right.inverses[0],atol=0,rtol=0)
