import copy
from types import SimpleNamespace
import pytest
import torch
from infogeo.activity_geometry import moment,transform,STATISTICS
from infogeo.local_preconditioning import condition_local_update
from experiments.run_ndfa_submission_benchmark import CachedManualMLP
from scripts.ndfa_strengthening_20260925 import next_round as study


def test_full_operator_agrees_with_existing_batch_space_rule():
    generator=torch.Generator().manual_seed(31)
    a=torch.randn(7,11,generator=generator,dtype=torch.float64)+2
    d=torch.randn(7,5,generator=generator,dtype=torch.float64)
    raw=d.T@a/len(a)
    torch.testing.assert_close(transform(a,raw,.3,"full"),condition_local_update(a,d,activity_damping=.3,backend="sample"),atol=1e-11,rtol=1e-11)


def test_mean_preserving_control_retains_nonzero_off_diagonal_uncentered_moments():
    a=torch.tensor([[3.,4.],[3.,2.],[1.,4.],[1.,2.]],dtype=torch.float64)
    assert moment(a,"centered")[0,1]==0
    torch.testing.assert_close(moment(a,"full"),moment(a,"diagonal_covariance_plus_mean"))
    assert moment(a,"full")[0,1]==6
    assert moment(a,"diagonal")[0,1]==0
    torch.testing.assert_close(moment(a,"isotropic_covariance_plus_mean").trace(),moment(a,"full").trace())


@pytest.mark.parametrize("statistic",STATISTICS)
def test_controls_keep_raw_numerator_and_follow_independent_dense_solve(statistic):
    a=torch.tensor([[3.,1.,0.],[4.,2.,1.],[1.,3.,4.]],dtype=torch.float64)
    raw=torch.tensor([[1.,2.,-1.],[2.,-1.,3.]],dtype=torch.float64)
    mean=a.mean(0);cov=(a-mean).T@(a-mean)/3
    matrices=dict(full=a.T@a/3,diagonal=torch.diag((a.T@a/3).diag()),centered=cov,
                  diagonal_covariance_plus_mean=torch.diag(cov.diag())+torch.outer(mean,mean),
                  isotropic_covariance_plus_mean=torch.eye(3,dtype=a.dtype)*cov.trace()/3+torch.outer(mean,mean))
    expected=raw@torch.linalg.inv(matrices[statistic]+.2*torch.eye(3,dtype=a.dtype))
    torch.testing.assert_close(transform(a,raw,.2,statistic),expected,atol=1e-12,rtol=1e-12)


def small_config():
    return dict(dataset="synthetic",hidden_dims=[7,5],batch_size=8,epochs=4,threads=1,feedback_scale=1.,weight_decay=0.,warmup_epochs=1,
                validation_examples=24,evaluate_every_epochs=1,checkpoint_every_epochs=1,foof_inverse_period=3,foof_calibration_batches=5,
                cells={"test":dict(n_train=16,task_scale=.45,nuisance_scale=2.,interaction=False,input_noise=.15,label_noise=.2)})


def test_probe_does_not_change_training_and_geometry_resume_is_exact(tmp_path):
    cfg=small_config();case=dict(id="test",cell="test",credit="dfa",normalization="bn",operator="activity_geometry",statistic="diagonal_covariance_plus_mean",optimizer="sgd_momentum",lr=.003,damping=.3)
    plain=study.train(cfg,case,41,tmp_path/"plain",device="cpu")
    probed=copy.deepcopy(cfg);probed["geometry_probe"]=True
    full=study.train(probed,case,41,tmp_path/"full",device="cpu")
    study.train(probed,case,41,tmp_path/"resume",device="cpu",stop_after_epoch=2)
    resumed=study.train(probed,case,41,tmp_path/"resume",device="cpu")
    assert plain["final_validation"]==full["final_validation"]==resumed["final_validation"]
    assert plain["stream_sha256"]==full["stream_sha256"]==resumed["stream_sha256"]
    states=[torch.load(tmp_path/name/"final.pt",weights_only=False)["model"] for name in ["plain","full","resume"]]
    for key in ["weights","biases","bn_gamma","bn_beta","bn_running_mean","bn_running_var"]:
        for index in range(len(states[0][key])):
            for other in states[1:]:torch.testing.assert_close(states[0][key][index],other[key][index],atol=0,rtol=0)
