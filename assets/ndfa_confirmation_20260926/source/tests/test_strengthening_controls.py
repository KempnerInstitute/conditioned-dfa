import copy
import pytest
import torch
import torch.nn.functional as F
from infogeo.tanh_dfa import TanhSigmoidMLP
from infogeo.dfa import init_feedback
from infogeo.credit_alignment import measure,scheduled_operator
from experiments.run_ndfa_submission_benchmark import CachedManualMLP
from scripts.ndfa_strengthening_20260925 import next_round as study


def test_tanh_bp_matches_autograd_for_stable_summed_binary_loss():
    m=TanhSigmoidMLP(4,[5,3,3],2,seed=31)
    m.weights=[w.double() for w in m.weights];m.biases=[b.double() for b in m.biases]
    x=torch.randn(8,4,dtype=torch.float64);y=torch.tensor([0,1]*4)
    manual=m.bp_gradients(x,y)
    weights=[w.clone().requires_grad_() for w in m.weights];biases=[b.clone().requires_grad_() for b in m.biases]
    h=x
    for i,(w,b) in enumerate(zip(weights,biases)):
        h=h@w.T+b
        if i<len(weights)-1:h=h.tanh()
    loss=F.binary_cross_entropy_with_logits(h,F.one_hot(y,2).double(),reduction="sum")/len(y)
    actual=torch.autograd.grad(loss,weights+biases)
    for a,b in zip(actual,manual.weights+manual.biases):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_alignment_restores_bn_state_and_reports_same_raw_dfa_after_probe():
    model=CachedManualMLP(4,[5,3],2,seed=1,batchnorm=True);model.training=True
    fb=init_feedback(model,seed=3)
    x=torch.randn(8,4);y=torch.tensor([0,1]*4)
    original=copy.deepcopy(model)
    result=measure(model,fb,x,y,dict(damping=.3))
    for a,b in zip(model.bn_running_mean+model.bn_running_var,original.bn_running_mean+original.bn_running_var):torch.testing.assert_close(a,b,atol=0,rtol=0)
    a=model.dfa_gradients(x,y,fb);b=original.dfa_gradients(x,y,fb)
    for l,r in zip(a.weights,b.weights):torch.testing.assert_close(l,r,atol=0,rtol=0)
    assert len(result['layers'])==2
    for layer in result['layers']:assert layer['raw_norm']==pytest.approx(layer['conditioned_norm'],rel=1e-6)


def test_early_and_late_use_equal_conditioning_work_without_mutating_case():
    for n in [16,17,20]:
        for name in ['early','late']:
            case=dict(operator='activity',conditioning_schedule=name)
            active=[i for i in range(n) if scheduled_operator(case,i,n)['operator']=='activity']
            assert len(active)==n//4
            assert case['operator']=='activity'


@pytest.mark.parametrize('activation',['relu','tanh'])
def test_new_diagnostics_and_schedule_resume_preserve_training(tmp_path,activation):
    cfg=dict(dataset='synthetic',hidden_dims=[7,5],batch_size=8,epochs=4,threads=1,feedback_scale=1.,weight_decay=0.,warmup_epochs=1,
             validation_examples=24,evaluate_every_epochs=1,checkpoint_every_epochs=1,foof_inverse_period=3,foof_calibration_batches=5,
             cells={'test':dict(n_train=16,task_scale=.45,nuisance_scale=2.,interaction=False,input_noise=.15,label_noise=.2)})
    case=dict(id='test',cell='test',credit='dfa',normalization='none',activation=activation,operator='activity',optimizer='adamw',lr=.001,damping=.3,conditioning_schedule='early')
    plain=study.train(cfg,case,41,tmp_path/'plain',device='cpu')
    cfg['alignment_probe']=True;cfg['alignment_probe_steps']=[1,2,4,8]
    probed=study.train(cfg,case,41,tmp_path/'probe',device='cpu')
    study.train(cfg,case,41,tmp_path/'resume',device='cpu',stop_after_epoch=2)
    resumed=study.train(cfg,case,41,tmp_path/'resume',device='cpu')
    assert plain['final_validation']==probed['final_validation']==resumed['final_validation']
    a=torch.load(tmp_path/'probe/final.pt',weights_only=False);b=torch.load(tmp_path/'resume/final.pt',weights_only=False)
    assert a['alignment_history']==b['alignment_history']
    for x,y in zip(a['model']['weights'],b['model']['weights']):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_alignment_summary_handles_censoring_and_missing_window():
    from scripts.ndfa_strengthening_20260925.summarize_alignment import layer_outcomes
    history=[dict(step=t,layers=[dict(raw_cosine=c,raw_descent_projection=p)]) for t,c,p in [(0,-1.,-2.),(128,1.,-1.),(256,1.,0.),(512,1.,1.)]]
    result=layer_outcomes(history,0)
    assert result['sustained_raw_alignment_onset']==128
    assert result['negative_projection_area_0_256']==256
    assert layer_outcomes(history[:2],0)['negative_projection_area_0_256'] is None
    assert layer_outcomes(history[:2],0)['onset_censored']


def test_cifar_geometry_probe_restores_bn_and_reuses_fixed_augmentation():
    from types import SimpleNamespace
    from experiments.run_ndfa_submission_benchmark import Data
    images=torch.randint(0,256,(12,3,32,32),dtype=torch.uint8)
    labels=torch.arange(12)%2
    data=Data(images,labels,images,labels,torch.zeros(3),torch.ones(3),{})
    model=CachedManualMLP(3072,[5,3],2,seed=31,batchnorm=True)
    args=SimpleNamespace(device='cpu',batch_size=8)
    initial=copy.deepcopy(model)
    first=study.geometry_probe(model,data,args,1);second=study.geometry_probe(model,data,args,1)
    assert first==second
    for a,b in zip(model.bn_running_mean+model.bn_running_var,initial.bn_running_mean+initial.bn_running_var):torch.testing.assert_close(a,b,atol=0,rtol=0)
