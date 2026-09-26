import copy
import json
from types import SimpleNamespace
import pytest
import torch
from infogeo.dfa import Gradients
from scripts.ndfa_strengthening_20260925 import integrated_components as component
from scripts.ndfa_strengthening_20260925 import integrated_train as runner
from scripts.ndfa_strengthening_20260925.freeze_decision import sha


def config():
    return dict(dataset='fixture',hidden_dims=[7,5],batch_size=8,threads=1,epochs=4,warmup_epochs=0,
                split_seed=934005,feedback_scale=.1,weight_decay=1e-4,foof_inverse_period=3,foof_calibration_batches=3,
                budget_kind='updates',lr_schedule='constant',evaluate_every_epochs=1,work_warmup_fraction=.025)


def case(**kw):
    c=dict(id='case',family='test',cell='fixture',credit='dfa',operator='activity',normalization='bn',lr=.001,optimizer='adamw',rho=1.,damping=None)
    c.update(kw);return c


@pytest.mark.parametrize('shape',[(5,9,7),(10,5,7),(5,9,3)])
def test_full_error_matches_dense_two_sided_solve_and_scalar_equals_A(shape):
    n,i,o=shape;g=torch.Generator().manual_seed(72);a=torch.randn(n,i,generator=g,dtype=torch.float64);d=torch.randn(n,o,generator=g,dtype=torch.float64)
    raw=Gradients([d.T@a/n,torch.zeros(3,o,dtype=a.dtype)],[d.mean(0),torch.zeros(3,dtype=a.dtype)],[d/n,torch.zeros(n,3,dtype=a.dtype)],0.)
    args=SimpleNamespace(damping_floor=1e-6);c=case(error_operator='full',error_rho=10.)
    result=component.error_transform([a],raw,args,c)
    la=float(a.square().mean());le=10*float(d.square().mean())
    expected=torch.linalg.solve(d.T@d/n+le*torch.eye(o,dtype=a.dtype),torch.linalg.solve(a.T@a/n+la*torch.eye(i,dtype=a.dtype),raw.weights[0].T).T)
    expected*=raw.weights[0].norm()/expected.norm();torch.testing.assert_close(result.weights[0],expected,atol=1e-12,rtol=1e-12)
    scalar=component.error_transform([a],raw,args,dict(c,error_operator='scalar')).weights[0]
    aa=component.condition_local_update(a,d,activity_damping=la,mode='activity');aa*=raw.weights[0].norm()/aa.norm()
    torch.testing.assert_close(scalar,aa,atol=1e-12,rtol=1e-12)
    assert torch.equal(result.weights[-1],raw.weights[-1])


@pytest.mark.parametrize('operator', ['full','diagonal'])
def test_error_runner_exact_resume_and_direction_probe_neutrality(tmp_path,operator):
    cfg=config();c=case(error_operator=operator,error_rho=10.,error_schedule='early')
    full=runner.train(cfg,c,51,tmp_path/'full',device='cpu')
    assert runner.train(cfg,c,51,tmp_path/'resume',device='cpu',stop_after_step=5)['status']=='interrupted'
    resumed=runner.train(cfg,c,51,tmp_path/'resume',device='cpu')
    assert full['final_validation']==resumed['final_validation'] and full['stream_sha256']==resumed['stream_sha256']
    assert full['flops']==resumed['flops']
    history=json.loads((tmp_path/'full/history.json').read_text());assert history[0]['error_direction']['layer_cosines_to_activity']


def test_shared_checkpoint_preserves_optimizer_rng_and_equal_continuation(tmp_path):
    cfg=dict(config(),branch_fractions=[.25,.75]);c=case();runner.train(cfg,c,53,tmp_path/'prefix',device='cpu')
    cp=tmp_path/'prefix/branch_4.pt'
    fork=dict(config(),epochs=2,branch_from=dict(checkpoint=str(cp),sha256=sha(cp),step=4))
    continued=runner.train(fork,c,53,tmp_path/'fork',device='cpu')
    direct=runner.train(dict(config(),epochs=2),c,53,tmp_path/'direct',device='cpu')
    assert continued['completed_updates']==direct['completed_updates']==8
    assert continued['final_validation']==direct['final_validation'] and continued['stream_sha256']==direct['stream_sha256']
    for key in ['weights','biases','bn_running_mean','bn_running_var']:
        a=torch.load(tmp_path/'fork/final.pt',weights_only=False)['model'][key];b=torch.load(tmp_path/'direct/final.pt',weights_only=False)['model'][key]
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_foof_dfa_uses_local_credit_and_all_weight_layers(tmp_path):
    c=case(operator='foof',optimizer='sgd',lr=.003,damping=.3);cfg=config()
    result=runner.train(cfg,c,55,tmp_path/'foof',device='cpu')
    assert result['status']=='complete' and result['flops']['calibration']>0
    assert result['official_test_loaded'] is False


def test_error_window_keeps_A_and_constant_learning_rate():
    c=case(error_operator='full',error_rho=10.,error_schedule='early')
    first=component.active_case(c,.249);last=component.active_case(c,.25)
    assert first['error_operator']=='full' and last['error_operator'] is None
    assert first['operator']==last['operator']=='activity'
    assert runner.rate(c,config(),0.,0,100,0)==runner.rate(c,config(),.99,99,100,0)==.001


def test_matrix_flop_estimate_includes_error_solve_and_foof_refresh():
    model=SimpleNamespace(weights=[torch.zeros(7,9),torch.zeros(3,7)],output_dim=3)
    a=component.step_flops(model,case(),5)
    k=component.step_flops(model,case(error_operator='full'),5)
    assert k['conditioning']>a['conditioning']>0
    before=component.step_flops(model,case(operator='foof'),5,foof_step=0)
    after=component.step_flops(model,case(operator='foof'),5,foof_step=1)
    assert before['conditioning']-after['conditioning']==9**3+7**3


def test_joint_test_gate_refuses_missing_global_freeze(tmp_path,monkeypatch):
    from scripts.ndfa_strengthening_20260925 import integrated_control as ctl
    with pytest.raises(FileNotFoundError):ctl.test_gate({'root':str(tmp_path)},'confirm')


def test_benchmark_loader_rejects_tampered_training_cache(tmp_path):
    path=tmp_path/'data.pt';path.write_bytes(b'changed')
    cfg=dict(dataset='benchmark',benchmark_root=str(tmp_path),benchmark_inventory={'x':{'file':'data.pt','sha256':'0'*64}})
    with pytest.raises(AssertionError):component.load_data(cfg,{'cell':'x'},1,None)


@pytest.mark.parametrize('loss,direction,pairs,expected',[(.2,.002,10,True),(.4,.002,10,False),(.2,.001,10,False),(.2,.002,9,False)])
def test_error_decision_requires_loss_direction_controls_and_every_seed(tmp_path,loss,direction,pairs,expected):
    from scripts.ndfa_strengthening_20260925.integrated_control import report
    names=['confirm_foof_cifar','confirm_foof_nuisance','confirm_error_cifar','confirm_error_mnist','confirm_benchmarks','confirm_geometry','confirm_timing','confirm_epochs']
    for name in names:
        folder=tmp_path/name;folder.mkdir();(folder/'test_evaluation').mkdir()
        (folder/'config.json').write_text(json.dumps({'budget_kind':'work'}))
        rows=[]
        if name=='confirm_error_cifar':
            for family in ['activity','early_diagonal','early_k']:
                for seed in range(pairs if family=='early_k' else 10):
                    metrics={'accuracy':.8 if family=='early_k' else .7,'loss':loss if family=='early_k' else .3}
                    rows.append(dict(case={'family':family,'cell':'cifar10'},seed=seed,status='complete',attempts=1,metrics={'best':metrics,'final':metrics},mean_early_direction_change=direction,training_seconds=200.,flops={'matrix':1.}))
        (folder/'test_evaluation/summary.json').write_text(json.dumps({'rows':rows}))
    (tmp_path/'joint_confirmation.json').write_text(json.dumps({'configs':dict.fromkeys(names,{})}))
    result=report({'root':str(tmp_path),'confirmation_seeds':list(range(10)),'alpha':.05})
    assert result['error_factor_pass'] is expected
