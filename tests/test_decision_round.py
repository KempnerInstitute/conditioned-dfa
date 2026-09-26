import copy
import json
import pytest
import torch
from scripts.ndfa_strengthening_20260925 import decision_train as runner


def cfg():
    return dict(dataset='fixture',hidden_dims=[7,5],batch_size=8,threads=1,epochs=4,warmup_epochs=1,
        split_seed=931005,feedback_scale=.1,weight_decay=1e-4,foof_inverse_period=3,foof_calibration_batches=3,
        budget_kind='updates',lr_schedule='cosine',evaluate_every_epochs=1,work_warmup_fraction=.025)


@pytest.mark.parametrize('operator,norm,optimizer',[('none','bn','adamw'),('activity','bn','adamw'),('foof','bn','sgd'),('none','fd','adamw'),('activity_geometry','bn','adamw')])
def test_exact_mid_epoch_resume_and_validation_selection(tmp_path,operator,norm,optimizer):
    config=cfg();case=dict(id='case',family='test',credit='bp' if operator=='foof' else 'dfa',operator=operator,normalization=norm,
        lr=.001,optimizer=optimizer,rho=1.,damping=.3,statistic='diagonal_covariance_plus_mean',conditioning_schedule='early' if operator=='activity' else 'always')
    full=runner.train(config,case,41,tmp_path/'full',device='cpu')
    partial=runner.train(config,case,41,tmp_path/'resume',device='cpu',stop_after_step=5)
    assert partial['status']=='interrupted'
    resumed=runner.train(config,case,41,tmp_path/'resume',device='cpu')
    assert full['final_validation']==resumed['final_validation']
    assert full['best_validation']==resumed['best_validation']
    assert full['best_step']==resumed['best_step']
    assert full['stream_sha256']==resumed['stream_sha256']
    assert full['conditioned_updates']==resumed['conditioned_updates']
    for file in ['best.pt','final.pt']:
        a=torch.load(tmp_path/'full'/file,weights_only=False)['model'];b=torch.load(tmp_path/'resume'/file,weights_only=False)['model']
        for name in ['weights','biases','bn_running_mean','bn_running_var']:
            for x,y in zip(a[name],b[name]):torch.testing.assert_close(x,y,atol=0,rtol=0)
    history=json.loads((tmp_path/'full/history.json').read_text())
    assert full['best_validation']['loss']==min(h['validation']['loss'] for h in history)
    assert full['completed_updates']==16


def test_constant_learning_rate_matches_early_and_late_windows():
    config=dict(cfg(),lr_schedule='constant');case=dict(lr=.0003,operator='activity')
    for schedule in ['early','late']:
        steps=[s for s in range(100) if runner.active_case(dict(case,conditioning_schedule=schedule),s/100)['operator']=='activity']
        assert len(steps)==25
        assert sum(runner.rate(case,config,s/100,s,100,5) for s in steps)==pytest.approx(.0075)


def test_work_budget_charges_calibration_and_saves_only_available_checkpoints(tmp_path,monkeypatch):
    ticks=iter(i*.01 for i in range(10000))
    monkeypatch.setattr(runner.time,'perf_counter',lambda:next(ticks))
    config=dict(cfg(),budget_kind='work',work_budget_seconds=.115)
    case=dict(id='case',family='test',credit='bp',operator='foof',normalization='none',optimizer='sgd',lr=.001,damping=.3)
    result=runner.train(config,case,41,tmp_path/'work',device='cpu')
    assert result['status']=='complete'
    assert result['calibration_seconds']>0
    assert result['training_seconds']>=.115
    assert result['budget_overshoot_seconds']<=.01000001
    assert result['best_training_seconds']<=result['training_seconds']
    assert result['official_test_loaded'] is False


def test_holm_correction_and_paired_interval():
    from scripts.ndfa_strengthening_20260925.decision_control import holm,paired_summary
    tests=dict(a=dict(p=.01),b=dict(p=.04),c=dict(p=.03))
    holm(tests)
    assert tests['a']['holm_p']==pytest.approx(.03)
    assert tests['b']['holm_p']==pytest.approx(.06)
    assert tests['c']['holm_p']==pytest.approx(.06)
    r=paired_summary([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
    assert r['mean']==5.5 and r['low']<5.5<r['high'] and r['n']==10


def test_test_evaluation_refuses_development_stage_before_loading_data(tmp_path,monkeypatch):
    from scripts.ndfa_strengthening_20260925 import decision_control as control
    monkeypatch.setattr(control,'audit',lambda plan,study:({'stage':'validation_development'},{}))
    with pytest.raises(AssertionError):control.evaluate({'root':str(tmp_path)},'development')


def test_test_evaluation_gate_requires_both_frozen_selections(tmp_path):
    from scripts.ndfa_strengthening_20260925.decision_control import evaluation_gate
    with pytest.raises(FileNotFoundError):evaluation_gate({'root':str(tmp_path)})


def test_execution_claim_excludes_a_second_live_attempt(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with runner.execution_claim(tmp_path):
        with pytest.raises(RuntimeError,match='owns this run'):
            with runner.execution_claim(tmp_path):pass
    with runner.execution_claim(tmp_path):pass


def test_scheduler_receipts_are_idempotent_and_conflicts_fail(tmp_path):
    from scripts.ndfa_strengthening_20260925.freeze_decision import workflow
    plan={'root':str(tmp_path)}
    workflow(plan,lambda s:s.update({'a:train':{'job':'123'}}))
    workflow(plan,lambda s:s.update({'b:train':{'job':'456'}}))
    assert workflow(plan)=={'a:train':{'job':'123'},'b:train':{'job':'456'}}
    with pytest.raises(AssertionError,match='Conflicting'):
        workflow(plan,lambda s:s.update({'a:train':{'job':'999'}}))


def test_parallel_audit_publication_uses_complete_unique_temporary_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from scripts.ndfa_strengthening_20260925.freeze_decision import read,write
    path=tmp_path/'audit.json'
    def publish(i):
        write(path,{'worker':i,'payload':[i]*1000})
        result=read(path)
        assert result['payload']==[result['worker']]*1000
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(publish,range(32)))
