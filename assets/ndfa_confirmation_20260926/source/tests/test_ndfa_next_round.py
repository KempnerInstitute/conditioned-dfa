import copy
from pathlib import Path
import pytest
import torch
from scripts.ndfa_strengthening_20260925 import next_round as study


def config():
    return dict(dataset="synthetic",hidden_dims=[7,5],batch_size=8,epochs=4,
                threads=1,feedback_scale=1.,weight_decay=1e-4,warmup_epochs=1,
                validation_examples=24,evaluate_every_epochs=1,checkpoint_every_epochs=1,
                foof_inverse_period=3,foof_calibration_batches=5,
                cells={"test":dict(n_train=16,task_scale=.45,nuisance_scale=2.,
                interaction=False,input_noise=.15,label_noise=.2)})


@pytest.mark.parametrize("operator,optimizer",[("none","adamw"),("activity","sgd_momentum"),("foof","sgd")])
@pytest.mark.parametrize("normalization",["none","bn"])
def test_interrupted_training_is_identical_after_resume(tmp_path,operator,optimizer,normalization):
    cfg=config();case=dict(id="test",cell="test",credit="bp" if operator=="foof" else "dfa",
                          normalization=normalization,operator=operator,optimizer=optimizer,lr=.003,damping=.3)
    full=study.train(cfg,case,41,tmp_path/"full",device="cpu")
    partial=study.train(cfg,case,41,tmp_path/"resume",device="cpu",stop_after_epoch=2)
    assert partial["status"]=="interrupted" and not (tmp_path/"resume/endpoint.json").exists()
    resumed=study.train(cfg,case,41,tmp_path/"resume",device="cpu")
    assert full["status"]==resumed["status"]=="complete"
    assert full["final_validation"]==resumed["final_validation"]
    assert full["stream_sha256"]==resumed["stream_sha256"]
    a=torch.load(tmp_path/"full/final.pt",weights_only=False)
    b=torch.load(tmp_path/"resume/final.pt",weights_only=False)
    for name in ["weights","biases","bn_gamma","bn_beta","bn_running_mean","bn_running_var"]:
        for left,right in zip(a["model"][name],b["model"][name]):
            torch.testing.assert_close(left,right,atol=0,rtol=0)
    assert study.train(cfg,case,41,tmp_path/"resume",device="cpu")==resumed


def test_validation_size_does_not_change_train_or_label_corruption():
    cfg=config();case=dict(cell="test")
    a=study.synthetic_data(cfg,case,123)
    cfg["validation_examples"]=101
    b=study.synthetic_data(cfg,case,123)
    torch.testing.assert_close(a.train,b.train,rtol=0,atol=0)
    torch.testing.assert_close(a.labels,b.labels,rtol=0,atol=0)
    assert a.provenance["test_generated"] is False


def test_resume_rejects_changed_scientific_configuration(tmp_path):
    cfg=config();case=dict(id="test",cell="test",credit="dfa",normalization="bn",operator="activity",optimizer="adamw",lr=.003,damping=.3)
    study.train(cfg,case,41,tmp_path/"run",device="cpu",stop_after_epoch=2)
    changed=copy.deepcopy(cfg);changed["epochs"]=5
    with pytest.raises(AssertionError,match="Resume identity changed"):
        study.train(changed,case,41,tmp_path/"run",device="cpu")
