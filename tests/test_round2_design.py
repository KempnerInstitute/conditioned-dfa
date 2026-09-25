import pytest
from scripts.ndfa_strengthening_20260925.freeze_round2 import extension_cases,horizon_cases
from scripts.ndfa_strengthening_20260925.summarize_round2 import pool


def test_boundary_extension_is_symmetric_unique_and_excludes_old_points():
    cases=[]
    for family in ["bp_none_activity","dfa_none_activity"]:
        for lr in [.001,.01,.1]:
            for damping in [.03,.3]:
                cases.append(dict(id=f"case_{len(cases):03d}",cell="nuisance",family=family,optimizer="sgd_momentum",lr=lr,damping=damping))
    selections={}
    for family in ["bp_none_activity","dfa_none_activity"]:
        winner=next(c for c in cases if c["family"]==family and c["lr"]==.001 and c["damping"]==.3)
        selections[family]=dict(selected=winner,boundary_flags={"lr":dict(selected=.001,range=[.001,.1]),"damping":dict(selected=.3,range=[.03,.3])})
    added,ledger=extension_cases(dict(cases=cases),dict(complete=True,selections=selections))
    assert [l["new_candidates"] for l in ledger]==[12,12]
    points=lambda rows:{(c["family"],c["lr"],c["damping"]) for c in rows}
    assert not points(added)&points(cases)
    assert len(points(added))==len(added)
    assert len({c["id"] for c in cases+added})==len(cases+added)
    for family in selections:
        assert {c["damping"] for c in added if c["family"]==family}>={.9,3.}


def test_horizon_includes_every_family_and_rejects_incomplete_parent():
    s=dict(complete=True,selections={"a":dict(selected=dict(id="case_003",family="bp")),"b":dict(selected=dict(id="case_009",family="dfa"))})
    cases=horizon_cases({},s)
    assert [c["family"] for c in cases]==["bp","dfa"]
    assert s["selections"]["a"]["selected"]["id"]=="case_003"
    with pytest.raises(AssertionError):horizon_cases({},dict(s,complete=False))


def test_pooled_selection_keeps_parent_winner_and_waits_for_new_grid():
    old=dict(case=dict(id="case_000",cell="nuisance",family="dfa",optimizer="sgd",lr=.001),status="complete",mean_validation_loss=.5,mean_validation_accuracy=.6,runs=[])
    new=dict(case=dict(old["case"],id="case_384",lr=.0001),status="complete",mean_validation_loss=.8,mean_validation_accuracy=.7,runs=[])
    result=pool(dict(complete=True,candidates=[old]),dict(candidates=[new]))
    assert result['nuisance/dfa']['selected']['id']=='case_000'
    new['status']='incomplete'
    assert pool(dict(complete=True,candidates=[old]),dict(candidates=[new]))['nuisance/dfa']['selected'] is None


@pytest.mark.parametrize("name",["rho","decor_lr"])
def test_extension_supports_cifar_relative_damping_and_fd_rate(name):
    cases=[dict(id=f"case_{i}",family="dfa",optimizer="adamw",lr=.001,**{name:x}) for i,x in enumerate([1.,30.])]
    summary=dict(complete=True,selections={"cifar10/dfa":dict(selected=cases[0],boundary_flags={name:dict(selected=1.,range=[1.,30.])})})
    added,_=extension_cases(dict(cases=cases),summary)
    assert len(added)==2 and all(c[name]<1. for c in added)
