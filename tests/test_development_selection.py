from scripts.ndfa_strengthening_20260925.summarize_development import choose


def test_incomplete_grid_cannot_select_an_apparently_favorable_candidate():
    specs=[dict(id="a",optimizer="sgd",lr=.01),dict(id="b",optimizer="sgd",lr=.1)]
    rows=[dict(case=specs[0],status="complete",mean_validation_loss=.1,mean_validation_accuracy=.9,runs=[]),
          dict(case=specs[1],status="incomplete")]
    assert choose(rows,specs)==dict(status="incomplete",selected=None)


def test_selection_uses_ce_retains_boundary_and_excludes_numerical_failures():
    specs=[dict(id=str(i),optimizer="sgd",lr=lr) for i,lr in enumerate([.001,.01,.1])]
    rows=[dict(case=specs[0],status="complete",mean_validation_loss=.4,mean_validation_accuracy=.8,runs=[]),
          dict(case=specs[1],status="complete",mean_validation_loss=.5,mean_validation_accuracy=.9,runs=[]),
          dict(case=specs[2],status="numerical_failure")]
    selected=choose(rows,specs)
    assert selected["selected"]==specs[0]
    assert selected["boundary_flags"]==dict(lr=dict(selected=.001,range=[.001,.1]))
    assert selected["confirmation_ready"] is False
