"""Recompute correction summaries from the portable endpoint records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets/ndfa_strengthening_20260925"


def same(a, b):
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)


def main():
    checks = []
    d = pd.read_csv(DATA/"synthetic_endpoints.csv")
    expected = pd.read_csv(DATA/"synthetic_stability.csv").set_index(["condition","method"])
    for key,g in d.groupby(["condition","method"]):
        row=expected.loc[key]
        same([g.test_acc.mean(),g.loss.median(),len(g),g.loss.gt(10).mean()],
             [row.accuracy,row.median_loss,row.runs,row.loss_above_10])
    checks.append("synthetic final-loss distributions and original-sweep BP")
    d=pd.read_csv(DATA/"adam_selected_endpoints.csv")
    expected=pd.read_csv(DATA/"adam_seed_summary.csv").set_index(["cell","method"])
    for key,g in d.groupby(["cell","method"]):
        assert len(g)==10 and len(g.seed.unique())==5
        v=100*g.groupby("seed").test_acc.mean()
        same([v.mean(),v.sem()],expected.loc[key,["accuracy_pct","sem_pp"]].values.astype(float))
    checks.append("Adam control: feedback averaged before five-seed SEM")
    d=pd.read_csv(DATA/"bn_endpoints.csv")
    keys=["condition","n_train","train_label_noise","input_noise","seed","feedback_seed"]
    w=d.pivot(index=keys,columns="method",values="test_acc")
    assert not w.isna().any().any()
    expected=pd.read_csv(DATA/"bn_headtohead.csv").set_index("condition")
    for condition,g in w.groupby(level="condition"):
        v=100*(g.ndfa_random-g.dfa_random).groupby(level="seed").mean()
        assert len(v)==5
        lo,hi=stats.t.interval(.95,4,loc=v.mean(),scale=v.sem())
        same([v.mean(),v.sem(),lo,hi],expected.loc[condition,["delta_pp","sem_pp","t95_low_pp","t95_high_pp"]].values)
    checks.append("within-BN contrasts: complete global seeds and t intervals")
    expected=pd.read_csv(DATA/"synthetic_t_intervals.csv").set_index("condition")
    for condition,g in pd.read_csv(DATA/"synthetic_seed_deltas.csv").groupby("condition"):
        v=g.mean_delta_pp; assert len(v)==5
        lo,hi=stats.t.interval(.95,4,loc=v.mean(),scale=v.sem())
        same([v.mean(),lo,hi],expected.loc[condition,["mean_delta_pp","t95_low_pp","t95_high_pp"]].values)
    checks.append("synthetic intervals: five independent seed summaries")
    expected=pd.read_csv(DATA/"factor_cohort_summary.csv").set_index(["study","cohort","method"])
    for stem,n in [("dfa_stall_threefactor",5),("dfa_stall_fashion_threefactor",5),("dfa_relu_mnist_threefactor",8)]:
        original=pd.read_csv(DATA/f"{stem}_original_seed_means.csv")
        extension=pd.read_csv(DATA/f"{stem}_extension_seed_means.csv")
        assert len(original.seed.unique())==n and len(extension.seed.unique())==5
        assert not set(original.seed)&set(extension.seed)
        pooled=pd.read_csv(DATA/f"{stem}_pooled_seed_means.csv")
        cols=["seed","method"]
        pd.testing.assert_frame_equal(pd.concat([original,extension]).sort_values(cols).reset_index(drop=True),pooled.sort_values(cols).reset_index(drop=True))
        for cohort in ["original","extension","pooled"]:
            d=pd.read_csv(DATA/f"{stem}_{cohort}_seed_means.csv")
            for method,g in d.groupby("method"):
                same([100*g.test_acc.mean(),100*g.test_acc.sem(),g.test_loss.mean(),g.test_loss.sem()],
                     expected.loc[(stem,cohort,method),["accuracy_pct","accuracy_sem_pp","loss","loss_sem"]].values.astype(float))
    checks.append("original, extension and pooled cohorts: disjointness and all means/SEMs")
    d=pd.read_csv(DATA/"original_vision_summary.csv")
    assert len(d)==192 and d.feedback_rank.eq(0).all()
    assert d.groupby(["dataset","n_train","label_noise","method"]).size().eq(1).all()
    checks.append("original vision sweep: full rank and one configuration per method/cell")
    # Table generation consumes the same verified exports. Reject a hand-edited
    # number in any inserted table while allowing surrounding prose to evolve.
    paper=ROOT/"drafts/Info-DFA/supplement.tex"
    if not paper.exists():
        paper=ROOT/"paper/supplement.tex"
    if paper.exists():
        text=paper.read_text()
        for name in ["stability","adam","bn","extensions"]:
            assert (DATA/f"table_{name}.tex").read_text().strip() in text, name
        checks.append("four generated correction tables match the manuscript exactly")
    print(json.dumps(dict(passed=True,checks=checks),indent=2))


if __name__ == "__main__":
    main()
