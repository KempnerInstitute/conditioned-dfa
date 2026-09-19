import math
import numpy as np
from analysis.ndfa_revision_math import two_block_minimum, paired_seed_summary, whole_seed_bootstrap


def test_gain_in_strong_separation_limit():
    raw=two_block_minimum(1,1,.1,math.inf)
    conditioned=two_block_minimum(1,1,.1,1)
    assert np.isclose(raw,1)
    assert np.isclose(conditioned,11/21)
    assert np.isclose(raw-conditioned,10/21)
    assert not np.isclose(raw-conditioned,341/441)


def test_complete_risk_monotonicity_and_no_noise():
    risks=[two_block_minimum(.8,.6,.1,r) for r in [.25,.5,1,2,8]]
    assert np.all(np.diff(risks)>0)
    assert two_block_minimum(1,0,0,8)==0


def test_uncertainty_depends_on_seeds_not_duplicated_grid():
    grid=np.array([[1,2,4,5,8],[3,4,6,7,10]],float)
    a=paired_seed_summary(grid.mean(0))
    b=paired_seed_summary(np.tile(grid,(30,1)).mean(0))
    assert a==b and a['n']==5
    assert np.allclose(whole_seed_bootstrap(grid.mean(0)),whole_seed_bootstrap(np.tile(grid,(30,1)).mean(0)))
