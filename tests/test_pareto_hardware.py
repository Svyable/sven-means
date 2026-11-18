"""Exact 3D Pareto/HV evaluation and guarded hardware replay fixture."""
import csv
import numpy as np
import pytest
from benchmarks.pareto_hardware_study import _pareto_mask3, hv3, truth_metrics


def brute_front(points):
    p=np.asarray(points,dtype=float)
    return np.array([
        not np.any(np.all(p<=p[i],axis=1)&np.any(p<p[i],axis=1))
        for i in range(len(p))
    ])


@pytest.mark.parametrize("seed", [0,11,33])
def test_fenwick_front_matches_strict_brute_with_duplicates(seed):
    rng=np.random.default_rng(seed)
    p=rng.integers(0,13,size=(100,3)).astype(float)
    p[-3:]=p[0]
    assert np.array_equal(_pareto_mask3(p),brute_front(p))


def test_hypervolume_exact_single_and_two_rectangles():
    p=np.array([[.2,.3,.4]])
    assert hv3(p,reference=(1,1,1))==pytest.approx(.8*.7*.6)
    # Nested dominated rectangle adds no new volume.
    assert hv3(np.r_[p,[[.4,.5,.6]]],reference=(1,1,1))==pytest.approx(.8*.7*.6)
    a=np.array([[0.,.5,.5],[.5,0.,.5]])
    # Union 2*0.25 minus overlap 0.125 = 0.375.
    assert hv3(a,reference=(1,1,1))==pytest.approx(.375)
    assert hv3(np.empty((0,3)))==0.


def test_hv3_equal_to_3d_unit_grid_brute_small():
    rng=np.random.default_rng(22)
    p=rng.choice(np.linspace(.1,.9,9),size=(7,3),replace=True)
    truth=0.
    grid=(np.arange(100)+.5)/100
    # Approximation error <= voxel resolution surface area.
    for x in grid:
        for y in grid:
            for z in grid:
                if np.any(np.all(p<=np.array([x,y,z]),axis=1)):
                    truth+=1e-6
    assert abs(hv3(p,reference=(1,1,1))-truth)<.025


def test_truth_scoring_uses_real_feasible_and_never_double_counts():
    y=np.array([
        [1.,2.,500.,94.], [2.,1.,450.,95.],
        [3.,3.,200.,70.], [.5,.2,1010.,99.],
    ])
    out=truth_metrics(y,np.arange(4),[0,1],accuracy_floor=90,flash_cap=900)
    assert out["true_feasible_designs"]==2
    assert out["front_exact_recall"]==1.
    assert out["hypervolume_ratio"]==pytest.approx(1.)
    no=truth_metrics(y,np.arange(4),[2],accuracy_floor=90,flash_cap=900)
    assert no["hypervolume_ratio"]==0.
    with pytest.raises(ValueError):
        truth_metrics(y,np.arange(4),[0,0],accuracy_floor=90,flash_cap=900)
