"""No expensive oracle leakage, streaming archive and conditional-cover tests."""
import numpy as np
import pytest

from benchmarks.pareto_archive import (
    OPS, topology_features, fit_linear_proxy, normalized_proxy_costs,
    choose_candidates, greedy_feature_diversity, streaming_proxy_grid,
)


def fake_arches(n=150, seed=12):
    rng=np.random.default_rng(seed)
    ops=np.array(OPS)
    ids=rng.integers(0,len(ops),size=(n,6))
    top=["+".join("|"+ops[e]+"~"+str(j%3)+"|" for j,e in enumerate(row))
         for row in ids]
    size=rng.uniform(100,710,size=n)
    return top,size


def test_features_use_only_topology_and_preconversion_size():
    tops,size=fake_arches()
    x,f=topology_features(tops,size)
    assert x.shape==(len(tops),31)
    assert len(f)==len(tops)
    assert np.allclose(x[:,:30].sum(1),6)
    assert np.allclose(x[:,-1],np.log1p(size))
    with pytest.raises(ValueError):
        topology_features(["not a valid topology"],[500])


def test_proxy_uses_only_paid_calibration_labels():
    tops,size=fake_arches()
    x,f=topology_features(tops,size)
    n=len(x)
    rng=np.random.default_rng(42)
    truth=np.column_stack((rng.normal(-2,1,n),rng.normal(1,1,n),
                           rng.normal(85,8,n)))
    cal=np.arange(25)
    baseline,_=fit_linear_proxy(x,truth,cal)
    changed=truth.copy()
    changed[25:]+=1000000
    alt,_=fit_linear_proxy(x,changed,cal)
    assert np.array_equal(baseline,alt)
    others=np.arange(25,n)
    for method in ("random","stratified","kmeans++","farthest",
                   "proxy_rank","proxy_grid"):
        a=choose_candidates(method,x,baseline,size,f,cal,others,17,seed=4)
        b=choose_candidates(method,x,alt,size,f,cal,others,17,seed=4)
        assert a==b
        assert len(a)==len(set(a))==17
        assert not set(a) & set(cal)


@pytest.mark.parametrize("budget", [1,9,40,100])
def test_archive_caps_capacity_and_is_deterministic(budget):
    tops,size=fake_arches()
    x,f=topology_features(tops,size)
    rng=np.random.default_rng(30)
    proxy=np.column_stack((
        rng.uniform(-2,2,len(x)),rng.uniform(0,3,len(x)),
        rng.uniform(75,95,len(x))))
    norm=normalized_proxy_costs(proxy,size,np.arange(len(x)))
    a=streaming_proxy_grid(proxy,norm,size,np.arange(len(x)),f,budget,seed=91)
    b=streaming_proxy_grid(proxy,norm,size,np.arange(len(x)),f,budget,seed=91)
    assert a==b
    assert len(a)==len(set(a))==budget


def test_conditional_proxy_grid_cover_no_evictions():
    # A held representative in the same cell differs in each proxy
    # coordinate by < side width; the theorem adds 2*error bound.
    proxy=np.array([[1.0,2.,91.],[1.01,2.01,91.01],
                    [4.,7.,70.],[4.01,7.01,70.01]])
    normalized=np.array([[.10,.10,.10],[.11,.12,.11],
                         [.60,.60,.60],[.61,.62,.61]])
    size=np.ones(4)*300
    fam=np.zeros(4,dtype=int)
    ids=streaming_proxy_grid(proxy,normalized,size,np.arange(4),fam,
                             budget=2,seed=1,cell_width=.15)
    assert set(ids)&{0,1}
    assert set(ids)&{2,3}
    for i in range(4):
        chosen=[j for j in ids if np.array_equal(
            np.floor(normalized[i]/.15),
            np.floor(normalized[j]/.15))]
        assert chosen
        assert max(abs(normalized[i]-normalized[chosen[0]])) <.15+1e-12


def test_invalid_pools_and_calibration():
    tops,size=fake_arches()
    x,f=topology_features(tops,size)
    proxy=np.zeros((len(x),3))
    with pytest.raises(ValueError):
        choose_candidates("random",x,proxy,size,f,[0],[0,1,2],2,9)
    with pytest.raises(ValueError):
        choose_candidates("invented",x,proxy,size,f,[0],[1,2],1,9)
    with pytest.raises(ValueError):
        streaming_proxy_grid(proxy,np.zeros((len(x),3)),size,
                             np.arange(4),f,5,9)
