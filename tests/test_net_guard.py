"""Exact replacement-cost evaluator tests and opt-in net-holdout integration."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from svenmeans import SMeans, SMeansConfig, smeans
from svenmeans._net_guard import (
    exact_top_three, lower_confidence_gate, net_split_gain_samples,
    reserve_validation,
)


def brute_net(data, centroids, src, tgt, delta):
    old = (data[:, None, :] - centroids[None, :, :]).square().sum(2).min(1).values
    altered = centroids.clone()
    old_source = centroids[src].clone()
    altered[src] = old_source - delta
    altered[tgt] = old_source + delta
    new = (data[:, None, :] - altered[None, :, :]).square().sum(2).min(1).values
    return old - new


@pytest.mark.parametrize("k", [2, 3, 7, 37])
def test_top_three_exact_with_deleted_source_target(k):
    gen = torch.Generator().manual_seed(44 + k)
    data = torch.randn(73, 5, generator=gen)
    centers = torch.randn(k, 5, generator=gen)
    centers[-1] = centers[0] + .03
    values, ids = exact_top_three(data, centers, row_tile=9, centroid_tile=4)
    for src, tgt in [(0, k-1), (k-1, 0), (0, 1)]:
        delta = torch.randn(5, generator=gen)*.05
        estimated = net_split_gain_samples(
            data, values, ids, src, tgt, centers[src], delta
        )
        exact = brute_net(data, centers, src, tgt, delta)
        assert torch.allclose(estimated, exact, atol=2e-5), (k,src,tgt)


def test_large_translation_and_true_rare_target_cost():
    g = torch.Generator().manual_seed(5)
    centers = torch.tensor([[0.,0.],[.1,0.],[200.,0.]], dtype=torch.float32)
    data = torch.cat((
        .05 * torch.randn(220,2,generator=g),
        torch.tensor([[200.,0.]])*torch.ones(20,1)
    ))
    q, ids = exact_top_three(data,centers)
    gain = net_split_gain_samples(
        data,q,ids,0,2,centers[0],torch.tensor([.4,0.])
    )
    assert gain[:220].mean() < 0
    assert gain[220:].mean() < -39000
    ok, mean, lcb = lower_confidence_gate(gain)
    assert not ok and mean < 0 and lcb < mean

    shifted = centers + 2000.
    translated = data + 2000.
    q2, ids2 = exact_top_three(translated, shifted)
    gain2 = net_split_gain_samples(
        translated,q2,ids2,0,2,shifted[0],torch.tensor([.4,0.])
    )
    assert torch.allclose(gain2,gain,rtol=1e-3,atol=.1)


def test_exact_gain_handles_duplicate_centroids_ties():
    c = torch.tensor([[0.,0.],[0.,0.],[0.,0.],[2.,2.]])
    q = torch.tensor([[0.,0.],[1.,0.],[2.,2.]],dtype=torch.float32)
    v,i=exact_top_three(q,c,row_tile=1,centroid_tile=2)
    got=net_split_gain_samples(q,v,i,0,1,c[0],torch.tensor([.5,0.]))
    assert torch.allclose(got,brute_net(q,c,0,1,torch.tensor([.5,0.])),atol=1e-6)


def test_lcb_positive_negative_and_nonfinite():
    x = torch.ones(80,dtype=torch.float32)*2
    ok,mean,lcb=lower_confidence_gate(x)
    assert ok and mean==pytest.approx(2) and lcb==pytest.approx(2)
    assert not lower_confidence_gate(-x)[0]
    assert not lower_confidence_gate(torch.tensor([float("nan"),1.]))[0]
    with pytest.raises(ValueError):
        lower_confidence_gate(torch.tensor([1.]))


def test_reserved_pool_is_disjoint_and_deterministic():
    x=torch.arange(1000).float().reshape(500,2)
    fit,val=reserve_validation(x,seed=11,fraction=.1)
    fit2,val2=reserve_validation(x,seed=11,fraction=.1)
    assert torch.equal(fit,fit2) and torch.equal(val,val2)
    assert len(fit)==450 and len(val)==50
    assert not set(fit[:,0].tolist()) & set(val[:,0].tolist())
    assert len(torch.unique(torch.cat((fit,val),dim=0),dim=0))==len(x)


@pytest.mark.parametrize("geometry,protected", [
    ("global","none"),("global","distance"),
    ("local_direction","none"),("local_direction","distance")
])
def test_holdout_fit_deterministic_and_keeps_full_output_labels(geometry,protected):
    g=torch.Generator().manual_seed(19)
    x=torch.randn(2200,12,generator=g)
    cfg=SMeansConfig(
        revive_geometry=geometry,revive_protection=protected,
        revive_acceptance="net_holdout",net_validation_fraction=.12,
        net_max_proposals=3,net_lcb_z=1.28,
    )
    a=smeans(x,45,nepochs=2.,seed=7,device="cpu",
             config=cfg,compute_labels=True,return_info=True)
    b=smeans(x,45,nepochs=2.,seed=7,device="cpu",
             config=cfg,compute_labels=True,return_info=True)
    assert torch.equal(a.centroids,b.centroids)
    assert a.labels is not None and len(a.labels)==len(x)
    assert a.net_validation_rows==pytest.approx(264,abs=1)
    assert a.net_proposed==a.net_rejected+a.net_accepted
    assert a.net_accepted<=a.net_proposed
    assert a.total_splits==b.total_splits
    assert a.mse is not None and np.isfinite(a.mse)


def test_default_mode_same_as_explicit_no_net():
    x=torch.randn(750,10,generator=torch.Generator().manual_seed(9))
    cfg=SMeansConfig(revive_geometry="local_direction")
    explicit=SMeansConfig(revive_geometry="local_direction",
                          revive_acceptance="none")
    a=smeans(x,30,nepochs=1.5,seed=3,config=cfg,device="cpu")
    b=smeans(x,30,nepochs=1.5,seed=3,config=explicit,device="cpu")
    assert torch.equal(a,b)


def test_sklearn_exposes_net_gate():
    x=np.random.default_rng(14).normal(size=(700,9)).astype("float32")
    est=SMeans(n_clusters=24,nepochs=1.5,random_state=8,
               revive_acceptance="net_holdout",net_validation_fraction=.1).fit(x)
    assert np.isfinite(est.inertia_)
    assert est.get_params()["revive_acceptance"]=="net_holdout"


@pytest.mark.parametrize("cfg,pattern", [
    (SMeansConfig(revive_acceptance="wrong"),"revive_acceptance"),
    (SMeansConfig(net_validation_fraction=0),"net_validation_fraction"),
    (SMeansConfig(net_lcb_z=-1),"net LCB"),
    (SMeansConfig(net_max_proposals=0),"net LCB"),
])
def test_invalid_net_configs_rejected(cfg,pattern):
    x=torch.randn(100,8)
    with pytest.raises(ValueError,match=pattern):
        smeans(x,8,config=cfg,device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(),reason="no GPU on CPU CI")
def test_unvalidated_gpu_mode_fails():
    with pytest.raises(NotImplementedError, match="CPU-only"):
        smeans(torch.randn(200,8,device="cuda"),8,device="cuda",
               config=SMeansConfig(revive_acceptance="net_holdout"))
