"""Checks for attributed distance-aware rare-centroid preservation."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from svenmeans import SMeans, SMeansConfig, smeans
from svenmeans._rare_guard import (
    distance_aware_removable,
    nearest_other_center_sqdist,
)


def test_nearby_candidates_revived_and_far_rare_centroid_protected():
    centers = torch.tensor([[0., 0.], [.3, 0.], [100., 0.]])
    candidates = torch.tensor([False, True, True])
    seen = torch.tensor([20, 20, 20])
    eligible = distance_aware_removable(
        centers, candidates, mse=1.0, near_factor=4.0,
        last_seen=seen, step=30, abandon_steps=100,
    )
    assert eligible.tolist() == [False, True, False]


def test_abandonment_reclaims_long_unvisited_far_centroid():
    centers = torch.tensor([[0., 0.], [.3, 0.], [100., 0.]])
    candidates = torch.tensor([False, True, True])
    seen = torch.tensor([20, 20, 20])
    eligible = distance_aware_removable(
        centers, candidates, mse=1.0, near_factor=4.0,
        last_seen=seen, step=120, abandon_steps=100,
    )
    assert eligible.tolist() == [False, True, True]


def test_nearest_other_center_tiling_exact_with_shift():
    g = torch.Generator().manual_seed(43)
    c = torch.randn(31, 9, generator=g) * 10
    ids = torch.tensor([0, 4, 8, 16, 30], dtype=torch.int64)
    expected = torch.cdist(c, c).square()
    expected.fill_diagonal_(float("inf"))
    want = expected[ids].min(dim=1).values
    observed = nearest_other_center_sqdist(c, ids, row_tile=2, center_tile=3)
    assert torch.allclose(observed, want, atol=1e-3)
    shifted = nearest_other_center_sqdist(
        c + 100_000., ids, row_tile=1, center_tile=7
    )
    assert torch.allclose(shifted, want, rtol=1e-3, atol=0.1)


def test_no_candidates_and_single_center_are_safe():
    c = torch.tensor([[2., 3.]])
    assert math.isinf(nearest_other_center_sqdist(c, torch.tensor([0]))[0])
    eligible = distance_aware_removable(
        c, torch.tensor([False]), mse=1.0, near_factor=4.,
        last_seen=torch.tensor([0]), step=30, abandon_steps=10,
    )
    assert eligible.tolist() == [False]


@pytest.mark.parametrize("geometry", ["global", "local_direction"])
def test_opt_in_cpu_fit_finite_reproducible(geometry):
    g = torch.Generator().manual_seed(18)
    x = torch.randn(1450, 12, generator=g)
    cfg = SMeansConfig(revive_geometry=geometry, revive_protection="distance")
    a = smeans(x, 56, nepochs=2.0, seed=3, device="cpu",
               config=cfg, return_info=True)
    b = smeans(x, 56, nepochs=2.0, seed=3, device="cpu",
               config=cfg, return_info=True)
    assert torch.equal(a.centroids, b.centroids)
    assert a.guarded_candidate_checks == b.guarded_candidate_checks
    assert a.guarded_candidate_checks >= 0
    assert a.mse is not None and np.isfinite(a.mse)


def test_existing_default_is_unchanged():
    g = torch.Generator().manual_seed(14)
    x = torch.randn(700, 11, generator=g)
    a = smeans(x, 33, nepochs=1.5, seed=4, device="cpu")
    b = smeans(x, 33, nepochs=1.5, seed=4, device="cpu",
               config=SMeansConfig(revive_protection="none"))
    assert torch.equal(a, b)


def test_sklearn_and_invalid_arguments():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(300, 8)).astype("float32")
    est = SMeans(
        n_clusters=24, nepochs=1.0, random_state=2,
        revive_protection="distance", revive_near_factor=4.,
        revive_abandon_epochs=2.0,
    ).fit(x)
    assert np.isfinite(est.inertia_)
    assert est.get_params()["revive_protection"] == "distance"

    for cfg, pattern in [
        (SMeansConfig(revive_protection="invented"), "revive_protection"),
        (SMeansConfig(revive_protection="distance", revive_near_factor=0), "revive_near_factor"),
        (SMeansConfig(revive_protection="distance", revive_abandon_epochs=-1), "revive_abandon_epochs"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            smeans(x, 8, device="cpu", config=cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU in CPU CI")
def test_unvalidated_gpu_path_rejected():
    with pytest.raises(NotImplementedError, match="CPU-only"):
        smeans(torch.randn(100, 8, device="cuda"), 8, device="cuda",
               config=SMeansConfig(revive_protection="distance"))
