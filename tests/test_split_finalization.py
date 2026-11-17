"""Resetting final-pass statistics is a controlled, self-comparison ablation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from svenmeans import SMeans, SMeansConfig, smeans
from svenmeans._finalization import invalidate_split_statistics


def test_only_changed_clusters_lose_final_pass_evidence():
    sums = torch.arange(24, dtype=torch.float64).reshape(6, 4).clone()
    counts = torch.arange(6, dtype=torch.float64) + 10
    before_sums = sums.clone()
    before_counts = counts.clone()
    invalidate_split_statistics(sums, counts, torch.tensor([1, 4]), torch.tensor([3, 5]))
    for i in range(6):
        if i in (1, 3, 4, 5):
            assert sums[i].count_nonzero() == 0
            assert counts[i] == 0
        else:
            assert torch.equal(sums[i], before_sums[i])
            assert counts[i] == before_counts[i]


def test_empty_split_is_no_op():
    sums = torch.randn(4, 3)
    counts = torch.randn(4)
    old_sums, old_counts = sums.clone(), counts.clone()
    invalidate_split_statistics(sums, counts, torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64))
    assert torch.equal(sums, old_sums)
    assert torch.equal(counts, old_counts)


def test_shape_mismatch_is_detected():
    with pytest.raises(ValueError, match="same shape"):
        invalidate_split_statistics(torch.zeros(5, 3), torch.zeros(5),
                                    torch.tensor([1, 2]), torch.tensor([3]))


def test_opt_in_not_a_default_regression():
    x = torch.randn(600, 12, generator=torch.Generator().manual_seed(3))
    control = smeans(x, 18, nepochs=2.0, seed=17, device="cpu")
    explicit = smeans(x, 18, nepochs=2.0, seed=17, device="cpu",
                      config=SMeansConfig(reset_pass_on_split=False))
    assert torch.equal(control, explicit)


@pytest.mark.parametrize("geometry", ["global", "local_direction"])
def test_opt_in_determinism(geometry):
    x = torch.randn(1300, 14, generator=torch.Generator().manual_seed(11))
    cfg = SMeansConfig(revive_geometry=geometry, reset_pass_on_split=True)
    a = smeans(x, 64, nepochs=2.0, seed=5, device="cpu",
               config=cfg, return_info=True)
    b = smeans(x, 64, nepochs=2.0, seed=5, device="cpu",
               config=cfg, return_info=True)
    assert torch.equal(a.centroids, b.centroids)
    assert np.isfinite(a.mse)


def test_sklearn_accepts_reset_parameter():
    X = np.random.default_rng(7).normal(size=(500, 9)).astype("float32")
    model = SMeans(n_clusters=32, nepochs=2.0, reset_pass_on_split=True,
                   random_state=7).fit(X)
    assert model.get_params()["reset_pass_on_split"] is True
    assert np.isfinite(model.inertia_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_reset_finite():
    x = torch.randn(1500, 32, device="cuda")
    cfg = SMeansConfig(reset_pass_on_split=True)
    fit = smeans(x, 64, nepochs=2.0, seed=3, device="cuda",
                 backend="native", config=cfg, return_info=True)
    assert torch.isfinite(fit.centroids).all()
