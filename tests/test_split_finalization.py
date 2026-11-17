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


def test_fractional_donor_retention_zeros_target_only():
    sums = torch.arange(20, dtype=torch.float64).reshape(5, 4).clone()
    counts = torch.arange(5, dtype=torch.float64) + 2
    initial_sums, initial_counts = sums.clone(), counts.clone()
    invalidate_split_statistics(sums, counts, torch.tensor([2]),
                                torch.tensor([4]), donor_retention=0.5)
    assert torch.equal(sums[2], initial_sums[2] * 0.5)
    assert counts[2] == initial_counts[2] * 0.5
    assert sums[4].count_nonzero() == 0 and counts[4] == 0
    for idx in (0, 1, 3):
        assert torch.equal(sums[idx], initial_sums[idx])
        assert counts[idx] == initial_counts[idx]


def test_target_only_preserves_donor_statistics():
    sums = torch.randn(5, 4, generator=torch.Generator().manual_seed(7))
    counts = torch.arange(5, dtype=torch.float32) + 5
    old_sums, old_counts = sums.clone(), counts.clone()
    invalidate_split_statistics(sums, counts, torch.tensor([1]),
                                torch.tensor([3]), donor_retention=1.0)
    assert torch.equal(sums[1], old_sums[1])
    assert counts[1] == old_counts[1]
    assert sums[3].count_nonzero() == 0 and counts[3] == 0


def test_invalid_retention_rejected_even_when_reset_disabled():
    with pytest.raises(ValueError, match="donor_pass_retention"):
        smeans(torch.randn(100, 4), 5, device="cpu",
               config=SMeansConfig(donor_pass_retention=-0.1))
    with pytest.raises(ValueError, match="donor_retention"):
        invalidate_split_statistics(torch.zeros(2, 2), torch.zeros(2),
                                    torch.tensor([0]), torch.tensor([1]),
                                    donor_retention=1.1)


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
