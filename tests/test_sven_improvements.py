"""Tests for sven-means' deliberately changed behavior, not just inherited APIs."""
import numpy as np
import pytest
import torch

from svenmeans import SMeans, SMeansConfig, assign_clusters, smeans
from svenmeans._recipe import _sample_coreset_indices, subsampled_d15_init


def test_initializer_unique_indices_and_determinism():
    x = torch.arange(3200, dtype=torch.float32).reshape(-1, 1)
    a = subsampled_d15_init(x, k=150, seed=17, max_samples=256, tile_cols=9)
    b = subsampled_d15_init(x, k=150, seed=17, max_samples=256, tile_cols=9)
    assert a.shape == (150, 1)
    assert torch.equal(a, b)
    assert len(torch.unique(a)) == 150


def test_no_giant_n_permutation_for_small_coreset(monkeypatch):
    original = torch.randperm

    def guarded(n, *args, **kwargs):
        assert n < 100_000, "allocated full dataset permutation"
        return original(n, *args, **kwargs)

    monkeypatch.setattr(torch, "randperm", guarded)
    rng = torch.Generator().manual_seed(6)
    indexes = _sample_coreset_indices(1_000_000, 4096, rng, torch.device("cpu"))
    assert indexes.shape == (4096,)
    assert indexes.unique().numel() == 4096


def test_initializer_bounds_distance_tile(monkeypatch):
    original = torch.addmm
    widths = []

    def monitored(input, mat1, mat2, *args, **kwargs):
        widths.append(mat2.shape[1])
        return original(input, mat1, mat2, *args, **kwargs)

    monkeypatch.setattr(torch, "addmm", monitored)
    x = torch.randn(1800, 8, generator=torch.Generator().manual_seed(7))
    out = subsampled_d15_init(x, k=96, seed=0, max_samples=400, tile_cols=11)
    assert out.shape == (96, 8)
    assert widths and max(widths) <= 11


def test_zero_distances_cannot_reselect_identical_index():
    x = torch.zeros(500, 3)
    # Distinct selected indices aren't observable from identical rows,
    # but the routine must terminate without multinomial zero-probability errors.
    assert subsampled_d15_init(x, k=80, seed=3, tile_cols=7).shape == (80, 3)


def test_translation_stable_assignments():
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]]) + 100_000
    points = torch.tensor([[0.1, 0.0], [1.9, 0.0]]) + 100_000
    labels, sq = assign_clusters(points, centers)
    assert labels.tolist() == [0, 1]
    assert torch.allclose(sq, torch.tensor([0.01, 0.01]), atol=0.01)


def test_cpu_centroids_only_skips_extra_assignment(monkeypatch):
    import svenmeans._cpu as cpu

    original = cpu.tiled_assign
    calls = []

    def monitored(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cpu, "tiled_assign", monitored)
    x = torch.randn(900, 6, generator=torch.Generator().manual_seed(1))
    a = smeans(x, 12, nepochs=1.0, seed=2, device="cpu",
               compute_labels=False, return_info=True)
    without = len(calls)
    assert a.labels is None and a.inertia is None and a.mse is None
    calls.clear()
    b = smeans(x, 12, nepochs=1.0, seed=2, device="cpu",
               compute_labels=True, return_info=True)
    assert len(calls) == without + 1
    assert torch.allclose(a.centroids, b.centroids)
    assert b.labels.shape == (900,) and b.inertia >= 0


def test_center_fit_shifted_data_has_finite_quality():
    rng = torch.Generator().manual_seed(8)
    x = torch.cat([torch.randn(200, 4, generator=rng) * 0.2,
                   torch.randn(200, 4, generator=rng) * 0.2 + 7.0])
    shifted = x + 100_000.0
    cfg = SMeansConfig(center_data=True, init_tile_cols=5)
    a = smeans(x, 2, nepochs=2, seed=10, device="cpu", config=cfg, return_info=True)
    b = smeans(shifted, 2, nepochs=2, seed=10, device="cpu", config=cfg, return_info=True)
    assert np.isfinite(a.mse) and np.isfinite(b.mse)
    assert abs(a.mse - b.mse) < 0.1
    assert torch.allclose(a.centroids, b.centroids - 100_000.0, atol=0.10)


def test_sklearn_optional_new_parameters():
    x = np.random.default_rng(4).normal(size=(200, 4)).astype(np.float32) + 100_000
    est = SMeans(n_clusters=4, nepochs=2, random_state=0,
                 center_data=True, init_max_samples=48, init_tile_cols=7)
    est.fit(x)
    assert est.cluster_centers_.shape == (4, 4)
    assert np.isfinite(est.inertia_)
    assert est.predict(x[:20]).shape == (20,)


@pytest.mark.parametrize("kwargs", [
    {"rounds": 1}, {"max_samples": 0}, {"tile_cols": 0}, {"power": -1}
])
def test_invalid_initialization_config(kwargs):
    with pytest.raises(ValueError):
        subsampled_d15_init(torch.randn(100, 3), 5, seed=0, **kwargs)
