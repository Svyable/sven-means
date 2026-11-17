"""Checks power diagram embedding, assignment budget and real FAISS recall."""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
from benchmarks.power_routing import (
    build_power_ivf, calibrate_power_prices, pairwise_sqdist,
    power_embedding,
)


def test_prices_preserve_weighted_voronoi_routing_exactly():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(160, 6)).astype("float32")
    c = rng.normal(size=(9, 6)).astype("float32")
    prices = np.array([0., .1, -.5, 2., .04, 0., 1.2, -.3, .5], dtype="float32")
    target = np.argmin(pairwise_sqdist(x, c) + prices[None, :], axis=1)
    data, centers = power_embedding(x, c, prices)
    q = faiss.IndexFlatL2(data.shape[1])
    q.add(centers)
    _, actual = q.search(data, 1)
    assert np.array_equal(target, actual[:, 0])
    # Data-to-query exact L2 squared distances unchanged by adding zeros.
    other, _ = power_embedding(x[:2], c, prices)
    assert np.allclose(
        ((other[0] - other[1])**2).sum(),
        ((x[0] - x[1])**2).sum(),
    )


def test_calibration_never_worsens_cv_or_exceeds_train_distortion_budget():
    rng = np.random.default_rng(12)
    # Strong imbalance, 9 wide near-zero groups and one distant point.
    x = np.r_[rng.normal(0, .7, (150, 8)), rng.normal(4, .4, (50, 8))].astype("float32")
    c = np.r_[rng.normal(0, .25, (8, 8)), rng.normal(4, .25, (2, 8))].astype("float32")
    result = calibrate_power_prices(
        x, c, max_distortion_increase=0.01, iterations=80
    )
    assert result["new_cv"] <= result["baseline_cv"] + 1e-8
    assert result["assigned_mse"] <= result["baseline_mse"] * 1.01 + 1e-7
    assert result["prices"].shape == (len(c),)
    second = calibrate_power_prices(x, c, max_distortion_increase=0.01)
    assert np.array_equal(result["prices"], second["prices"])


def test_unweighted_route_matches_original_ivf_and_full_scan_is_exact():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 5)).astype("float32")
    centers = x[:10].copy()
    index = build_power_ivf(x, centers, np.zeros(len(centers)))
    ref = faiss.IndexFlatL2(5)
    ref.add(x)
    query = rng.normal(size=(18, 5)).astype("float32")
    ground_truth = ref.search(query, 10)[1]
    queries, _ = power_embedding(query, centers, np.zeros(len(centers)))
    index.nprobe = len(centers)
    _, retrieved = index.search(queries, 10)
    assert np.array_equal(np.sort(retrieved, axis=1), np.sort(ground_truth, axis=1))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        calibrate_power_prices(np.ones((5, 3)), np.ones((2, 3)),
                               max_distortion_increase=-.1)
    with pytest.raises(ValueError):
        power_embedding(np.ones((5, 2)), np.ones((3, 2)), np.ones(4))
