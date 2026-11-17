"""Prove basic scan-cost accounting and interpolation invariants."""
import pytest
import torch

from benchmarks.scan_budget import interpolated_recall_at_budget, scan_curve


def test_all_lists_probe_all_points_and_all_neighbors():
    train = torch.tensor([[0.], [.2], [10.], [10.2]])
    test = torch.tensor([[.1], [10.1]])
    centroids = torch.tensor([[0.], [10.]])
    curve = scan_curve(train, test, centroids, probes=(1, 2),
                       true_neighbors=2, query_count=2)
    assert [p["nprobe"] for p in curve] == [1, 2]
    assert curve[-1]["mean_scan_fraction"] == pytest.approx(1.0)
    assert curve[-1]["mean_candidate_recall_at_10"] == pytest.approx(1.0)
    assert curve[0]["mean_scan_fraction"] == pytest.approx(.5)


def test_mean_scan_and_recall_monotone_with_probe_count():
    rng = torch.Generator().manual_seed(44)
    train = torch.randn(90, 5, generator=rng)
    test = torch.randn(20, 5, generator=rng)
    centers = train[torch.randperm(90, generator=rng)[:12]]
    curve = scan_curve(train, test, centers, probes=(1, 2, 4, 8, 12))
    costs = [p["mean_scan_fraction"] for p in curve]
    rec = [p["mean_candidate_recall_at_10"] for p in curve]
    assert costs == sorted(costs)
    assert rec == sorted(rec)
    assert costs[-1] == pytest.approx(1.0)
    assert rec[-1] == pytest.approx(1.0)


def test_interpolation_is_expected_mixture_not_extrapolation():
    curve = [
        {"mean_scan_fraction": .2, "mean_candidate_recall_at_10": .4},
        {"mean_scan_fraction": .6, "mean_candidate_recall_at_10": .8},
        {"mean_scan_fraction": 1.0, "mean_candidate_recall_at_10": 1.0},
    ]
    assert interpolated_recall_at_budget(curve, .4) == pytest.approx(.6)
    assert interpolated_recall_at_budget(curve, .8) == pytest.approx(.9)
    assert interpolated_recall_at_budget(curve, .1) is None
    assert interpolated_recall_at_budget(curve, .2) == pytest.approx(.4)


def test_invalid_data_and_budget_rejected():
    with pytest.raises(ValueError):
        interpolated_recall_at_budget([], 0.0)
    with pytest.raises(ValueError):
        scan_curve(torch.randn(20, 4), torch.randn(4, 3), torch.randn(3, 4))


def test_unequal_cluster_occupancy_is_explicitly_charged():
    train = torch.tensor([[0.], [.01], [.02], [.03], [.04], [10.]])
    queries = torch.tensor([[.01], [10.]])
    centers = torch.tensor([[0.], [10.]])
    curve = scan_curve(train, queries, centers, probes=(1, 2),
                       true_neighbors=1)
    # nprobe=1 scans five of six candidates for the first query, but only
    # one of six for the second. "one probed list" is not constant cost.
    assert curve[0]["mean_scan_fraction"] == pytest.approx(.5)
    assert curve[0]["max_scan_fraction"] == pytest.approx(5 / 6)
