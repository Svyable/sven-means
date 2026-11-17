"""Unit and integration tests for experimental rank-one residual revival."""
import math

import numpy as np
import pytest
import torch

from svenmeans import SMeans, SMeansConfig, smeans
from svenmeans._directional import DirectionalRevival


def make_tracker(k=2, d=2, beta=0.0, observations=1):
    return DirectionalRevival(
        k, d, seed=9, device=torch.device("cpu"),
        beta=beta, min_observations=observations,
    )


def test_projected_gain_matches_exact_two_center_sse_improvement():
    # Force a known unit direction. With r = +/-2e1, proposed rho=1.1,
    # improvement for each row is 2*rho*2 - rho^2 = 3.19.
    tracker = make_tracker()
    tracker.q[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = torch.tensor([[-2., 0.], [2., 0.], [-2., 0.], [2., 0.]])
    labels = torch.zeros(4, dtype=torch.int64)
    centers = torch.zeros(2, 2)
    counts = torch.tensor([4., 0.])
    tracker.observe(x, labels, centers, counts)

    src, delta = tracker.propose(
        torch.tensor([True, False]), 1, torch.tensor([4., 0.]),
        split_mult=0.55,
    )
    assert src.tolist() == [0]
    assert torch.allclose(delta, torch.tensor([[1.1, 0.0]]), atol=1e-5)
    original = x.square().sum(dim=1)
    moved = torch.minimum(
        (x - delta[0]).square().sum(dim=1),
        (x + delta[0]).square().sum(dim=1),
    )
    actual = (original - moved).mean()
    expected = 2 * 1.1 * tracker.mean_abs[0] - 1.1**2
    assert torch.allclose(actual, expected, atol=1e-5)
    assert actual.item() > 3.1


def test_zero_residual_batch_cannot_destroy_direction():
    tracker = make_tracker(k=1, d=3, beta=0.0)
    old = tracker.q.clone()
    tracker.observe(torch.zeros(5, 3), torch.zeros(5, dtype=torch.int64),
                    torch.zeros(1, 3), torch.tensor([5.]))
    assert torch.allclose(tracker.q, old, atol=1e-6)
    assert tracker.q.norm().item() > 0.99


def test_mean_absolute_radius_is_conditional_sse_optimum():
    tracker = make_tracker()
    tracker.q[:] = torch.tensor([[1., 0.], [0., 1.]])
    x = torch.tensor([[-2., 0.], [2., 0.]])
    tracker.observe(x, torch.zeros(2, dtype=torch.int64),
                    torch.zeros(2, 2), torch.tensor([2., 0.]))
    src, delta = tracker.propose(
        torch.tensor([True, False]), 1, torch.tensor([2., 0.]),
        split_mult=0.55, radius_policy="mean_abs",
    )
    assert src.tolist() == [0]
    assert torch.allclose(delta, torch.tensor([[2., 0.]]), atol=1e-5)
    # Radius 2 is the exact minimizing radius for these two fixed points.
    for rho in (1., 1.9, 2.1, 3.):
        candidate = torch.tensor([rho, 0.])
        loss = torch.minimum((x - candidate).square().sum(1),
                             (x + candidate).square().sum(1)).mean()
        assert loss >= 0.0
    assert torch.minimum((x - delta[0]).square().sum(1),
                         (x + delta[0]).square().sum(1)).mean() == 0.0


def test_learns_rotated_residual_direction_with_rank_one_sketch():
    rng = torch.Generator().manual_seed(42)
    t = torch.randn(1000, 1, generator=rng) * 5.0
    x = t * torch.tensor([[1., 1.]]) / math.sqrt(2)
    x += torch.randn(1000, 2, generator=rng) * 0.05
    tracker = make_tracker(k=1, d=2, beta=0.4)
    labels = torch.zeros(1000, dtype=torch.int64)
    counts = torch.tensor([1000.0])
    for _ in range(30):
        tracker.observe(x, labels, torch.zeros(1, 2), counts)
    truth = torch.tensor([1., 1.]) / math.sqrt(2)
    cosine = torch.abs(torch.dot(tracker.q[0], truth)).item()
    assert cosine > 0.98


def test_proposal_filters_cold_uninformative_and_negative_gain():
    tracker = make_tracker(k=3, d=2, observations=100)
    tracker.q[:] = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    x = torch.tensor([[-1., 0.], [1., 0.]])
    tracker.observe(x, torch.zeros(2, dtype=torch.int64),
                    torch.zeros(3, 2), torch.tensor([2., 0., 0.]))
    src, delta = tracker.propose(torch.tensor([True, True, True]), 2,
                                 torch.tensor([2., 2., 2.]), 0.55)
    assert src.numel() == 0 and delta.shape == (0, 2)

    tracker.observed[0] = 100.
    src, delta = tracker.propose(torch.tensor([True, False, False]), 1,
                                 torch.tensor([2., 2., 2.]), 0.55,
                                 gain_floor=100.0)
    assert src.numel() == 0


def test_split_resets_ema_and_copies_direction():
    tracker = make_tracker()
    tracker.observed[:] = torch.tensor([50., 2.])
    tracker.mean_sq[:] = torch.tensor([8., 9.])
    tracker.mean_abs[:] = torch.tensor([2., 3.])
    before = tracker.q[0].clone()
    tracker.on_split(torch.tensor([0]), torch.tensor([1]))
    assert torch.equal(tracker.q[0], before)
    assert torch.equal(tracker.q[1], before)
    assert tracker.observed.sum().item() == 0
    assert tracker.mean_sq.sum().item() == 0
    assert tracker.mean_abs.sum().item() == 0


def test_local_mode_end_to_end_determinism_and_finiteness():
    rng = torch.Generator().manual_seed(77)
    d, ktrue, n = 6, 5, 2000
    centers = torch.randn(ktrue, d, generator=rng) * 5.
    assigned = torch.randint(ktrue, (n,), generator=rng)
    x = centers[assigned] + torch.randn(n, d, generator=rng) * 0.7
    cfg = SMeansConfig(
        revive_geometry="local_direction",
        local_beta=0.60,
        local_min_observations=8,
        center_data=True,
    )
    a = smeans(x, 20, nepochs=2.0, seed=5, device="cpu",
               config=cfg, return_info=True)
    b = smeans(x, 20, nepochs=2.0, seed=5, device="cpu",
               config=cfg, return_info=True)
    assert a.centroids.shape == (20, d)
    assert torch.allclose(a.centroids, b.centroids, atol=1e-6)
    assert math.isfinite(a.mse)
    assert a.total_splits >= 0
    assert a.inertia >= 0


def test_sklearn_exposes_local_parameters():
    X = np.random.default_rng(2).normal(size=(300, 4)).astype(np.float32)
    est = SMeans(
        n_clusters=12, nepochs=1, random_state=7,
        revive_geometry="local_direction",
        local_beta=0.5, local_min_observations=4,
        local_gain_floor=0.0,
    ).fit(X)
    assert np.isfinite(est.inertia_)
    assert est.get_params()["revive_geometry"] == "local_direction"


def test_sketch_observation_stride_is_deterministic():
    x = torch.randn(900, 6, generator=torch.Generator().manual_seed(19))
    cfg = SMeansConfig(revive_geometry="local_direction", local_observe_every=4)
    a = smeans(x, 20, nepochs=2, device="cpu", seed=3, config=cfg)
    b = smeans(x, 20, nepochs=2, device="cpu", seed=3, config=cfg)
    assert torch.equal(a, b)


def test_reject_invalid_local_mode_and_bad_hyperparameters():
    x = torch.randn(100, 4)
    with pytest.raises(ValueError, match="revive_geometry"):
        smeans(x, 4, config=SMeansConfig(revive_geometry="magic"), device="cpu")
    with pytest.raises(ValueError, match="beta"):
        smeans(x, 4, config=SMeansConfig(revive_geometry="local_direction", local_beta=1.0),
               device="cpu")
    with pytest.raises(ValueError, match="min_observations"):
        smeans(x, 4, config=SMeansConfig(revive_geometry="local_direction",
                                         local_min_observations=0), device="cpu")
    with pytest.raises(ValueError, match="local_observe_every"):
        smeans(x, 4, device="cpu",
               config=SMeansConfig(revive_geometry="local_direction", local_observe_every=0))
    with pytest.raises(ValueError, match="local_radius_policy"):
        smeans(x, 4, device="cpu",
               config=SMeansConfig(revive_geometry="local_direction", local_radius_policy="invalid"))
    with pytest.raises(ValueError, match="gain_floor"):
        tracker = make_tracker()
        tracker.propose(torch.tensor([True, True]), 1, torch.ones(2), 0.5, -1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present on CPU CI")
def test_cuda_rejects_unvalidated_mode():
    x = torch.randn(100, 4, device="cuda")
    with pytest.raises(NotImplementedError, match="CPU-only"):
        smeans(x, 8, device="cuda",
               config=SMeansConfig(revive_geometry="local_direction"))
