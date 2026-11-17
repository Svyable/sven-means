"""Unit proofs and invariants for objective-aligned streaming split directions."""
import math

import numpy as np
import pytest
import torch

from svenmeans import SMeans, SMeansConfig, smeans
from svenmeans._directional import DirectionalRevival


def tracker(objective, d=2, beta=0.0):
    x = DirectionalRevival(
        1, d, seed=8, device=torch.device("cpu"), beta=beta,
        min_observations=1, direction_objective=objective,
    )
    x.q[0] = torch.tensor([1.] + [0.] * (d-1))
    return x


def test_exact_sign_gradient_and_covariance_gradient_diverge():
    # Every point projects positively onto the old direction e1.
    # E[r * sign(r.q)] = [2, -.5] while E[r*(r.q)] = [7, -5.5].
    # The sign gradient better targets E|r.q| on this heavy-tail example.
    residual = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [5., -5.]])
    centers = torch.zeros((1, 2))
    ids = torch.zeros(4, dtype=torch.int64)
    counts = torch.tensor([4.])
    l1 = tracker("split_gain")
    l2 = tracker("covariance")
    l1.observe(residual, ids, centers, counts)
    l2.observe(residual, ids, centers, counts)

    want_l1 = torch.nn.functional.normalize(torch.tensor([2., -.5]), dim=0)
    want_l2 = torch.nn.functional.normalize(torch.tensor([7., -5.5]), dim=0)
    assert torch.allclose(l1.q[0], want_l1, atol=1e-6)
    assert torch.allclose(l2.q[0], want_l2, atol=1e-6)
    assert not torch.allclose(l1.q[0], l2.q[0])
    s1 = (residual @ l1.q[0]).abs().mean()
    s2 = (residual @ l2.q[0]).abs().mean()
    assert s1 > s2
    assert torch.allclose(l1.mean_abs[0], s1, atol=1e-6)
    # The original covariance path retains the pre-update moment semantics.
    assert l2.mean_abs[0] == pytest.approx(2.0)


def test_fixed_direction_optimal_radius_follows_absolute_moment():
    residual = torch.tensor([[-2., 0.], [2., 0.], [0., 1.]])
    q = torch.tensor([1., 0.])
    z = residual @ q
    radius = z.abs().mean()
    objective = lambda rho: ((residual * residual).sum(1)
                             + rho*rho - 2*rho*z.abs()).mean()
    assert objective(radius) <= objective(radius * .5) + 1e-5
    assert objective(radius) <= objective(radius * 1.5) + 1e-5
    expected_gain = radius.square()
    observed_gain = (residual.square().sum(1).mean() - objective(radius))
    assert torch.allclose(expected_gain, observed_gain)


def test_zero_projection_does_not_erase_l1_direction():
    t = tracker("split_gain", d=3)
    before = t.q.clone()
    t.observe(torch.zeros(10, 3), torch.zeros(10, dtype=torch.int64),
              torch.zeros((1, 3)), torch.tensor([10.]))
    assert torch.allclose(t.q, before)


@pytest.mark.parametrize("objective,radius", [
    ("covariance", "rms_scaled"),
    ("covariance", "mean_abs"),
    ("split_gain", "mean_abs"),
    ("split_gain", "rms_scaled"),
])
def test_end_to_end_determinism_and_finite_metrics(objective, radius):
    g = torch.Generator().manual_seed(31)
    x = torch.randn(1800, 11, generator=g)
    cfg = SMeansConfig(
        revive_geometry="local_direction",
        local_direction_objective=objective, local_radius_policy=radius,
        local_observe_every=4,
    )
    a = smeans(x, 48, nepochs=2.0, seed=5, device="cpu",
               config=cfg, return_info=True)
    b = smeans(x, 48, nepochs=2.0, seed=5, device="cpu",
               config=cfg, return_info=True)
    assert torch.equal(a.centroids, b.centroids)
    assert a.mse is not None and np.isfinite(a.mse)
    assert a.total_splits == b.total_splits


def test_default_and_explicit_covariance_are_identical():
    x = torch.randn(1100, 8, generator=torch.Generator().manual_seed(14))
    a = smeans(x, 32, nepochs=2.0, seed=12, device="cpu",
               config=SMeansConfig(revive_geometry="local_direction"))
    b = smeans(x, 32, nepochs=2.0, seed=12, device="cpu",
               config=SMeansConfig(revive_geometry="local_direction",
                                   local_direction_objective="covariance"))
    assert torch.equal(a, b)


def test_sklearn_exposes_new_objective():
    X = np.random.default_rng(3).normal(size=(600, 7)).astype("float32")
    est = SMeans(n_clusters=18, nepochs=2.0, random_state=5,
                 revive_geometry="local_direction",
                 local_direction_objective="split_gain",
                 local_radius_policy="mean_abs").fit(X)
    assert est.get_params()["local_direction_objective"] == "split_gain"
    assert np.isfinite(est.inertia_)


def test_invalid_objective_fails_explicitly():
    with pytest.raises(ValueError, match="direction_objective"):
        tracker("invented")
    x = torch.randn(100, 8)
    with pytest.raises(ValueError, match="local_direction_objective"):
        smeans(x, 8, device="cpu",
               config=SMeansConfig(local_direction_objective="invented"))
