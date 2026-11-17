"""Train-only, trust-region calibrated power-diagram routing for FAISS IVF-Flat.

PRIOR ART: Additively weighted Voronoi/power diagrams and capacity-balanced
partitions are established methods (e.g. Lévy 2016; Zheng et al. 2021).
This code does not claim to invent them. We explore a *train-only*
routing-calibration layer for fixed centroids from sven-means.

For given centroids c_j and prices p_j, route x to
  argmin_j ||x-c_j||^2 + p_j.
An exact L2 nearest-centroid search can implement this by adding one
coordinate: append zero to data/queries and sqrt(p_j - min(p)) to centers.
All *data-to-query* L2 distances remain unchanged, so IVF-Flat reranking
is exact over scanned candidates.

This is a small-CPU proof of concept with O(nk) distance scratch,
NOT a distributed-billion-vector algorithm. Prices are constrained by a
maximum additional training-assignment SSE (default 1%). The constraint
does not prove improved search recall on held-out queries.
"""
from __future__ import annotations

import time

import faiss
import numpy as np


def pairwise_sqdist(x, centers):
    x = np.ascontiguousarray(x, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("expected matching [n,d] data and [k,d] centroids")
    if not len(x) or not len(centers):
        raise ValueError("nonempty data and centroid arrays required")
    # Center around a common anchor to reduce cancellation under shifts.
    anchor = centers[:1]
    xc, cc = x - anchor, centers - anchor
    values = (
        (xc*xc).sum(axis=1, keepdims=True) +
        (cc*cc).sum(axis=1)[None, :] - 2*(xc @ cc.T)
    )
    return np.maximum(values, 0.0)


def _objective(dists, prices):
    assignment = np.argmin(dists + prices[None, :], axis=1)
    n, k = dists.shape
    occupancy = np.bincount(assignment, minlength=k)
    unweighted_mse = float(dists[np.arange(n), assignment].mean())
    occupancy_cv = float(occupancy.std() / max(1e-9, occupancy.mean()))
    return assignment, occupancy, unweighted_mse, occupancy_cv


def calibrate_power_prices(
    data,
    centers,
    *,
    max_distortion_increase: float = 0.01,
    iterations: int = 80,
    gain: float = 0.5,
) -> dict:
    """Try to reduce occupancy variation without exceeding training SSE budget.

    This is a bounded, deterministic dual-price heuristic. It is NOT a
    guaranteed optimizer of the capacitated Voronoi objective. The
    constraint is tested using *unweighted* train-to-assigned-centroid SSE.
    Stores only the best feasible set of prices found.
    """
    if (not 0.0 <= max_distortion_increase <= 0.25
            or iterations < 1 or not 0.0 < gain <= 10):
        raise ValueError("invalid distortion budget, iterations or gain")
    started = time.perf_counter()
    distances = pairwise_sqdist(data, centers)
    n, k = distances.shape
    zeros = np.zeros(k, dtype=np.float64)
    _, counts, mse, cv = _objective(distances, zeros)
    if k == 1:
        return {
            "prices": zeros.astype(np.float32), "baseline_mse": mse,
            "assigned_mse": mse, "baseline_cv": cv, "new_cv": cv,
            "calibration_ms": 1000*(time.perf_counter()-started),
            "iterations": 0,
        }

    best_prices = zeros.copy()
    best_mse, best_cv = mse, cv
    current = zeros.copy()
    step = float(np.median(
        np.partition(distances, 1, axis=1)[:, 1]
        - np.partition(distances, 1, axis=1)[:, 0]
    )) * gain
    step = max(step, 1e-7)
    accepted_steps = 0
    for _ in range(iterations):
        _, occ, _, _ = _objective(distances, current)
        # Positive price is a penalty: overfull cells contract, underfull
        # cells expand. The log ratio avoids spending an entire step on
        # large cells when the occupancy is already near target.
        target = n/k
        gradient = np.log((occ + 0.5) / (target + 0.5))
        proposal = current + step * gradient
        proposal -= proposal.mean()
        _, _, new_mse, new_cv = _objective(distances, proposal)
        if new_mse <= mse*(1.0+max_distortion_increase)+1e-8:
            current = proposal
            accepted_steps += 1
            if new_cv + 1e-10 < best_cv:
                best_cv = new_cv
                best_mse = new_mse
                best_prices = proposal.copy()
        else:
            # Backtrack from current state rather than silently accepting
            # a solution that violates the registered distortion budget.
            step *= 0.5
            if step < 1e-10:
                break
    # All prices can be shifted by a common constant. The extra coordinate
    # uses the nonnegative shifted values; no change to induced assignments.
    prices = best_prices - best_prices.min()
    if not np.isfinite(prices).all():
        raise AssertionError("invalid calibrated prices")
    return {
        "prices": prices.astype(np.float32),
        "baseline_mse": mse, "assigned_mse": best_mse,
        "baseline_cv": cv, "new_cv": best_cv,
        "calibration_ms": 1000*(time.perf_counter()-started),
        "iterations": accepted_steps,
    }


def power_embedding(data, centers, prices):
    """Embed additively weighted Voronoi routing as ordinary L2 geometry."""
    x = np.ascontiguousarray(data, dtype=np.float32)
    c = np.ascontiguousarray(centers, dtype=np.float32)
    prices = np.asarray(prices, dtype=np.float32).reshape(-1)
    if prices.shape != (len(c),) or not np.isfinite(prices).all():
        raise ValueError("prices must be finite and have one value per centroid")
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError("data and centroid dimensions must match")
    bias = np.maximum(prices - prices.min(), 0)
    x_aug = np.ascontiguousarray(
        np.column_stack([x, np.zeros(len(x), dtype=np.float32)]),
        dtype=np.float32,
    )
    c_aug = np.ascontiguousarray(
        np.column_stack([c, np.sqrt(bias)]), dtype=np.float32,
    )
    return x_aug, c_aug


def build_power_ivf(train, centers, prices):
    """FAISS IVF-Flat with unchanged stored neighbor distances (extra 0)."""
    data, biased_centers = power_embedding(train, centers, prices)
    quantizer = faiss.IndexFlatL2(data.shape[1])
    index = faiss.IndexIVFFlat(
        quantizer, data.shape[1], len(centers), faiss.METRIC_L2
    )
    index.quantizer.add(biased_centers)
    index.is_trained = True
    index.add_with_ids(data, np.arange(len(train), dtype=np.int64))
    if index.ntotal != len(train):
        raise AssertionError("Faiss dropped training vectors")
    return index
