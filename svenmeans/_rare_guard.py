"""Attributed rare-cluster protection for optional sven-means revival.

Source (not sven-means' own invention):
Eyal Weiss (2026), "Small Is Not Dead: Distance-Aware Cluster Revival for
e-means", https://arxiv.org/abs/2611.18238.

Low-usage centers are removed only when their squared distance to the nearest
other center is <= tau*running-MSE, or when abandoned for two epoch lengths.
We adapt this published rule to the sven-means update, with either global
or local-direction splitting, as an independently measured combination.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def nearest_other_center_sqdist(
    centroids: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    row_tile: int = 16,
    center_tile: int = 128,
) -> torch.Tensor:
    """Exact nearest-other squared L2, bounded tile scratch.

    Cost O(u*k*d) for u candidates. Scratch O(row_tile*center_tile*d).
    This CPU proof of concept is NOT a billion-centroid implementation.
    """
    if centroids.ndim != 2 or candidate_indices.ndim != 1:
        raise ValueError("expected centroids [k,d] and indices [u]")
    if row_tile < 1 or center_tile < 1:
        raise ValueError("tile sizes must be positive")
    k = len(centroids)
    if len(candidate_indices) == 0:
        return centroids.new_empty((0,))
    if k < 2:
        return centroids.new_full((len(candidate_indices),), float("inf"))
    if bool((candidate_indices.min() < 0) | (candidate_indices.max() >= k)):
        raise ValueError("candidate indices must be within [0,k)")
    nearest_dist = centroids.new_full((len(candidate_indices),), float("inf"))
    for a in range(0, len(candidate_indices), row_tile):
        ids = candidate_indices[a:a+row_tile]
        queries = centroids[ids]
        best = centroids.new_full((len(ids),), float("inf"))
        for b in range(0, k, center_tile):
            stop = min(k, b + center_tile)
            # Use direct differences to avoid norm cancellation under shifts.
            residual = queries[:, None, :] - centroids[None, b:stop, :]
            dist = residual.square().sum(dim=-1)
            cols = torch.arange(b, stop, device=centroids.device)
            dist.masked_fill_(ids[:, None] == cols[None, :], float("inf"))
            best = torch.minimum(best, dist.min(dim=1).values)
        nearest_dist[a:a+len(ids)] = best
    return nearest_dist


@torch.no_grad()
def distance_aware_removable(
    centroids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    mse: float,
    near_factor: float,
    last_seen: torch.Tensor,
    step: int,
    abandon_steps: int,
) -> torch.Tensor:
    """Only redundant or sufficiently long-abandoned low-count centers."""
    k = len(centroids)
    if candidates.shape != (k,) or last_seen.shape != (k,):
        raise ValueError("candidate and last-seen vectors must have length k")
    if candidates.dtype != torch.bool:
        raise ValueError("candidate mask must be boolean")
    if (candidates.device != centroids.device
            or last_seen.device != centroids.device):
        raise ValueError("guard tensors must reside on one device")
    if not math.isfinite(mse) or mse < 0:
        raise ValueError("mse must be finite and nonnegative")
    if (not math.isfinite(near_factor) or near_factor <= 0
            or abandon_steps < 1):
        raise ValueError("near_factor > 0 and abandon_steps >= 1 required")
    removable = torch.zeros_like(candidates)
    ids = torch.nonzero(candidates, as_tuple=True)[0]
    if ids.numel() == 0:
        return removable
    nearest_sq = nearest_other_center_sqdist(centroids, ids)
    abandoned = (step - last_seen[ids]) >= abandon_steps
    removable[ids] = (nearest_sq <= near_factor * mse) | abandoned
    return removable
