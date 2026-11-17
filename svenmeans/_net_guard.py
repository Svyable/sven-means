"""Independent holdout net-SSE gate for experimental centroid recycling.

This is a *diagnostic*, not a statistically calibrated confidence procedure.
The validation rows are reserved BEFORE centroid seeding and are never
included in EMA, donor selection, or training assignments. Reuse of the
same holdout across decisions can adaptively overfit it. A normal
lower-confidence bound is used as a heuristic, not a coverage guarantee.

The evaluator is exact for **all** k centers on the reserved vectors:
when one old source and one old target are replaced by two children,
the nearest unaffected center is among the old three closest centers.
This removes the O(k*d) competitor search *per proposal*; one tiled
top-3 pass each time current centers change is still O(n_val*k*d).
"""
from __future__ import annotations

import math
import torch


def reserve_validation(
    data: torch.Tensor,
    seed: int,
    fraction: float,
    *,
    min_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic disjoint train/holdout split; preserves data dtype.

    An opt-in run trains ONLY on returned train; callers benchmarking
    unguarded controls must provide that exact train subset as input.
    """
    if data.ndim != 2 or not 0.0 < fraction <= 0.4 or min_rows < 2:
        raise ValueError("net validation requires 2D data and 0<fraction<=0.4")
    n = len(data)
    size = max(min_rows, math.ceil(n * fraction))
    if size >= n:
        raise ValueError("not enough input points to reserve net validation")
    generator = torch.Generator(device=data.device)
    generator.manual_seed((int(seed) ^ 0x6E3779B1) % (2**63 - 1))
    perm = torch.randperm(n, generator=generator, device=data.device)
    return data[perm[size:]].contiguous(), data[perm[:size]].contiguous()


@torch.no_grad()
def exact_top_three(
    data: torch.Tensor,
    centers: torch.Tensor,
    *,
    row_tile: int = 32,
    centroid_tile: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted squared distances/indices of top-three full L2 centers.

    Direct subtraction is robust under common large coordinate shifts.
    Scratch O(row_tile*centroid_tile*d), outputs O(n_val*3).
    """
    if data.ndim != 2 or centers.ndim != 2 or data.shape[1] != centers.shape[1]:
        raise ValueError("data and centers must have shape [n,d] and [k,d]")
    if len(centers) < 2 or row_tile < 1 or centroid_tile < 1:
        raise ValueError("two centroids and positive tile sizes required")
    n, k = int(data.shape[0]), int(centers.shape[0])
    out_dist = data.new_full((n, 3), float("inf"))
    out_ids = torch.full((n, 3), -1, dtype=torch.int64, device=data.device)
    for a in range(0, n, row_tile):
        stop = min(n, a + row_tile)
        x = data[a:stop]
        best = x.new_full((len(x), 3), float("inf"))
        ids = torch.full((len(x), 3), -1, dtype=torch.int64, device=x.device)
        for b in range(0, k, centroid_tile):
            end = min(k, b + centroid_tile)
            block = (x[:, None, :] - centers[None, b:end, :]).square().sum(dim=2)
            loc = torch.arange(b, end, dtype=torch.int64, device=x.device)
            loc = loc[None, :].expand(len(x), -1)
            combined = torch.cat((best, block), dim=1)
            combined_ids = torch.cat((ids, loc), dim=1)
            best, top = combined.topk(3, largest=False, sorted=True)
            ids = combined_ids.gather(1, top)
        out_dist[a:stop], out_ids[a:stop] = best, ids
    return out_dist, out_ids


@torch.no_grad()
def net_split_gain_samples(
    validation: torch.Tensor,
    top_dist: torch.Tensor,
    top_ids: torch.Tensor,
    source: int,
    target: int,
    parent: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Exact old-best minus new-best squared L2, each holdout row.

    The proposed replacement deletes BOTH old source and target and
    installs parent-delta and parent+delta. Keeping only old top 2 is
    insufficient if those happen to be the two deleted centers.
    """
    if source == target or source < 0 or target < 0:
        raise ValueError("must replace distinct nonnegative centers")
    if top_dist.shape != (len(validation), 3) or top_ids.shape != top_dist.shape:
        raise ValueError("top3 shape mismatch")
    surviving = (top_ids != source) & (top_ids != target) & (top_ids >= 0)
    other = torch.where(surviving, top_dist, float("inf")).min(dim=1).values
    plus = (validation - (parent + delta)).square().sum(dim=1)
    minus = (validation - (parent - delta)).square().sum(dim=1)
    new = torch.minimum(other, torch.minimum(plus, minus))
    return top_dist[:, 0] - new


@torch.no_grad()
def lower_confidence_gate(
    individual_gains: torch.Tensor,
    z: float = 1.645,
    gain_floor: float = 0.,
) -> tuple[bool, float, float]:
    """Heuristic one-sided normal LCB on mean net reduction.

    NOT a formal frequentist guarantee when repeatedly applied to the
    same holdout, or under non-normal, dependent or heavy-tailed gains.
    """
    if individual_gains.ndim != 1 or len(individual_gains) < 2:
        raise ValueError("at least two row-level gains required")
    if z < 0 or gain_floor < 0 or not (math.isfinite(z) and math.isfinite(gain_floor)):
        raise ValueError("z and gain_floor must be finite and nonnegative")
    v = individual_gains.to(torch.float64)
    if not bool(torch.isfinite(v).all()):
        return False, float("nan"), float("-inf")
    mean = float(v.mean().item())
    se = float(v.std(unbiased=True).item() / math.sqrt(len(v)))
    low = mean - z * se
    return low > gain_floor, mean, low
