"""Cost-normalized coarse IVF candidate-recall diagnostics.

The key confounder: nprobe fixes number of lists, NOT number of vectors
examined. Two partitions can have radically different list occupancy.
Report both mean scanned fraction and candidate recall at each nprobe.

At an expected scan budget between two operating points, a randomized
probe-count policy can mix their results. Linear interpolation is *exact
in expectation* for that mixture, not a claim of attainable deterministic
per-query constant work. Out-of-range budgets are unavailable, not extrapolated.
"""
from __future__ import annotations

import torch

from svenmeans import assign_clusters


@torch.no_grad()
def scan_curve(
    train: torch.Tensor,
    test: torch.Tensor,
    centroids: torch.Tensor,
    *,
    probes=(1, 2, 4, 8, 16, 32, 64),
    true_neighbors: int = 10,
    query_count: int = 96,
) -> list[dict]:
    if train.ndim != 2 or test.ndim != 2 or centroids.ndim != 2:
        raise ValueError("expected 2D train, test and centroid tensors")
    if train.shape[1] != test.shape[1] or train.shape[1] != centroids.shape[1]:
        raise ValueError("embedding dimensions must match")
    if len(train) < 1 or len(test) < 1 or len(centroids) < 1:
        raise ValueError("nonempty train, test and centroids required")
    if true_neighbors < 1 or query_count < 1 or not probes or min(probes) < 1:
        raise ValueError("positive query, neighbor and probe counts required")

    q = test[:query_count]
    closest_train = torch.cdist(q, train).topk(
        min(true_neighbors, len(train)), largest=False, dim=1
    ).indices
    labels, _ = assign_clusters(train, centroids)
    list_sizes = torch.bincount(labels, minlength=len(centroids))
    bins_of_neighbors = labels[closest_train]
    ranks = torch.cdist(q, centroids).argsort(dim=1)

    results = []
    for p in sorted({min(int(p), len(centroids)) for p in probes}):
        visited = ranks[:, :p]
        scan_counts = list_sizes[visited].sum(dim=1)
        hits = (
            bins_of_neighbors[:, :, None] == visited[:, None, :]
        ).any(dim=2)
        fraction = scan_counts.float() / len(train)
        results.append({
            "nprobe": p,
            "mean_scan_fraction": float(fraction.mean()),
            "median_scan_fraction": float(fraction.median()),
            "max_scan_fraction": float(fraction.max()),
            "mean_candidate_recall_at_10": float(hits.float().mean()),
            "mean_scanned_vectors": float(scan_counts.float().mean()),
        })
    return results


def interpolated_recall_at_budget(curve: list[dict], budget: float) -> float | None:
    """Expected recall from randomized mixing of two adjacent probe counts.

    No extrapolation: None means *this partition cannot use whole-list probing
    to achieve this small expected scan budget with provided probe choices*.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0,1]")
    if not curve:
        return None
    points = sorted(
        (row["mean_scan_fraction"], row["mean_candidate_recall_at_10"])
        for row in curve
    )
    if budget < points[0][0] - 1e-7 or budget > points[-1][0] + 1e-7:
        return None
    for i, (cost, recall) in enumerate(points):
        if abs(budget - cost) <= 1e-7:
            return recall
        if i == 0 or cost < budget:
            continue
        previous_cost, previous_recall = points[i - 1]
        if cost - previous_cost <= 1e-12:
            return max(recall, previous_recall)
        prob_upper = (budget - previous_cost) / (cost - previous_cost)
        return previous_recall * (1 - prob_upper) + recall * prob_upper
    return points[-1][1] if abs(budget - points[-1][0]) <= 1e-7 else None
