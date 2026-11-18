"""Experimental streaming proxy-grid archive for *expensive-evaluation selection*.

This is NOT a new Pareto archive or a state-of-the-art optimization claim:
epsilon-grid archiving, power-law diversity, and Pareto ranking have prior art.

The candidate selector sees ONLY cheap architecture features and proxy
predictions trained from explicitly paid-for calibration labels. Oracle
accuracy / latency / energy of unselected architectures are hidden.

A grid cell representative has a conditional covering property:
if (i) the archive retains one candidate for EVERY occupied proxy cell,
(ii) each normalized proxy error is bounded by epsilon uniformly, and
(iii) a cell has side width h, then any true normalized objective vector
is within h+2*epsilon (L-infinity) of that cell's representative.
When memory causes cell eviction or the proxy bound is not certified,
NO such guarantee applies. No statistical confidence claim is made.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")


def topology_features(topologies, sizes):
    """Free features: six edge operators + converted INT8 flatbuffer size.

    No actual power, latency, energy, post-deployment ROM or accuracy enters
    the feature calculation. The converted model's size must have been known
    *before* expensive physical-device profiling.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    if sizes.ndim != 1 or len(sizes) != len(topologies):
        raise ValueError("one converted model size per topology required")
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("converted model sizes must be positive finite")
    x = np.zeros((len(topologies), 6 * len(OPS) + 1), dtype=np.float64)
    families = np.empty(len(topologies), dtype=np.int32)
    for i, s in enumerate(topologies):
        ops = re.findall(r"([A-Za-z0-9_]+)~\d+", s)
        if len(ops) != 6 or any(op not in OPS for op in ops):
            raise ValueError(f"unexpected NATS-Bench cell topology at row {i}: {s!r}")
        for j, op in enumerate(ops):
            x[i, j * len(OPS) + OPS.index(op)] = 1.0
        families[i] = OPS.index(ops[0])
    x[:, -1] = np.log1p(sizes)
    return x, families


def fit_linear_proxy(features, labels, calibration_ids, ridge=10.0):
    """Three-output ridge, trained only on paid calibration evaluations.

    Output columns: log latency_seconds, log energy_mJ, CIFAR10 accuracy.
    A small fixed regularizer is used without tuning to held-out truth.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    ids = np.asarray(calibration_ids, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x), 3) or len(ids) < 10:
        raise ValueError("bad dimensions or insufficient calibration observations")
    if len(set(ids.tolist())) != len(ids) or min(ids) < 0 or max(ids) >= len(x):
        raise ValueError("calibration indices must be distinct and in range")
    if not np.isfinite(x).all() or not np.isfinite(y[ids]).all():
        raise ValueError("finite calibration observations required")
    mean = x[ids].mean(0)
    scale = x[ids].std(0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    a = (x[ids] - mean) / scale
    a = np.column_stack((np.ones(len(ids)), a))
    beta = np.linalg.solve(
        a.T @ a + ridge * np.diag(np.r_[0., np.ones(a.shape[1] - 1)]),
        a.T @ y[ids],
    )
    all_x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    proxy = all_x @ beta
    residual = y[ids] - a @ beta
    return proxy, {
        "calibration_rmse": np.sqrt(np.mean(residual * residual, axis=0)).tolist(),
        "ridge": float(ridge),
        "num_calibration": len(ids),
    }


def normalized_proxy_costs(proxy, sizes, allowed_ids):
    """Cheap normalized costs: log latency, log energy, negative accuracy.

    The conversion-size field is used as a *known feasibility precheck*,
    not an objective; 3D oracle metrics remain hidden until scoring.
    """
    proxy = np.asarray(proxy, dtype=np.float64)
    allowed = np.asarray(allowed_ids, dtype=np.int64)
    if proxy.ndim != 2 or proxy.shape[1] != 3 or not len(allowed):
        raise ValueError("proxy must have [n,3] and an allowed candidate pool")
    cost = np.column_stack((proxy[:, 0], proxy[:, 1], -proxy[:, 2]))
    lo = np.quantile(cost[allowed], 0.01, axis=0)
    hi = np.quantile(cost[allowed], 0.99, axis=0)
    return np.clip((cost - lo) / np.maximum(hi - lo, 1e-8), -0.5, 1.5)


def _rank_scores(proxy, normalized, sizes, ids, family, quality_threshold=90.0):
    """Only proxy estimates and converted model size can rank candidates."""
    ids = np.asarray(ids, dtype=np.int64)
    family = np.asarray(family, dtype=np.int64)
    freq = np.bincount(family[ids], minlength=len(OPS))
    rarity = 1.0 / np.sqrt(np.maximum(1, freq))
    predicted_quality = proxy[ids, 2]
    optimistic = predicted_quality >= quality_threshold - 2.0
    known_flash_ok = np.asarray(sizes)[ids] <= 746.0
    # Three-objective compromise and a modest rare-topology-family bonus.
    utility = -(
        0.36 * normalized[ids, 0] + 0.36 * normalized[ids, 1]
        + 0.28 * normalized[ids, 2]
    ) + 0.22 * rarity[family[ids]]
    utility += 0.8 * optimistic.astype(np.float64)
    utility -= 4.0 * (~known_flash_ok).astype(np.float64)
    return utility


def greedy_feature_diversity(features, calibration_ids, candidate_ids, budget, seed, mode):
    """Classical k-center farthest-first or D^2 seeding; no true metrics."""
    if mode not in ("farthest", "kmeans++"):
        raise ValueError("unknown feature-diversity policy")
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if budget < 0 or budget > len(ids):
        raise ValueError("invalid query budget")
    if not budget:
        return []
    x = np.asarray(features, dtype=np.float64)
    # Normalize each feature by standard deviation on cheap public features.
    center = x.mean(0)
    spread = np.where(x.std(0) > 1e-8, x.std(0), 1)
    z = ((x-center) / spread).astype(np.float32)
    candidate = z[ids]
    observed = z[np.asarray(calibration_ids, dtype=np.int64)]
    if len(observed):
        d2 = np.full(len(ids), np.inf, dtype=np.float32)
        for row in observed:
            d2 = np.minimum(d2, ((candidate-row)**2).sum(axis=1))
    else:
        d2 = np.ones(len(ids), dtype=np.float32)
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(ids), dtype=bool)
    output = []
    for _ in range(budget):
        if mode == "farthest":
            chosen = int(np.argmax(np.where(remaining, d2, -1)))
        else:
            weights = np.where(remaining, np.maximum(d2, 0), 0)
            total = weights.sum(dtype=np.float64)
            if not np.isfinite(total) or total <= 1e-12:
                chosen = int(rng.choice(np.flatnonzero(remaining)))
            else:
                chosen = int(rng.choice(len(ids), p=weights/total))
        output.append(int(ids[chosen]))
        remaining[chosen] = False
        d2 = np.minimum(d2, ((candidate-candidate[chosen])**2).sum(axis=1))
    return output


def stratified_uniform(candidate_ids, family, budget, seed):
    """Round-robin uniform random family sampling."""
    ids = np.asarray(candidate_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = {}
    for f in sorted(set(np.asarray(family)[ids].tolist())):
        group = ids[np.asarray(family)[ids] == f]
        groups[f] = list(rng.permutation(group).astype(int))
    result = []
    while len(result) < budget:
        moved = False
        for f in groups:
            if groups[f] and len(result) < budget:
                result.append(groups[f].pop())
                moved = True
        if not moved:
            break
    return result


def streaming_proxy_grid(proxy, normalized, sizes, candidate_ids, family, budget,
                         seed, cell_width=0.15):
    """One-pass capacity-B grid archive with B-element spillover.

    The archive itself holds at most B representatives, plus B spillover
    candidates used to spend the complete evaluation budget. It does not
    store the full incoming stream, though proxy arrays are precomputed
    cheaply over the full enumerable search space.

    This is an empirical cell-priority heuristic, *not* an optimal or
    certified constrained-Pareto selector.
    """
    if budget < 0 or not 0 < cell_width <= 1.0:
        raise ValueError("invalid capacity or grid side")
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if budget > len(ids):
        raise ValueError("budget exceeds candidate count")
    if budget == 0:
        return []
    rng = np.random.default_rng(seed)
    scores = _rank_scores(proxy, normalized, sizes, ids, family)
    order = rng.permutation(len(ids))
    cells = {}
    overflow = {}
    archived = set()
    def spill(identifier, score):
        if identifier in archived:
            return
        if identifier in overflow:
            overflow[identifier] = max(score, overflow[identifier])
        elif len(overflow) < budget:
            overflow[identifier] = score
        else:
            smallest = min(overflow, key=lambda j: (overflow[j], -j))
            if score > overflow[smallest]:
                del overflow[smallest]
                overflow[identifier] = score

    for idx in order:
        i = int(ids[idx])
        score = float(scores[idx])
        cell = (int(family[i]),) + tuple(
            np.floor(normalized[i]/cell_width).astype(int).tolist()
        )
        if cell in cells:
            old_id, old_score = cells[cell]
            if score > old_score:
                archived.discard(old_id)
                spill(old_id, old_score)
                overflow.pop(i, None)
                cells[cell] = (i, score)
                archived.add(i)
            else:
                spill(i, score)
        elif len(cells) < budget:
            overflow.pop(i, None)
            cells[cell] = (i, score)
            archived.add(i)
        else:
            evict = min(cells, key=lambda j: (cells[j][1], cells[j][0]))
            old_id, old_score = cells[evict]
            if score > old_score:
                del cells[evict]
                archived.discard(old_id)
                spill(old_id, old_score)
                overflow.pop(i, None)
                cells[cell] = (i, score)
                archived.add(i)
            else:
                spill(i, score)
    winners = [v[0] for v in cells.values()]
    # The archive's diversity representatives retain their budget slots.
    # Spend any leftover capacity on the best non-archive spillover.
    remaining = budget - len(winners)
    more = sorted(
        ((j, s) for j, s in overflow.items() if j not in archived),
        key=lambda item: (-item[1], item[0]),
    )[:remaining]
    winners.extend(j for j, _ in more)
    if len(winners) != budget or len(set(winners)) != len(winners):
        raise AssertionError("streaming archive failed to spend unique oracle budget")
    return winners


def choose_candidates(policy, features, proxy, sizes, family,
                      calibration_ids, candidates, budget, seed):
    """All selection policies have the same cheap information."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(np.intersect1d(calibration_ids, candidates)):
        raise ValueError("calibration and unqueried candidate pools overlap")
    if budget < 0 or budget > len(candidates):
        raise ValueError("invalid oracle budget")
    rng = np.random.default_rng(seed)
    if policy == "random":
        return rng.permutation(candidates)[:budget].astype(int).tolist()
    if policy == "stratified":
        return stratified_uniform(candidates, family, budget, seed)
    if policy in ("kmeans++", "farthest"):
        return greedy_feature_diversity(
            features, calibration_ids, candidates, budget, seed, policy
        )
    norm = normalized_proxy_costs(proxy, sizes, candidates)
    if policy == "proxy_rank":
        score = _rank_scores(proxy, norm, sizes, candidates, family)
        # Stable tie-break and deterministic ranking.
        return candidates[np.lexsort((candidates, -score))[:budget]].astype(int).tolist()
    if policy == "proxy_grid":
        return streaming_proxy_grid(
            proxy, norm, sizes, candidates, family, budget, seed
        )
    raise ValueError(f"Unknown selection policy {policy}")
