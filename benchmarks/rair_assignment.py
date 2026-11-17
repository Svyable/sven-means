"""SIGMOD 2026 RAIRS AIR assignment reference *formula* for IVF-Flat.

This is NOT an original algorithm or a reproduction of the authors'
Shared-Cell Enhanced IVF Lists (SEIL). The Amplified Inverse Residual
(AIR) scoring rule is by Zehai Yang & Shimin Chen, SIGMOD 2026,
https://arxiv.org/abs/2601.07183. Their source is
https://github.com/schencoding/rairs (C++/FAISS).

Let r=c_primary-x and r'=c_candidate-x, and choose a second list by
  ||r'||² + 0.5 * r·r'
within the ten nearest centers. SRAIR excludes the primary; RAIR
allows it, in which case the original point is NOT duplicated.

The selected row set can separately be limited to the 25%/50% quotas
predeclared in the preceding SOAR experiment. This row preselection is
a *local research extension* to the original RAIR formulation and does
not reproduce its native full-data assignment policy.
"""
from __future__ import annotations

import numpy as np

from benchmarks.orthogonal_spill import nearest_centroids


def air_secondary(
    data: np.ndarray,
    centroids: np.ndarray,
    *,
    air_lambda: float = 0.5,
    shortlist: int = 10,
) -> dict:
    """Strict/non-strict published AIR scores, with fixed shortlist.

    Returns:
      primary - original closest center
      strict - closest AIR-scored alternative excluding primary (SRAIR)
      optional - minimizes AIR over all candidates (RAIR);
                 if equal to primary, skip replica
      full_scores - score for each shortlisted center, useful for invariants
    """
    if not np.isfinite(air_lambda) or air_lambda < 0:
        raise ValueError("AIR lambda must be finite and nonnegative")
    if shortlist < 2:
        raise ValueError("shortlist must be at least two centers")
    x = np.ascontiguousarray(data, dtype=np.float32)
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    _, candidate_ids = nearest_centroids(x, c, shortlist=shortlist)
    primary = candidate_ids[:, 0]
    # Accumulate scoring in FP64; residual subtraction directly is stable
    # for input arrays with large shared offsets.
    residual = c[candidate_ids].astype(np.float64) - x.astype(np.float64)[:, None, :]
    primary_residual = residual[:, 0, :]
    distances = np.einsum("ncd,ncd->nc", residual, residual)
    correlation = np.einsum("nd,ncd->nc", primary_residual, residual)
    losses = distances + air_lambda * correlation
    strict = candidate_ids[
        np.arange(len(x)), 1 + losses[:, 1:].argmin(axis=1)
    ]
    optional = candidate_ids[
        np.arange(len(x)), losses.argmin(axis=1)
    ]
    if np.any(strict == primary):
        raise AssertionError("strict SRAIR must not reassign to same center")
    return {
        "primary": np.asarray(primary, dtype=np.int64),
        "strict": np.asarray(strict, dtype=np.int64),
        "optional": np.asarray(optional, dtype=np.int64),
        "shortlist_centers": np.asarray(candidate_ids, dtype=np.int64),
        "air_scores": np.asarray(losses, dtype=np.float64),
        "abstention_fraction": float(np.mean(optional == primary)),
    }
