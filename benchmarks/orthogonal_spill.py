"""An experimental FAISS IVF-Flat *indexing* layer inspired by Google's SOAR.

PRIOR ART: Sun et al., SOAR: Improved Indexing for Approximate Nearest
Neighbor Search, NeurIPS 2023. Orthogonality-amplified spilling is theirs.
Our *budgeted* comparison tests how much selectively duplicated inverted
list entries improve actual L2 recall and search work after Sven training.

This is not a modification to k-means training or the k-means loss: the
coarse centroids are held fixed. A secondary routing assignment is added
for a chosen fraction of training vectors. The original nearest-center
(primary) assignment is retained. Search results can contain duplicate
vector IDs and are deduplicated before calculating recall@10.

Use exact FAISS IndexIVFFlat add_core with a prescribed list number and
test it against exact IndexFlatL2 at nprobe=k. This is the ACTUAL FAISS
search and its actual index data, not a theoretical candidate oracle.

An L2-specific orthogonal residual score selects secondary center j:
  ||x - c_j||^2 + lambda * ((x-c_j) dot (x-c_primary))^2
                              / (||x-c_primary||^2 + eps).
The score is inspired by SOAR but is *not* claimed to reproduce all of
Google's tuned ScaNN machinery. We preregister lambda=2 and shortlist=8.

CPU-only prototype. Additional list entries increase memory and index
build cost. Do not claim improved k-means MSE, GPU performance or an
independent new indexing algorithm from this baseline adaptation.
"""
from __future__ import annotations

import time

import faiss
import numpy as np

SOAR_LAMBDA = 2.0
SHORTLIST = 8
NEIGHBORS = 10


def nearest_centroids(
    data: np.ndarray, centers: np.ndarray, shortlist: int = SHORTLIST
) -> tuple[np.ndarray, np.ndarray]:
    data = np.ascontiguousarray(data, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    if data.ndim != 2 or centers.ndim != 2 or data.shape[1] != centers.shape[1]:
        raise ValueError("expected data[n,d] and centroids[k,d] with matching d")
    if len(data) < 1 or len(centers) < 2 or shortlist < 2:
        raise ValueError("nonempty data, k>=2 and shortlist>=2 required")
    if not np.isfinite(data).all() or not np.isfinite(centers).all():
        raise ValueError("data and centers must be finite")
    coarse = faiss.IndexFlatL2(data.shape[1])
    coarse.add(centers)
    dists, ids = coarse.search(data, min(shortlist, len(centers)))
    return dists, ids


def secondary_candidates(
    data: np.ndarray,
    centers: np.ndarray,
    *,
    weight: float = SOAR_LAMBDA,
    shortlist: int = SHORTLIST,
) -> dict:
    """Precompute a *shared* shortlist for all secondary selection policies.

    The primary assignment is always the Euclidean-nearest center.
    The secondary never equals the primary. The orthogonal loss is
    assessed only among the same L2-nearest shortlist that nearest-
    secondary uses, so the method changes a finite set of choices.
    """
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("orthogonality weight must be finite and nonnegative")
    data = np.ascontiguousarray(data, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    distances, ids = nearest_centroids(data, centers, shortlist)
    primary = ids[:, 0]
    closest_other = ids[:, 1]
    r0 = data.astype(np.float64) - centers[primary].astype(np.float64)
    cands = ids[:, 1:]
    alternatives = data.astype(np.float64)[:, None, :] - centers[cands].astype(np.float64)
    proj = np.einsum("nld,nd->nl", alternatives, r0)
    norm_sq = np.einsum("nd,nd->n", r0, r0)
    # If the primary residual is zero (point exactly at its centroid)
    # the orthogonality term is undefined, so select nearest second.
    penalty = np.where(
        norm_sq[:, None] > 1e-15,
        proj**2 / np.maximum(norm_sq[:, None], 1e-15),
        0.0,
    )
    # FAISS shortlist squared distances are rounded FP32. Compute both
    # terms in FP64 before taking the secondary-penalty argmin.
    square = np.einsum("nld,nld->nl", alternatives, alternatives)
    residual_score = square + weight * penalty
    chosen = cands[np.arange(len(data)), residual_score.argmin(axis=1)]
    if np.any(chosen == primary) or np.any(closest_other == primary):
        raise AssertionError("secondary center must differ from primary")
    # Small margin = close to a Voronoi boundary, a train-only heuristic
    # for which vectors receive additional index storage.
    margin = (distances[:, 1].astype(np.float64)
              - distances[:, 0].astype(np.float64)) / np.maximum(
        distances[:, 0].astype(np.float64), 1e-9
    )
    return {
        "primary": primary.astype(np.int64),
        "nearest": closest_other.astype(np.int64),
        "orthogonal": chosen.astype(np.int64),
        "relative_boundary_margin": margin,
        "shortlist": ids,
        "shortlist_sqdist": distances,
    }


def select_spill_rows(
    relative_boundary_margin: np.ndarray,
    fraction: float,
    *,
    policy: str = "boundary",
    seed: int = 0,
) -> np.ndarray:
    """Fixed additional storage budget; selection uses training rows only."""
    margin = np.asarray(relative_boundary_margin)
    if margin.ndim != 1 or not np.isfinite(margin).all():
        raise ValueError("margin must be a finite vector")
    if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
        raise ValueError("spill fraction must be in [0,1]")
    if policy not in ("boundary", "random"):
        raise ValueError("row selection must be boundary or random")
    n_spill = int(np.floor(len(margin) * fraction + 1e-9))
    if not n_spill:
        return np.empty((0,), dtype=np.int64)
    if policy == "boundary":
        ranked = np.argsort(margin, kind="stable")
    else:
        ranked = np.random.default_rng(seed).permutation(len(margin))
    return np.ascontiguousarray(ranked[:n_spill], dtype=np.int64)


def build_spill_ivf(
    train: np.ndarray,
    centers: np.ndarray,
    *,
    secondary_ids: np.ndarray | None = None,
    selected_rows: np.ndarray | None = None,
) -> tuple[faiss.IndexIVFFlat, float]:
    """Build the actual FAISS IVF list structure with predetermined assignments.

    FAISS stores duplicate IDs in different inverted lists; direct_map is
    intentionally disabled. IDs refer to original row positions.
    """
    start = time.perf_counter()
    train = np.ascontiguousarray(train, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    dists, ids = nearest_centroids(train, centers, shortlist=2)
    del dists
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatL2(train.shape[1]), train.shape[1],
        len(centers), faiss.METRIC_L2,
    )
    index.quantizer.add(centers)
    index.is_trained = True
    all_ids = np.arange(len(train), dtype=np.int64)
    primary = np.ascontiguousarray(ids[:, 0], dtype=np.int64)
    index.add_core(
        len(train), faiss.swig_ptr(train),
        faiss.swig_ptr(all_ids), faiss.swig_ptr(primary)
    )
    if (secondary_ids is None) != (selected_rows is None):
        raise ValueError("secondary_ids and selected_rows must be passed together")
    replicated = 0
    if secondary_ids is not None:
        s = np.ascontiguousarray(selected_rows, dtype=np.int64)
        alternatives = np.ascontiguousarray(secondary_ids, dtype=np.int64)
        if alternatives.shape != (len(s),):
            raise ValueError("one secondary center needed per selected row")
        if len(s):
            if (s.min() < 0 or s.max() >= len(train)
                    or alternatives.min() < 0
                    or alternatives.max() >= len(centers)):
                raise ValueError("secondary rows/centers out of bounds")
            if len(np.unique(s)) != len(s):
                raise ValueError("selected rows must be unique")
            if np.any(alternatives == primary[s]):
                raise ValueError("secondary must not equal primary")
            vectors = np.ascontiguousarray(train[s], dtype=np.float32)
            index.add_core(
                len(s), faiss.swig_ptr(vectors), faiss.swig_ptr(s),
                faiss.swig_ptr(alternatives)
            )
            replicated = len(s)
    if index.ntotal != len(train) + replicated:
        raise AssertionError("FAISS discarded inserted entries")
    return index, 1000.0*(time.perf_counter()-start)


def deduplicate_ranked_ids(retrieved_ids: np.ndarray, k: int = NEIGHBORS):
    """First occurrence is nearest; return at most k unique original IDs.

    Every input is stored at most twice, so searching 2*k FAISS entries
    suffices to expose the first k distinct neighbors whenever at least
    k distinct candidates have been scanned.
    """
    keep = []
    seen = set()
    for idx in retrieved_ids:
        val = int(idx)
        if val >= 0 and val not in seen:
            seen.add(val)
            keep.append(val)
            if len(keep) >= k:
                break
    return keep


def evaluate_spill_index(
    index: faiss.IndexIVFFlat,
    centers: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    exact_sqdist: np.ndarray,
    *,
    probes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    repeats: int = 3,
    original_n: int,
) -> dict:
    """Real IVF search/latency/entries scanned; exact ground truth, no oracle."""
    if repeats < 1 or original_n < 1:
        raise ValueError("repeats and original_n must be positive")
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if (queries.shape[0] != truth.shape[0]
            or truth.shape[1] < NEIGHBORS
            or exact_sqdist.shape != truth.shape):
        raise ValueError("truth ID/distance shapes must match queries and top 10")
    if not np.isfinite(exact_sqdist).all():
        raise ValueError("reference distances must be finite")
    qindex = faiss.IndexFlatL2(centers.shape[1])
    qindex.add(centers)
    lengths = np.array(
        [index.invlists.list_size(j) for j in range(len(centers))],
        dtype=np.int64,
    )
    if lengths.sum() != index.ntotal:
        raise AssertionError("inverted-list counts do not match entries")
    curve = []
    settings = sorted({min(len(centers), int(p)) for p in (*probes, len(centers))})
    for nprobe in settings:
        index.nprobe = nprobe
        _, visited = qindex.search(queries, nprobe)
        scanned = lengths[visited].sum(axis=1)
        # Fairly charge **both** plain and spilled indexes for the same
        # FAISS output heap length of 2*KNN, including dedup overhead.
        index.search(queries, 2*NEIGHBORS)
        elapsed = []
        retrieved = None
        found_dist = None
        for _ in range(repeats):
            started = time.perf_counter()
            found_dist, retrieved = index.search(queries, 2*NEIGHBORS)
            elapsed.append(
                1000.*(time.perf_counter()-started)/len(queries)
            )
        assert found_dist is not None and retrieved is not None

        # In some real vector sets, exactly tied distances can make
        # IndexFlatL2 and IndexIVFFlat return different equally-valid IDs
        # at the boundary of top10. Therefore standard ID-recall may
        # be <1 even when all lists are scanned. Keep it as raw evidence,
        # while using the same explicit distance-tie-aware reference
        # threshold for comparable candidate-recall curves.
        id_hits = []
        radius_hits = []
        exact_order_distances = []
        for i in range(len(queries)):
            reference_radius = float(exact_sqdist[i, NEIGHBORS-1])
            tolerance = 1e-5 + 1e-5*max(1., abs(reference_radius))
            seen = set()
            unique = []
            distances_unique = []
            for value, distance in zip(retrieved[i], found_dist[i]):
                idx = int(value)
                if idx >= 0 and idx not in seen:
                    seen.add(idx)
                    unique.append(idx)
                    distances_unique.append(float(distance))
                    if len(unique) >= NEIGHBORS:
                        break
            id_hits.append(
                len(set(int(x) for x in truth[i, :NEIGHBORS]).intersection(unique))
                / NEIGHBORS
            )
            radius_hits.append(
                sum(distance <= reference_radius+tolerance
                    for distance in distances_unique) / NEIGHBORS
            )
            exact_order_distances.append(distances_unique)
        recall = float(np.mean(id_hits))
        radius_recall = float(np.mean(radius_hits))
        curve.append({
            "nprobe": nprobe,
            "mean_scan_fraction": float(scanned.mean()/original_n),
            "mean_scanned_entries": float(scanned.mean()),
            "max_scanned_entries": int(scanned.max()),
            "recall_at_10": recall,
            "tie_aware_recall_at_10": radius_recall,
            "mean_candidate_recall_at_10": radius_recall,
            "median_search_ms_per_query": float(np.median(elapsed)),
        })
    expected_full_fraction = float(index.ntotal/original_n)
    if not np.isclose(curve[-1]["mean_scan_fraction"], expected_full_fraction):
        raise AssertionError("full-list scan did not include all index entries")
    if curve[-1]["tie_aware_recall_at_10"] < 0.99999:
        # If this fires, the full IVF scan is genuinely missing a neighbor
        # whose distance is within the exact 10th-distance boundary.
        # Do not mask this by waiving the full-scan correctness invariant.
        raise AssertionError(
            "full-scan FAISS nearest-neighbor distances disagree with exhaustive search "
            f"(tie-aware recall={curve[-1]['tie_aware_recall_at_10']:.6f}, "
            f"raw ID recall={curve[-1]['recall_at_10']:.6f})"
        )
    return {
        "num_stored_entries": int(index.ntotal),
        "storage_amplification": expected_full_fraction,
        "serialized_bytes": len(faiss.serialize_index(index)),
        "list_cv": float(lengths.std()/lengths.mean()),
        "curve": curve,
    }
