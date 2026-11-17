"""Test that actual FAISS IndexIVFFlat uses supplied centers and scan counts."""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from benchmarks.neural_faiss import (
    build_ivf_index, evaluated_faiss_curve, summarize_paired, checksum_vectors
)


def test_exact_if_all_lists_probed():
    rng = np.random.default_rng(44)
    train = rng.normal(size=(80, 8)).astype("float32")
    queries = rng.normal(size=(12, 8)).astype("float32")
    centroids = train[:8].copy()
    gt = faiss.IndexFlatL2(8)
    gt.add(train)
    exact = gt.search(queries, 10)[1]
    index = build_ivf_index(train, centroids)
    assert index.nlist == len(centroids)
    assert index.ntotal == len(train)
    curve = evaluated_faiss_curve(
        train, queries, centroids, exact, probes=(1, 2, 4, 8), repeats=1
    )
    assert curve[-1]["nprobe"] == 8
    assert curve[-1]["mean_scan_fraction"] == pytest.approx(1.0)
    assert curve[-1]["recall_at_10"] == pytest.approx(1.0)
    assert [p["mean_scan_fraction"] for p in curve] == sorted(
        p["mean_scan_fraction"] for p in curve
    )
    assert checksum_vectors(train) == checksum_vectors(train.copy())


def test_duplicates_have_no_unscanned_phantom_neighbors():
    train = np.array([[0.0], [.1], [.3], [4.0], [5.0]], dtype="float32")
    test = np.array([[.2]], dtype="float32")
    centers = np.array([[0.0], [5.0]], dtype="float32")
    idx = faiss.IndexFlatL2(1)
    idx.add(train)
    # Only five true neighbors exist, so this tiny dataset checks IVF vector
    # preservation without attempting recall@10: build index and inspect bins.
    ivf = build_ivf_index(train, centers)
    assert ivf.invlists.list_size(0) + ivf.invlists.list_size(1) == 5
    ivf.nprobe = 2
    _, found = ivf.search(test, 5)
    assert sorted(found[0].tolist()) == list(range(5))
