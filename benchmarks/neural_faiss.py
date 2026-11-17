"""Pretrained neural embeddings + actual FAISS IVF-Flat benchmark.

NO data are included with this project. The runner downloads two independent
public text corpora and an Apache-2.0 pretrained text embedding model into
the Actions runner. Explicit source SHA revisions and vector hashes are
recorded. No neural encoder training occurs. Embedding time is excluded
from clustering fit time (and reported separately).

Run:
  python -m benchmarks.neural_faiss --dataset banking77 --output /tmp/banking.json
  python -m benchmarks.neural_faiss --dataset 20news --output /tmp/news.json

Full search recall at equal expected scanned fraction is interpolated between
actual FAISS IVF-Flat nprobe operating points. Interpolation represents a
randomized probe-count policy in expectation, NOT a per-query scan cap.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import platform
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
import sklearn
import torch

from benchmarks.scan_budget import interpolated_recall_at_budget
from svenmeans import SMeansConfig, assign_clusters, smeans

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# First pilot resolved these immutable revisions. Freeze them for replication.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BANKING = "PolyAI/banking77"
BANKING_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
PROBES = (1, 2, 4, 8, 16, 32, 64, 96)
BUDGETS = (0.05, 0.10, 0.15, 0.25)
KNN = 10


def checksum_vectors(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _unique_documents(documents, limit, rng, excluded=()):
    """Unique nonempty texts, deterministic shuffle, disjoint from excluded."""
    excluded_set = set(excluded)
    permutation = rng.permutation(len(documents))
    seen = set(excluded_set)
    chosen = []
    for i in permutation:
        item = " ".join(str(documents[int(i)]).split())
        if len(item) < 15 or item in seen:
            continue
        chosen.append(item)
        seen.add(item)
        if len(chosen) == limit:
            break
    if len(chosen) < limit:
        raise ValueError(f"Corpus supplied only {len(chosen)} distinct usable texts; need {limit}")
    return chosen


def load_texts(dataset: str, n_train: int, n_queries: int, seed: int):
    """Respect official train/test partitions; dedupe query/train overlap."""
    rng = np.random.default_rng(seed)
    if dataset == "banking77":
        from datasets import load_dataset
        revision = BANKING_REVISION
        train_source = load_dataset(BANKING, split="train", revision=revision)
        query_source = load_dataset(BANKING, split="test", revision=revision)
        train = _unique_documents(train_source["text"], n_train, rng)
        queries = _unique_documents(query_source["text"], n_queries, rng, excluded=train)
        metadata = {
            "dataset": BANKING, "dataset_sha": revision,
            "dataset_license": "cc-by-4.0", "partition": "official train/test",
        }
    elif dataset == "20news":
        from sklearn.datasets import fetch_20newsgroups

        opts = {"remove": ("headers", "footers", "quotes")}
        train_source = fetch_20newsgroups(subset="train", **opts).data
        query_source = fetch_20newsgroups(subset="test", **opts).data
        train = _unique_documents(train_source, n_train, rng)
        queries = _unique_documents(query_source, n_queries, rng, excluded=train)
        metadata = {
            "dataset": "sklearn.datasets.fetch_20newsgroups",
            "dataset_revision": "scikit-learn-20newsgroups-subset",
            "dataset_license": "review underlying corpus usage; raw text not redistributed",
            "partition": "official train/test",
            "scikit_learn_version": sklearn.__version__,
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if set(train) & set(queries):
        raise AssertionError("text overlap between train and evaluation")
    metadata.update({
        "text_train_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
        "text_test_sha256": hashlib.sha256("\n".join(queries).encode()).hexdigest(),
        "n_train": len(train), "n_test": len(queries),
    })
    return train, queries, metadata


def embed_texts(train_texts, test_texts, *, batch_size: int):
    from sentence_transformers import SentenceTransformer

    pinned_revision = MODEL_REVISION
    t0 = time.perf_counter()
    encoder = SentenceTransformer(
        MODEL, revision=pinned_revision, device="cpu", trust_remote_code=False
    )
    train = np.asarray(
        encoder.encode(train_texts, batch_size=batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False), dtype=np.float32
    )
    test = np.asarray(
        encoder.encode(test_texts, batch_size=batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False), dtype=np.float32
    )
    elapsed = time.perf_counter() - t0
    if train.ndim != 2 or test.shape[1] != train.shape[1]:
        raise AssertionError("bad embedding model shape")
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise AssertionError("nonfinite neural embeddings")
    details = {
        "model": MODEL,
        "model_revision": pinned_revision,
        "model_license": "apache-2.0",
        "vector_dimension": int(train.shape[1]),
        "normalization": "L2 unit vectors via sentence-transformers",
        "train_sha256": checksum_vectors(train),
        "test_sha256": checksum_vectors(test),
        "embedding_wall_sec": elapsed,
        "encoder_max_seq_length": int(encoder.max_seq_length),
    }
    return np.ascontiguousarray(train), np.ascontiguousarray(test), details


def build_ivf_index(train, centers):
    """Use exact supplied centers in IVF-Flat; do NOT let FAISS retrain them."""
    d = int(train.shape[1])
    if centers.shape[1] != d or len(centers) < 1:
        raise ValueError("centroid dimensions differ from training vectors")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, len(centers), faiss.METRIC_L2)
    index.quantizer.add(np.ascontiguousarray(centers, dtype=np.float32))
    index.is_trained = True
    index.add_with_ids(
        np.ascontiguousarray(train, dtype=np.float32),
        np.arange(len(train), dtype=np.int64)
    )
    if index.ntotal != len(train):
        raise AssertionError("IVF lost vectors during indexing")
    return index


def evaluated_faiss_curve(
    train, test, centers, exact_neighbor_ids,
    probes=PROBES, repeats=3,
):
    """Actual FAISS IndexIVFFlat recall and latency, with exact scan accounting."""
    if repeats < 1:
        raise ValueError("repeats must be >=1")
    index = build_ivf_index(train, centers)
    q_index = faiss.IndexFlatL2(train.shape[1])
    q_index.add(np.ascontiguousarray(centers))
    measurements = []
    test = np.ascontiguousarray(test, dtype=np.float32)
    for nprobe in sorted({min(len(centers), int(p)) for p in probes}):
        index.nprobe = nprobe
        ids_of_lists = q_index.search(test, nprobe)[1]
        lengths = np.array([index.invlists.list_size(i)
                            for i in range(len(centers))], dtype=np.int64)
        scans = lengths[ids_of_lists].sum(axis=1)
        index.search(test, KNN)  # warmup
        elapsed = []
        retrieved = None
        for _ in range(repeats):
            start = time.perf_counter()
            _, ids = index.search(test, KNN)
            elapsed.append(1000.0 * (time.perf_counter() - start) / len(test))
            retrieved = ids
        assert retrieved is not None
        recall = float(np.mean([
            len(set(exact_neighbor_ids[i]).intersection(int(v) for v in retrieved[i] if v >= 0))
            / KNN
            for i in range(len(test))
        ]))
        measurements.append({
            "nprobe": nprobe,
            "recall_at_10": recall,
            "mean_scan_fraction": float(scans.mean() / len(train)),
            "mean_scanned_vectors": float(scans.mean()),
            "max_scanned_vectors": int(scans.max()),
            "median_search_ms_per_query": statistics.median(elapsed),
            # Interpolator requires same name as candidate-recall curve.
            "mean_candidate_recall_at_10": recall,
        })
    assert measurements[-1]["nprobe"] == len(centers)
    assert measurements[-1]["mean_scan_fraction"] == 1.0
    if measurements[-1]["recall_at_10"] < 0.999:
        raise AssertionError("all-list IVF-Flat recall should match exact FAISS L2 neighbors")
    return measurements


def make_centers(train, k, epochs, seed, arm):
    started = time.perf_counter()
    if arm == "sven_global":
        cfg = SMeansConfig(revive_geometry="global")
        fitted = smeans(torch.from_numpy(train), k, nepochs=epochs,
                        seed=seed, device="cpu", config=cfg,
                        compute_labels=False, return_info=True)
        centers = fitted.centroids.numpy()
        splits = fitted.total_splits
    elif arm == "sven_local":
        cfg = SMeansConfig(revive_geometry="local_direction",
                           local_observe_every=4)
        fitted = smeans(torch.from_numpy(train), k, nepochs=epochs,
                        seed=seed, device="cpu", config=cfg,
                        compute_labels=False, return_info=True)
        centers = fitted.centroids.numpy()
        splits = fitted.total_splits
    elif arm.startswith("faiss_kmeans_"):
        iters = int(arm.rsplit("_", 1)[1])
        faiss_model = faiss.Kmeans(
            train.shape[1], k, niter=iters, nredo=1, seed=seed,
            verbose=False, gpu=False, min_points_per_centroid=1
        )
        faiss_model.train(np.ascontiguousarray(train, dtype=np.float32))
        centers = faiss_model.centroids
        splits = None
    else:
        raise ValueError("unknown arm " + arm)
    fit_ms = (time.perf_counter() - started) * 1000
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    if centers.shape != (k, train.shape[1]):
        raise AssertionError("unexpected center shape")
    if not np.isfinite(centers).all():
        raise AssertionError("nonfinite centroids")
    return centers, fit_ms, splits


def train_and_measure(train, test, k, epochs, seed, arm, exact_ids):
    centers, fit_ms, splits = make_centers(train, k, epochs, seed, arm)
    _, errors = assign_clusters(torch.from_numpy(test), torch.from_numpy(centers))
    curve = evaluated_faiss_curve(train, test, centers, exact_ids)
    return {
        "arm": arm, "model_seed": seed,
        "test_mse": float(errors.double().mean()),
        "fit_ms": fit_ms,
        "total_splits": splits,
        "curve": curve,
        "expected_recall_by_scan_fraction": {
            str(b): interpolated_recall_at_budget(curve, b) for b in BUDGETS
        },
    }


def summarize_paired(rows):
    """No post-hoc winner selection; compare all arms to same-seed Sven global."""
    baseline = {r["model_seed"]: r for r in rows if r["arm"] == "sven_global"}
    out = {}
    for arm in ("sven_local", "faiss_kmeans_2", "faiss_kmeans_20"):
        subset = [r for r in rows if r["arm"] == arm]
        pairs = []
        for r in subset:
            b = baseline[r["model_seed"]]
            d = {
                "model_seed": r["model_seed"],
                "mse_gain_pct": 100.0 * (1 - r["test_mse"] / b["test_mse"]),
                "fit_runtime_ratio": r["fit_ms"] / b["fit_ms"],
            }
            for x in BUDGETS:
                a_recall = r["expected_recall_by_scan_fraction"][str(x)]
                b_recall = b["expected_recall_by_scan_fraction"][str(x)]
                d[f"scan_{x}_gain_points"] = (
                    100 * (a_recall - b_recall)
                    if a_recall is not None and b_recall is not None else None
                )
            pairs.append(d)
        stats = {
            "mean_mse_gain_pct": statistics.mean(p["mse_gain_pct"] for p in pairs),
            "mse_wins": sum(p["mse_gain_pct"] > 0 for p in pairs),
            "median_fit_runtime_ratio": statistics.median(
                p["fit_runtime_ratio"] for p in pairs
            ),
            "num_pairs": len(pairs),
        }
        for x in BUDGETS:
            values = [p[f"scan_{x}_gain_points"] for p in pairs
                      if p[f"scan_{x}_gain_points"] is not None]
            stats[f"scan_{x}_gain_points"] = (
                statistics.mean(values) if values else None
            )
            stats[f"scan_{x}_valid_pairs"] = len(values)
        out[arm] = {"summary": stats, "paired": pairs}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("banking77", "20news"), required=True)
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-test", type=int, default=160)
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--data-seeds", type=int, default=2)
    ap.add_argument("--model-seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if min(args.n_train, args.n_test, args.clusters, args.data_seeds,
           args.model_seeds, args.batch_size, args.threads) < 1:
        ap.error("all integer options must be positive")
    if args.clusters >= args.n_train or args.epochs <= 0:
        ap.error("clusters must be smaller than n_train; epochs positive")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    result = {
        "protocol": {
            "stage": "pretrained neural embedding pilot",
            "source": args.dataset, "L2_normalized": True,
            "faiss_index": "IndexIVFFlat, provided centroids, no retraining",
            "expected_budget": "randomized mix of two adjacent nprobe counts",
            "comparisons": ["sven_global", "sven_local",
                            "faiss_kmeans_2", "faiss_kmeans_20"],
            "ground_truth": "IndexFlatL2 exhaustive exact top-10",
            "nprobe_grid": list(PROBES), "scan_budget_grid": list(BUDGETS),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "sentence_transformers": package_version("sentence-transformers"),
            "datasets": package_version("datasets"),
            "huggingface_hub": package_version("huggingface-hub"),
            "threads": args.threads,
        },
        "settings": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
        "data_seeds": [],
    }
    for data_seed in range(args.data_seeds):
        train_texts, query_texts, corpus = load_texts(
            args.dataset, args.n_train, args.n_test, 4901 + data_seed
        )
        train, test, model_details = embed_texts(
            train_texts, query_texts, batch_size=args.batch_size
        )
        index = faiss.IndexFlatL2(train.shape[1])
        index.add(train)
        exact_ids = index.search(test, KNN)[1]
        arms = ("sven_global", "sven_local", "faiss_kmeans_2", "faiss_kmeans_20")
        rows = []
        for model_seed in range(args.model_seeds):
            rotated = arms[model_seed % len(arms):] + arms[:model_seed % len(arms)]
            for arm in rotated:
                rows.append(train_and_measure(
                    train, test, args.clusters, args.epochs, model_seed, arm, exact_ids
                ))
        result["data_seeds"].append({
            "data_seed": data_seed,
            "corpus": corpus, "embedding": model_details,
            "summary": summarize_paired(rows), "raw_runs": rows,
        })
        # Preserve evidence if later data download fails or CI is interrupted.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Actual FAISS IVF-Flat evaluation completed")
    for split in result["data_seeds"]:
        print("DATA SEED", split["data_seed"], "MODEL SHA", split["embedding"]["model_revision"])
        for arm, measures in split["summary"].items():
            s = measures["summary"]
            print(arm, "MSE gain", round(s["mean_mse_gain_pct"], 3),
                  "5% scan recall gain (pp)", s["scan_0.05_gain_points"],
                  "10% scan recall gain (pp)", s["scan_0.1_gain_points"],
                  "time ratio", round(s["median_fit_runtime_ratio"], 3))


if __name__ == "__main__":
    main()
