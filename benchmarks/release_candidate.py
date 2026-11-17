"""Paired release-candidate study: test MSE, candidate recall, fit latency.

Candidate A = current sven-means with final-pass stats carried through splits.
Candidate B = the SAME model with post-split final-pass statistics invalidated.

Arms:
  global/control, global/reset, local/control, local/reset

Datasets: rotated and Zipf synthetic generators; real digits and breast-cancer
feature vectors (bundled in scikit-learn); real 20 Newsgroups documents embedded
by train-only TF-IDF + TruncatedSVD (requires an explicitly requested download).

Examples:
  python benchmarks/release_candidate.py --dataset all-offline --output /tmp/offline.json
  python benchmarks/release_candidate.py --dataset news_lsa --output /tmp/news.json

This is an observational A/B study, not a pass/fail test for superiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer, load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

from svenmeans import SMeansConfig, assign_clusters, smeans


def dataset_arrays(name: str, data_seed: int, n: int, test_n: int, d: int):
    if name in ("digits", "cancer"):
        X = (load_digits().data if name == "digits" else load_breast_cancer().data)
        train, test = train_test_split(X, test_size=0.25, random_state=data_seed)
        scaler = StandardScaler().fit(train)  # strictly train-only
        return (
            torch.tensor(scaler.transform(train), dtype=torch.float32),
            torch.tensor(scaler.transform(test), dtype=torch.float32),
            {"source": "scikit-learn bundled real tabular data", "normalization": "train-only StandardScaler"},
        )

    if name == "news_lsa":
        # Public documents with boilerplate removed; the vectorizer and SVD
        # are fitted ONLY on training documents to avoid evaluation leakage.
        corpus = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes"),
        ).data
        ids = np.random.default_rng(data_seed).permutation(len(corpus))
        use = ids[:min(len(ids), n + test_n)]
        ntr = min(n, len(use) - 1)
        train_docs = [corpus[i] for i in use[:ntr]]
        test_docs = [corpus[i] for i in use[ntr:]]
        vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.9, max_features=10000, stop_words="english",
            dtype=np.float32,
        )
        tr_sparse = vectorizer.fit_transform(train_docs)
        te_sparse = vectorizer.transform(test_docs)
        nc = max(2, min(d, tr_sparse.shape[0] - 1, tr_sparse.shape[1] - 1))
        svd = TruncatedSVD(n_components=nc, n_iter=5, random_state=data_seed)
        tr = normalize(svd.fit_transform(tr_sparse), norm="l2")
        te = normalize(svd.transform(te_sparse), norm="l2")
        return (
            torch.from_numpy(np.ascontiguousarray(tr, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(te, dtype=np.float32)),
            {
                "source": "scikit-learn 20 Newsgroups, train subset; download required",
                "representation": "TF-IDF then train-only TruncatedSVD then L2 normalization",
                "vocabulary_size": int(tr_sparse.shape[1]),
                "latent_dimensions": int(nc),
            },
        )

    if name in ("rotated", "zipf"):
        rng = torch.Generator().manual_seed(data_seed)
        groups = 10
        means = torch.randn(groups, d, generator=rng) * 6
        directions = torch.nn.functional.normalize(
            torch.randn(groups, d, generator=rng), dim=1,
        )
        weights = torch.ones(groups)
        if name == "zipf":
            weights = torch.arange(1, groups + 1, dtype=torch.float32).pow(-1.2)
        weights /= weights.sum()

        def draw(count):
            groups_index = torch.multinomial(weights, count, generator=rng, replacement=True)
            points = means[groups_index] + torch.randn(count, d, generator=rng) * 0.55
            if name == "rotated":
                stretch = torch.randn(count, 1, generator=rng) * 4.0
                points += stretch * directions[groups_index]
            return points.contiguous()

        return draw(n), draw(test_n), {
            "source": "seeded Gaussian mixture generator",
            "correlation": "rank-one local elongation" if name == "rotated" else "isotropic",
        }
    raise ValueError(f"Unsupported dataset: {name}")


def recall_of_inverted_lists(
    train: torch.Tensor,
    test: torch.Tensor,
    centers: torch.Tensor,
    probes=(1, 4, 8),
    *,
    neighbors: int = 10,
    query_count: int = 96,
) -> dict[str, float]:
    """Fraction of exact top-10 neighbors present in visited coarse IVF lists.

    This is *candidate recall*, not end-to-end FAISS IVF query latency, nor
    exact rerank throughput. It tests whether the centroid partition retains
    relevant nearest neighbors when a query probes only 1/4/8 lists.
    """
    q = test[:min(query_count, test.shape[0])]
    true_neighbors = torch.cdist(q, train).topk(
        min(neighbors, train.shape[0]), largest=False, dim=1
    ).indices
    train_labels, _ = assign_clusters(train, centers)
    neighbor_labels = train_labels[true_neighbors]
    probe_ids = torch.cdist(q, centers).topk(
        min(max(probes), centers.shape[0]), largest=False, dim=1
    ).indices
    return {
        f"candidate_recall_at_{neighbors}_nprobe_{p}": float(
            (neighbor_labels.unsqueeze(-1) == probe_ids[:, None, :min(p, centers.shape[0])])
            .any(dim=-1).float().mean().item()
        )
        for p in probes
    }


def train_one(train, test, k: int, seed: int, nepochs: float,
              geometry: str, reset: bool):
    cfg = SMeansConfig(
        revive_geometry=geometry, reset_pass_on_split=reset,
        local_observe_every=4,
    )
    start = time.perf_counter()
    result = smeans(
        train, k, nepochs=nepochs, seed=seed, device="cpu",
        config=cfg, compute_labels=False, return_info=True,
    )
    elapsed = (time.perf_counter() - start) * 1000
    _, distances = assign_clusters(test, result.centroids)
    return {
        "arm": f"{geometry}/{'reset' if reset else 'carry'}",
        "model_seed": seed,
        "mse": float(distances.double().mean()),
        "fit_ms": elapsed,
        "dead_clusters": result.dead_clusters,
        "splits": result.total_splits,
        "recall": recall_of_inverted_lists(train, test, result.centroids),
    }


def evaluate(name, args):
    runs = []
    source_records = []
    arms = [
        ("global", False), ("global", True),
        ("local_direction", False), ("local_direction", True),
    ]
    for data_seed in range(args.data_seeds):
        tr, te, provenance = dataset_arrays(
            name, args.seed_offset + data_seed, args.n, args.test_n, args.d
        )
        if tr.shape[0] <= args.k:
            raise ValueError(f"{name}: n_train must exceed k")
        source_records.append({
            "data_seed": args.seed_offset + data_seed,
            "shape_train": list(tr.shape),
            "shape_test": list(te.shape),
            "train_sha256": hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
            "test_sha256": hashlib.sha256(te.numpy().tobytes()).hexdigest(),
            **provenance,
        })
        for model_seed in range(args.model_seeds):
            # Rotate order to reduce unfair warm cache effects on one arm.
            ordered = arms[model_seed % 4:] + arms[:model_seed % 4]
            for geom, reset in ordered:
                row = train_one(tr, te, args.k, model_seed, args.epochs, geom, reset)
                row["data_seed"] = args.seed_offset + data_seed
                runs.append(row)

    summary = {}
    comparison = {}
    for geometry in ("global", "local_direction"):
        control_key = f"{geometry}/carry"
        reset_key = f"{geometry}/reset"
        base = [r for r in runs if r["arm"] == control_key]
        new = [r for r in runs if r["arm"] == reset_key]
        paired = []
        for b in base:
            after = next(r for r in new if r["data_seed"] == b["data_seed"]
                         and r["model_seed"] == b["model_seed"])
            paired.append({
                "data_seed": b["data_seed"],
                "model_seed": b["model_seed"],
                "mse_gain_pct": 100 * (1 - after["mse"] / b["mse"]),
                "runtime_ratio": after["fit_ms"] / b["fit_ms"],
                "recall_at10_nprobe4_gain_points": 100 * (
                    after["recall"]["candidate_recall_at_10_nprobe_4"]
                    - b["recall"]["candidate_recall_at_10_nprobe_4"]
                ),
            })
        comparison[geometry] = {
            "mean_mse_gain_pct": statistics.mean(p["mse_gain_pct"] for p in paired),
            "median_mse_gain_pct": statistics.median(p["mse_gain_pct"] for p in paired),
            "mse_wins": sum(p["mse_gain_pct"] > 0 for p in paired),
            "total_pairs": len(paired),
            "median_runtime_ratio": statistics.median(p["runtime_ratio"] for p in paired),
            "mean_recall_gain_points": statistics.mean(
                p["recall_at10_nprobe4_gain_points"] for p in paired
            ),
            "pairs": paired,
        }
        for arm, sub in ((control_key, base), (reset_key, new)):
            summary[arm] = {
                "mean_test_mse": statistics.mean(r["mse"] for r in sub),
                "median_fit_ms": statistics.median(r["fit_ms"] for r in sub),
                "mean_recall_at10_nprobe4": statistics.mean(
                    r["recall"]["candidate_recall_at_10_nprobe_4"] for r in sub
                ),
                "mean_split_count": statistics.mean(r["splits"] for r in sub),
            }

    return {
        "dataset": name,
        "provenance": source_records,
        "summary": summary,
        "paired": comparison,
        "raw": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[
        "all-offline", "rotated", "zipf", "digits", "cancer", "news_lsa",
    ], default="all-offline")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--test-n", type=int, default=1200)
    parser.add_argument("--d", type=int, default=96)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--model-seeds", type=int, default=4)
    parser.add_argument("--data-seeds", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=1903)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.n, args.test_n, args.d, args.k, args.model_seeds,
           args.data_seeds, args.threads) < 1 or args.k > args.n or args.epochs <= 0:
        parser.error("positive counts and epochs required, with k<=n")
    torch.set_num_threads(args.threads)
    names = (["rotated", "zipf", "digits", "cancer"]
             if args.dataset == "all-offline" else [args.dataset])
    ds = [evaluate(name, args) for name in names]
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "cpu_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()},
        "datasets": ds,
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    for result in ds:
        print(f"=== {result['dataset']} ===")
        for geom, data in result["paired"].items():
            print(f"{geom:16s} reset vs carry: mean MSE gain "
                  f"{data['mean_mse_gain_pct']:+.3f}%, "
                  f"MSE wins {data['mse_wins']}/{data['total_pairs']}, "
                  f"median time {data['median_runtime_ratio']:.3f}x, "
                  f"candidate recall@10 (nprobe=4) gain "
                  f"{data['mean_recall_gain_points']:+.3f} percentage points")
    if not args.output:
        print(payload)


if __name__ == "__main__":
    main()
