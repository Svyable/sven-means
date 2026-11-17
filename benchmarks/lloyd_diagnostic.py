"""Experiment 008: does one ordinary Lloyd step explain FAISS's advantage?

A *diagnostic* ablation, NOT a novel centroid-update method:
- Fit Sven global and local once with equal seeds.
- Starting from each resulting centroid set, apply 0, 1, or 2 standard
  full-batch Lloyd updates on TRAIN data only.
- Benchmark the resulting coarse centroids via actual FAISS IndexIVFFlat,
  exact IndexFlatL2 search ground truth, and equal expected vector-scan
  budgets, plus clustering fit/refinement time.
- Include FAISS KMeans 2 and 20 iterations as reference arms.

This experiment is on existing *development corpora*; do not relabel
the results independent or use to make new superiority claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from benchmarks.neural_faiss import (
    embed_texts, evaluated_faiss_curve, load_texts,
)
from benchmarks.scan_budget import interpolated_recall_at_budget
from svenmeans import SMeansConfig, smeans


BUDGETS = (0.05, 0.10, 0.15, 0.25)
STEPS = (0, 1, 2)


def assign_faiss(train: np.ndarray, centers: np.ndarray):
    """Exact squared L2 assignments on training vectors only."""
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(np.ascontiguousarray(centers, dtype=np.float32))
    distances, labels = index.search(np.ascontiguousarray(train, dtype=np.float32), 1)
    return labels[:, 0], distances[:, 0]


def lloyd_trace(train: np.ndarray, initial_centers: np.ndarray, steps: int = 2):
    """Deterministic exact full-batch Lloyd sequence, preserving empty centers.

    Each update is the exact minimizer for fixed assignments, and the next
    reassignment can only further lower the TRAIN objective, up to float
    rounding. No test/query vectors are examined here.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    x = np.ascontiguousarray(train, dtype=np.float32)
    c = np.ascontiguousarray(initial_centers, dtype=np.float32).copy()
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1] or not len(c):
        raise ValueError("expected nonempty [n,d] data and [k,d] centroids")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("input must be finite")
    out = []
    total_ms = 0.
    for step in range(steps + 1):
        start = time.perf_counter()
        label, sqdist = assign_faiss(x, c)
        if step == 0:
            total_ms = 0.
        else:
            # Charge the current-step assignment; the previous update is
            # already included in the stopwatch for that previous iteration.
            total_ms += (time.perf_counter() - start) * 1000
        out.append({
            "steps": step,
            "centers": c.copy(),
            "train_mse": float(np.mean(sqdist, dtype=np.float64)),
            "refinement_ms": float(total_ms),
            "empty_centers": int((np.bincount(label, minlength=len(c)) == 0).sum()),
        })
        if step == steps:
            break
        began = time.perf_counter()
        sums = np.zeros_like(c, dtype=np.float64)
        np.add.at(sums, label, x.astype(np.float64, copy=False))
        counts = np.bincount(label, minlength=len(c))
        nonempty = counts > 0
        c[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        total_ms += (time.perf_counter() - began) * 1000
    return out


def train_sven(train: np.ndarray, k: int, epochs: float, seed: int, geometry: str):
    start = time.perf_counter()
    cfg = SMeansConfig(revive_geometry=geometry, local_observe_every=4)
    result = smeans(
        torch.from_numpy(train), k, nepochs=epochs, device="cpu",
        seed=seed, config=cfg, compute_labels=False, return_info=True
    )
    return (
        np.ascontiguousarray(result.centroids.numpy(), dtype=np.float32),
        (time.perf_counter() - start)*1000,
        result.total_splits,
    )


def train_faiss(train: np.ndarray, k: int, seed: int, iterations: int):
    start = time.perf_counter()
    model = faiss.Kmeans(
        train.shape[1], k, niter=iterations, nredo=1, seed=seed,
        gpu=False, verbose=False, min_points_per_centroid=1
    )
    model.train(np.ascontiguousarray(train, dtype=np.float32))
    return (
        np.ascontiguousarray(model.centroids, dtype=np.float32),
        (time.perf_counter() - start)*1000
    )


def evaluate_centers(train, query, centers, exact, *,
                     label, model_seed, fit_ms, refinement_ms, splits):
    curve = evaluated_faiss_curve(
        train, query, centers, exact, repeats=3,
    )
    _, holdout_dist = assign_faiss(query, centers)
    return {
        "arm": label,
        "model_seed": model_seed,
        "train_fit_ms": fit_ms,
        "refinement_ms": refinement_ms,
        "total_fit_ms": fit_ms + refinement_ms,
        "heldout_mse": float(np.mean(holdout_dist, dtype=np.float64)),
        "splits": splits,
        "curve": curve,
        "recall_at_equal_expected_scan": {
            str(b): interpolated_recall_at_budget(curve, b) for b in BUDGETS
        },
    }


def pairs_against_global(rows):
    base = {r["model_seed"]: r for r in rows if r["arm"] == "sven_global/0"}
    summary = {}
    for arm in sorted(set(r["arm"] for r in rows) - {"sven_global/0"}):
        vals = []
        for row in (r for r in rows if r["arm"] == arm):
            origin = base[row["model_seed"]]
            pair = {
                "model_seed": row["model_seed"],
                "mse_improvement_pct": 100*(1-row["heldout_mse"]/origin["heldout_mse"]),
                "total_fit_time_ratio": row["total_fit_ms"]/origin["total_fit_ms"],
            }
            for budget in BUDGETS:
                a = row["recall_at_equal_expected_scan"][str(budget)]
                b = origin["recall_at_equal_expected_scan"][str(budget)]
                pair[f"recall_{budget}_gain_points"] = (
                    100*(a-b) if a is not None and b is not None else None
                )
            vals.append(pair)
        out = {
            "num_pairs": len(vals),
            "mean_mse_improvement_pct": statistics.mean(x["mse_improvement_pct"] for x in vals),
            "mse_wins": sum(x["mse_improvement_pct"] > 0 for x in vals),
            "median_total_fit_time_ratio": statistics.median(x["total_fit_time_ratio"] for x in vals),
        }
        for budget in BUDGETS:
            available = [x[f"recall_{budget}_gain_points"] for x in vals
                         if x[f"recall_{budget}_gain_points"] is not None]
            out[f"recall_{budget}_mean_gain_points"] = (
                statistics.mean(available) if available else None
            )
            out[f"recall_{budget}_num_comparable"] = len(available)
            out[f"recall_{budget}_wins"] = sum(x > 0 for x in available)
        summary[arm] = {"summary": out, "paired": vals}
    return summary


def run_dataset(name, args):
    records = []
    for data_seed in range(args.data_seeds):
        # There is no train/query leakage; official corpus partitions.
        train_text, query_text, provenance = load_texts(
            name, args.n_train, args.n_test, 4901 + data_seed
        )
        train, query, encoder_details = embed_texts(
            train_text, query_text, batch_size=args.batch_size
        )
        truth = faiss.IndexFlatL2(train.shape[1])
        truth.add(train)
        exact = truth.search(query, 10)[1]
        rows = []
        for seed in range(args.model_seeds):
            for geometry in ("global", "local_direction"):
                fitted, fit_ms, splits = train_sven(
                    train, args.clusters, args.epochs, seed, geometry
                )
                trace = lloyd_trace(train, fitted, steps=max(STEPS))
                for item in trace:
                    rows.append(evaluate_centers(
                        train, query, item["centers"], exact,
                        label=f"sven_{geometry}/{item['steps']}",
                        model_seed=seed, fit_ms=fit_ms,
                        refinement_ms=item["refinement_ms"], splits=splits,
                    ))
            for iterations in (2, 20):
                centers, fit_ms = train_faiss(train, args.clusters, seed, iterations)
                rows.append(evaluate_centers(
                    train, query, centers, exact,
                    label=f"faiss_{iterations}", model_seed=seed,
                    fit_ms=fit_ms, refinement_ms=0.0, splits=None
                ))
        records.append({
            "data_seed": data_seed,
            "source": provenance,
            "embedding": encoder_details,
            "summary": pairs_against_global(rows),
            "raw_runs": rows,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "dataset": name,
            "settings": {k: str(v) if isinstance(v, Path) else v
                         for k,v in vars(args).items()},
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "records": records,
        }, indent=2) + "\n")
    for record in records:
        print("DATA SEED", record["data_seed"])
        for arm, stat in record["summary"].items():
            s = stat["summary"]
            print(arm, "MSE improvement",round(s["mean_mse_improvement_pct"],3),
                  "recall at 10% scanned",s["recall_0.1_mean_gain_points"],
                  "fit-time ratio",round(s["median_total_fit_time_ratio"],3))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("banking77","20news"), required=True)
    parser.add_argument("--n-train", type=int, default=1600)
    parser.add_argument("--n-test", type=int, default=160)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=2.)
    parser.add_argument("--model-seeds", type=int, default=2)
    parser.add_argument("--data-seeds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    a=parser.parse_args()
    if min(a.n_train,a.n_test,a.clusters,a.model_seeds,a.data_seeds,a.threads,a.batch_size) <1:
        parser.error("count and seed parameters must be positive")
    if a.clusters >= a.n_train or a.epochs<=0:
        parser.error("clusters<n_train and epochs>0")
    torch.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    run_dataset(a.dataset,a)


if __name__=="__main__":
    main()
