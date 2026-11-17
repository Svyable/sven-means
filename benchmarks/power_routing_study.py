"""Factorial test: local residual revival × train-only priced IVF routing.

Exploratory development data only. Calibration uses TRAIN vectors and no
held-out queries. FAISS IndexIVFFlat performs *real* coarse search and exact
L2 scanning of visited vectors, with full-list recall asserted to be 1.
The 1D coordinate extension implements a known power diagram, not an
original theoretical concept.

Run:
  python -m benchmarks.power_routing_study --dataset all-offline --output /tmp/power.json
  python -m benchmarks.power_routing_study --dataset news_lsa --output /tmp/news.json
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
import sklearn
import torch

from benchmarks.power_routing import (
    build_power_ivf, calibrate_power_prices, pairwise_sqdist,
    power_embedding,
)
from benchmarks.release_candidate import dataset_arrays
from benchmarks.scan_budget import interpolated_recall_at_budget
from svenmeans import SMeansConfig, smeans

PROBES = (1, 2, 4, 8, 16, 32, 64)
BUDGETS = (0.05, 0.10, 0.25)
NEIGHBORS = 10


def measure_ivf(train, query, centers, prices, ground_truth):
    index = build_power_ivf(train, centers, prices)
    test, biased_centers = power_embedding(query, centers, prices)
    coarse = faiss.IndexFlatL2(biased_centers.shape[1])
    coarse.add(biased_centers)
    sizes = np.array(
        [index.invlists.list_size(j) for j in range(len(centers))],
        dtype=np.int64,
    )
    curve = []
    for p in sorted({min(int(p), len(centers)) for p in PROBES}):
        index.nprobe = p
        visited = coarse.search(test, p)[1]
        scanned = sizes[visited].sum(axis=1)
        index.search(test, NEIGHBORS)  # warmup
        timings = []
        retrieved = None
        for _ in range(3):
            t0 = time.perf_counter()
            _, retrieved = index.search(test, NEIGHBORS)
            timings.append(1000 * (time.perf_counter() - t0) / len(test))
        recall = float(np.mean([
            len(set(ground_truth[i]).intersection(
                int(v) for v in retrieved[i] if v >= 0
            ))/NEIGHBORS
            for i in range(len(test))
        ]))
        curve.append({
            "nprobe": p,
            "mean_scan_fraction": float(scanned.mean()/len(train)),
            "mean_candidate_recall_at_10": recall,
            "median_query_ms": statistics.median(timings),
            "mean_scanned_vectors": float(scanned.mean()),
        })
    if curve[-1]["mean_scan_fraction"] < 1 - 1e-7 or curve[-1]["mean_candidate_recall_at_10"] < .999:
        raise AssertionError("full-list FAISS IVF-Flat must equal exact search")
    cv = float(sizes.std()/sizes.mean())
    return {
        "curve": curve,
        "list_cv": cv,
        "max_list_size": int(sizes.max()),
        "min_list_size": int(sizes.min()),
        "budgeted_recall": {
            str(b): interpolated_recall_at_budget(curve, b) for b in BUDGETS
        },
    }


def evaluate(name, args):
    data_records, rows = [], []
    for dseed in range(args.data_seeds):
        seed = args.seed_offset+dseed
        train, all_queries, origin = dataset_arrays(
            name, seed, args.n, args.test_n, args.d
        )
        train = np.ascontiguousarray(train.numpy(), dtype=np.float32)
        queries = np.ascontiguousarray(
            all_queries[:min(96, len(all_queries))].numpy(), dtype=np.float32
        )
        index = faiss.IndexFlatL2(train.shape[1])
        index.add(train)
        exact = index.search(queries, NEIGHBORS)[1]
        data_records.append({
            "data_seed": seed, "n_train": len(train), "n_query": len(queries),
            "vector_dim": train.shape[1],
            "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
            "query_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
            **origin,
        })
        for mseed in range(args.model_seeds):
            for geometry in ("global", "local_direction"):
                t0 = time.perf_counter()
                cfg = SMeansConfig(
                    revive_geometry=geometry, local_observe_every=4
                )
                fit = smeans(
                    torch.from_numpy(train), args.k, nepochs=args.epochs,
                    seed=mseed, device="cpu", config=cfg,
                    compute_labels=False, return_info=True,
                )
                fit_ms = (time.perf_counter()-t0)*1000
                centers = np.ascontiguousarray(fit.centroids.numpy())
                cal = calibrate_power_prices(
                    train, centers, max_distortion_increase=args.distortion_budget,
                    iterations=args.pricing_iterations,
                )
                for pricing, prices in (
                    ("plain", np.zeros(len(centers), dtype=np.float32)),
                    ("priced", cal["prices"]),
                ):
                    measured = measure_ivf(train, queries, centers, prices, exact)
                    row = {
                        "data_seed": seed, "model_seed": mseed,
                        "arm": f"{'global' if geometry=='global' else 'local'}/{pricing}",
                        "fit_ms": fit_ms,
                        "pricing_ms": cal["calibration_ms"] if pricing == "priced" else 0.,
                        "calibration_train_mse_increase_pct": (
                            100*(cal["assigned_mse"]/cal["baseline_mse"]-1)
                            if pricing == "priced" else 0.
                        ),
                        "calibration_train_cv_before": cal["baseline_cv"],
                        "calibration_train_cv_after": (
                            cal["new_cv"] if pricing == "priced" else cal["baseline_cv"]
                        ),
                        **measured,
                    }
                    rows.append(row)

    comparisons = {}
    for candidate, control in (
        ("global/priced", "global/plain"),
        ("local/priced", "local/plain"),
        ("local/plain", "global/plain"),
        ("local/priced", "global/plain"),
    ):
        matched = []
        for row in (x for x in rows if x["arm"] == candidate):
            base = next(x for x in rows if x["arm"] == control
                        and x["data_seed"] == row["data_seed"]
                        and x["model_seed"] == row["model_seed"])
            pair = {
                "data_seed": row["data_seed"], "model_seed": row["model_seed"],
                "candidate": candidate, "control": control,
                "list_cv_improvement": base["list_cv"]-row["list_cv"],
                "extra_calibration_ms": row["pricing_ms"]-base["pricing_ms"],
            }
            for b in BUDGETS:
                c = row["budgeted_recall"][str(b)]
                a = base["budgeted_recall"][str(b)]
                pair[f"recall_{b}_gain_points"] = (
                    100*(c-a) if c is not None and a is not None else None
                )
            matched.append(pair)
        summary = {
            "mean_list_cv_improvement": statistics.mean(
                p["list_cv_improvement"] for p in matched
            ),
            "mean_extra_calibration_ms": statistics.mean(
                p["extra_calibration_ms"] for p in matched
            ),
            "pairs": len(matched),
        }
        for b in BUDGETS:
            valid = [p[f"recall_{b}_gain_points"] for p in matched
                     if p[f"recall_{b}_gain_points"] is not None]
            summary[f"recall_{b}_gain_points"] = statistics.mean(valid) if valid else None
            summary[f"valid_{b}_pairs"] = len(valid)
            summary[f"wins_{b}"] = sum(v > 0 for v in valid)
        comparisons[candidate+" vs "+control] = {
            "summary": summary, "raw_pairs": matched
        }
    return {
        "dataset": name, "provenance": data_records,
        "comparisons": comparisons, "raw": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=(
        "all-offline", "zipf", "rotated", "digits", "cancer", "news_lsa"
    ), default="all-offline")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--test-n", type=int, default=1200)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2.)
    ap.add_argument("--data-seeds", type=int, default=2)
    ap.add_argument("--model-seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=2907)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--distortion-budget", type=float, default=.01)
    ap.add_argument("--pricing-iterations", type=int, default=80)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if min(a.n,a.test_n,a.d,a.k,a.data_seeds,a.model_seeds,a.threads,a.pricing_iterations) < 1:
        ap.error("counts, dimensions, seeds, iterations must be positive")
    if a.k >= a.n or a.epochs <= 0 or not 0<=a.distortion_budget<=.25:
        ap.error("k<n, epochs>0, distortion budget in [0,.25] required")
    torch.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    names = ["rotated","zipf","digits","cancer"] if a.dataset=="all-offline" else [a.dataset]
    results = [evaluate(n,a) for n in names]
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "faiss": getattr(faiss,"__version__","unknown"),
            "threads": a.threads,
        },
        "config": {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "results": results,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    for result in results:
        print("===",result["dataset"],"===")
        for name, obj in result["comparisons"].items():
            s=obj["summary"]
            print(name, "CV improvement",round(s["mean_list_cv_improvement"],3),
                  "recall@10 at 10% scans",s["recall_0.1_gain_points"],
                  "valid",s["valid_0.1_pairs"], "pricing ms",
                  round(s["mean_extra_calibration_ms"],2))


if __name__=="__main__":
    main()
