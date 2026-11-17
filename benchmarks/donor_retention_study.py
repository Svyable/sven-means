"""Post-hoc Stage B: how much *donor* final-pass evidence should survive a split?

Stages are honest: Stage A observed reset-both improve coarse candidate recall
on skewed synthetic Zipf but worsen held-out SSE. This separately registered
experiment compares carry, reset-both, 50% donor retention, and target-only
using identical training/test data and optimizer seeds.

Run: python benchmarks/donor_retention_study.py --dataset all-offline
     python benchmarks/donor_retention_study.py --dataset news_lsa
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from release_candidate import dataset_arrays, recall_of_inverted_lists
from svenmeans import SMeansConfig, assign_clusters, smeans


ARMS = [
    ("carry", False, 0.0),
    ("both", True, 0.0),
    ("half", True, 0.5),
    ("target_only", True, 1.0),
]


def train_one(train, test, n_clusters, nepochs, model_seed, mode, reset, retention):
    config = SMeansConfig(
        reset_pass_on_split=reset,
        donor_pass_retention=retention,
        revive_geometry="global",
    )
    started = time.perf_counter()
    fitted = smeans(
        train, n_clusters, nepochs=nepochs, seed=model_seed,
        device="cpu", config=config, return_info=True, compute_labels=False,
    )
    fit_ms = 1000.0 * (time.perf_counter() - started)
    _, dist = assign_clusters(test, fitted.centroids)
    labels, _ = assign_clusters(train, fitted.centroids)
    counts = torch.bincount(labels, minlength=n_clusters).double()
    cv = float(counts.std(unbiased=False).div(counts.mean().clamp_min(1)))
    return {
        "arm": mode,
        "model_seed": model_seed,
        "test_mse": float(dist.double().mean()),
        "fit_ms": fit_ms,
        "recall": recall_of_inverted_lists(train, test, fitted.centroids),
        "occupancy_cv": cv,
        "empty_list_fraction": float((counts == 0).double().mean()),
        "splits": int(fitted.total_splits),
    }


def evaluate(dataset, args):
    rows, provenance = [], []
    for offset in range(args.data_seeds):
        seed = args.seed_offset + offset
        train, test, origin = dataset_arrays(dataset, seed, args.n, args.test_n, args.d)
        if train.shape[0] <= args.k:
            raise ValueError("n_train must exceed k")
        provenance.append({
            "data_seed": seed, "train_shape": list(train.shape),
            "test_shape": list(test.shape), **origin,
        })
        for model_seed in range(args.model_seeds):
            order = ARMS[model_seed % len(ARMS):] + ARMS[:model_seed % len(ARMS)]
            for name, reset, retention in order:
                result = train_one(
                    train, test, args.k, args.epochs, model_seed, name, reset, retention
                )
                result["data_seed"] = seed
                rows.append(result)

    baseline = [r for r in rows if r["arm"] == "carry"]
    comparisons = {}
    for name, _, _ in ARMS[1:]:
        candidate = [r for r in rows if r["arm"] == name]
        matched = []
        for b in baseline:
            alt = next(
                r for r in candidate if r["data_seed"] == b["data_seed"]
                and r["model_seed"] == b["model_seed"]
            )
            matched.append({
                "data_seed": b["data_seed"],
                "model_seed": b["model_seed"],
                "mse_gain_pct": 100 * (1 - alt["test_mse"] / b["test_mse"]),
                "candidate_recall_gain_points": 100 * (
                    alt["recall"]["candidate_recall_at_10_nprobe_4"]
                    - b["recall"]["candidate_recall_at_10_nprobe_4"]
                ),
                "occupancy_cv_change": alt["occupancy_cv"] - b["occupancy_cv"],
                "runtime_ratio": alt["fit_ms"] / b["fit_ms"],
            })
        comparisons[name] = {
            "mean_mse_gain_pct": statistics.mean(r["mse_gain_pct"] for r in matched),
            "median_mse_gain_pct": statistics.median(r["mse_gain_pct"] for r in matched),
            "mse_wins": sum(r["mse_gain_pct"] > 0 for r in matched),
            "mean_recall_gain_points": statistics.mean(
                r["candidate_recall_gain_points"] for r in matched
            ),
            "mean_occupancy_cv_change": statistics.mean(
                r["occupancy_cv_change"] for r in matched
            ),
            "median_runtime_ratio": statistics.median(r["runtime_ratio"] for r in matched),
            "paired": matched,
        }

    return {
        "dataset": dataset,
        "provenance": provenance,
        "results": comparisons,
        "raw_runs": rows,
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
        parser.error("positive counts required, k<=n and epochs>0")
    torch.set_num_threads(args.threads)

    datasets = ["rotated", "zipf", "digits", "cancer"] if args.dataset == "all-offline" else [args.dataset]
    outcomes = [evaluate(name, args) for name in datasets]
    summary = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
        },
        "parameters": {key: str(val) if isinstance(val, Path) else val
                       for key, val in vars(args).items()},
        "datasets": outcomes,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for outcome in outcomes:
        print("===", outcome["dataset"], "===")
        for policy, data in outcome["results"].items():
            print(
                f"{policy:12s} MSE gain {data['mean_mse_gain_pct']:+.3f}% "
                f"(wins {data['mse_wins']}/{len(data['paired'])}) "
                f"recall@10 nprobe4 {data['mean_recall_gain_points']:+.3f} pp, "
                f"occupancy CV {data['mean_occupancy_cv_change']:+.3f}, "
                f"runtime {data['median_runtime_ratio']:.3f}x"
            )
    if not args.output:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
