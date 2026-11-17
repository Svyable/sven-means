"""Paired evaluation of quality at equal *candidate-vector scan budgets*.

Unlike fixed nprobe, this accounts for how many points belong to probed
centroid lists. The interpolated curves represent an *expected* operating
point from randomized switching between two whole-list probe counts.

Run: python benchmarks/scan_budget_study.py --dataset all-offline
     python benchmarks/scan_budget_study.py --dataset news_lsa
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from release_candidate import dataset_arrays
from scan_budget import interpolated_recall_at_budget, scan_curve
from svenmeans import SMeansConfig, assign_clusters, smeans


ARMS = [
    ("global", "global", False),
    ("local", "local_direction", False),
    ("global_reset", "global", True),
    ("local_reset", "local_direction", True),
]
BUDGETS = (0.05, 0.10, 0.15, 0.25, 0.50, 0.75)


def fit_one(train, test, k, nepochs, seed, name, geometry, reset):
    cfg = SMeansConfig(revive_geometry=geometry, reset_pass_on_split=reset,
                       local_observe_every=4)
    started = time.perf_counter()
    model = smeans(train, k, nepochs=nepochs, seed=seed, device="cpu",
                   config=cfg, compute_labels=False, return_info=True)
    runtime = (time.perf_counter() - started) * 1000
    _, sq = assign_clusters(test, model.centroids)
    curve = scan_curve(train, test, model.centroids)
    return {
        "arm": name, "model_seed": seed,
        "test_mse": float(sq.double().mean()),
        "fit_ms": runtime,
        "splits": model.total_splits,
        "curve": curve,
        "budgeted_candidate_recall": {
            str(b): interpolated_recall_at_budget(curve, b) for b in BUDGETS
        },
    }


def evaluate(dataset, args):
    runs = []
    provenance = []
    for idx in range(args.data_seeds):
        dseed = args.seed_offset + idx
        tr, te, orig = dataset_arrays(
            dataset, dseed, args.n, args.test_n, args.d
        )
        provenance.append({
            "data_seed": dseed,
            "train_shape": list(tr.shape), "test_shape": list(te.shape),
            "train_sha256": hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
            "test_sha256": hashlib.sha256(te.numpy().tobytes()).hexdigest(),
            **orig,
        })
        for model_seed in range(args.model_seeds):
            ordered = ARMS[model_seed % len(ARMS):] + ARMS[:model_seed % len(ARMS)]
            for name, geometry, reset in ordered:
                row = fit_one(tr, te, args.k, args.epochs, model_seed,
                              name, geometry, reset)
                row["data_seed"] = dseed
                runs.append(row)
    pairs = {}
    base = [r for r in runs if r["arm"] == "global"]
    for name, _, _ in ARMS[1:]:
        candidate = [r for r in runs if r["arm"] == name]
        paired = []
        for b in base:
            x = next(
                r for r in candidate
                if r["data_seed"] == b["data_seed"]
                and r["model_seed"] == b["model_seed"]
            )
            b4 = next(row for row in b["curve"] if row["nprobe"] == 4)
            x4 = next(row for row in x["curve"] if row["nprobe"] == 4)
            row = {
                "data_seed": b["data_seed"],
                "model_seed": b["model_seed"],
                "mse_gain_pct": 100 * (1 - x["test_mse"] / b["test_mse"]),
                "fixed_4probe_recall_gain_points": 100 * (
                    x4["mean_candidate_recall_at_10"] - b4["mean_candidate_recall_at_10"]
                ),
                "fixed_4probe_scan_fraction_change_points": 100 * (
                    x4["mean_scan_fraction"] - b4["mean_scan_fraction"]
                ),
                "runtime_ratio": x["fit_ms"] / b["fit_ms"],
            }
            for budget in BUDGETS:
                bc = b["budgeted_candidate_recall"][str(budget)]
                xc = x["budgeted_candidate_recall"][str(budget)]
                row[f"budget_{budget}_gain_points"] = (
                    100 * (xc - bc) if bc is not None and xc is not None else None
                )
            paired.append(row)
        budget_summary = {}
        for budget in BUDGETS:
            vals = [p[f"budget_{budget}_gain_points"] for p in paired
                    if p[f"budget_{budget}_gain_points"] is not None]
            budget_summary[str(budget)] = {
                "paired_count": len(vals),
                "mean_gain_points": statistics.mean(vals) if vals else None,
                "wins": sum(v > 0 for v in vals),
            }
        pairs[name] = {
            "mean_mse_gain_pct": statistics.mean(p["mse_gain_pct"] for p in paired),
            "mse_wins": sum(p["mse_gain_pct"] > 0 for p in paired),
            "fixed_4probe_recall_gain_points": statistics.mean(
                p["fixed_4probe_recall_gain_points"] for p in paired
            ),
            "fixed_4probe_scan_change_points": statistics.mean(
                p["fixed_4probe_scan_fraction_change_points"] for p in paired
            ),
            "median_fit_time_ratio": statistics.median(
                p["runtime_ratio"] for p in paired
            ),
            "budgeted": budget_summary,
            "paired": paired,
        }
    return {"dataset": dataset, "provenance": provenance, "results": pairs, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[
        "all-offline", "rotated", "zipf", "digits", "cancer", "news_lsa"
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
        parser.error("all counts and epochs positive, with k<=n")
    torch.set_num_threads(args.threads)
    datasets = (
        ["rotated", "zipf", "digits", "cancer"]
        if args.dataset == "all-offline" else [args.dataset]
    )
    results = [evaluate(name, args) for name in datasets]
    report = {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "threads": torch.get_num_threads(),
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "study": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for ds in results:
        print(f"=== {ds['dataset']} ===")
        for name, v in ds["results"].items():
            cols = " ".join(
                f"{b}: {s['mean_gain_points']:+.2f}pp / {s['paired_count']} pairs"
                if s["mean_gain_points"] is not None else f"{b}: not comparable"
                for b, s in v["budgeted"].items()
            )
            print(
                f"{name:14s} MSE {v['mean_mse_gain_pct']:+.2f}%, "
                f"nprobe4 recall {v['fixed_4probe_recall_gain_points']:+.2f}pp / "
                f"scan cost {v['fixed_4probe_scan_change_points']:+.2f}pp; "
                f"equal budget {cols}"
            )
    if not args.output:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
