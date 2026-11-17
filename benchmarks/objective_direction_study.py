"""Experiment 010: compare covariance-PCA to objective-aligned split direction.

Design registered before observing numerical outcomes.
Old development workloads ONLY. Candidate recall is the ideal coarse
list inclusion proxy at equal expected scanned vector fraction,
NOT actual FAISS latency. A later independent pretrained corpus/actual
FAISS comparison is required for a broad publishable claim.
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

from benchmarks.release_candidate import dataset_arrays
from benchmarks.rare_guard_study import rare_islands
from benchmarks.scan_budget import scan_curve, interpolated_recall_at_budget
from svenmeans import SMeansConfig, assign_clusters, smeans


ARMS = [
    ("global", "global", "covariance", "rms_scaled"),
    ("cov_rms", "local_direction", "covariance", "rms_scaled"),
    ("cov_abs", "local_direction", "covariance", "mean_abs"),
    ("gain_rms", "local_direction", "split_gain", "rms_scaled"),
    ("gain_abs", "local_direction", "split_gain", "mean_abs"),
]
BUDGETS = (0.05, 0.10, 0.25)


def train_one(train, test, args, name, geometry, direction, radius, seed):
    cfg = SMeansConfig(
        revive_geometry=geometry,
        revive_protection="none",
        local_direction_objective=direction,
        local_radius_policy=radius,
        local_observe_every=4,
    )
    started = time.perf_counter()
    res = smeans(
        train, args.k, nepochs=args.epochs, device="cpu", seed=seed,
        config=cfg, return_info=True, compute_labels=False,
    )
    elapsed = 1000. * (time.perf_counter() - started)
    _, dist = assign_clusters(test, res.centroids)
    curve = scan_curve(train, test, res.centroids, query_count=96)
    return {
        "arm": name, "model_seed": seed,
        "test_mse": float(dist.double().mean()),
        "fit_ms": elapsed,
        "splits": int(res.total_splits),
        "curve": curve,
        "recall_by_scan_budget": {
            str(b): interpolated_recall_at_budget(curve, b) for b in BUDGETS
        },
    }


def evaluate(dataset, args):
    records = []
    for i in range(args.data_seeds):
        dseed = args.seed_offset + i
        if dataset == "rare_islands":
            train, test, provenance = rare_islands(
                args.n, args.test_n, args.d, dseed
            )
        else:
            train, test, provenance = dataset_arrays(
                dataset, dseed, args.n, args.test_n, args.d
            )
        rows = []
        for seed in range(args.model_seeds):
            # Change run order across seeds to reduce warm-cache asymmetry.
            ordered = ARMS[seed%len(ARMS):]+ARMS[:seed%len(ARMS)]
            for name, geo, direction, radius in ordered:
                rows.append(train_one(
                    train, test, args, name, geo, direction, radius, seed
                ))
        baseline = {r["model_seed"]:r for r in rows if r["arm"] == "cov_rms"}
        comparisons = {}
        for arm in [a[0] for a in ARMS if a[0] != "cov_rms"]:
            paired=[]
            for candidate in [r for r in rows if r["arm"] == arm]:
                control=baseline[candidate["model_seed"]]
                row={
                    "model_seed": candidate["model_seed"],
                    "mse_gain_vs_cov_pct": 100*(1-candidate["test_mse"]/control["test_mse"]),
                    "runtime_vs_cov_ratio": candidate["fit_ms"]/control["fit_ms"],
                    "split_delta": candidate["splits"]-control["splits"],
                }
                for b in BUDGETS:
                    a= candidate["recall_by_scan_budget"][str(b)]
                    c= control["recall_by_scan_budget"][str(b)]
                    row[f"recall_{b}_gain_points"] = (
                        100*(a-c) if a is not None and c is not None else None
                    )
                paired.append(row)
            comparisons[arm]={
                "mean_mse_gain_pct": statistics.mean(p["mse_gain_vs_cov_pct"] for p in paired),
                "mse_wins": sum(p["mse_gain_vs_cov_pct"]>0 for p in paired),
                "median_runtime_ratio": statistics.median(
                    p["runtime_vs_cov_ratio"] for p in paired
                ),
                "mean_split_delta": statistics.mean(p["split_delta"] for p in paired),
                "pairs": paired,
            }
            for b in BUDGETS:
                v=[p[f"recall_{b}_gain_points"] for p in paired
                   if p[f"recall_{b}_gain_points"] is not None]
                comparisons[arm][f"recall_{b}_gain_points"] = (
                    statistics.mean(v) if v else None
                )
                comparisons[arm][f"recall_{b}_valid_pairs"] = len(v)
        records.append({
            "data_seed": dseed,
            "provenance": {
                **provenance,
                "train_shape":list(train.shape),"test_shape":list(test.shape),
                "train_sha256": hashlib.sha256(train.numpy().tobytes()).hexdigest(),
                "test_sha256": hashlib.sha256(test.numpy().tobytes()).hexdigest(),
            },
            "paired": comparisons, "raw": rows,
        })
    summary={}
    for arm in [a[0] for a in ARMS if a[0]!="cov_rms"]:
        allpairs=[p for record in records for p in record["paired"][arm]["pairs"]]
        summary[arm]={
            "mean_mse_gain_pct":statistics.mean(p["mse_gain_vs_cov_pct"] for p in allpairs),
            "mse_wins":sum(p["mse_gain_vs_cov_pct"]>0 for p in allpairs),
            "num_pairs":len(allpairs),
            "median_runtime_ratio":statistics.median(p["runtime_vs_cov_ratio"] for p in allpairs),
            "mean_split_delta":statistics.mean(p["split_delta"] for p in allpairs),
        }
        for b in BUDGETS:
            scores=[p[f"recall_{b}_gain_points"] for p in allpairs
                    if p[f"recall_{b}_gain_points"] is not None]
            summary[arm][f"recall_{b}_gain_points"]=statistics.mean(scores) if scores else None
            summary[arm][f"recall_{b}_paired_count"]=len(scores)
    return {"dataset":dataset,"summary":summary,"records":records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=(
        "all-offline","rare_islands","zipf","rotated","digits","cancer",
        "news_lsa"
    ),default="all-offline")
    p.add_argument("--n",type=int,default=4000)
    p.add_argument("--test-n",type=int,default=1200)
    p.add_argument("--d",type=int,default=96)
    p.add_argument("--k",type=int,default=64)
    p.add_argument("--epochs",type=float,default=2.)
    p.add_argument("--data-seeds",type=int,default=2)
    p.add_argument("--model-seeds",type=int,default=4)
    p.add_argument("--seed-offset",type=int,default=3307)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if min(args.n,args.test_n,args.d,args.k,args.data_seeds,args.model_seeds,args.threads)<1:
        p.error("n, test n, dimensions, k, seeds, threads must be positive")
    if args.k>=args.n or args.epochs<=0:
        p.error("n>k and positive epochs required")
    torch.set_num_threads(args.threads)
    datasets=(
        ["rotated","zipf","digits","cancer","rare_islands"]
        if args.dataset=="all-offline" else [args.dataset]
    )
    results=[evaluate(d,args) for d in datasets]
    report={
        "environment":{"torch":torch.__version__,"numpy":np.__version__,
                       "sklearn":sklearn.__version__,"python":platform.python_version(),
                       "threads":torch.get_num_threads()},
        "config":{key:str(v) if isinstance(v,Path) else v
                  for key,v in vars(args).items()},
        "results":results,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    for r in results:
        print("===",r["dataset"],"===")
        for name,obj in r["summary"].items():
            print(f"{name:12s} vs cov/rms: MSE {obj['mean_mse_gain_pct']:+7.3f}% "
                  f"wins {obj['mse_wins']}/{obj['num_pairs']}, "
                  f"recall 10% scanned {obj['recall_0.1_gain_points']}, "
                  f"runtime {obj['median_runtime_ratio']:.2f}x")


if __name__=="__main__":
    main()
