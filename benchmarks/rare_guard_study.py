"""Exploratory combination: distance-protected revival + residual directions.

ATTRIBUTION: Distance protection is adapted from Eyal Weiss (2026),
arXiv:2611.18238, not invented by Sven. This experiment tests the
combination with the independent cluster-local residual proposal.

For dataset development stress tests, run:
  python -m benchmarks.rare_guard_study --dataset all-offline --output /tmp/rare.json
  python -m benchmarks.rare_guard_study --dataset news_lsa --output /tmp/news.json
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
from benchmarks.scan_budget import scan_curve, interpolated_recall_at_budget
from svenmeans import SMeansConfig, assign_clusters, smeans

ARMS = [
    ("global", "global", "none"),
    ("global_protected", "global", "distance"),
    ("local", "local_direction", "none"),
    ("local_protected", "local_direction", "distance"),
]
BUDGETS = (0.05, 0.10, 0.25)


def rare_islands(n: int, test_n: int, d: int, seed: int):
    """Small isolated groups below revive count threshold.

    The injected rare groups appear in train with 4 points each and at
    matching frequency in test. Do not confuse with independently sampled
    real embeddings or a new implementation of Weiss's exact experiments.
    """
    g = torch.Generator().manual_seed(seed)
    groups = 20
    common = torch.randn(groups, d, generator=g) * 6.0
    rare = torch.randn(3, d, generator=g)
    rare = torch.nn.functional.normalize(rare, dim=1) * 100.
    rare_counts = 4
    n_common = n - len(rare) * rare_counts
    if n_common < groups:
        raise ValueError("Need at least 20 common points and 12 rare points")
    labels = torch.randint(groups, (n_common,), generator=g)
    train = common[labels] + torch.randn(n_common, d, generator=g) * 0.4
    for center in rare:
        train = torch.cat((train, center + torch.randn(rare_counts, d, generator=g)*0.2))
    train = train[torch.randperm(len(train), generator=g)]
    prevalence = len(rare)*rare_counts/n
    choose_rare = torch.rand(test_n, generator=g) < prevalence
    test = common[torch.randint(groups, (test_n,), generator=g)]
    test += torch.randn(test_n, d, generator=g) * 0.4
    if int(choose_rare.sum()) > 0:
        indices = torch.randint(len(rare), (int(choose_rare.sum()),), generator=g)
        test[choose_rare] = (
            rare[indices] + torch.randn(len(indices), d, generator=g)*0.2
        )
    return train.contiguous(), test.contiguous(), {
        "generator": "20 Gaussian common groups + 3 rare distant groups",
        "rare_train_points_per_group": rare_counts,
        "rare_centroid_distance_scale": 100,
    }


def evaluate(dataset, args):
    provenance, rows = [], []
    for i in range(args.data_seeds):
        seed = args.seed_offset + i
        if dataset == "rare_islands":
            train, test, source = rare_islands(args.n, args.test_n, args.d, seed)
        else:
            train, test, source = dataset_arrays(
                dataset, seed, args.n, args.test_n, args.d
            )
        provenance.append({
            "data_seed": seed, "train_size": list(train.shape),
            "test_size": list(test.shape),
            "train_sha256": hashlib.sha256(train.numpy().tobytes()).hexdigest(),
            "test_sha256": hashlib.sha256(test.numpy().tobytes()).hexdigest(),
            **source,
        })
        for model_seed in range(args.model_seeds):
            order = ARMS[model_seed % 4:] + ARMS[:model_seed % 4]
            for name, geometry, protection in order:
                cfg = SMeansConfig(
                    revive_geometry=geometry,
                    revive_protection=protection,
                    local_observe_every=4,
                )
                t0 = time.perf_counter()
                res = smeans(
                    train, args.k, nepochs=args.epochs, seed=model_seed,
                    device="cpu", config=cfg,
                    compute_labels=False, return_info=True,
                )
                ms = (time.perf_counter() - t0) * 1000
                _, sq = assign_clusters(test, res.centroids)
                curve = scan_curve(train, test, res.centroids)
                row = {
                    "arm": name,
                    "data_seed": seed,
                    "model_seed": model_seed,
                    "heldout_mse": float(sq.double().mean()),
                    "fit_ms": ms,
                    "splits": res.total_splits,
                    "guarded_candidate_checks": res.guarded_candidate_checks,
                    "scan": curve,
                    "recall_by_budget": {
                        str(b): interpolated_recall_at_budget(curve, b)
                        for b in BUDGETS
                    },
                }
                rows.append(row)
    baseline = [r for r in rows if r["arm"] == "global"]
    paired_results = {}
    for name, _, _ in ARMS[1:]:
        sub = [r for r in rows if r["arm"] == name]
        matched = []
        for b in baseline:
            a = next(r for r in sub if r["data_seed"] == b["data_seed"]
                     and r["model_seed"] == b["model_seed"])
            rec = {
                "data_seed": b["data_seed"], "model_seed": b["model_seed"],
                "mse_gain_pct": 100*(1-a["heldout_mse"]/b["heldout_mse"]),
                "fit_runtime_ratio": a["fit_ms"]/b["fit_ms"],
                "split_delta": a["splits"]-b["splits"],
                "guarded_candidate_checks": a["guarded_candidate_checks"],
            }
            for budget in BUDGETS:
                ac, bc = a["recall_by_budget"][str(budget)], b["recall_by_budget"][str(budget)]
                rec[f"recall_{budget}_gain_points"] = (
                    100*(ac-bc) if ac is not None and bc is not None else None
                )
            matched.append(rec)
        summary = {
            "mse_gain_pct": statistics.mean(p["mse_gain_pct"] for p in matched),
            "mse_wins": sum(p["mse_gain_pct"] > 0 for p in matched),
            "median_fit_runtime_ratio": statistics.median(
                p["fit_runtime_ratio"] for p in matched
            ),
            "mean_guarded_checks": statistics.mean(
                p["guarded_candidate_checks"] for p in matched
            ),
        }
        for b in BUDGETS:
            vals = [p[f"recall_{b}_gain_points"] for p in matched
                    if p[f"recall_{b}_gain_points"] is not None]
            summary[f"recall_{b}_gain_points"] = (
                statistics.mean(vals) if vals else None
            )
            summary[f"recall_{b}_comparisons"] = len(vals)
        paired_results[name] = {"summary": summary, "matched": matched}
    return {
        "dataset": dataset, "provenance": provenance,
        "pairs": paired_results, "raw": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=(
        "all-offline", "rare_islands", "rotated", "zipf", "digits",
        "cancer", "news_lsa",
    ), default="all-offline")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--test-n", type=int, default=1200)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2.)
    ap.add_argument("--data-seeds", type=int, default=2)
    ap.add_argument("--model-seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=2303)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if min(args.n,args.test_n,args.d,args.k,args.data_seeds,args.model_seeds,args.threads) < 1:
        ap.error("sample counts, dimensions, k and seeds must be positive")
    if args.k >= args.n or args.epochs <= 0:
        ap.error("n must exceed k, epochs must be positive")
    torch.set_num_threads(args.threads)
    datasets = (
        ["rare_islands", "zipf", "digits", "cancer"]
        if args.dataset == "all-offline" else [args.dataset]
    )
    results = [evaluate(name, args) for name in datasets]
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print("===", result["dataset"], "===")
        for arm, item in result["pairs"].items():
            s = item["summary"]
            print(f"{arm:17s} MSE {s['mse_gain_pct']:+7.2f}% "
                  f"(wins {s['mse_wins']}/{args.data_seeds*args.model_seeds}), "
                  f"fit {s['median_fit_runtime_ratio']:.2f}x, "
                  f"guarded candidate checks {s['mean_guarded_checks']:.1f}, "
                  f"recall at 10% scan "
                  f"{s['recall_0.1_gain_points']}")
if __name__ == "__main__":
    main()
