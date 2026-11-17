"""Experiment 011: budgeted SOAR-inspired redundancy, actual FAISS IVF.

Preregistered ablation before results:
  plain: 1 entry per training vector
  nearest/boundary: second nearest centroid for boundary rows
  orthogonal/boundary: shortlist residual orthogonality secondary center
  nearest/random: second nearest centroid for random rows

Additional entries: exactly 25% or 50% of train vectors, the same quota
for all three spilling policies. Same trained centroids for every paired
comparison. Real FAISS IVFFlat search, duplicate-aware top10, full exact
nearest-neighbor ground truth; compare at equal *scanned entries* rather
than equal nprobe. The 2*K search heap is equal for all methods.

This is not invented SOAR. The L2 residual scoring here is a small-scale
adaptation of an established Google ScaNN technique (NeurIPS 2023).
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

from benchmarks.orthogonal_spill import (
    build_spill_ivf, evaluate_spill_index, secondary_candidates,
    select_spill_rows,
)
from benchmarks.release_candidate import dataset_arrays
from benchmarks.scan_budget import interpolated_recall_at_budget
from svenmeans import SMeansConfig, smeans

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
MODELS = ("sven_global", "sven_local", "faiss_lloyd2", "meta_emeans2")
POLICIES = (
    ("plain", 0., "boundary", "nearest"),
    ("nearest_boundary25", .25, "boundary", "nearest"),
    ("orthogonal_boundary25", .25, "boundary", "orthogonal"),
    ("nearest_random25", .25, "random", "nearest"),
    ("nearest_boundary50", .5, "boundary", "nearest"),
    ("orthogonal_boundary50", .5, "boundary", "orthogonal"),
    ("nearest_random50", .5, "random", "nearest"),
)


def fit_centers(train, k, epochs, seed, model):
    """All training is on train vectors only; no query/time lookahead."""
    started = time.perf_counter()
    if model in ("sven_global", "sven_local"):
        cfg = SMeansConfig(
            revive_geometry=(
                "global" if model == "sven_global" else "local_direction"
            ),
            local_observe_every=4,
        )
        fit = smeans(
            torch.from_numpy(train), k, nepochs=epochs, seed=seed,
            device="cpu", config=cfg, compute_labels=False,
            return_info=True,
        )
        centers = fit.centroids.numpy()
    elif model == "faiss_lloyd2":
        native = faiss.Kmeans(
            train.shape[1], k, niter=2, nredo=1,
            seed=seed, gpu=False, verbose=False,
            min_points_per_centroid=1,
        )
        native.train(train)
        centers = native.centroids
    elif model == "meta_emeans2":
        import faiss.contrib
        dirname = Path(__file__).resolve().parents[1]/"faiss-meta"/"contrib"
        if not dirname.is_dir():
            raise RuntimeError(
                "Meta e-means pinned source missing; check out "
                "facebookresearch/faiss@c8f283867285af547db7a701c1b8dfa89631b3ea"
            )
        faiss.contrib.__path__.insert(0, str(dirname))
        from faiss.contrib.e_means import NumpyEmeans

        meta = NumpyEmeans(
            train.shape[1], k, nepochs=2, seed=seed,
            verbose=False, capture_assignments=False,
        )
        centers, _ = meta.train(train)
    else:
        raise ValueError("invalid center model " + model)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    if centers.shape != (k, train.shape[1]) or not np.isfinite(centers).all():
        raise AssertionError("invalid fitted coarse centroids")
    return centers, 1000.*(time.perf_counter()-started)


def make_truth(train, query):
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    return index.search(query, 10)


def compare_matched(rows):
    """Paired controls include same *storage budget* and same centroids."""
    comps = {}
    refs = {
        "orthogonal_boundary25": "nearest_boundary25",
        "orthogonal_boundary50": "nearest_boundary50",
        "nearest_boundary25": "plain",
        "nearest_boundary50": "plain",
        "orthogonal_boundary25_vs_plain": "plain",
        "orthogonal_boundary50_vs_plain": "plain",
        "nearest_boundary25_vs_random": "nearest_random25",
        "nearest_boundary50_vs_random": "nearest_random50",
    }
    for label, control in refs.items():
        candidate_name = label.replace("_vs_plain", "").replace("_vs_random", "")
        candidate = next(x for x in rows if x["arm"] == candidate_name)
        base = next(x for x in rows if x["arm"] == control)
        record = {
            "candidate": candidate_name, "control": control,
            "entry_storage_ratio": candidate["num_stored_entries"] /
                                   base["num_stored_entries"],
            "serialized_index_ratio": candidate["serialized_bytes"] /
                                      base["serialized_bytes"],
            "index_build_ms_delta": candidate["index_build_ms"] -
                                    base["index_build_ms"],
        }
        for fraction in BUDGETS:
            c = candidate["recall_at_equal_scanned_entries"][str(fraction)]
            b = base["recall_at_equal_scanned_entries"][str(fraction)]
            record[f"recall_{fraction}_gain_points"] = (
                100*(c-b) if c is not None and b is not None else None
            )
        comps[label+" vs "+control] = record
    return comps


def study(name, args):
    data_records = []
    for dataseed in range(args.data_seeds):
        train_t, test_t, prov = dataset_arrays(
            name, args.seed_offset+dataseed, args.n,
            args.test_n, args.d
        )
        train = np.ascontiguousarray(train_t.numpy(), dtype=np.float32)
        test = np.ascontiguousarray(
            test_t[:min(args.queries, len(test_t))].numpy(), dtype=np.float32
        )
        exact_sqdist, truth = make_truth(train, test)
        records = []
        for seed in range(args.model_seeds):
            for model in MODELS:
                centers, fit_ms = fit_centers(
                    train, args.k, args.epochs, seed, model
                )
                started = time.perf_counter()
                options = secondary_candidates(
                    train, centers, weight=args.orth_lambda,
                    shortlist=args.shortlist
                )
                secondary_ms = 1000.*(time.perf_counter()-started)
                policies = POLICIES[seed % len(POLICIES):] + POLICIES[:seed % len(POLICIES)]
                rows=[]
                for arm, fraction, row_policy, sec_method in policies:
                    if arm == "plain":
                        index, index_ms = build_spill_ivf(train, centers)
                    else:
                        selection = select_spill_rows(
                            options["relative_boundary_margin"], fraction,
                            policy=row_policy, seed=9000+seed+dataseed*101
                        )
                        index, index_ms = build_spill_ivf(
                            train, centers, selected_rows=selection,
                            secondary_ids=options[sec_method][selection],
                        )
                    stats = evaluate_spill_index(
                        index, centers, test, truth, exact_sqdist,
                        original_n=len(train),
                        repeats=args.repeats,
                    )
                    rows.append({
                        "arm": arm, "center_model": model,
                        "data_seed": dataseed, "fit_seed": seed,
                        "center_fit_ms": fit_ms,
                        "secondary_candidate_ms": secondary_ms if arm != "plain" else 0.,
                        "index_build_ms": index_ms,
                        **stats,
                        "recall_at_equal_scanned_entries": {
                            str(b): interpolated_recall_at_budget(
                                stats["curve"], b
                            ) for b in BUDGETS
                        },
                    })
                records.append({
                    "fit_seed": seed, "center_model": model,
                    "paired": compare_matched(rows), "raw": rows,
                })
        data_records.append({
            "data_seed": dataseed,
            "provenance": {
                **prov,
                "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
                "train_shape": list(train.shape),
                "query_shape": list(test.shape),
            },
            "models": records,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "protocol": "SOAR-inspired spill, actual FAISS IVF-Flat; all controls",
            "meta_faiss_sha": "c8f283867285af547db7a701c1b8dfa89631b3ea",
            "settings": {k:str(v) if isinstance(v,Path) else v
                         for k,v in vars(args).items()},
            "environment": {
                "python":platform.python_version(),
                "faiss":getattr(faiss,"__version__","unknown"),
                "numpy":np.__version__,
                "torch":torch.__version__,
                "threads":torch.get_num_threads(),
            },
            "datasets":data_records,
        }, indent=2)+"\n")
    print("===",name,"===")
    for model in MODELS:
        matches=[
            item for record in data_records for item in record["models"]
            if item["center_model"] == model
        ]
        for pair_label in [
            "orthogonal_boundary25 vs nearest_boundary25",
            "orthogonal_boundary50 vs nearest_boundary50",
            "nearest_boundary25 vs plain",
            "orthogonal_boundary25_vs_plain vs plain",
        ]:
            pairs=[x["paired"][pair_label] for x in matches]
            scores=[p["recall_0.1_gain_points"] for p in pairs
                    if p["recall_0.1_gain_points"] is not None]
            print(model,pair_label,
                  "mean recall gain at equal 10% scanned entries",
                  statistics.mean(scores) if scores else None,
                  "valid",len(scores),
                  "index bytes ratio",round(statistics.mean(
                      p["serialized_index_ratio"] for p in pairs
                  ),3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=(
        "news_lsa","rotated","zipf","digits","cancer","all-offline"
    ),default="news_lsa")
    p.add_argument("--n",type=int,default=4000)
    p.add_argument("--test-n",type=int,default=1200)
    p.add_argument("--queries",type=int,default=96)
    p.add_argument("--d",type=int,default=96)
    p.add_argument("--k",type=int,default=64)
    p.add_argument("--epochs",type=float,default=2.0)
    p.add_argument("--data-seeds",type=int,default=2)
    p.add_argument("--model-seeds",type=int,default=3)
    p.add_argument("--seed-offset",type=int,default=3781)
    p.add_argument("--shortlist",type=int,default=8)
    p.add_argument("--orth-lambda",type=float,default=2.0)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if min(a.n,a.test_n,a.queries,a.d,a.k,a.data_seeds,a.model_seeds,
           a.shortlist,a.threads,a.repeats)<1 or a.k>=a.n:
        p.error("counts positive, k<n")
    if a.k<2 or a.shortlist<2 or a.epochs<=0:
        p.error("k>=2, shortlist>=2, epochs>0")
    if not np.isfinite(a.orth_lambda) or a.orth_lambda <0:
        p.error("orth-lambda must be finite and nonnegative")
    torch.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    datasets=["rotated","zipf","digits","cancer"] if a.dataset=="all-offline" else [a.dataset]
    for name in datasets:
        # One output artifact per dataset to avoid accidental replacement.
        dest=(a.output if len(datasets)==1 else
              a.output.with_name(a.output.stem+"-"+name+".json"))
        from argparse import Namespace
        config=Namespace(**{**vars(a),"output":dest})
        study(name,config)


if __name__=="__main__":
    main()
