# Experiment 007 — trust-region power-diagram correction for IVF

**Status:** train-only research prototype. No production changes.

## Prior art and what is NOT ours

Power diagrams with weighted Voronoi assignments, capacity-constrained
partitions and semi-discrete optimal transport have a long literature:
- Lévy, *Centroidal Power Diagrams with Capacity Constraints* (2016),
  https://doi.org/10.1145/2980179.2982428
- Zheng et al., *Hybrid capacity constrained centroidal power diagrams*
  (2021), https://doi.org/10.1016/j.cag.2021.04.007
- Faiss IVF coarse quantizers already expose how unbalanced list
  occupancy affects scan cost (see https://faiss.ai/).

This study **does not invent weighted Voronoi cells**. It explores
whether a small, train-only and bounded-error correction on top of Sven's
global or local-direction centers improves exact IVF-Flat recall
at equal scanned-vector budgets, without retraining centroid coordinates.

## Exact routing equivalence

For centers c_j and prices p_j choose
`argmin_j (||x-c_j||² + p_j)`. The prices need not be positive.

Let p_min=min_j p_j, and augment each query and training vector with
one zero coordinate, while augmenting center j with
`sqrt(p_j-p_min)`. Then

`||[x,0] - [c_j,sqrt(p_j-p_min)]||² = ||x-c_j||² + p_j-p_min`.

Thus a standard **FAISS IndexIVFFlat** with an extra dimension performs
exactly the desired weighted coarse routing. Since both queries and
stored database vectors append **zero**, final candidate-to-query
Euclidean distances are **unchanged**. The extra dimension costs
one float per indexed vector; it is not a zero-memory method.

## Price calibration

Use only the **training** dataset, no evaluation queries.

Start p_j=0. Precompute train-to-center squared L2 distances for this
CPU pilot (O(nk) storage; explicitly *not* billion-scale). For each
iteration, compute assignments to minimize squared distance + price.
Increase prices for overfull lists and lower for underfull lists using
`log((count_j+0.5)/(n/k+0.5))`. Backtrack when the actual unweighted
training assignment SSE rises above `1.01×` its unweighted nearest
centroid baseline; record the most balanced feasible price vector.
The step size comes from median first/second-nearest centroid gap.
This is a deterministic heuristic under an empirical trust region,
**not** a proven optimal-transport solver or a guarantee of recall.

## Preregistered comparison

Two-by-two factorial using the same actual FAISS IVF-Flat engine:
- Sven global × ordinary / power-priced routing
- Sven local-direction × ordinary / power-priced routing

Each pair has two independent data splits × three training seeds;
k=64, two centroid-training epochs. Development corpora are rotated,
Zipf, real digits, real breast cancer and 20 Newsgroups *LSA*
vectors. Model/centroid training is unchanged between ordinary vs
priced, and pricing takes place before seeing the held-out queries.

Metrics:
- actual FAISS IVF-Flat recall@10 vs exact `IndexFlatL2`;
- equal **expected** 5%, 10%, 25% candidate fractions with full
  nprobe-to-scan curves, not just equal nprobe;
- inverted-list occupancy coefficient of variation and max size;
- calibration overhead and actual *assigned* centroid MSE on training,
  constrained to <=1% above nearest-center MSE;
- unchanged centroid-fit time, separate from calibration and search.

No public novelty or superiority claim can follow from these old
development datasets. Only if the Pareto tradeoff is favorable should
a later **unseen pretrained third corpus** (not used to select the
pricing algorithm) be evaluated. The previously used 20 Newsgroups and
BANKING77 neural pilot splits are no longer unseen.

A final deployment study must replace O(nk) calibration storage by
streaming/approximate assignments and measure GPU/FAISS memory and
timing at much larger n,k.


## Observed exploratory results (November 17, 2026 UTC)

[CI run 44021411968](https://github.com/Svyable/sven-means/actions/runs/44021411968) passed all power-routing invariant tests and both real/experimental benchmark jobs. The results compare each priced option with the **same** centroids and plain (zero-priced) routing, with two source seeds × three optimizer seeds (**6 paired cases** per workload). All five families use *actual FAISS IndexIVFFlat* candidate scanning, and recall is interpolated to an equal expected **10% of training vectors scanned**. A positive gain means better recall, not just better fixed-nprobe coverage.

| Workload | Global-price list CV improvement | Global-price recall@10 gain (pp) | Local-price list CV improvement | Local-price recall@10 gain (pp) |
|---|---:|---:|---:|---:|
| Rotated synthetic | +0.159 | **−0.067** | +0.138 | +0.003 |
| Zipf synthetic | +0.369 | **+1.921** | +0.216 | **−0.902** |
| Real digits | +0.112 | **−0.183** | +0.099 | **−0.156** |
| Real breast-cancer features | +0.115 | **−0.221** | +0.096 | +0.432 |
| 20 Newsgroups LSA document vectors | +0.271 | **−0.282** | +0.279 | **−1.139** |

Pricing reliably **reduced list-size dispersion**, but often **reduced retrieval quality** at the same expected candidate cost. Its extra calibration time was approximately 11–67 milliseconds in these small CPU experiments, beyond clustering fit time. All methods share the **same train-only ≤1% additional squared assignment error budget**. Full per-seed, curve, CV, timing and distortion records are downloadable from the cited CI run.

### Decision and publication boundary

The hypothesis that reducing list imbalance while bounding mean training-assignment distortion would generally improve FAISS nearest-neighbor recall is **falsified by this dataset collection**. Power pricing is **not recommended as a default**, especially for the local-direction centroids where it lost −1.139 pp on real LSA documents at a 10% expected scan budget. The +1.921 pp Zipf gain is only on a synthetic skewed workload and is not evidence of general superiority.

The likely missing ingredient is *local nearest-neighbor affinity*: a balanced partition can split semantically or geometrically close points across list boundaries. A future experiment should account for candidate-neighbor preservation using a train-only surrogate while reporting the resulting distortions and overhead. That would be a **new exploratory hypothesis** selected after observing this study, not a preregistered result or a claim to invent power diagrams.

These development datasets already informed earlier Sven experiments. Independent 384D pretrained neural embeddings and fair FAISS/Meta baseline comparisons remain necessary. No publication or deployment claims about throughput, GPU scaling, or billions of vectors follow from these results.
