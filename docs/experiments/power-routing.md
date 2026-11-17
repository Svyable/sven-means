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
