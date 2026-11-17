# Experiment 012 — RAIRS AIR assignment formula against SOAR and vanilla IVF

**Status: preregistered development benchmark, initial results outstanding.**

This experiment arises from our SOAR-style pilot [Experiment 011](orthogonal-spill.md).
Its results indicated that orthogonal secondary assignments outperform naive
second-nearest assignments at equal replication quota, but often **fail to
beat unreplicated IVF** when total scanned entries are held constant.

An important stronger prior-art baseline is **RAIRS** by Zehai Yang and
Shimin Chen, *RAIRS: Optimizing Redundant Assignment and List Layout for
IVF-Based ANN Search*, [SIGMOD 2026 / arXiv:2601.07183](https://arxiv.org/abs/2601.07183);
[official C++ repository](https://github.com/schencoding/rairs).
RAIRS demonstrated exactly the failure of naïve second-list assignment
and addressed it using two published contributions:

- **AIR** (Amplified Inverse Residual): for primary center c0,
  choose alternative center cj by minimizing
  `||cj-x||² + 0.5 * ((c0-x)·(cj-x))`.
  **SRAIR** excludes the primary from candidates (exactly two lists).
  **RAIR** can select the primary again, avoiding a duplicate for that
  vector if no sufficiently good second list exists.
  The published default is `lambda=0.5`, `N_CANDS=10`.
- **SEIL**: a shared-cell layout avoids duplicated storage and
  redundant distance evaluation when the same vector belongs to both
  visited lists. This is a **separate index-engine technology**.

Our code implements **only the published AIR formula** under
ordinary FAISS IndexIVFFlat with actual duplicate list entries.
It explicitly does **NOT** reproduce SEIL, RAIRS C++ throughput or
their IVF-PQ FastScan+Refine index. Comparing throughput to RAIRS
requires running [their C++ implementation](https://github.com/schencoding/rairs)
on matched hardware, compression and query workload.

## Fixed ablation

On two already-used development datasets (20 Newsgroups LSA and Zipf),
freeze 2 source draws, 3 optimizer seeds and these centroid producers:

- Sven global, Sven experimental local
- FAISS Lloyd 2 iterations
- pinned genuine Meta FAIR NumpyEmeans at 2 epochs

For each fixed centroid set, compute a **single common shortlist**
of the ten nearest centroids for every training vector. Select
exactly 25% or 50% of training vectors with the smallest relative
nearest/second-nearest centroid distance gap; selection uses
training vectors only, never heldout queries. Score the
selected rows by:

- nearest-second, SOAR-inspired L2 squared-projection penalty (lambda=2)
- **SRAIR** published AIR lambda=.5, excluding the primary
- **RAIR** published AIR lambda=.5, including primary; abstain from
  replication if primary is best (actual index size may be smaller)

Report the unreplicated IVF index as **first-class control**.
For RAIR, index only rows that did not abstain and report *actual*
extra entries, not the quota. All methods search the exact same
`IndexIVFFlat` system with top-20 entry heap, duplicate-aware top10,
and exact `IndexFlatL2` reference. Tied-distance cases use
a documented tolerance on the true tenth squared-distance radius.
Full nprobe=k recall must be geometrically exact.

Primary metric is actual FAISS recall@10 at **equal expected scanned
entries**, not fixed nprobe, at 5/10/20/30/50% budgets.
Report actual serialized index bytes, list counts, per-query
search time, secondary-computation and index construction time
separately. The strict SRAIR/SOAR/nearest-secondary 25% and 50%
comparisons are same selected rows, same entries and index bytes;
non-strict RAIR is budget-*limited* but should use whatever fewer
entries its abstention rule selects.

## Decision gate

Any advantage of RAIR or SRAIR belongs to Yang & Chen's **published
SIGMOD 2026 work**, not to Sven. The experimental purpose is to
establish a must-beat SOTA control before pursuing **truly new**
train-only neighbor-coverage-per-byte placement, streaming
residual-uncertainty allocations, or rare-cluster-aware candidate
selection. Development datasets already used by Sven cannot be
reclassified as independent holdout confirmation. Any paper claim
requires new pretrained embeddings, larger k/d/n and a direct
comparison to authors' C++ RAIRS+SEIL and Google's original SOAR
at matched memory, CPU/GPU latency and accuracy.
