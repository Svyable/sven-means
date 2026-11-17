# Experiment 012 — RAIRS AIR assignment formula against SOAR and vanilla IVF

**Status: completed initial development pilot, green CI; negative/conditional results.**

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


## First measured results — November 17, 2026 UTC

[GitHub Actions run 44109571162](https://github.com/Svyable/sven-means/actions/runs/44109571162)
finished with **all three benchmark jobs passing** (formula/unit tests, real
20 Newsgroups LSA vectors, Zipf synthetic vectors); the accompanying Python
3.10–3.13 CPU regression checks also passed. The CI artifacts contain
complete per-seed FAISS query-time, index-bytes, stored-entry, scanned-entry,
raw-ID and distance-tie-aware recall curves and inputs' SHA-256 hashes.

Tables below compare **actual FAISS IVF-Flat tie-aware recall@10** at
equal expected **10% of the original database size in scanned index entries**.
Each value is the mean of **six paired tests (two source draws × three
optimizer seeds)**. Positive recall gains favor the candidate arm; *zero*
is no improvement. The primary controls share fixed centroids and selected
rows; all strict 25%-replicated methods have **identical index bytes**.

### Development 20 Newsgroups LSA — 25% maximum extra entries

| Coarse-centroid producer | Strict AIR vs SOAR | Optional AIR vs SOAR | Optional AIR vs unreplicated index |
|---|---:|---:|---:|
| Sven global | −1.129 pp | −0.320 pp | −1.243 pp |
| Sven local | −1.133 pp | −0.251 pp | −0.861 pp |
| FAISS Lloyd, 2 iterations | −1.480 pp | −0.899 pp | −0.064 pp |
| Pinned Meta NumpyEmeans, 2 epochs | −1.123 pp | −0.422 pp | −1.510 pp |

Strict AIR **beats naive nearest-secondary** (+0.55 to +1.85 pp) while
still trailing the **SOAR L2 scoring** used in Experiment 011 across
all four centroid producers. Optional RAIR also beats naive but improves
relative to strict by **abstaining from some second assignments**:
about 2–3% fewer serialized index bytes than the strict methods
(0.967× to 0.980×). Relative to ordinary unreplicated IVF, it still
uses ~1.20–1.22× serialized index bytes and has **lower recall** at
the selected 10%-scan operating point.

### Development Zipf synthetic — 25% maximum extra entries

| Coarse-centroid producer | Strict AIR vs SOAR | Optional AIR vs SOAR | Optional AIR vs unreplicated index |
|---|---:|---:|---:|
| Sven global | −1.538 pp | −0.202 pp | −1.802 pp |
| Sven local | −0.003 pp | +0.824 pp | −0.149 pp |
| FAISS Lloyd, 2 iterations | −0.329 pp | +0.725 pp | −0.305 pp |
| Pinned Meta NumpyEmeans, 2 epochs | −0.307 pp | +1.217 pp | −0.588 pp |

Optional RAIR sometimes outperforms SOAR at **less index storage**
(~0.936–0.962× the strict method's bytes) on skewed Zipf. However,
**every** optional arm still has lower recall than unreplicated IVF at
equal scanned entries. These findings are a useful counterexample to
the idea that all redundancy helps low-scan retrieval.

### Interpretation and caveats

1. **AIR and abstention are Yang & Chen's published SIGMOD 2026
   methods**, not new contributions or independent validation of their
   1.33× throughput headline. We use their published score
   `||r'||²+0.5*r·r'`, nearest-10 shortlist and optional/strict
   behavior, but only within a **train-selected 25%/50% duplication
   budget**, which is an experimental modification of their full
   assignment system.
2. The paper's critical **SEIL shared-cell layout** and
   IVF-PQ-FastScan/refinement are *not* implemented. Our actual
   FAISS IndexIVFFlat stores duplicate vector codes and may pay their
   distance-computation cost; the published RAIRS throughput results
   are not comparable to these small CPU FAISS-Flat tests.
3. A modest positive **SOAR-versus-strict-AIR** result on existing
   real LSA data does not prove SOAR generally wins: the official
   RAIRS paper evaluated many larger datasets and different latency
   budgets. The current two datasets have already informed Sven work,
   so they are *exploratory*, not untouched confirmation.
4. **Decision:** do not change Sven's default training or search
   index. Neither optional AIR nor SRAIR currently provides an
   end-to-end quality–storage–scanning Pareto improvement over
   unreplicated IVF on these workloads. Novelty requires a new
   neighbor-coverage policy that also beats SOAR, RAIRS, and
   their original optimized engines on genuinely unseen neural
   embeddings at matched resource costs.
