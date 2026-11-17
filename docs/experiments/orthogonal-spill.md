# Experiment 011 — selective SOAR-style orthogonal spilling

**Status:** pre-result, CPU only, separate index experiment; none of
Sven's clustering defaults, centroid training, or sklearn APIs change.

## Why this is higher-priority than arXiv:2611.11598

Cai et al. (2026), *Perturbing Inputs Improves Pretraining Distillation*
https://arxiv.org/abs/2611.11598, trains small language models to
imitate teacher predictions on corrupted/blended tokens. It supplies
no direct method or empirical evidence for faster k-means or improved
nearest-neighbor retrieval. We are **not** adding teacher-derived
synthetic embeddings or noise to the centroid learning objective.

Instead, target the *measured* failure of IVF at low scan fractions:
nearest neighbors may be routed to a coarse list that a query misses.

## Existing state of the art, NOT our idea

Google's [SOAR: Improved Indexing for Approximate Nearest Neighbor Search
(Sun et al., NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/0973524e02a712af33325d0688ae6f49-Abstract-Conference.html)
introduced spilling with **orthogonality-amplified residuals** for
approximate nearest-neighbor search. A secondary assignment is chosen
so its vector-to-center residual can compensate for the primary
assignment's residual when queries are near a partition boundary.
See Google's [explanation](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/).

The original SOAR method and ScaNN implementation are **prior art**.
Our experimental code is a small L2 adaptation with a preregistered
`lambda=2` and 8-nearest-center shortlist; it is not a complete
reimplementation of Google's index, theoretical guarantees or tuning.

## Experimental idea

1. Train coarse centroids with 2-epoch Sven global, 2-epoch Sven local,
   2-iteration FAISS Lloyd, or pinned **actual Meta NumpyEmeans**
   at two epochs. Same training vectors and seeds in each comparison.
2. Assign each training vector to its nearest center as usual.
3. Select a fixed fraction of training rows to duplicate based on
   **relative 1st/2nd centroid distance gap** (smaller gap first).
   As an essential control, select the same number of rows at random.
4. Choose secondary center as either:
   - second-nearest center (naive spill, established prior art), or
   - among eight nearest, minimize
     `||x-c_j||² + 2 * ((x-c_j)·(x-c_primary))² /
                            max(||x-c_primary||², eps)`
     (SOAR-inspired orthogonality-amplified residual).
5. **Actually insert** the same vectors under both original and
   prescribed secondary list IDs into a FAISS `IndexIVFFlat` via
   `add_core`, *not* a simulated candidates-only oracle.
6. Search with FAISS at the same probe counts, requesting 20 ranked
   entries for **every** arm; deduplicate IDs to recover top-10.
   Validate with exact `IndexFlatL2` top-10. Full-list recall must
   equal 1 on each arm.

We test **0%, 25% and 50% additional index entries**. The row-selection
policy and secondary center policy are ablated independently; the
primary comparison is SOAR-inspired vs nearest-secondary with the
**same selected rows**, same centers, same storage quota, and same
FAISS top-20 search heap.

## Metrics

- Actual FAISS recall@10 at **equal expected total inverted-list
  entries scanned** (5%, 10%, 20%, 30%, 50% of original number of
  vectors). Duplicates count as additional scanned entries. Interpolate
  only between measured integer `nprobe` operating points; unavailable
  fractions remain unavailable.
- **Tie-aware recall:** when many original vectors have exactly equal
  query distances, FAISS IndexFlatL2 and IndexIVFFlat may legally
  return different sets of IDs at the equal-distance top-10 boundary.
  Record *raw ID-based recall* separately but use a deterministic
  tolerance on the exact 10th-neighbor squared-distance radius for
  primary scan-budget comparisons. At full nprobe=k, all returned
  unique distances must lie within that exact boundary (within
  documented FP32 numerical tolerance). Never waive a real distance
  mismatch or count duplicate IDs as new neighbors.
- Actual median 3-repeat search milliseconds/query, per-query max
  entries scanned, index build milliseconds, serialized FAISS index
  bytes, centroid training milliseconds, list occupancy CV.
- Two document selections × three centroid seeds per dataset, across
  real 20 Newsgroups LSA document vectors and two synthetic geometries.
  *These are development datasets already used in prior sven-means work.*
- Original data array SHA-256, exact pinned Meta source SHA, FAISS
  package version, per-fit and per-probe raw rows in downloadable CI JSON.

The index entries and serialized index **grow** when vectors are
duplicated. An equal scanned-entries budget does not imply equal
memory, index construction time, latency, or query centroid search
overhead. None are suppressed from the report. Lower k-means MSE is
not expected because centroid coordinates are fixed.

## Decision and next steps

If the orthogonality-amplified secondary assignment improves actual
recall versus nearest-secondary at the **same spill quota** across
multiple development datasets without a prohibitive search-time cost,
freeze the method and evaluate on an entirely **new pretrained neural
corpus**, *then* compare to Google's ScaNN SOAR implementation (not
only our simple nearest-secondary control), and expand to large k/GPU.
A favorable result against vanilla IVF alone **does not establish
novelty**: SOAR already proposed orthogonal spill.

More original work could use Sven's streaming residual moments to
choose per-cluster spill quotas under an end-to-end memory/latency
constraint, instead of computing every training row's entire
centroid shortlist. That is a **hypothesis**, not an implemented win.

Use:
```bash
pip install -e ".[dev]"
pip install faiss-cpu==1.16.0
python -m pytest -q tests/test_orthogonal_spill.py
python -m benchmarks.orthogonal_spill_study --dataset news_lsa --output /tmp/soar.json
```


## Measured pilot: Nov 17, 2026 UTC — negative/conditional evidence

[Actions run 44106459030](https://github.com/Svyable/sven-means/actions/runs/44106459030) **passed the 12-case real FAISS IVF invariant suite**, both the LSA development-corpus benchmark, and both rotated/Zipf synthetic controls. The experiment uses two independent source draws × three training seeds per model (**six paired observations per row**), with exact source-array SHA-256, all nprobe/scan/latency observations, and full per-fit raw JSON available as CI artifacts.

At **equal expected scanning of 10% of the original database size in index entries**, this table reports the mean actual FAISS tie-aware recall@10 percentage-point gain from **25% spill** compared with ordinary second-nearest spill on exactly the same selected vectors, followed by gain vs a **non-spilled index** with the same centroids:

| Development dataset | Centroid producer | Orthogonal vs nearest-secondary, +25% entries | Orthogonal vs no spill, +25% entries |
|---|---|---:|---:|
| Real 20 Newsgroups LSA | Sven global | **+1.754 pp** | **−0.805 pp** |
| Real 20 Newsgroups LSA | Sven local | **+2.057 pp** | **−0.929 pp** |
| Real 20 Newsgroups LSA | FAISS Lloyd, 2 iters | **+2.544 pp** | **+0.407 pp** |
| Real 20 Newsgroups LSA | Meta NumpyEmeans, 2 epochs | **+1.564 pp** | **−1.021 pp** |
| Rotated synthetic | Sven global | +0.032 pp | −0.007 pp |
| Rotated synthetic | Sven local | +0.000 pp | −0.014 pp |
| Rotated synthetic | FAISS Lloyd, 2 iters | +0.009 pp | +0.138 pp |
| Rotated synthetic | Meta NumpyEmeans, 2 epochs | **−0.013 pp** | −0.039 pp |
| Skewed Zipf synthetic | Sven global | +1.324 pp | **−1.709 pp** |
| Skewed Zipf synthetic | Sven local | −0.001 pp | −0.829 pp |
| Skewed Zipf synthetic | FAISS Lloyd, 2 iters | +0.929 pp | −2.092 pp |
| Skewed Zipf synthetic | Meta NumpyEmeans, 2 epochs | +0.370 pp | −1.261 pp |

All quoted comparison pairs have **identical serialized FAISS index size** within each 25% replication quota. Compared with no spilling, 25% replication increased serialized index size about **1.246×**. Actual search/latency and index build times are in the raw run artifacts; equal scanned entries is *not* equivalent to equal wall-clock search time or memory.

For a **50%** replication quota on real LSA document vectors, orthogonality-aware spilling gains against the *same-storage nearest-secondary* control were **+2.115 pp (Sven global), +2.126 pp (Sven local), +4.372 pp (FAISS2), +1.463 pp (Meta2)**. Versus the *unreplicated* index, however, the same 50% orthogonal spill delivered respectively **−0.783, −1.202, +1.204, and −0.792 pp** at 10% expected scanned entries. Thus even the larger replicated index usually did not beat non-replicated routing for the Sven or Meta centroids.

### Correctness nuance: distance ties

The initial experiment failed on a strict `full nprobe=k` **ID-based** recall assertion. Some development vectors yield exactly or nearly equal squared distances to queries; the exhaustive FlatL2 and full-list IVF implementations can legally break such ties differently and return distinct valid IDs. To retain a strict geometric correctness check, the benchmark was changed to report **both** raw exact-ID recall and **tie-aware** recall using the exhaustively computed tenth-nearest squared-distance radius plus a documented FP32 tolerance. Full-list **tie-aware recall must be 100%**, or the benchmark fails. The changed source and full CI rerun both pass.

### Falsification / decision

- **Conditional positive:** the SOAR-inspired secondary assignment is more effective than naive nearest-secondary at identical replication and scan work on the real LSA development vectors across all four centroid producers.
- **Important negative:** on real LSA, *three of four* 25%-replicated orthogonal arms have **lower** recall than non-replicated IVF at equal scanned entries; on the skewed synthetic dataset all four regress vs no spill. Some gains against naive spill merely recover harm from naive replication.
- **No general upgrade and no publication novelty:** Google **already** invented orthogonal spilling. The L2 adaptation here does not improve Sven centroid MSE or solve the CPU/GPU training-speed gap to Meta e-means. Leave it as an opt-in research benchmark, **not** a default index layer.
- The genuine next question is how to **selectively purchase coverage of otherwise missed neighbors under an entry/latency budget** using train-only neighbor relationships or a held-out calibration split. Existing boundary-margin selection is weak. Any supervised/retrieval-aware adjustment must be tested on new, never-tuned embeddings and benchmarked against *original ScaNN SOAR*, not only our nearest-secondary control.
