# Experiment 009 — Does classical Lloyd repair explain the FAISS gap?

**Status:** pre-result, post-pilot diagnostic on two previously observed
neural-vector corpora. No change to sven-means defaults. The update is
standard Lloyd's algorithm, **not invented by Sven**.

## Question

On both 20 Newsgroups and GSM8K, 2-iteration FAISS Lloyd k-means
outperformed the new local-direction Sven method in holdout MSE and
IVF recall while training considerably faster on CPU. What part of
the gap can a *single exact centroid mean update* recover?

## Registered ablation

Take exactly the same trained coarse centroids from:
- global Sven (2 epoch budget)
- experimental local-direction Sven (2 epoch budget, sketch every 4 batches)

Do not re-seed or retrain either Sven model. Apply independently
0,1,2 **full-batch Lloyd mean updates using training assignments**,
keeping any empty centroid unchanged. No held-out query enters the
update. This produces 6 arms (global/local × 0/1/2).

Compare with FAISS's two-iteration and 20-iteration Lloyd k-means
and **Meta FAIR's actual `NumpyEmeans` CPU reference** at two and five
training epochs. Its source is pinned to [facebookresearch/faiss commit
`c8f283867285af547db7a701c1b8dfa89631b3ea`](https://github.com/facebookresearch/faiss/tree/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means).
The CPU study checks out this exact external source into
`faiss-meta/contrib/e_means`, appends it to `faiss.contrib` import
paths, and verifies the real `NumpyEmeans` API. No reimplementation
or cherry-picked claims from the Meta paper.

Use same pretrained Apache-2.0 MiniLM 384D representation, official
train/test corpus splits, two data selections × two centroid seeds
on *development* corpora BANKING77 and 20 Newsgroups.

The metric is **actual FAISS IndexIVFFlat** search recall@10 at equal
expected scanned-vector fractions, paired held-out centroid MSE,
CPU centroid training time including Lloyd repair (measured
separately), list scan curves and per-query search timing.
Correctness unit tests assert *training SSE cannot increase*
after each full Lloyd iteration, within float32 rounding.

This has not been benchmarked on the independent GSM8K third-domain
corpus and is **not** a preregistered pre-GSM hypothesis. These two
development corpora may not be used to claim out-of-domain
generalization after choosing refinement count.

## Interpretation gates

- If even two Lloyd updates fail to close the FAISS2 gap, centroid
  initialization and weighting/centroid allocation deserve scrutiny.
- If one update closes the accuracy/recall gap but makes total CPU
  training much slower than FAISS2 or Meta's 2-epoch NumpyEmeans,
  that is a useful **diagnostic**, not a competitive speed breakthrough.
- The 5-epoch Meta and 20-iteration FAISS arms are *higher-compute*
  references, not equal-epoch/clock head-to-head competitors. Record
  each arm's actual CPU wall time independently.
- Only a genuine Pareto gain at equal wall-clock time on *new* corpora
  and GPU hardware would justify general performance or paper claims.

The metric `expected scanned fraction` is randomized interpolation
between adjacent nprobe values and **not a deterministic per-query cap**.

Reproduce via
```bash
pip install -e ".[dev]"
pip install -r benchmarks/requirements-neural.txt
pytest -q tests/test_lloyd_diagnostic.py
python -m benchmarks.lloyd_diagnostic --dataset banking77 --output /tmp/lloyd-banking.json
python -m benchmarks.lloyd_diagnostic --dataset 20news --output /tmp/lloyd-news.json
```


## Observed direct Meta / FAISS CPU comparison (November 17, 2026 UTC)

[Actions run 44022848657](https://github.com/Svyable/sven-means/actions/runs/44022848657) passed the pinned Meta NumpyEmeans smoke test and the BANKING77/20 Newsgroups real-neural benchmark jobs. Each corpus has two independent document selections × two optimizer seeds (**4 paired fits per arm**). Every arm uses the **same** neural model/data and identical FAISS IndexIVFFlat evaluation, with full per-seed raw JSON and environment versions archived.

The following figures are **relative to Sven global with zero additional Lloyd updates**. A positive percentage means lower held-out SSE; positive percentage points mean better real FAISS recall@10 at the same **expected 10% database vectors scanned**. Time ratios are median total centroid-fit wall times relative to Sven global; all extra Lloyd updates have been charged for their initial assignment, mean update and reassignment.

| Method | 20 Newsgroups MSE gain | 20 Newsgroups recall gain | 20 Newsgroups fit time | BANKING77 MSE gain | BANKING77 recall gain | BANKING77 fit time |
|---|---:|---:|---:|---:|---:|---:|
| Sven global + 1 Lloyd | +1.072% | +5.080 pp | 1.417× | +0.632% | +0.338 pp | 1.557× |
| Sven global + 2 Lloyd | +1.502% | +7.437 pp | 1.785× | +0.833% | +0.504 pp | 1.967× |
| Sven local + 0 Lloyd | +3.278% | +11.769 pp | 1.152× | −0.501% | −0.420 pp | 1.195× |
| Sven local + 1 Lloyd | +3.658% | +12.242 pp | 1.497× | −0.047% | −0.026 pp | 1.677× |
| Sven local + 2 Lloyd | +3.774% | +12.401 pp | 1.795× | +0.295% | +0.156 pp | 2.002× |
| **Meta FAIR NumpyEmeans, 2 epochs** | **+4.489%** | **+13.775 pp** | **0.640×** | −1.464% | −0.220 pp | 0.664× |
| Meta FAIR NumpyEmeans, 5 epochs | +4.431% | +15.478 pp | 1.030× | **+0.919%** | +0.214 pp | 1.131× |
| FAISS Lloyd, 2 iterations | +3.694% | **+16.108 pp** | **0.080×** | −4.609% | −0.782 pp | **0.174×** |
| FAISS Lloyd, 20 iterations | +4.526% | +17.976 pp | 0.463× | −1.284% | +0.008 pp | 1.185× |

### Interpretation

- On 20 Newsgroups, Sven local+2 Lloyd still has **worse MSE and recall** than the *actual* pinned Meta NumpyEmeans at two epochs, while using ~**2.8× Meta's CPU fit time**. FAISS Lloyd at two iterations wins recall at a tiny fraction of Sven's wall clock. **No general superiority claim is supportable.**
- On BANKING77, two Lloyd steps applied to Sven global improve MSE and recall over Sven global, but almost **double total fit time**. Meta five-epoch improves distortion further for less time, while the strongest candidate-recall at 10% scan in this table is Sven global+2 Lloyd, with limited paired trials. This is a tradeoff, not dominance.
- One/two classical Lloyd refinements do help centroid quality somewhat but **do not close the competitive quality–time gap**. The next scientifically valuable target is initialization and training dynamics, not a cleanup-only default.
- The 5-epoch Meta and 20-iteration FAISS arms are intentionally *different-budget reference points*. This table reports measured wall clock, not an assertion that one epoch equals one Lloyd iteration or that cross-language implementation timings will scale similarly to GPU.
- These two corpora have already been used for exploratory work and cannot be reused as a fresh confirmatory dataset. The previous GSM8K corpus has also now been examined. Publication requires a new pretrained-embedding domain, adequate uncertainty estimates, pinned-source Meta comparison at equal time, and independently tested GPU memory/speed.

### Full provenance

Meta FAIR source revision `c8f283867285af547db7a701c1b8dfa89631b3ea` is sparse-checked-out into the test job and imported as **Meta's own** `NumpyEmeans`; no equivalent was reimplemented. Frozen MiniLM SHA `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and BANKING77 SHA `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`. Each Actions artifact records complete per-seed observations, including per-nprobe search time and scanned candidate fractions.
