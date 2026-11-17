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
