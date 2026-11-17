# Experiment 004 — recall versus candidate vectors scanned

**Preregistered comparison** following a confound discovered in
[Experiment 003](donor-retention.md). Code does not change any
clustering default, centroid update, or split strategy.

## Why fixed nprobe is insufficient

At fixed `nprobe=4`, a query over an uneven partition can visit much larger
centroid lists than a query over a balanced partition. The reported +11.823
percentage-point Zipf candidate-recall gain for reset-both came with a
+0.990 increase in the coefficient of variation of list sizes. That
suggests an unfair search-work comparison, not necessarily a real
retrieval improvement.

## Definitions

For a query, sort centroids by L2 distance. For each `nprobe`, count **all
training vectors in the visited lists**. Divide by total training size to
get `scan_fraction`. Also count how many of the query's true exact
top-10 nearest training vectors lie in those lists; this is
`candidate_recall@10`, equivalent to ideal exact re-ranking of those
candidates ignoring search overhead.

Compute mean candidate recall and mean scan fraction across 96 independent
held-out queries, for `nprobe in {1,2,4,8,16,32,64}`.

For an equal **expected** scan budget `b in {0.25, 0.50, 0.75}`, linearly
interpolate between neighboring measured operating points. This has a
clear operational meaning: randomly choose the upper or lower probe count
with a probability yielding the desired *average* scanned fraction.
It is **not** a deterministic per-query scan cap, nor a measurement of
FAISS implementation time. If a budget is below a model's smallest
measured cost, return unavailable and **do not extrapolate**.

## Comparisons

- Global baseline (current production default).
- Local-direction revival, with 4-batch sketch cadence.
- Global reset-both final-pass history.
- Local-direction reset-both final-pass history.

Training and evaluation protocols follow the exact held-out data,
model seeds, and train-only document-vector preprocessing used in
[Experiment 002](split-pass-invalidation.md); two independent data seeds,
four model seeds per dataset (8 paired comparisons).
Synthetic rotated and Zipf, real digits and breast-cancer feature data,
plus 20 Newsgroups TF-IDF/SVD document vectors.

For each model we also publish held-out distortion and fit time.
A positive fixed-nprobe recall gain is not a success if it requires
substantially more candidate scans. The primary outcome is
**candidate recall at equal expected scanned fractions**, separately
from test MSE and CPU fit latency.

## Decision criteria

No general retrieval claim unless gains persist on multiple real
datasets across scan budgets, with no unacceptable distortion or
training-time regression. Real FAISS IVF index timing, cache effects,
SIMD and storage locality are still missing; this is a cost-normalized
algorithmic proxy, not an end-to-end deployment benchmark.

Reproduction:
```bash
python -m pytest -q tests/test_scan_budget.py
python benchmarks/scan_budget_study.py --dataset all-offline --output /tmp/scan-offline.json
python benchmarks/scan_budget_study.py --dataset news_lsa --output /tmp/scan-documents.json
```
