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

For an equal **expected** scan budget `b in {0.05, 0.10, 0.15, 0.25, 0.50, 0.75}`, linearly
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

## Disclosure: additional early operating points

An initial CI run showed that multiple families had near-perfect recall by
25% scanned, making the originally chosen 25/50/75% budget grid nearly
uninformative. We therefore added 5/10/15% budgets to inspect the unsaturated
portion **after observing the initial cost curves**. This is disclosed rather
than framed as preregistered, and the full 1/2/4/8/16/32/64-probe curves are
retained so reviewers can choose their own operating point. We will not
select one favorable budget as universal proof.

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


## Observed paired results — first full-curve CI run

Source: [GitHub Actions 44001747114](https://github.com/Svyable/sven-means/actions/runs/44001747114), with downloadable full per-seed JSON, original training/test dataset SHA-256 hashes, environment metadata and all seven probe-count operating points. Both jobs passed. Each row compares **local-direction revival with our own global-revival default** on 2 independent data seeds × 4 model seeds.

Positive gain means more exact top-10 neighbors are contained in the scanned candidate sets **at the same expected scan fraction**. Interpolated points are expected values of a randomized whole-list probe policy, not deterministic per-query scan limits.

| Dataset | Held-out MSE gain (%) | Candidate recall gain at 5% scan (points) | At 10% scan | At 15% scan |
|---|---:|---:|---:|---:|
| Rotated synthetic | +1.10 | +1.17 | +0.14 | +0.00 |
| Zipf synthetic | +0.62 | +9.10 | +2.27 | −3.16 |
| Real digits (standardized) | −4.44 | +0.10 | −0.06 | +0.29 |
| Real breast-cancer features | −0.03 | **−5.29** | **−4.99** | −2.48 |
| Real 20 Newsgroups LSA documents | **+0.96** | **+1.99** | +0.80 | +0.53 |

Across the real document-vector eight-pair set, local revival improved MSE in **7/8** runs and recall at 5% expected scan in **7/8** runs, with ~**1.21× median training time**. On Zipf, local revival actually improves recall at small scan budgets, despite its poor fixed-four-probe recall, because the local method scans fewer candidates per probe. The advantage reverses at higher budgets. On real breast-cancer feature vectors, the same local policy materially *loses* cost-normalized recall at small scan budgets despite an apparent fixed-probe gain.

### Reset-both confound verified

On Zipf, global reset-both increased fixed-four-probe candidate recall by **+11.82 percentage points**, but increased mean scan fraction by **+5.82 points**. At equal expected scan fraction its benefit shrinks to **+0.52 points** by 25% scan, and ~zero by 50%; at 10% scan its mean effect is **−5.22 points** over six comparable pairs (two others lack a valid whole-list operating point that low). Thus fixed-nprobe gains must not be marketed as performance gains.

### Research and release decision

**Keep cluster-local residual revival opt-in.** The LSA document-vector result is a legitimate positive research observation but not universal: raw real-digit distortion and breast-cancer low-scan recall get worse, and the CPU fit takes ~20% longer. Report full Pareto curves, not a single cherry-picked operating point. Reconfirm on independent pretrained vector embeddings (semantic, visual, multimodal), at multiple k and memory/compute budgets, then benchmark a real FAISS IVF index.

No current evidence justifies claiming that the fork beats Meta e-means across GPU efficiency, billion-scale feasibility, or retrieval quality. The full curve and negative cases are essential parts of the result.
