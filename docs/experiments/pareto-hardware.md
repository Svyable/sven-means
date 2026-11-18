# Experiment 012 — streaming proxy-grid selection against physical hardware measurements

**Status: preregistered pilot before seeing empirical outcomes, 2026-11-18.**
This is not Exstill's 5 nm design data and not an Exstill token-energy claim.

## Source and immutable data provenance

Use **NATS-Bench-MCU v2**, by Sebastian Zimmermann, René Groh, and
Andreas M. Kist, Zenodo DOI
[10.5281/zenodo.21644470](https://doi.org/10.5281/zenodo.21644470),
licensed **CC BY 4.0**. 14,281 neural-cell designs were successfully
deployed to an nRF5340 microcontroller and measured on actual hardware;
1,344 additional designs failed a flash feasibility precheck. The latter
have no measurement-based latency/energy values, so this pilot evaluates
selection **within the successfully profiled design universe only**.
It does NOT report discovering designs across all 15,625 possibilities.

The corrected hardware CSV and separately sourced NATS-Bench *test*
accuracy CSV are downloaded by CI from the authors' public GitHub
repository at a **fixed source SHA**:
c92adc0ec112bdee6c73358cc52f3e026367b5a0.
The full JSON archive records both CSV SHA-256 hashes and selected
architecture IDs. No third-party dataset bytes are committed into Sven.
The source contains full deployment provenance, original paper and CSV
column definitions. We cannot infer Exstill's actual 5nm search
landscape from a microcontroller-cell topology benchmark.

## Information and "oracle" contract

Cheap, available BEFORE an expensive evaluation:
- six categorical NATS-Bench cell-edge operations from arch_str;
- the converted INT8 flatbuffer's size in KB (before physical testing).
This is encoded as 30 one-hot edge-operation columns plus log1p(size).
First-edge operation defines five disjoint architecture families.

**Withheld until paid for (or until final scoring):**
- NATS-Bench CIFAR-10 test accuracy (%), used here as an expensive
  label. No policy sees it except on paid calibration designs.
- Actual measured on-device inference energy (mJ) and latency (s).
- Measured deployment ROM (KB); preconversion INT8 model size is
  cheap but final ROM is withheld from the policies.

For all policies, 64 randomly selected initial expensive oracle queries
are reserved for a *shared* three-output linear ridge surrogate
(lambda=10, no tuning on held-out outcomes). Its targets are log
latency, log energy, and raw CIFAR-10 accuracy. It uses only those
64 paid labels; no other true outcomes enter predictions, candidate
selection or ranking. Full code includes a **counterfactual no-leak test**:
alter every unpaid label and selected candidate IDs must remain identical.

This is **offline replay** over a finished public dataset. New expensive
hardware measurements were NOT actually run. The evaluation budget is
a count of virtual oracle lookups; report computation time separately,
never call it physical-chip evaluation time saved.

## Objective, constraints and evaluation

Strict feasibility on the held-out truth:
- CIFAR-10 test accuracy >= 90%;
- measured deployment ROM <= 900 KB.

For feasible designs minimize **3 measured hardware objectives**:
latency seconds, energy mJ, deployed ROM KB. Actual energy and latency
are correlated, but not interchangeable, and ROM adds a third axis.

Exact true 3D Pareto masks computed with a Fenwick sweep; exact
3D dominated hypervolume computed by 2D sweep slices, benchmarked
against exhaustive finite-grid examples. Reference point is the full
measured feasible universe's per-axis maximum normalized to 1, plus
0.05, and lower bound per-axis minimum. These bounds are used **only
after selection for evaluation**, never by the policies.

Outcomes: normalized hypervolume captured, exact frontier hit-rate,
5% epsilon-front coverage in the metric-normalized three dimensions,
best feasible energy-delay product relative to global feasible best,
number of feasible designs found, actual query count and CPU selection
wall time including the common surrogate training overhead.

## Comparators, all with the same expensive-query budget

1. Uniform random (strong budget-matched reference).
2. Family-stratified random (round-robin five topology families).
3. Farthest-first k-center in cheap 31D structure features.
4. K-means++ D^2 seeding in the same cheap features.
5. Cheap-surrogate scalarized accuracy/energy/latency score ordering
   (strong proxy exploitation baseline).
6. **Experimental bounded streaming proxy-grid**, allocating at most B
   representative cells and B overflow candidates, each from
   predicted objectives and cheap architecture-family ID only. On
   cell collisions retain the better cheap-score candidate, evict
   lowest-score cells when full. There is no oracle-guided selection.

Primary all-family experiment: n=14,281 deployed, three random
calibration and selection seeds **31/51/73**, 64 paid calibration
architectures each, total budgets **1%, 5%, 10%** of the candidate
universe (including the 64). These are 143/715/1429 actual queries
after rounding. Results across seeds are reported separately.

**Out-of-family stress test:** choose the first-edge
nor_conv_3x3 cell family as the candidate universe. Train the
surrogate using 64 paid labels **exclusively from the other four
families**, and charge them to the candidate search budget even
though they cannot contribute to within-family Pareto recovery.
Two preselected seeds **31/51** and 5%, 10%, 20% total-query fractions
of that family. This deliberately tests extrapolation; selection is
allowed to see cheap family ID and topology, not unseen measured
results. Do not replace family choice after seeing the outcome.

## Conditional proxy-covering lemma, not a new theory claim

Suppose a three-objective normalized **proxy** cell has side h and
the archive has sufficient capacity to retain at least one
representative from *every occupied cell* without eviction. If for
all candidates a uniform bound ||proxy - true||_infinity <= epsilon
actually holds, then for every candidate c there is a retained
representative r with

    ||true(c) - true(r)||_infinity < h + 2 * epsilon.

Proof: same-cell proxy distance < h coordinatewise; apply the
triangle inequality to c's proxy error, proxy difference, r's error.

This is **conditional**, supplies *no* feasibility-margin or
Pareto-optimality guarantee by itself and is known epsilon-grid
archive-style reasoning. Our fixed-B archive generally **does evict
cells**; the empirical model has *no certified uniform error bound*.
Consequently **none of the paper claims get a formal guarantee from
this lemma in the tested capacity-limited setting**. We include it
only to expose assumptions and prevent false guarantees.

Prior art includes Laumanns, Thiele, Deb, Zitzler,
[Combining convergence and diversity in evolutionary multiobjective
optimization](https://pubmed.ncbi.nlm.nih.gov/12227996/) (2002),
ε-dominance archiving; standard k-center/farthest-first, Arthur &
Vassilvitskii k-means++ (2007), and multiobjective Bayesian
optimization / Pareto-aware active learning. This experiment is NOT
an invention claim for grid archives or Pareto-preserving selection.

## Promotion and publication rule

- No apparent advantage if performance is indistinguishable from
  random, farthest-first, kmeans++, or exploitation baseline at the
  same total query count, or if advantage disappears in family-holdout.
- No claims of "saving evaluations" without actually measuring the
  true bottleneck, whole search overhead, and expensive evaluation
  accuracy on the target hardware/model system.
- No Exstill chip energy, fabricated power, latency or area claims:
  5nm co-design is distinct from NATS-Bench-MCU's Nordic nRF5340.
- No statistical generalization claim from one hardware platform,
  one architecture search space, or three seeds. This is a measured
  pilot to justify (or reject) larger multicorpus work.
- A meaningful next step would require stronger existing
  multiobjective active-learning baselines, held-out architecture
  families, more hardware platforms and explicit expensive
  quality/simulation cost measurement. Ordinary clustering SSE is
  *not* the objective.

Reproduce:
    pip install numpy pytest
    python -m pytest -q tests/test_pareto_archive.py tests/test_pareto_hardware.py
    python -m benchmarks.pareto_hardware_study --hardware /path/to/hw_metrics.csv --accuracies /path/to/accuracies.csv --output /tmp/hw.json

Full source data can be obtained from:
https://github.com/ankilab/NATS-Bench-MCU/tree/c92adc0ec112bdee6c73358cc52f3e026367b5a0/export_corrected
