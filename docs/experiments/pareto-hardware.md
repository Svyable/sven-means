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


## Post-first-run methodological correction and expanded follow-up

**Disclosure: the following changes were made AFTER inspecting the
first CI run, not part of the initial preregistered design.**

The initial run [44318746830](https://github.com/Svyable/sven-means/actions/runs/44318746830)
passed all correctness and real-hardware jobs. The first 14,281-design
evaluation demonstrated that the globally feasible strict 3D Pareto
front contains only **four** designs, and the preselected
nor_conv_3x3 family contains only **three**. Hypervolume is thus a
highly low-cardinality statistic; do not interpret small changes
across 2–3 seeds as robust evidence of a new method.

The initial code independently seeded each method as
seed + 37 * evaluation_budget. This unintentionally made
random/stratified/diversity policies choose unrelated sets at
1/5/10% budgets, producing mathematically confusing
non-monotone hypervolume (e.g. random global mean 0.9113
at 5% vs 0.8712 at 10%). **This is not a failure of hypervolume**;
it reflects entirely different random subsets. The corrected
protocol uses the SAME method seed at every budget. Classical
random/stratified/farthest-first/k-means++/surrogate-rank policies
are now **nested by construction**. A runtime invariant rejects
a run if their *true oracle* hypervolume declines with added
evaluations. The fixed-capacity proxy-grid archive is **not**
nested because resizing may evict cells; report nonmonotonic
results honestly if any.

No selection hyperparameter, model ridge penalty, feasibility
limit, objective, data-split seed, or candidate score was changed
after the initial outcomes. All estimates from the first run
must be labeled **superseded for budget-curve comparisons**.

Because one prechosen held-out family has just 3 true frontier
points, the research follow-up is expanded to test **ALL FIVE**
first-edge operation families as separate extrapolation tests:
none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3.
Two fixed seeds 31/51 and 5/10/20% total budgets for each,
with calibration labels exclusively from the other four
families. This all-family follow-up was chosen *after* seeing the
first family; it is more resistant to cherry picking but is NOT
a fully independent pre-result confirmation of the algorithm.

### Decisions to take from the corrected run

- If the grid archive does not exceed simple proxy-ranked or
  farthest-first selection at equal expensive-query budgets across
  the family sweep, state clearly **no breakthrough**.
- The same 3–4-point strict Pareto frontier may make the entire
  dataset unsuitable for detecting subtle frontier preservation.
  A future study must choose another benchmark based on *a
  registered independent problem*, not relax thresholds and
  relabel the current data as fresh.
- For a real science contribution, tie benefits to **expensive
  evaluation reduction**, not cheap CSV lookup runtime; record
  strongest multiobjective active-learning baselines, uncertainty
  calibration, and family-disjoint generalization.


## Corrected measured results — November 18, 2026 UTC

[**Green run 44319398154**](https://github.com/Svyable/sven-means/actions/runs/44319398154)
executed the corrected nested-budget protocol. **Seven jobs passed**:
independent correctness checks, full 14,281-design replay, and all
five externally held-out architecture-family extrapolations. Raw
architecture ID selections, individual seed metrics, source SHA-256
and wall-clock costs are in separate GitHub Actions artifacts.

Pinned authors' source revision:
c92adc0ec112bdee6c73358cc52f3e026367b5a0.

- hw_metrics.csv SHA256:
  ae318c606d6095ec31ac7383c20e330a6b1e850c8430f6ba20974e00c5d4f693
- accuracies.csv SHA256:
  948fa57fa643dbf749dc394f9f38ab20639737de1a2054d5a72e1be15dd76fae

### Full 14,281 deployed architectures

The strict feasible 3D Pareto front has **four** designs. The
table is mean exact 3D hypervolume recovered from three independent
policy/calibration seeds (31/51/73), with 64 shared calibration
queries INCLUDED in every budget. Larger is better:

| Policy | 143 evaluations (1%) | 715 (5%) | 1429 (10%) |
|---|---:|---:|---:|
| Random | 0.7152 | 0.9082 | 0.9107 |
| Family-stratified random | 0.7815 | 0.8290 | **0.9304** |
| Farthest-first | 0.8006 | 0.8720 | 0.9072 |
| K-means++ D² | 0.7563 | 0.8802 | 0.8820 |
| Surrogate-only exploitation rank | **0.8251** | **0.9186** | 0.9186 |
| Bounded streaming proxy-grid archive | 0.8217 | 0.8730 | 0.9186 |

**No persistent superiority** for the proposed grid. At 1% the
simple exploitation baseline slightly wins; at 5% the grid loses
to exploitation and *random*; at 10% stratified random exceeds both
grid and exploitation. All baseline policies except capacity-specific
grid produce nondecreasing hypervolume by construction and explicit
test as their budget grows.

At 1429 total queries, the measured mean *selection-only* CPU time
for grid was **~2138 milliseconds**, compared with **~5 ms** for
surrogate rank and ~1027 ms for farthest-first. Shared initial
surrogate fitting is reported separately in JSON. These are
small-CPU software costs, **not hardware deployment times**.

### Five held-out first-edge topology families

Surrogate was trained using 64 **paid** examples from the other four
topology families, and each row below is selected exclusively from
the held-out family. Two fixed seeds 31/51, 10% of the held-out
family as the total query budget, including those 64 evaluations.
Strict feasible true front size varies by family: generally **2–3
architectures**, further reducing statistical power.

| Unseen family | Random | Farthest-first | Surrogate rank | Streaming grid |
|---|---:|---:|---:|---:|
| none | 0.7627 | **0.8482** | 0.7639 | 0.7614 |
| skip_connect | 0.7193 | 0.7117 | **0.9799** | 0.9655 |
| nor_conv_1x1 | 0.7518 | **0.9362** | 0.8400 | 0.8401 |
| nor_conv_3x3 | 0.5136 | 0.8904 | 0.7284 | **0.8941** |
| avg_pool_3x3 | 0.8284 | 0.8416 | **0.9583** | 0.9551 |

At the 10% held-out budget the grid exceeds random in **three or
four** families (depending on rounding of the none family: it is
0.7614 versus random 0.7627, so **four of five**).
It exceeds farthest-first on **three of five**.
However it exceeds the **best of the strong baselines in only one
of five** (nor_conv_3x3, with a tiny 0.0037 absolute hypervolume
difference over farthest-first). In three families either
farthest-first or surrogate-only selection wins by a larger margin;
in the fifth surrogate-only slightly wins.

All family-level raw results include 5%, 10%, 20% budgets, exact
Pareto-front hit rates, epsilon coverage, discovered feasible
architecture counts and time. The broad family sweep was explicitly
post-first-run: not independently preregistered confirmation.

### Interpretation, prior art and decision

These are **measurements on real silicon** supplied by the
NATS-Bench-MCU authors, reused in an **offline replay**. We have
not fabricated a chip, run Exstill's proprietary design model,
simulated the 5nm gate-level co-design primitives, trained a new
architecture or performed any new physical deployments.

The grid's archive state is capacity bounded, and its conditional
epsilon-cell lemma is valid only under conditions not established
in this experiment. Grid archiving and epsilon-dominance are
well-established prior art (Laumanns et al., 2002), and this
study does not justify a new algorithmic priority claim.

**Outcome:** do NOT present the streaming grid as a general
breakthrough or ship it as an Exstill integration. The experiment
does provide a **reproducible leakage-audited evaluation harness**,
source-pinned physically measured objectives, exact 3D hypervolume,
budget-charged calibration, tough cheap-feature baselines and
family-disjoint extrapolation — useful infrastructure for subsequent
rigorous proposals.

A follow-up would need:
1. A design-space benchmark with a *larger, meaningfully distributed*
   true feasible frontier, without post-hoc loosening of this dataset's
   thresholds and calling it independent.
2. Sequential multiobjective active-learning / Bayesian search
   baselines including expensive-evaluation cost, uncertainty
   calibration, candidate-family generalization, and true workload
   evaluation costs.
3. A genuinely **new mechanism** that outperforms those controls
   across multiple hardware platforms and seed/data splits, with
   theoretical qualifications correctly bounded.
4. A direct applicability test to Exstill's actual bottleneck; its
   10M costed designs are **not** 10M full routed circuit simulations.

**Publication decision:** no new general performance or theoretical
breakthrough claim based on Experiment 012.
