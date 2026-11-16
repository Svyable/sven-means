# Feistel epoch shuffling: memory for statistical caution

The upstream s-means CPU and CUDA loops use `torch.randperm(n)` once per epoch. This allocates O(n) **additional integer indices** even if a logical batch is only a few thousand rows.

An affine modular permutation `(a*i+b)%n` is cheap but can preserve undesirable regularities in already-ordered datasets. An experimental alternative uses a six-round balanced [Feistel network](https://en.wikipedia.org/wiki/Feistel_cipher) with keyed round functions and cycle-walking.

## Construction

1. Choose the smallest *even* bit width `w` with `2**w >= n`. Define `M=2**w` and half-width `h=w/2`.
2. Apply six keyed Feistel rounds to a logical position `i`. The result is a permutation of the full `[0,M)` space because a Feistel round is invertible irrespective of the round-function details.
3. If the result is `>=n`, apply the **same** full-domain permutation again until it falls in `[0,n)`. This is cycle-walking.
4. For successive batches, generate logical indices `begin:end` on the fly and apply the mapping.

**Correctness:** Follow any cycle of the full-domain permutation. Its points belonging to `[0,n)` appear in a circular order. Mapping each in-range element to the *next* in-range element along its cycle is a bijection. Thus cycle-walking never repeats or drops an index in one epoch, even for non-power-of-two dataset sizes. Tests enumerate all indices for many odd/composite/power-of-two `n` and compare CPU/GPU on CUDA-capable runners.

**Memory:** O(batch) extra indices per step, rather than O(n). This does not eliminate the need to hold a full dataset in the current CPU/GPU runner. It is groundwork for a future streaming API.

**Statistical limitation:** This **is not** a uniformly random permutation over all `n!` possibilities, unlike `randperm`. Feistel round functions here are intentionally inexpensive and not audited cryptographic primitives. Adversarially ordered datasets might show bias, worse convergence, or poorer test MSE. The cycle walk's expected depth for well-mixed permutations is below 4 because `M/n<4`, but there is no small worst-case bound.

**Speed limitation:** The implementation uses a GPU synchronization per cycle-walking check. Depending on batch size and hardware, ordinary `randperm` can be faster. This mode is opt-in, not a blanket upgrade.

## How to evaluate

```bash
python benchmarks/compare.py --kind gaussian --n 20000 --k 256 --shuffle-mode randperm --seeds 5
python benchmarks/compare.py --kind gaussian --n 20000 --k 256 --shuffle-mode feistel --seeds 5
python benchmarks/compare.py --kind zipf --n 20000 --k 256 --shuffle-mode feistel --seeds 5
```

Compare median fit time, held-out MSE distribution and peak memory; repeat on actual embeddings whose input order may be meaningful. Only promote Feistel to default after this evidence is convincing.
