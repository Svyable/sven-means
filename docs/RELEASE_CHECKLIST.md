# Release candidate readiness

**Current state:** experimental alpha. The repository is public, but this
workflow does not upload packages to PyPI or create release tags.

## Release gates

- [x] Source attribution: original s-means copyright and MIT license retained
  in [LICENSE](../../LICENSE), with derivative provenance in
  [NOTICE.md](../../NOTICE.md).
- [x] Native wheel and source distribution built in isolated environment;
  `twine check --strict`; LICENSE and NOTICE present in both archives.
- [x] Import version equals package metadata; smoke fit with installed wheel.
- [x] CPU regression tests on Python 3.11 and 3.12.
- [x] Paired benchmark harness with test data held out and archived JSON.
- [ ] Real high-dimensional *pretrained* embedding evaluation, not just
  latent semantic analysis of text documents.
- [ ] GPU native and Triton correctness/performance matrix on actual hardware.
- [ ] End-to-end FAISS IVF recall/latency, not only coarse candidate recall.
- [ ] Independent reproduction and performance comparisons at equal
  quality **and** wall-clock budgets.
- [ ] Choose package version, generate signed release provenance and check
  the exact downloadable artifacts before tagging.

## Public language

A defensible alpha description is: *a research fork of s-means featuring
bounded-workspace seeding, optional constant-memory epoch shuffling,
numerically stable assignments, and experimental local residual revival,
with reproducible CPU experiments.*

Do not advertise “beats Meta”, “universally faster”, “billions of vectors”,
or “zero additional memory”. Those assertions are unproven for this fork.

## How to validate a candidate

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m build
python -m twine check --strict dist/*
python benchmarks/release_candidate.py --dataset all-offline --output /tmp/self.json
```

Review negative and positive datasets, seed-level variance and the availability
of artifacts. Version bumps, release tags, PyPI credentials, and publishing
remain deliberately manual and require explicit authorization.
