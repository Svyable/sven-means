# Changelog

## Unreleased — experimental alpha

- Added split-aware final-pass accumulator invalidation as an opt-in A/B
  candidate (evaluated separately; default preserved until verified).
- Added cluster-local covariance-vector residual revival as an opt-in CPU
  experiment, with a measured 4-batch update cadence and independent
  radius-policy ablations.
- Added bounded-workspace coreset initialization, large-offset-stable
  distance inference, and optional full-data centering.
- Added affine and Feistel epoch traversal options that avoid a full-length
  permutation index tensor, leaving uniform random shuffle as default.
- Added extensive CPU tests, reproducible experimental workflows and
  attribution notices.
- Aligned public package version and normalized PEP 639 MIT license metadata.
- Added source distribution / wheel validation and release criteria.

See `docs/experiments` for exact protocols, datasets, complete CI artifacts
and limitations. No production-quality or GPU superiority claim is made.
