# Changelog

## 2026-03-14

- Replaced the duplicated scalar accumulation in `submission.py` with chunk-local Helion `hl.dot(...)` calls over the full WY tile for both `w` and `u`.
- Precomputed per-chunk scaled key and value tiles once per block instead of recomputing per column.
- Updated the static kernel config to `num_warps=8` and `num_stages=2`.
- Verified correctness with `python eval.py test gated_deltanet_recompute_w_u_py` passing all provided tests.
- Improved the local benchmark from `0.1741 / 0.1576 / 0.1583 ms` to `0.0080 / 0.0086 / 0.0093 ms` on the benchmark shapes in `task.yml`.
