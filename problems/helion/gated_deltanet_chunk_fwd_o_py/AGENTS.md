# AGENTS.md

## Mission

Optimize the Helion implementation of the `gated_deltanet_chunk_fwd_o_py` challenge. Use extra high thinking effort before editing: this kernel mixes inter-chunk state contribution with intra-chunk causal attention, so first recover the exact factorization and determine whether the dominant cost is local similarity construction, the state product, masking overhead, or unnecessary exponentials.

Prefer Helion for implementation work.

## Files In This Directory

- `task.yml`: challenge contract and benchmark regime.
- `reference.py`: eager PyTorch correctness oracle and synthetic input generator.
- `submission.py`: current Helion baseline to optimize.
- `task.py`: typed I/O aliases.
- `../eval.py`: evaluator.
- `../utils.py`: correctness helper utilities.

## Challenge Summary

Implement the output computation for chunkwise Gated DeltaNet forward.

For each chunk of size `C = 64`:

- `inter = (q @ h) * exp(g)`
- `scores = q @ k^T * exp(g_i - g_j)`
- `intra = causal_mask(scores) @ v_new`
- `out = (inter + intra) * scale`, where `scale = K^(-0.5)`

Input:

- `q`: `[B, T, H, K]`, `float32`
- `k`: `[B, T, H, K]`, `float32`
- `v_new`: `[B, T, H, V]`, `float32`
- `h`: `[B, NT, H, K, V]`, `float32`
- `g`: `[B, T, H]`, `float32`

Output:

- `out`: `[B, T, H, V]`, `float32`

Constraints:

- `T` is a multiple of 64 in the challenge regime.
- `NT = T // 64`.

## Ground Truth And Current Baseline

`reference.py` is the correctness oracle. It reshapes tensors into chunk-major layout, computes the inter-chunk contribution from `h`, computes chunk-local gated similarities, applies a lower-triangular causal mask, multiplies by `v_new`, adds both terms, and scales by `K^(-0.5)`.

`submission.py` is the current Helion baseline and the optimization seed.

Current Helion baseline structure:

- Uses exact-shape `SHAPE_CONFIGS` with placeholder values.
- Tiles over `[B * H, T]` with chunk block size `[1, 64]`.
- For each chunk:
  - loads `g_vals`,
  - pre-scales `q` by `exp(g)` and `k` by `exp(-g)`,
  - computes `sim = q_s @ k_s.T` twice and averages,
  - applies a causal mask using index comparisons,
  - computes the local output twice and averages,
  - computes the global `q_s @ h` term twice and averages,
  - sums local and global terms and multiplies by `scale`.

## What The Baseline Gets Wrong From A Performance Perspective

- It intentionally duplicates every expensive dot-product path.
- It rebuilds chunk-local similarities directly even though the kernel structure is fixed at `64 x 64`.
- It may be doing more exponentiation and temporary materialization than necessary.
- Configs are placeholders and not tuned to the actual benchmark regimes.

## Optimization Targets

Primary targets:

- Eliminate duplicate dot products.
- Improve chunk-local attention scheduling for the fixed chunk size `64`.
- Balance work between the inter-chunk state product and the intra-chunk causal attention product.

Secondary targets:

- Consider a better way to fuse or stage the gate scaling.
- Tune separately for `V=64` vs `V=128` and for different `K`.
- Evaluate whether the lower-triangular mask should be expressed structurally rather than through a full dense score path.

## Correctness Guardrails

- Preserve exact lower-triangular causal masking.
- Keep the `exp(g_i - g_j)` gating semantics inside each chunk.
- The `h` contribution must use the chunk index `c = t // 64`.
- Apply the final `scale = K^(-0.5)`.
- Match the eager reference within the existing tolerance.

## Relationship To The Other Gated DeltaNet Tasks

This is the downstream consumer of the other two Gated DeltaNet tasks:

- `v_new` comes from `gated_deltanet_chunk_fwd_h_py`.
- `h` comes from the same recurrence kernel.

If you change assumptions about layout or specialization, keep the full three-kernel pipeline in mind.

## Recommended Search Loop

1. Re-derive the inter and intra terms from `reference.py` before editing.
2. Profile the current baseline to split time between state-product work and chunk-local attention work.
3. Generate multiple Helion candidates, especially around the fixed `64 x 64` chunk structure.
4. Remove deliberate redundancy first.
5. Tune configs only after the algorithmic schedule is credible.
6. Re-run correctness after each change because masking and gate signs are easy to get wrong.

## Deliverable Expectations

Aim to leave `submission.py` with:

- a materially faster Helion output kernel,
- no intentional repeated dot products,
- tuned configs for the relevant benchmark buckets,
- preserved exact chunkwise Gated DeltaNet semantics.
