# AGENTS.md

## Mission

Optimize the Helion implementation of the `gated_deltanet_recompute_w_u_py` challenge. Use extra high thinking effort before editing: this kernel is part of a larger Gated DeltaNet forward pipeline, so first recover the exact per-chunk math, tensor reshaping, and chunk-local dependency structure.

Prefer Helion for new implementations.

## Files In This Directory

- `task.yml`: formal challenge statement and benchmark regime.
- `reference.py`: eager PyTorch correctness oracle plus synthetic input generation.
- `submission.py`: current Helion baseline to improve.
- `task.py`: typed I/O aliases.
- `../eval.py`: evaluator.
- `../utils.py`: comparison helpers.

## Challenge Summary

Implement the forward recomputation of WY-transformed keys and values for Gated DeltaNet.

For each chunk of size `BT = 64`:

- `u = A @ (v * beta[..., None])`
- `w = A @ (k * beta[..., None] * exp(g)[..., None])`

Input:

- `k`: `[B, T, H, K]`, `float32`
- `v`: `[B, T, H, V]`, `float32`
- `beta`: `[B, T, H]`, `float32`
- `A`: `[B, T, H, BT]`, `float32`
- `g`: `[B, T, H]`, `float32`

Output:

- `w`: `[B, T, H, K]`, `float32`
- `u`: `[B, T, H, V]`, `float32`

Constraints:

- `T` is a multiple of 64 in the challenge regime.
- `A.shape[-1]` is the chunk size.

Important benchmark regimes:

- `(B, T, H, K, V)` includes long-sequence cases such as `(2, 1024, 3, 64, 64)`.
- Extra hardcoded shapes also exist in the baseline config map beyond the official task set.

## Ground Truth And Current Baseline

`reference.py` is the correctness oracle. It reshapes tensors into chunk-major layouts, performs chunk-local batched matrix products for `u_c` and `w_c`, then reshapes back.

`submission.py` is the current Helion baseline and should be treated as the optimization seed.

Current Helion baseline structure:

- Specializes on exact `(B, T, H, K, V)` shapes via `SHAPE_CONFIGS`.
- Tiles over `[B * H, T]` with block size `[1, C]`, where `C = A.shape[-1]`.
- For each chunk tile, initializes `w_acc1/u_acc1` and `w_acc2/u_acc2`.
- Iterates once from `0..C-1` and again from `C-1..0`, doing the same accumulation work twice, then averages the results.
- Loads one chunk column `a_col`, one scalar `beta`, one scalar `exp(g)`, and one full `k_ci`/`v_ci` vector per inner step.

## What The Baseline Gets Wrong From A Performance Perspective

- It intentionally doubles the entire accumulation path.
- It appears to use scalar-by-vector outer-product style accumulation instead of a more direct matrix-oriented chunk computation.
- The current schedule may underutilize reuse of `A`, `beta`, and `exp(g)` within a chunk.
- Configs are placeholder values and not tuned.

## Optimization Targets

Primary targets:

- Remove duplicated forward and reverse accumulation loops.
- Express the chunk computation in a way that better matches the real math and available Helion dot primitives.
- Improve reuse of chunk-local data and reduce redundant exponentials or scalar loads.

Secondary targets:

- Consider precombining `beta` and `exp(g)` within the kernel if it reduces repeated work.
- Tune shapes separately for `V=64` and `V=128` if needed.
- Reassess whether the best decomposition is over `(B, H, chunk)` or over output feature tiles.

## Correctness Guardrails

- Preserve exact chunk-local semantics with chunk size 64.
- `w` and `u` must match the eager reference to `rtol=1e-2`, `atol=1e-2`.
- Do not change input generation or public API.
- Be careful about the meaning of `g`: the generated inputs already pass cumulative chunk-local gate values into this kernel.

## Relationship To The Other Gated DeltaNet Tasks

This kernel feeds the `chunk_fwd_h` task. Improvements here should respect that downstream consumer:

- `w` is later multiplied against the running hidden state.
- `u` becomes the chunk-local corrected value source before recurrence.

Understanding the full three-kernel pipeline helps avoid optimizing the wrong abstraction.

## Recommended Search Loop

1. Read `reference.py` carefully and reconstruct the chunk-major layout transformations.
2. Profile the current baseline on at least one short and one long sequence benchmark shape.
3. Classify whether runtime is dominated by global memory bandwidth, exponentiation overhead, or poor chunk GEMV scheduling.
4. Generate several Helion rewrites, not just config tweaks.
5. Keep algorithmic rewrites and autotuning in separate commits or artifacts.
6. Re-run correctness after every rewrite.

## Deliverable Expectations

Aim to leave `submission.py` with:

- a cleaner chunk-local Helion implementation of the WY transform,
- no deliberate duplicate work,
- tuned configs for the important benchmark buckets,
- comments only where the chunk scheduling or data layout is subtle.
