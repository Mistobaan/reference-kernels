# AGENTS.md

## Mission

Optimize the Helion implementation of the `gated_deltanet_chunk_fwd_h_py` challenge. Use extra high thinking effort before changing anything: this is the sequential inter-chunk recurrence in Gated DeltaNet, and the hard part is understanding which dimensions can be parallelized without violating the state update semantics.

Prefer Helion unless you hit a real expressivity limit.

## Files In This Directory

- `task.yml`: challenge statement, constraints, tests, benchmarks.
- `reference.py`: eager PyTorch correctness oracle and input generator.
- `submission.py`: current Helion baseline to optimize.
- `task.py`: I/O type aliases.
- `../eval.py`: evaluation harness.
- `../utils.py`: correctness helpers.

## Challenge Summary

Implement the chunkwise hidden-state forward recurrence for Gated DeltaNet.

For each `(b, h)` pair, with chunk size `C = 64` and initial state `h_state = 0`:

1. Store current state into `h_out` for this chunk.
2. Compute corrected values: `v_new = u - w @ h_state`
3. Compute gate factors: `exp(g_last - g_t)` within the chunk
4. Decay the recurrent state by `exp(g_last)`
5. Update state with `k^T @ v_gated`

Input:

- `k`: `[B, T, H, K]`, `float32`
- `w`: `[B, T, H, K]`, `float32`
- `u`: `[B, T, H, V]`, `float32`
- `g`: `[B, T, H]`, `float32`

Output:

- `h`: `[B, NT, H, K, V]`, `float32`
- `v_new`: `[B, T, H, V]`, `float32`

Constraints:

- `T` is a multiple of 64 in the task regime.
- Processing is sequential across chunks, but parallel within `(B, H)` and feature tiles.

## Ground Truth And Current Baseline

`reference.py` is the correctness oracle. It reshapes tensors to chunk-major form, then performs a Python `for c in range(NT)` recurrence that stores the pre-update state, computes `v_new`, forms gated values, decays the running state, and updates via `k^T @ v_gated`.

`submission.py` is the current Helion baseline and the optimization seed.

Current Helion baseline structure:

- Specializes on exact problem shapes with placeholder `SHAPE_CONFIGS`.
- Tiles over `[B * H, V]` with block size `[1, 8]`.
- Keeps a per-tile `state` of shape `[K, tv]` in float32.
- Iterates chunk-by-chunk using `hl.tile(T, block_size=C)`.
- Computes the projection `w @ state` twice and averages it.
- Computes the update `k_adj.T @ diff` twice and averages it.
- Stores `h_out` before each chunk update and writes `v_out` for the tile.

## What Makes This Kernel Hard

- The recurrence across chunks is real and cannot be removed.
- The state has shape `[K, V]`, which can be large enough to stress registers/shared memory and force careful tiling.
- The kernel returns both the per-chunk state snapshots and the per-token corrected values, so write bandwidth matters in addition to recurrence cost.

## What The Baseline Gets Wrong From A Performance Perspective

- It intentionally duplicates both the projection and update dot products.
- The current tiling over only `V` may be too narrow for some `(K, V)` regimes.
- Placeholder configs likely leave occupancy and pipeline depth on the table.
- The kernel may be paying heavily for output writes and state materialization; profile before assuming math is the only bottleneck.

## Optimization Targets

Primary targets:

- Remove duplicate dot products.
- Choose a better state tiling strategy across `K` and `V`.
- Reduce overhead in the chunk loop while preserving exact recurrence semantics.

Secondary targets:

- Consider whether `h_out` storage can be scheduled more efficiently.
- Tune configs separately for `V=64` and `V=128` if the optimal tile differs.
- Verify whether exponentials and gating can be hoisted or reused inside the chunk.

## Correctness Guardrails

- `h_out[:, c]` must contain the state before processing chunk `c`.
- `v_new` must be computed from that pre-update state.
- The update order is not interchangeable: store state, compute `v_new`, apply gate, decay, then update.
- Preserve float32 accumulation behavior and pass the existing tolerances.
- Do not break the public `custom_kernel(data)` API.

## Relationship To The Other Gated DeltaNet Tasks

This kernel sits between the WY recomputation and the output kernel:

- Inputs `w` and `u` come from `gated_deltanet_recompute_w_u_py`.
- Outputs `h` and `v_new` feed `gated_deltanet_chunk_fwd_o_py`.

Pipeline awareness matters. If you optimize this kernel with a layout assumption, verify that it still aligns with the surrounding tasks.

## Recommended Search Loop

1. Read the recurrence in `reference.py` until you can explain the order of operations without looking.
2. Profile the baseline to separate recurrence cost from output-write cost.
3. Propose multiple tiling schemes for the `[K, V]` state.
4. Remove deliberate redundancy first, then autotune.
5. Re-run correctness after every edit because recurrence bugs can look numerically plausible.

## Deliverable Expectations

Aim to leave `submission.py` with:

- a materially better Helion recurrence kernel,
- clearer tiling around the `[K, V]` state,
- tuned configs for the benchmark buckets,
- preserved semantics for both returned outputs.
