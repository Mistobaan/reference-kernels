# AGENTS.md

## Mission

Optimize the Helion implementation of the `causal_conv1d_py` challenge in an iterative profile-guided loop. Use extra high thinking effort before changing the kernel: first reconstruct the exact math, dataflow, and shape regime, then decide whether the bottleneck is memory traffic, instruction redundancy, launch configuration, or an avoidable layout/padding cost.

MUST use Helion as the authoring backend.

## Files In This Directory

- `task.yml`: challenge statement, correctness contract, test shapes, benchmark shapes, timeouts.
- `reference.py`: eager PyTorch ground truth used for correctness checking.
- `submission.py`: current Helion baseline to improve.
- `task.py`: type aliases for the challenge I/O.
- `../eval.py`: evaluator used by the harness.
- `../utils.py`: correctness helpers.

## Challenge Summary

Implement a causal depthwise 1D convolution:

`out[b, d, t] = bias[d] + sum_{k=0}^{W-1} weight[d, k] * x[b, d, t - W + 1 + k]`

with left zero-padding and independent per-channel filters.

Input:

- `x`: `[B, D, S]`, `float32`
- `weight`: `[D, W]`, `float32`
- `bias`: `[D]`, `float32`

Output:

- `out`: `[B, D, S]`, `float32`

Important regimes from `task.yml`:

- Tests include small and medium shapes such as `(B, D, S, W) = (1, 64, 64, 4)` and `(1, 128, 64, 8)`.
- Benchmarks focus on large-channel long-sequence cases such as `(1, 768, 2048, 4)` and `(1, 1536, 2048, 4)`.
- `W` is small in the provided shapes. Exploit that.

## Ground Truth And Current Baseline

`reference.py` is the correctness oracle. It pads `x` on the left with `W - 1` zeros and calls grouped `torch.nn.functional.conv1d` with `groups=D`.

`submission.py` is the current Helion baseline. Treat it as the optimization starting point, not as the correctness oracle.

Current Helion baseline structure:

- Chooses a hardcoded `helion.Config` per exact shape via `SHAPE_CONFIGS`.
- Builds an explicitly padded tensor in Python with `torch.cat`.
- Launches a kernel over tiles of `[B, D, N]` with block size `[1, None, None]`.
- For each output position, loops over `j in range(W)` and repeats the same load/multiply/accumulate path three times into `acc1`, `acc2`, `acc3`, then averages them.
- Adds bias after accumulation and writes one batch slice at a time.

## What The Baseline Gets Wrong From A Performance Perspective

- It intentionally performs the same work three times.
- It materializes padded input instead of handling causal bounds in-kernel.
- It does not exploit that the operation is depthwise with tiny filter width.
- The block configuration placeholders are not tuned.
- The current schedule does not show any explicit attempt to improve locality across sequence positions or channels.

These are expected inefficiencies, not bugs.

## Optimization Targets

Primary targets:

- Remove redundant work completely.
- Reduce global memory traffic.
- Choose a tile shape that fits the provided benchmark regimes.
- Keep a stable specialization story for small `W`.

Secondary targets:

- Consider in-kernel causal masking instead of explicit host-side padding if it reduces traffic and launch overhead.
- Consider specializing separate kernel paths for common `W` values in the task set.
- Tune `block_sizes`, `num_warps`, `num_stages`, and optionally an `advanced_controls_file`.

## Correctness Guardrails

- Preserve exact causal semantics: no output may depend on future timesteps.
- Output shape must remain `[B, D, S]`.
- Keep `float32` behavior and pass the existing `rtol=1e-2`, `atol=1e-2` check.
- Do not change the public entry point `custom_kernel(data)`.
- If you remove explicit padding, verify boundary handling carefully for `t < W - 1`.

## Recommended Search Loop

1. Read `task.yml`, `reference.py`, and `submission.py` together before editing.
2. Profile the current Helion baseline on at least one short-sequence and one long-sequence benchmark shape.
3. Classify the bottleneck: memory-bound, compute-bound, launch-bound, or shape-config limited.
4. Generate multiple Helion candidates before choosing one direction.
5. Keep per-shape config changes separate from algorithmic changes so gains are attributable.
6. Re-run correctness after every kernel edit.
7. Benchmark only after correctness is restored.

## Deliverable Expectations

Aim to leave `submission.py` with:

- a materially faster Helion kernel,
- cleaned-up kernel logic without intentional redundancy,
- updated `SHAPE_CONFIGS` for the challenge shapes you tuned,
- concise comments only where the scheduling choice is non-obvious.
