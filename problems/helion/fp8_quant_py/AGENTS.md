# AGENTS.md

## Mission

Optimize the Helion implementation of the `fp8_quant_py` challenge through a profile-guided search loop. Use extra high thinking effort before editing: understand the quantization contract, the grouping layout, and whether the kernel is dominated by reduction cost, memory traffic, or poor work decomposition.

Prefer Helion for all new kernel work.

## Files In This Directory

- `task.yml`: challenge definition, test shapes, benchmark shapes, evaluation policy.
- `reference.py`: eager PyTorch correctness oracle.
- `submission.py`: current Helion baseline to optimize.
- `task.py`: typed I/O aliases.
- `../eval.py`: evaluation harness.
- `../utils.py`: correctness helpers.

## Challenge Summary

Implement per-token-group FP8 E4M3 quantization.

For each contiguous group of `group_size` values:

1. `absmax = max(abs(x_group))`
2. `scale = max(absmax, eps) / 448.0`
3. `x_q = clamp(x_group / scale, -448.0, 448.0)`

The evaluator uses float32 outputs for compatibility:

- `x_q`: quantized values stored as float32
- `x_s`: per-group scales stored as float32

Input:

- `x`: `[num_tokens, hidden_dim]`, `float32`
- `x_q`: preallocated `[num_tokens, hidden_dim]`, `float32`
- `x_s`: preallocated `[num_tokens, hidden_dim // group_size]`, `float32`

Output:

- `(x_q, x_s)`

Important regimes from `task.yml`:

- Test shapes include tiny to moderate token counts with hidden sizes up to 4096.
- Benchmarks include `(1, 4096, 128)`, `(16, 4096, 128)`, and `(256, 4096, 128)`.
- Group size is fixed at 64 or 128 in the provided shapes.

## Ground Truth And Current Baseline

`reference.py` is the correctness oracle. It reshapes the input to `[num_tokens, num_groups, group_size]`, computes a per-group absmax reduction, forms scales, quantizes, and writes into the provided output buffers.

`submission.py` is the current Helion baseline and the actual optimization starting point.

Current Helion baseline structure:

- Flattens `[num_tokens, hidden_dim]` into `N = num_tokens * num_groups` rows of width `group_size`.
- Runs one Helion row kernel over `[N, group_size]`.
- Computes `abs(row)` and `amax(row)` three times, averages the identical results, then computes quantized outputs three times and averages those identical results.
- Writes the quantized row to `qout` and the single scale to `scales_out`.
- Relies on `custom_kernel` to reshape results back into `x_q` and `x_s`.

## What The Baseline Gets Wrong From A Performance Perspective

- It intentionally triples the reduction and quantization work.
- It does not obviously exploit vector width or a reduction-friendly tile for `group_size` 64/128.
- Configs are placeholders and not tuned for the benchmark regimes.
- The Python-side flatten/reshape is acceptable semantically but should still be evaluated for overhead relative to the kernel runtime.

## Optimization Targets

Primary targets:

- Collapse the triple-redundant reduction and quantization path into one pass.
- Tune the row/group kernel shape for `group_size` 64 and 128.
- Improve reduction efficiency and memory coalescing.

Secondary targets:

- Check whether the best schedule differs between latency-oriented small `num_tokens` and throughput-oriented large `num_tokens`.
- Consider specialization by `group_size`.
- Preserve the preallocated-output contract without adding extra allocations.

## Correctness Guardrails

- `x_q` and `x_s` must be written into the provided output buffers and returned.
- Scale must use `max(absmax, 1e-10) / 448.0`.
- Quantized values must respect the FP8 range contract `[-448.0, 448.0]`.
- Keep output dtypes as float32.
- Pass the separate value and scale checks in `reference.py`.

## Recommended Search Loop

1. Reconstruct the row-major flattening used by `custom_kernel`.
2. Profile the baseline separately on a single-token and many-token benchmark shape.
3. Decide whether the bottleneck is the per-row reduction, output write bandwidth, or launch/config overhead.
4. Generate a few Helion candidates with different reduction schedules.
5. Tune config values only after removing obvious redundant work.
6. Re-run correctness after every edit because scale mistakes are easy to hide.

## Deliverable Expectations

Aim to leave `submission.py` with:

- a single-pass Helion quantization kernel,
- tuned per-shape or per-bucket configs,
- unchanged public API and buffer semantics,
- no intentional redundant computations.
