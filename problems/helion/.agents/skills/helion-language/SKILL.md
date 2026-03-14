---
name: helion-language
description: Use this skill when authoring, reviewing, debugging, or porting Helion kernels in this repository, especially for `@helion.kernel`, `helion.language as hl`, autotuning/configs, jagged or masked indexing, matmul or attention kernels, and AOT autotuning workflows.
---

# Helion Language

Helion in this repository is best approached as "PyTorch with tiles". Start from
the closest example in `examples/`, preserve the existing style, and only reach
for lower-level Helion primitives when direct tile indexing is not enough.

## First Pass

1. Read only the one or two reference files below that match the task.
2. Keep host work outside `hl.tile` loops: shape math, asserts, output
   allocation, closures, and `torch.autograd.Function` wrappers.
3. Keep device work inside `hl.tile` or `hl.grid` loops and express math with
   normal PyTorch ops first.
4. Use FP32 accumulators for reductions and dot products unless the kernel is
   intentionally lower precision.
5. Validate against an eager PyTorch baseline with `helion._testing.run_example`
   or a targeted `pytest` node.

## Repository Conventions

- Import Helion as:
  ```python
  import helion
  import helion.language as hl
  ```
- Examples should define `main()`.
- Do not add `print()` inside kernels.
- `hl.tile` preserves dimensions, so `x[tile_m, tile_n]` stays rank-2 and
  `x[tile]` means "the whole tile", not "the first element".
- Prefer explicit host-side output allocation and dtype promotion so behavior
  matches PyTorch.
- Avoid defensive noise such as `hasattr`, `getattr`, or broad `except` blocks
  unless an existing pattern in the repository requires it.

## Choose Settings Early

- `static_shapes=True` is the default and usually the right choice for
  shape-stable dense kernels such as GEMM, attention, and fused norms.
- `static_shapes=False` is better when the same kernel must serve many shapes,
  for jagged workloads, dropout, or AOT heuristic training.
- Use `hl.register_block_size(...)` when different loop dimensions should have
  independent tuneable tile sizes.
- Use `hl.register_tunable(name, fragment)` for non-block choices such as
  split-K or algorithm variants.
- Use `hl.specialize(value)` when a size should become a compile-time constant
  inside the kernel body.

## Reference Map

- Dense kernel skeletons and host/device split:
  [references/kernel-structure.md](references/kernel-structure.md)
- Elementwise, broadcasting, views, and simple flattening:
  [references/elementwise-and-broadcasting.md](references/elementwise-and-broadcasting.md)
- Reductions, softmax, and normalization:
  [references/reductions-and-normalization.md](references/reductions-and-normalization.md)
- GEMM, epilogues, and attention:
  [references/matmul-and-attention.md](references/matmul-and-attention.md)
- Jagged, masked, and irregular memory access:
  [references/jagged-masked-and-irregular.md](references/jagged-masked-and-irregular.md)
- RNG, atomics, barriers, and persistent-worker patterns:
  [references/advanced-ops.md](references/advanced-ops.md)
- Autotuning, AOT heuristics, and debugging:
  [references/tuning-aot-and-debugging.md](references/tuning-aot-and-debugging.md)

## Authoring Checklist

- Does the outermost `hl.tile` describe the intended grid?
- Are outputs allocated on the host with the right shape and dtype?
- If a reduction spans a long dimension, did you choose the right pattern:
  persistent, looped, split-K, or a two-pass stable algorithm?
- If indexing is irregular, are `hl.load` and `hl.store` masks correct for edge
  tiles?
- If the kernel is used in training, does it need a matching backward kernel or
  a `torch.autograd.Function` wrapper?

## Validation Loop

- Fast iteration: `pytest <file> -x -vv -s`
- Narrow with `pytest <file> -k <pattern>`
- Useful env vars:
  - `HELION_PRINT_OUTPUT_CODE=1`
  - `HELION_DEBUG_DTYPE_ASSERTS=1`
  - `HELION_LOGS=all`
  - `HELION_USE_DEFAULT_CONFIG=1` for targeted local iteration only
- Do not run the whole suite with `HELION_USE_DEFAULT_CONFIG=1`.

## Core Docs

- API surface: `docs/api/language.md`
- Kernel object and binding behavior: `docs/api/kernel.md`
- Config knobs: `docs/api/config.md`
- Settings and debug env vars: `docs/api/settings.md`
