# Tuning, AOT, And Debugging

Use this when the work is about performance control, stable deployment, or
debugging generated kernels rather than the math itself.

## Choosing `static_shapes`

- Keep `static_shapes=True` for dense kernels with stable shapes. This is the
  default and usually helps GEMM and attention.
- Switch to `static_shapes=False` when one compiled kernel should serve many
  shapes or when you want heuristic-driven config reuse.

Examples in this repo:

- `examples/matmul.py`: `static_shapes=True`
- `examples/attention.py`: `static_shapes=True`
- `examples/low_mem_dropout.py`: `static_shapes=False`
- `examples/grouped_gemm.py`: `static_shapes=False`

## Pattern: Expose Block Sizes To The Autotuner

Inspired by `examples/softmax.py` and `examples/aot_example.py`.

```python
block_m = hl.register_block_size(m)
block_n = hl.register_block_size(n)

for tile_m in hl.tile(m, block_size=block_m):
    for tile_n in hl.tile(n, block_size=block_n):
        ...
```

Use `hl.register_block_size(...)` when the best tile size depends on the problem
shape or hardware.

## Pattern: Pin Or Save A Config

Inspired by `examples/matmul.py`.

```python
best_config = matmul.autotune(args, force=True)
best_config.save("best_config.json")

@helion.kernel(config=helion.Config(...))
def tuned_kernel(...):
    ...
```

For production-style flows, prefer saving the winning config or using AOT
heuristics instead of re-running a long autotune search at runtime.

## Pattern: AOT Autotuning

Inspired by `examples/aot_example.py`.

```python
@helion.experimental.aot_kernel()
def vector_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    ...


@helion.experimental.aot_kernel(batched=[[0, None], None])
def rms_norm_batched(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    ...


def _matmul_key(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int, int]:
    m, k = a.shape
    _, n = b.shape
    return (m, n, k, a.element_size())


@helion.experimental.aot_kernel(key=_matmul_key)
def matmul_custom_key(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ...
```

Choose between:

- plain `@helion.experimental.aot_kernel()` for automatic AOT setup
- `batched=...` when some dimensions are batch dimensions and should not create
  distinct heuristics
- `key=...` when you know exactly which features should drive config selection
- `collect_fn=` and `measure_fn=` when the tuning shapes are known upfront

## Debugging Loop

Use the repository's recommended loop:

- `pytest test/<file>.py::TestName::test_case -x -vv -s`
- `pytest -ra` to see skips
- `HELION_PRINT_OUTPUT_CODE=1` to inspect generated Triton
- `HELION_DEBUG_DTYPE_ASSERTS=1` to check dtype lowering
- `HELION_USE_DEFAULT_CONFIG=1` only for targeted local iteration

Relevant docs:

- `docs/api/kernel.md`
- `docs/api/config.md`
- `docs/api/settings.md`
- `docs/deployment_autotuning.md`

## Sources

- `examples/matmul.py`
- `examples/softmax.py`
- `examples/attention.py`
- `examples/low_mem_dropout.py`
- `examples/grouped_gemm.py`
- `examples/aot_example.py`
- `docs/api/kernel.md`
- `docs/api/config.md`
- `docs/api/settings.md`
