# Kernel Structure

Use this when writing a new dense kernel or reviewing whether host and device
logic are split cleanly.

## Pattern: Host Setup, Then Device Tiles

Inspired by `examples/add.py` and `examples/matmul.py`.

```python
import torch

import helion
import helion.language as hl


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out
```

What matters:

- Everything before the `hl.tile(...)` loop is ordinary host-side Python.
- The outermost `hl.tile(...)` loop becomes the GPU grid.
- `tile` is a tile object, not a scalar index. `x[tile]` loads the whole tile.

## Pattern: Nested Tiling for Reductions

Inspired by `examples/matmul.py`.

```python
@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty((m, n), dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)

    return out
```

What matters:

- The first `hl.tile([m, n])` loop is parallel across blocks.
- The nested `hl.tile(k)` loop is a sequential loop inside each block.
- Scratch tensors such as `acc` should usually accumulate in FP32.

## Tile Object Cheatsheet

Common tile attributes used in this repository:

- `tile.begin`: first logical index covered by the tile
- `tile.end`: exclusive end index
- `tile.index`: vector of element indices inside the tile
- `tile.id`: tile ordinal within that loop
- `tile.block_size`: chosen block size for that loop

These show up in jagged indexing, persistent kernels, split-K kernels, and
barrier-based reductions.

## When to Use `hl.grid`

Inspired by `examples/grouped_gemm.py`.

Use `hl.grid(n)` when you want scalar-style iteration over workers, groups, or
other small integer spaces rather than tile objects:

```python
for worker_id in hl.grid(num_workers):
    ...
```

This is common in persistent kernels and metadata iteration. Use `hl.tile`
instead when the loop body naturally operates on tiles of tensor elements.

## Sources

- `examples/add.py`
- `examples/matmul.py`
- `examples/grouped_gemm.py`
