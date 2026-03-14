# Advanced Ops

Use this when the kernel needs randomness, atomics, barriers, or a
persistent-worker style that is more structured than a plain tiled loop.

## Pattern: Stateless RNG In Device Code

Inspired by `examples/low_mem_dropout.py` and `test/test_random.py`.

```python
@helion.kernel(static_shapes=False)
def low_mem_dropout(p: float, x: torch.Tensor, seed: int) -> torch.Tensor:
    scale = 1.0 / (1.0 - p)
    n = x.numel()
    x_flat = x.view(-1)
    out_flat = torch.empty_like(x_flat)

    for tile_n in hl.tile(n):
        xi = x_flat[tile_n].to(torch.float32)
        r = hl.rand([tile_n], seed=seed)
        keep = r > p
        out_flat[tile_n] = torch.where(keep, xi * scale, 0.0).to(x.dtype)

    return out_flat.view_as(x)
```

Notes:

- `hl.rand(...)` is keyed by logical indices plus the seed.
- The test suite checks that the sequence is deterministic across different
  block sizes for the same logical shape and seed.
- Flattening is often the easiest way to make dropout or RNG-driven pointwise
  kernels independent of the original rank.

## Pattern: Split-K With Atomics

Inspired by `examples/matmul_split_k.py`.

```python
split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
...
hl.atomic_add(out, [tile_m, tile_n], acc)
```

Use atomics when multiple blocks contribute partial sums to the same output tile
and you want the simplest implementation. The tradeoff is non-deterministic
reduction order.

## Pattern: Deterministic Two-Stage Reduction With A Barrier

Inspired by `examples/split_k_barrier.py`.

```python
for tile_m, tile_n, tile_k_outer in hl.tile([m, n, k], block_size=[None, None, block_k]):
    ...
    tmp[tile_m, tile_n, tile_k_outer.id] = acc

hl.barrier()

for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)
```

Use this when deterministic results matter more than the extra temporary buffer
and global synchronization.

## Pattern: Persistent Workers

Inspired by `examples/grouped_gemm.py`.

Persistent kernels usually look like:

1. compute a worker count on the host
2. loop over `worker_id` with `hl.grid(num_workers)`
3. assign tiles dynamically inside the kernel

This is the right tool for jagged workloads, grouped work, and load balancing
problems where a flat dense grid leaves some blocks with much more work.

## Sources

- `examples/low_mem_dropout.py`
- `examples/matmul_split_k.py`
- `examples/split_k_barrier.py`
- `examples/grouped_gemm.py`
- `test/test_random.py`
