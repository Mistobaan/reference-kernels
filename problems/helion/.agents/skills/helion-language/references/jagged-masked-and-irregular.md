# Jagged, Masked, And Irregular Access

Use this when direct `x[tile]` indexing is not enough because the access pattern
is jagged, sparse, flattened, or masked at tile boundaries.

## Pattern: Build Explicit Indices And Use `hl.load`

Inspired by `examples/jagged_softmax.py`.

```python
base_indices = starts[:, None] + tile_k.index[None, :]
flat_indices = base_indices[:, :, None] * M + tile_m.index[None, None, :]
row_mask = tile_k.index[None, :] < seqlens[:, None]
combined_mask = row_mask[:, :, None] & (tile_m.index < M)[None, None, :]

x_slice = hl.load(
    x_flat,
    [flat_indices],
    extra_mask=combined_mask,
)
```

This is the standard recipe for irregular data:

1. compute explicit logical indices
2. build an `extra_mask` for valid lanes
3. call `hl.load(...)`
4. use `torch.where(...)` or masked reductions when invalid lanes should behave
   like `-inf`, `0`, or another sentinel

## Pattern: Masked Stores

Inspired by `examples/jagged_softmax.py` and `examples/grouped_gemm.py`.

```python
hl.store(
    out,
    [flat_indices],
    block_out,
    extra_mask=combined_mask,
)
```

Prefer `hl.store(...)` instead of direct indexing when:

- the write indices are computed manually
- the tile is partially out of bounds
- some lanes are logically absent because of ragged sequence lengths

## Pattern: Persistent Worker Loop With `hl.arange`

Inspired by `examples/grouped_gemm.py`.

```python
for worker_id in hl.grid(num_workers):
    ...
    row_idx = base_row + hl.arange(BLOCK_M)
    col_idx = base_col + hl.arange(BLOCK_N)
    rows_valid = row_idx < group_end
    cols_valid = col_idx < N
    ...
    a_blk = hl.load(A_packed, [row_idx, k_idx], extra_mask=rows_valid[:, None])
    b_blk = hl.load(B, [k_idx, col_idx], extra_mask=cols_valid[None, :])
```

Use this style when the kernel must assign work dynamically rather than relying
on a regular dense grid.

## Practical Rule

If dense tile indexing can express the kernel, prefer it. Move to explicit
`hl.load(...)` and `hl.store(...)` only for:

- jagged row offsets
- persistent worker schemes
- flattened or gathered indices
- boundary masking that cannot be represented by ordinary slices

## Sources

- `examples/jagged_softmax.py`
- `examples/grouped_gemm.py`
- `examples/jagged_layer_norm.py`
- `examples/jagged_dense_bmm.py`
