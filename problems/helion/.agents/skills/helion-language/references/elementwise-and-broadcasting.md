# Elementwise And Broadcasting

Use this when the kernel is mostly pointwise math, light broadcasting, or a
reshape or flatten around pointwise work.

## Pattern: Match PyTorch Broadcasting On The Host

Inspired by `examples/add.py`.

```python
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

Why this pattern is preferred:

- Broadcasting rules stay identical to eager PyTorch.
- Output shape and dtype are explicit before the kernel launches.
- The device loop stays simple and dense.

## Pattern: Broadcast Inside A Tile With `None`

Inspired by the style used in `examples/attention.py`, `examples/softmax.py`,
and `examples/rms_norm.py`.

```python
values = x[tile_m, tile_n]
row_max = torch.amax(values, dim=1)
centered = values - row_max[:, None]
```

The important part is that tile indexing preserves rank. If `values` is rank-2,
`row_max[:, None]` broadcasts over the second axis exactly like eager PyTorch.

## Pattern: Flatten To 1D For Uniform Elementwise Work

Inspired by `examples/low_mem_dropout.py`.

```python
@helion.kernel(static_shapes=False)
def scale_flat(x: torch.Tensor, scale: float) -> torch.Tensor:
    n = x.numel()
    x_flat = x.view(-1)
    out_flat = torch.empty_like(x_flat)
    for tile_n in hl.tile(n):
        out_flat[tile_n] = (x_flat[tile_n].to(torch.float32) * scale).to(x.dtype)
    return out_flat.view_as(x)
```

This is useful when the operation is layout-agnostic and a flattened traversal
is easier than reproducing the original rank.

## Pattern: Capture Small Host Values In Closures

Inspired by the epilogue style in `examples/matmul.py`.

If a pointwise transform depends on a bias, scale, or other auxiliary tensor,
prefer a closure or direct capture rather than manually threading extra scalar
state through every tile:

```python
def epilogue(acc: torch.Tensor, tile: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.relu(acc + bias[tile[1]])
```

Helion lifts the captured tensor into kernel arguments when needed.

## Sources

- `examples/add.py`
- `examples/low_mem_dropout.py`
- `examples/matmul.py`
- `examples/attention.py`
- `examples/softmax.py`
- `examples/rms_norm.py`
