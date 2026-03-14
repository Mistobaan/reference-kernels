# Reductions And Normalization

Use this when the kernel has a reduction axis, especially for softmax, sum,
RMS norm, or layer norm style code.

## Pattern: Two-Pass Numerically Stable Softmax

Inspired by `examples/softmax.py`.

```python
@helion.kernel()
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)

    for tile_m in hl.tile(m, block_size=block_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)

        for tile_n in hl.tile(n, block_size=block_n):
            values = x[tile_m, tile_n]
            local_max = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_max)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next

        for tile_n in hl.tile(n, block_size=block_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]

    return out
```

Why this pattern works:

- `hl.register_block_size(...)` exposes row and column blocking to autotuning.
- The first pass computes a stable running max and partition sum.
- The second pass writes normalized outputs once the final statistics are known.

## Pattern: Rowwise Normalization

Inspired by `examples/rms_norm.py`.

```python
@helion.kernel()
def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(x_tile * x_tile, dim=-1) + eps)
        out[tile_m, :] = (x_tile * inv_rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)

    return out
```

This is the default pattern for norms in this repo:

- load a logical row tile
- upcast to FP32
- compute statistics
- broadcast them back over the feature axis
- downcast only when storing

## Pattern: Specialize Feature Size For Backward Or Scratch Buffers

Inspired by `examples/rms_norm.py` and `examples/layer_norm.py`.

```python
m_block = hl.register_block_size(x.size(0))
weight_shape = hl.specialize(weight.size(0))
grad_weight = x.new_empty(
    [(x.size(0) + m_block - 1) // m_block, weight_shape],
    dtype=torch.float32,
)
```

Use `hl.specialize(...)` when a dimension is needed to size temporaries or
stabilize control flow inside the kernel body.

## Pattern: Separate Forward And Backward Kernels

Inspired by `examples/softmax.py` and `examples/rms_norm.py`.

The repository usually keeps forward and backward kernels separate, then wraps
them in `torch.autograd.Function`:

```python
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = softmax_two_pass(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return softmax_bwd(grad_output, y)
```

That keeps each Helion kernel focused on one device-level computation.

## Sources

- `examples/softmax.py`
- `examples/rms_norm.py`
- `examples/layer_norm.py`
