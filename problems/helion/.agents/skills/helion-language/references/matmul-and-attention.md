# Matmul And Attention

Use this when the kernel is dominated by dot products, matrix multiplications,
or streaming softmax attention.

## Pattern: GEMM With A Closure-Based Epilogue

Inspired by `examples/matmul.py`.

```python
@helion.kernel(static_shapes=True)
def matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    epilogue=lambda acc, tile: acc,
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty((m, n), dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))

    return out
```

Why this is the default GEMM pattern here:

- `static_shapes=True` usually helps dense GEMMs.
- `torch.addmm` keeps the kernel close to the eager math definition.
- The epilogue closure lets you fuse bias, activation, or other pointwise work
  without rewriting the reduction.

## Pattern: Split-K As A Tunable Algorithm Choice

Inspired by `examples/matmul_split_k.py`.

```python
split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
k_block = helion.next_power_of_2(helion.cdiv(k, split_k))

for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    for inner_k in hl.tile(outer_k.begin, outer_k.end):
        acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
    hl.atomic_add(out, [tile_m, tile_n], acc)
```

Use this when `K` is so large that more parallelism across the reduction axis is
worth the extra synchronization cost or loss of determinism.

## Pattern: Streaming Attention

Inspired by `examples/attention.py`.

```python
for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
    m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
    l_i = torch.full_like(m_i, 1.0)
    acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
    q = q_view[tile_b, tile_m, :]

    for tile_n in hl.tile(v_view.size(1)):
        k = k_view[tile_b, :, tile_n]
        qk = torch.bmm(q, k)
        ...
        acc = torch.baddbmm(acc, p.to(v.dtype), v_view[tile_b, tile_n, :])

    out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
```

Key ideas:

- reshape host-side tensors into a view that matches the kernel's tiling plan
- maintain running max and normalization terms exactly like streaming softmax
- accumulate the value projection in FP32

## `torch.*` Ops First, `hl.dot` Second

This repository often starts with `torch.addmm`, `torch.bmm`, and other regular
PyTorch ops inside tiles. Reach for `hl.dot(...)` only when you need explicit
control over the dot-product primitive, as in `examples/flex_attention.py` or
quantized GEMM examples.

## Sources

- `examples/matmul.py`
- `examples/matmul_split_k.py`
- `examples/attention.py`
- `examples/flex_attention.py`
