from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(block_sizes=[64], num_warps=2, num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    (1, 256, 256, 3): helion.Config(block_sizes=[256], num_warps=4, num_stages=2),
    (1, 128, 64, 8): helion.Config(block_sizes=[64], num_warps=2, num_stages=2),
    (4, 64, 128, 4): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 1536, 2048, 4): helion.Config(block_sizes=[256], num_warps=2, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_2.acf"),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[512], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_2.acf"),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[512], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_2.acf"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,  # (B, D, S) input
        w: torch.Tensor,  # (D, W) filter coefficients
        b: torch.Tensor,  # (D,) additive offset
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty_like(x)

        block_s = hl.register_block_size(32, S)
        for rbd, rs in hl.tile([B * D, S], block_size=[1, block_s]):
            flat_idx = rbd.begin
            b_idx = flat_idx // D
            d_idx = flat_idx % D

            acc = hl.zeros([rs], dtype=torch.float32)
            base_idx = rs.index - (W - 1)
            for j in range(W):
                x_idx = base_idx + j
                x_vals = hl.load(x, [b_idx, d_idx, x_idx], extra_mask=x_idx >= 0).to(
                    torch.float32
                )
                acc = acc + x_vals * w[d_idx, j].to(torch.float32)

            y[b_idx, d_idx, rs] = (acc + b[d_idx].to(torch.float32)).to(y.dtype)

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    return kernel(x, weight, bias)
