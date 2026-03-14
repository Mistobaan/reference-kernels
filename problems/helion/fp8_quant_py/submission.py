from task import input_t, output_t

import torch
import helion
import helion.language as hl


FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


# Per-shape configs: map (num_tokens, hidden_dim, group_size) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 256, 64): helion.Config(block_sizes=[4], num_warps=1, num_stages=1),
    (4, 512, 128): helion.Config(block_sizes=[8], num_warps=1, num_stages=1),
    (16, 1024, 64): helion.Config(block_sizes=[32], num_warps=2, num_stages=2),
    (1, 4096, 128): helion.Config(block_sizes=[16], num_warps=2, num_stages=2),
    (8, 4096, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    # Benchmark shapes
    # (1, 4096, 128) already covered above
    (16, 4096, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (256, 4096, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (256, 8192, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (4096, 7168, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/fp8_group_quant_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,       # [N, G] input rows
        qout: torch.Tensor,       # [N, G] quantized output
        scales_out: torch.Tensor,  # [N] output normalization factors
    ) -> None:
        nrows = data.size(0)
        block_r = hl.register_block_size(1, nrows)

        for rr in hl.tile(nrows, block_size=block_r):
            row = data[rr, :].to(torch.float32)
            amax = torch.amax(torch.abs(row), dim=1)
            amax = torch.clamp(amax, min=FP8_EPS)
            inv_scale = FP8_MAX / amax

            qout[rr, :] = torch.clamp(row * inv_scale[:, None], FP8_MIN, FP8_MAX)
            scales_out[rr] = amax / FP8_MAX

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    kernel = _KERNELS[(T, H, gsz)]

    flat_in = x.reshape(N, gsz)
    flat_q = x_q.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    kernel(flat_in, flat_q, flat_s)
    return x_q, x_s
