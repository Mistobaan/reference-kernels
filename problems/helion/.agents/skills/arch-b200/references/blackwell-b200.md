# Blackwell B200 Kernel Programming Reference

## Table of Contents

- [1. Core execution model](#1-core-execution-model)
- [2. Tensor Memory (TMEM)](#2-tensor-memory-tmem)
- [3. Memory hierarchy](#3-memory-hierarchy)
- [4. Tensor Core throughput](#4-tensor-core-throughput)
- [5. Arithmetic intensity targets](#5-arithmetic-intensity-targets)
- [6. Tensor Memory Accelerator (TMA)](#6-tensor-memory-accelerator-tma)
- [7. CTA clusters and CTA pairs](#7-cta-clusters-and-cta-pairs)
- [8. Kernel design patterns](#8-kernel-design-patterns)
- [9. Important bottlenecks on Blackwell](#9-important-bottlenecks-on-blackwell)
- [10. Profiling metrics](#10-profiling-metrics)
- [11. Practical kernel rules](#11-practical-kernel-rules)
- [12. Quick reference summary](#12-quick-reference-summary)
- [Sources](#sources)

Use this file for exact Blackwell B200 priors during roofline analysis, kernel design, and low-level CUDA or Triton tuning. Unless the project provides measured values, treat the throughput and bandwidth figures below as public-reference approximations for planning rather than guaranteed device truth.

## 1. Core execution model

### Warp

- `1 warp = 32 threads`
- Warps execute in SIMT lockstep.

### Warpgroup

Blackwell continues the Hopper-era warpgroup model for tensor-core execution.

- `1 warpgroup = 4 warps = 128 threads`
- Tensor-core matrix instructions operate at warpgroup granularity.
- One threadblock typically contains multiple warpgroups.

Typical GEMM mapping:

```text
threadblock
 ├── warpgroup 0
 ├── warpgroup 1
 ├── warpgroup 2
 └── warpgroup 3
```

### Warpgroup MMA

Blackwell introduces `tcgen05.mma` for large tensor-core work.

- Operates on shared-memory tiles
- Stores accumulation in Tensor Memory
- Exposes large matrix shapes such as `256 x 256 x 16`
- Can involve cooperation across two SMs in some implementations

Implication: Blackwell pushes tensor work toward larger tiles and higher arithmetic intensity than Hopper-era kernels built around register-resident accumulators alone.

## 2. Tensor Memory (TMEM)

Blackwell adds Tensor Memory, a dedicated accumulator storage path for tensor-core pipelines.

Properties:

- Roughly `256 KB per SM`
- Holds intermediate tensor accumulations
- Reduces the register pressure seen on large Hopper MMA pipelines

Example capacity often cited in public references:

```text
128 x 512 elements
32-bit accumulators
```

Typical data path:

```text
shared memory -> tensor core -> TMEM -> register writeback
```

Practical benefits:

- fewer register spills
- larger MMA tiles
- deeper compute pipelines

## 3. Memory hierarchy

### Global memory (HBM3e)

Public B200 system references cite approximately `64 TB/s` of HBM3e bandwidth for a full DGX B200 node. A common single-GPU planning estimate is:

```text
~8 TB/s per B200 GPU
```

Use this as a roofline prior when the exact SKU or measured bandwidth is unknown.

### L2 cache

Blackwell public references commonly cite:

```text
~64-65 MB L2 cache
```

### L1 and shared memory

Blackwell uses a unified L1 and shared-memory design.

- Up to `256 KB` unified per SM
- Shared-memory carve-outs commonly documented as:

```text
0 KB
8 KB
16 KB
32 KB
64 KB
100 KB
132 KB
164 KB
196 KB
228 KB
```

Maximum usable shared memory by one block:

```text
227 KB
```

The difference reflects CUDA-reserved space.

## 4. Tensor Core throughput

Approximate B200 peak throughput figures cited in public material:

| Precision | Peak |
| --- | --- |
| FP4 | `~9 PFLOPS dense` |
| FP8 | `~4.5 PFLOPS` |
| BF16 / FP16 | `~2.25 PFLOPS` |
| FP64 | `~40 TFLOPS` |

Sparse modes are often described as roughly `2x` the dense throughput.

Implication: tensor compute has scaled faster than memory bandwidth, so many real kernels bottleneck on memory movement, synchronization, or non-matmul work before they approach tensor peak.

## 5. Arithmetic intensity targets

For roofline modeling:

```text
AI = FLOPs / bytes
```

Using the common B200 planning priors:

```text
Peak FP16 tensor compute ~2.25e15 FLOP/s
HBM bandwidth           ~8e12 B/s
```

The rough FP16 tensor compute-bound threshold is:

```text
AI ~ 280 FLOP/byte
```

Interpretation:

- Kernels with `AI < ~280` are likely memory-bound
- Many transformer kernels land in the `10-100 FLOP/byte` range
- Shared-memory reuse, larger MMA tiles, and fewer global round-trips are usually mandatory to move B200 kernels toward compute-bound behavior

## 6. Tensor Memory Accelerator (TMA)

Blackwell continues the asynchronous TMA copy engine for moving tiles from global memory to shared memory.

Purpose:

```text
global memory -> shared memory
```

Features:

- asynchronous tiled loads
- multi-dimensional transfer support
- multicast to multiple CTAs

Typical pipeline:

```text
TMA load
↓
shared memory tile
↓
tensor core MMA
↓
TMEM accumulation
```

Warp specialization often separates roles:

```text
warp 0: TMA loads
warp 1-3: tensor compute
```

## 7. CTA clusters and CTA pairs

Blackwell extends threadblock cluster execution and adds CTA-pair cooperation patterns for tensor-core work.

Key idea:

- Two CTAs can cooperate on a single tensor-core operation
- This can increase tile size and arithmetic intensity
- Shared tensor-memory or cluster-level cooperation becomes part of the performance model

Example:

```text
CTA0 loads tile A
CTA1 loads tile B
paired MMA operation
```

Use this model when reasoning about advanced GEMM kernels that intentionally stretch beyond a single CTA's normal scope.

## 8. Kernel design patterns

### Warp specialization

Common role split:

```text
warpgroup 0 -> memory loads
warpgroup 1 -> tensor core MMA
warpgroup 2 -> reductions or epilogue
```

Complex kernels such as modern attention pipelines may use five or more specialized warp roles.

### Deep pipelines

Typical stages:

```text
stage 0: TMA load
stage 1: shared-memory staging
stage 2: tensor core MMA
stage 3: TMEM accumulation
stage 4: epilogue
```

Common pipeline depth:

```text
3-5 stages
```

### Representative tile sizes

Public examples of high-performance tensor kernels often use:

```text
threadblock tile: 128x128x64
warpgroup tile:   64x128x64
mma tile:         256x256x16
```

Treat these as search priors, not fixed rules.

## 9. Important bottlenecks on Blackwell

Typical limiting factors:

1. shared-memory bandwidth
2. synchronization latency and barrier design
3. register pressure despite TMEM support
4. non-matmul work such as softmax, exp, or reductions

Blackwell often makes tensor math cheap enough that these bottlenecks dominate earlier than expected.

## 10. Profiling metrics

Useful Nsight Compute metrics:

Compute:

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__inst_executed_pipe_tensor
```

Memory:

```text
dram__throughput.avg
l1tex__data_bank_conflicts
lts__throughput
```

Scheduler and issue health:

```text
smsp__warps_active
smsp__issue_active
```

Tensor-core activity:

```text
smsp__inst_executed_pipe_tensor
```

## 11. Practical kernel rules

Rules of thumb for B200 kernels:

1. Prefer warpgroup-oriented tensor mappings.
2. Use shared memory aggressively to hide HBM latency.
3. Keep MMA pipelines full; late loads waste a large amount of tensor throughput.
4. Favor larger tiles when the shared-memory and occupancy budget allows it.
5. Specialize warps or warpgroups for load, compute, and epilogue work.

## 12. Quick reference summary

```text
warp size:              32 threads
warpgroup:              4 warps = 128 threads

shared memory per SM:   up to 228 KB usable
L2 cache:               ~64-65 MB
tensor memory:          ~256 KB per SM

HBM bandwidth:          ~8 TB/s per GPU
FP16 tensor peak:       ~2.25 PFLOPS
FP4 tensor peak:        ~9 PFLOPS

AI compute bound:       ~280 FLOP/byte
```

## Sources

1. [Nvidia MMA instruction evolution overview](https://zhuanlan.zhihu.com/p/1987586644417745413?utm_source=chatgpt.com)
2. [Modular: Matrix Multiplication on Blackwell, Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction?utm_source=chatgpt.com)
3. [NVIDIA Developer Blog: Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/?utm_source=chatgpt.com)
4. [NVIDIA DGX B200 product page](https://www.nvidia.com/en-us/data-center/dgx-b200/?utm_source=chatgpt.com)
5. [Emergent Mind: Blackwell GPU Architecture](https://www.emergentmind.com/topics/blackwell-gpu-architecture?utm_source=chatgpt.com)
6. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html?utm_source=chatgpt.com)
7. [CUDO Compute: Blackwell architecture summary](https://www.cudocompute.com/blog/nvidias-blackwell-architecture-breaking-down-the-b100-b200-and-gb200?utm_source=chatgpt.com)
8. [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451?utm_source=chatgpt.com)
9. [Rohan Yadav: Warp specialization](https://rohany.github.io/blog/warp-specialization/?utm_source=chatgpt.com)
10. [Together AI: ThunderKittens on Blackwell](https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus?utm_source=chatgpt.com)
