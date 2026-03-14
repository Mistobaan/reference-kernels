---
name: arch-b200
description: Reason about NVIDIA Blackwell B200 architecture for AutoKernel roofline analysis and kernel optimization. Use when Codex needs Blackwell-specific priors for arithmetic intensity, warpgroup or tensor-core execution, shared-memory and TMA pipelines, TMEM pressure, CTA-pair cooperation, or B200 tuning decisions before handing work to idea-generation or kernel-authoring skills.
---

# Arch B200

Use this skill to turn raw Blackwell facts into an actionable bottleneck model for B200 kernels. Keep the output short: name the limiting resource, give one or two architecture-backed reasons, and route the next step to `gpu-roofline`, `gpu-memory-hierarchy`, `gpu-occupancy`, or a kernel authoring skill.

Read [references/blackwell-b200.md](references/blackwell-b200.md) when you need the exact Blackwell priors, rough throughput numbers, or memory-hierarchy details. Treat public peak numbers as planning priors, not measured truth; prefer repo-local benchmark or profiler evidence when it conflicts.

## Workflow

1. Classify the kernel as tensor-dominated, memory-dominated, reduction-heavy, or launch-limited.
2. Estimate arithmetic intensity and compare it against the B200 planning threshold. For FP16 tensor work, use `~280 FLOP/byte` as the rough compute-bound crossover unless the task provides better measured numbers.
3. Check whether the kernel structure matches Blackwell hardware units: `128-thread` warpgroups, TMA-fed shared-memory tiles, TMEM-backed accumulation, and cluster or CTA-pair cooperation for large tensor-core work.
4. Decide the first hard limit: HBM traffic, shared-memory bandwidth, synchronization, occupancy or register pressure, or non-matmul scalar work.
5. Turn the diagnosis into one concrete follow-up. Examples: increase shared-memory reuse, enlarge tensor tiles, split load and compute warps, reduce epilogue cost, or move to a warpgroup-oriented mapping.

## Guardrails

- Prefer warpgroup-level reasoning for tensor-core kernels. A `32-thread warp` is still the scheduling unit, but Blackwell tensor throughput is exposed through `128-thread` warpgroups.
- Do not call a kernel compute-bound from tensor peak alone. Compare achieved tensor issue rate against memory throughput and pipeline stalls.
- Expect many transformer subkernels to remain memory-bound on B200 even when tensor utilization looks healthy. The B200 tensor-to-bandwidth ratio is high enough that reuse usually dominates.
- Call out when shared-memory bandwidth or barriers, not HBM, are likely gating throughput. Blackwell shifts more kernels into on-chip bandwidth and pipeline-design limits.
- Use TMEM as an explanation for larger accumulator footprints and deeper pipelines, but still inspect register usage and spills. TMEM reduces pressure; it does not remove the register budget.
- When evidence points to softmax, exp, reductions, gathers, or epilogues, explicitly mark the kernel as non-matmul-limited and hand off to the relevant optimization skill instead of forcing a GEMM framing.

## Output

Produce:

- a one-line bottleneck label
- the 2-4 architecture facts that justify it
- one next optimization direction

If exact numbers or instruction names matter, quote them from the reference file instead of relying on memory.
