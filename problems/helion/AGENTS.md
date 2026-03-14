# Helion Challenge Workspace

## Mission

Optimize Helion submissions for the five problem folders in this workspace:

- `causal_conv1d_py`
- `fp8_quant_py`
- `gated_deltanet_chunk_fwd_h_py`
- `gated_deltanet_chunk_fwd_o_py`
- `gated_deltanet_recompute_w_u_py`

Target GPU and leaderboard environment:

- GPU: `B200_Nebius`
- Challenge: Helion Kernel Challenge
- `fp8_quant` is a warm-up problem only, but it is still in scope for the loop

Use Helion-first implementations. The submission must remain a single-file `submission.py` implementation for each problem.

## Required Files Per Problem

Before editing, read these files in the active problem directory:

- `task.yml`
- `reference.py`
- `task.py`
- `submission.py`

Treat these as read-only unless explicitly stated otherwise:

- `task.yml`
- `reference.py`
- `task.py`
- `../eval.py`
- `../utils.py`

Only `submission.py` in the active problem folder may be modified by Codex during the experiment loop.

## Skills

Use skills in this order before making kernel changes:

1. `arch-b200`
2. `lang-helion`

If the active problem folder contains a local Helion skill at `.agents/skills/helion-language/SKILL.md`, read and use it as problem-local guidance in addition to the repo-level skills.

## Experiment Loop

For each active problem:

1. Read `task.yml`, `reference.py`, `task.py`, and `submission.py`.
2. Reconstruct the correctness contract and important benchmark shapes from `task.yml`.
3. Establish a local baseline with:
   - `python eval.py test <problem_dir>`
   - `python eval.py benchmark <problem_dir>`
4. Diagnose the current bottleneck using the B200 and Helion skills.
5. Modify only `<problem_dir>/submission.py`.
6. Re-run:
   - `python eval.py test <problem_dir>`
   - `python eval.py benchmark <problem_dir>`
7. Continue iterating only while the kernel is converging toward a better local benchmark result.
8. Stop when one of these is true:
   - local tests pass and benchmark geomean improves enough to accept the change
   - no improvement is found
   - the attempt is blocked

## Kernel Rules

- Use Helion as the default language.
- Keep the public entry point in `submission.py` compatible with the task contract.
- Hardcode configs in `submission.py` for submission. Do not rely on autotuning on KernelBot.
- `hl.inline_triton()`, `hl.triton_kernel()`, and `hl.inline_asm_elementwise()` are allowed escape hatches, but the majority of the implementation should remain Helion.
- Prefer per-shape configs for all test and benchmark shapes in `task.yml`.
- Preserve correctness for all provided test shapes.

## Submission Policy

Once a candidate passes local tests and improves the local benchmark:

1. Emit a short summary of the change.
2. Emit a short branch memo suitable for a branch suffix.
3. Emit a commit message.
4. Mark whether the candidate is ready to submit.
5. Submit with:
   - `popcorn submit <problem>/submission.py --gpu B200_Nebius --leaderboard <leaderboard> --mode test --no-tui`
   - `popcorn submit <problem>/submission.py --gpu B200_Nebius --leaderboard <leaderboard> --mode leaderboard --no-tui`

Leaderboard names are derived from the folder name by removing the `_py` suffix.
