#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WORKSPACE_REL="${SCRIPT_DIR#$REPO_ROOT/}"
REPO_BASE_BRANCH="${REPO_BASE_BRANCH:-main}"
WORKTREES_ROOT="${WORKTREES_ROOT:-$(dirname "$REPO_ROOT")/worktrees}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CODEX_BIN="${CODEX_BIN:-codex}"
POPCORN_BIN="${POPCORN_BIN:-popcorn}"
MIN_SPEEDUP_PCT="${MIN_SPEEDUP_PCT:-1}"
MAX_EXPERIMENTS_PER_PROBLEM="${MAX_EXPERIMENTS_PER_PROBLEM:-6}"
MAX_STALL_EXPERIMENTS="${MAX_STALL_EXPERIMENTS:-3}"
HISTORY_RESULTS_LIMIT="${HISTORY_RESULTS_LIMIT:-8}"
HISTORY_CHANGELOG_LIMIT="${HISTORY_CHANGELOG_LIMIT:-3}"
DEFAULT_RUN_TAG="$(date '+%b%d' | tr '[:upper:]' '[:lower:]')"
STATE_ROOT="${STATE_ROOT:-$SCRIPT_DIR/.codex/helion-loop}"
VENV_PATH="${VENV_PATH:-}"

RUN_TAG="$DEFAULT_RUN_TAG"
PROBLEM_FILTER=""
SKIP_SUBMIT="true"
RESUME_RUN="false"

usage() {
  cat <<'EOF'
Usage: ./ralph.sh [run_tag] --problem <problem_dir> [--venv <path>] [--resume] [--submit]

Environment:
  WORKTREES_ROOT    Override worktree root. Defaults to sibling worktrees dir.
  MIN_SPEEDUP_PCT   Minimum required benchmark geomean improvement percentage.
  MAX_EXPERIMENTS_PER_PROBLEM  Maximum Codex experiment iterations per problem.
  MAX_STALL_EXPERIMENTS        Consecutive non-improving iterations before stopping.
  VENV_PATH         Virtualenv path to activate before running commands.
  PYTHON_BIN        Python executable to use. Defaults to "python".
  CODEX_BIN         Codex CLI executable. Defaults to "codex".
  POPCORN_BIN       Popcorn CLI executable. Defaults to "popcorn".
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --problem)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --problem requires a value" >&2
        exit 2
      fi
      PROBLEM_FILTER="$2"
      shift 2
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --venv requires a value" >&2
        exit 2
      fi
      VENV_PATH="$2"
      shift 2
      ;;
    --submit)
      SKIP_SUBMIT="false"
      shift
      ;;
    --resume)
      RESUME_RUN="true"
      shift
      ;;
    --skip-submit)
      SKIP_SUBMIT="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "ERROR: Unknown option $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "$RUN_TAG" == "$DEFAULT_RUN_TAG" ]]; then
        RUN_TAG="$1"
      else
        echo "ERROR: Unexpected positional argument $1" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

activate_virtualenv() {
  local requested_path="$1"
  local resolved_path

  if [[ -z "$requested_path" ]]; then
    return 0
  fi

  if [[ ! -d "$requested_path" ]]; then
    echo "ERROR: Virtualenv directory not found: $requested_path" >&2
    exit 2
  fi

  resolved_path="$(cd "$requested_path" && pwd -P)"
  if [[ ! -f "$resolved_path/bin/activate" ]]; then
    echo "ERROR: Virtualenv activate script not found: $resolved_path/bin/activate" >&2
    exit 2
  fi

  set +u
  if ! source "$resolved_path/bin/activate"; then
    set -u
    echo "ERROR: Failed to activate virtualenv: $resolved_path" >&2
    exit 2
  fi
  set -u

  echo "USING PYTHON $(which python)"

  VENV_PATH="$resolved_path"
}

activate_virtualenv "$VENV_PATH"

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $cmd" >&2
    exit 1
  fi
}

require_command git
require_command "$PYTHON_BIN"
require_command "$CODEX_BIN"

if [[ "$SKIP_SUBMIT" != "true" ]]; then
  require_command "$POPCORN_BIN"
fi

ARTIFACT_ROOT="$STATE_ROOT/runs/$RUN_TAG"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

humanize_duration_ms() {
  local duration_ms="$1"
  "$PYTHON_BIN" - "$duration_ms" <<'PY'
import sys

try:
    ms = float(sys.argv[1])
except Exception:
    print(sys.argv[1])
    raise SystemExit(0)

abs_ms = abs(ms)
if abs_ms >= 1000.0:
    value, unit = ms / 1000.0, "s"
elif abs_ms >= 1.0:
    value, unit = ms, "ms"
elif abs_ms >= 0.001:
    value, unit = ms * 1000.0, "us"
else:
    value, unit = ms * 1_000_000.0, "ns"

print(f"{value:.4f} {unit}")
PY
}

humanize_percent() {
  local percent_value="$1"
  "$PYTHON_BIN" - "$percent_value" <<'PY'
import sys

try:
    value = float(sys.argv[1])
except Exception:
    print(sys.argv[1])
    raise SystemExit(0)

print(f"{value:.4f}%")
PY
}

ensure_problem_history_files() {
  local problem="$1"
  local history_dir="$STATE_ROOT/history/$problem"
  local changelog_file="$history_dir/CHANGELOG.md"
  local results_file="$history_dir/results.tsv"

  mkdir -p "$history_dir"

  if [[ ! -f "$changelog_file" ]]; then
    cat > "$changelog_file" <<EOF
# $problem Change Log

Accepted changes for \`$problem\` are recorded here.
EOF
  fi

  if [[ ! -f "$results_file" ]]; then
    cat > "$results_file" <<'EOF'
timestamp	run_tag	problem	status	baseline_geomean_ms	candidate_geomean_ms	improvement_pct	tests_passed	ready_to_submit	branch	commit	commit_message	description	baseline_points_ms	candidate_points_ms	artifact_dir
EOF
  fi
}

problem_workspace_path() {
  local worktree_root="$1"
  printf '%s/%s\n' "$worktree_root" "$WORKSPACE_REL"
}

prepare_problem_workspace() {
  local worktree_root="$1"
  local problem="$2"
  local workspace_path
  local common_file

  workspace_path="$(problem_workspace_path "$worktree_root")"
  mkdir -p "$workspace_path"

  for common_file in AGENTS.md eval.py template.py utils.py; do
    if [[ -f "$SCRIPT_DIR/$common_file" ]]; then
      cp "$SCRIPT_DIR/$common_file" "$workspace_path/$common_file"
    fi
  done

  cp -R "$SCRIPT_DIR/$problem" "$workspace_path/$problem"
}

problem_autotune_acf_glob() {
  case "$1" in
    causal_conv1d_py)
      printf '%s\n' 'causal_conv_*.acf'
      ;;
    fp8_quant_py)
      printf '%s\n' 'fp8_group_quant_*.acf'
      ;;
    gated_deltanet_chunk_fwd_h_py)
      printf '%s\n' 'chunk_fwd_h_*.acf'
      ;;
    gated_deltanet_chunk_fwd_o_py)
      printf '%s\n' 'chunk_fwd_o_*.acf'
      ;;
    gated_deltanet_recompute_w_u_py)
      printf '%s\n' 'recompute_w_u_fwd_*.acf'
      ;;
  esac
}

problem_autotune_search_acf() {
  local problem="$1"
  local acf_glob

  acf_glob="$(problem_autotune_acf_glob "$problem")"
  if [[ -z "$acf_glob" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "$acf_glob" <<'PY'
import sys
from pathlib import Path

acf_glob = sys.argv[1]
files = sorted(str(path) for path in Path("/opt/booster_pack").glob(acf_glob))
print(",".join(files))
PY
}

discover_problems() {
  local -a found=()
  local dir
  for dir in "$SCRIPT_DIR"/*; do
    [[ -d "$dir" ]] || continue
    if [[ -f "$dir/submission.py" && -f "$dir/reference.py" && -f "$dir/task.yml" && -f "$dir/task.py" ]]; then
      found+=("$(basename "$dir")")
    fi
  done

  if [[ "${#found[@]}" -eq 0 ]]; then
    echo "ERROR: No problem directories found in $SCRIPT_DIR" >&2
    exit 1
  fi

  if [[ -n "$PROBLEM_FILTER" ]]; then
    local match="false"
    for dir in "${found[@]}"; do
      if [[ "$dir" == "$PROBLEM_FILTER" ]]; then
        match="true"
        break
      fi
    done
    if [[ "$match" != "true" ]]; then
      echo "ERROR: Problem directory not found: $PROBLEM_FILTER" >&2
      exit 1
    fi
    printf '%s\n' "$PROBLEM_FILTER"
    return
  fi

  printf '%s\n' "${found[@]}" | sort
}

parse_benchmark_geomean() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY'
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
values = [float(m.group(1)) for m in re.finditer(r"Benchmark \d+: ([0-9]+(?:\.[0-9]+)?) ms", text)]
if not values:
    raise SystemExit("no benchmark measurements found")
print(math.exp(sum(math.log(v) for v in values) / len(values)))
PY
}

parse_benchmark_points() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
values = [m.group(1) for m in re.finditer(r"Benchmark \d+: ([0-9]+(?:\.[0-9]+)?) ms", text)]
print(",".join(values))
PY
}

sanitize_branch_memo() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys

memo = sys.argv[1].strip().lower()
memo = re.sub(r"[^a-z0-9]+", "-", memo)
memo = re.sub(r"-{2,}", "-", memo).strip("-")
print(memo[:48] or "candidate")
PY
}

json_field() {
  local json_file="$1"
  local field_name="$2"
  "$PYTHON_BIN" - "$json_file" "$field_name" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
value = data[sys.argv[2]]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

json_field_optional() {
  local json_file="$1"
  local field_name="$2"
  "$PYTHON_BIN" - "$json_file" "$field_name" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
value = data.get(sys.argv[2], "")
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

validate_json_response() {
  local json_file="$1"
  "$PYTHON_BIN" - "$json_file" <<'PY'
import json
import sys
from pathlib import Path

required = {
    "status": str,
    "summary": str,
    "branch_memo": str,
    "commit_message": str,
    "baseline_geomean_ms": (int, float),
    "candidate_geomean_ms": (int, float),
    "tests_passed": bool,
    "ready_to_submit": bool,
}

data = json.loads(Path(sys.argv[1]).read_text())
for key, expected in required.items():
    if key not in data:
        raise SystemExit(f"missing field: {key}")
    if not isinstance(data[key], expected):
        raise SystemExit(f"invalid field type for {key}: {type(data[key]).__name__}")
PY
}

sanitize_text() {
  printf '%s' "$1" | tr '\t\r\n' '   ' | sed 's/  */ /g'
}

git_branch_exists() {
  local branch_name="$1"
  git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch_name"
}

log_indicates_resumable_failure() {
  local log_file="$1"
  if [[ ! -f "$log_file" ]]; then
    return 1
  fi

  "$PYTHON_BIN" - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore").lower()
patterns = [
    r"command not found",
    r"\bpython\b: not found",
    r"\bpython[0-9.]*\b: not found",
    r"bad interpreter",
    r"failed to activate virtualenv",
    r"cannot execute: required file not found",
    r"modulenotfounderror: no module named ['\"]torch['\"]",
    r"modulenotfounderror: no module named ['\"]helion['\"]",
    r"importerror: no module named ['\"]torch['\"]",
    r"importerror: no module named ['\"]helion['\"]",
    r"modulenotfounderror: no module named 'torch'",
    r'modulenotfounderror: no module named "torch"',
    r"modulenotfounderror: no module named 'helion'",
    r'modulenotfounderror: no module named "helion"',
    r"importerror: no module named 'torch'",
    r'importerror: no module named "torch"',
    r"importerror: no module named 'helion'",
    r'importerror: no module named "helion"',
]

raise SystemExit(0 if any(re.search(pattern, text) for pattern in patterns) else 1)
PY
}

write_output_schema() {
  local schema_file="$1"
  cat > "$schema_file" <<'EOF'
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "status",
    "summary",
    "branch_memo",
    "commit_message",
    "baseline_geomean_ms",
    "candidate_geomean_ms",
    "tests_passed",
    "ready_to_submit"
  ],
  "properties": {
    "status": {
      "type": "string",
      "enum": ["improved", "no_change", "blocked", "invalid"]
    },
    "summary": {
      "type": "string"
    },
    "branch_memo": {
      "type": "string"
    },
    "commit_message": {
      "type": "string"
    },
    "baseline_geomean_ms": {
      "type": "number"
    },
    "candidate_geomean_ms": {
      "type": "number"
    },
    "tests_passed": {
      "type": "boolean"
    },
    "ready_to_submit": {
      "type": "boolean"
    }
  }
}
EOF
}

append_results_row() {
  local results_file="$1"
  shift
  "$PYTHON_BIN" - "$results_file" "$@" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = sys.argv[2:]
with path.open("a", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(row)
PY
}

append_changelog_entry() {
  local changelog_file="$1"
  local timestamp="$2"
  local run_tag="$3"
  local branch_name="$4"
  local commit_hash="$5"
  local baseline_ms="$6"
  local candidate_ms="$7"
  local improvement="$8"
  local description="$9"
  local commit_summary="${10}"
  local baseline_human
  local candidate_human
  local improvement_human

  baseline_human="$(humanize_duration_ms "$baseline_ms")"
  candidate_human="$(humanize_duration_ms "$candidate_ms")"
  improvement_human="$(humanize_percent "$improvement")"

  {
    echo
    echo "## $timestamp"
    echo
    echo "- Run tag: \`$run_tag\`"
    echo "- Branch: \`$branch_name\`"
    echo "- Commit: \`$commit_hash\`"
    echo "- Baseline geomean: \`$baseline_human\` (\`$baseline_ms ms\`)"
    echo "- Candidate geomean: \`$candidate_human\` (\`$candidate_ms ms\`)"
    echo "- Improvement: \`$improvement_human\`"
    echo "- Commit message: $(sanitize_text "$commit_summary")"
    echo "- Summary: $(sanitize_text "$description")"
  } >> "$changelog_file"
}

write_history_context() {
  local out_file="$1"
  local problem="$2"
  local state_file="$3"
  local changelog_file="$4"
  local results_file="$5"
  "$PYTHON_BIN" - \
    "$out_file" \
    "$problem" \
    "$state_file" \
    "$changelog_file" \
    "$results_file" \
    "$HISTORY_RESULTS_LIMIT" \
    "$HISTORY_CHANGELOG_LIMIT" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_file, problem, state_file, changelog_file, results_file, results_limit, changelog_limit = sys.argv[1:]
results_limit = int(results_limit)
changelog_limit = int(changelog_limit)

state_path = Path(state_file)
state = json.loads(state_path.read_text()) if state_path.exists() else {}

def humanize_ms(ms: float | int | str | None) -> str:
    if ms in ("", None):
        return ""
    value = float(ms)
    abs_value = abs(value)
    if abs_value >= 1000.0:
        scaled, unit = value / 1000.0, "s"
    elif abs_value >= 1.0:
        scaled, unit = value, "ms"
    elif abs_value >= 0.001:
        scaled, unit = value * 1000.0, "us"
    else:
        scaled, unit = value * 1_000_000.0, "ns"
    return f"{scaled:.4f} {unit}"

def humanize_pct(value: float | int | str | None) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.4f}%"

lines: list[str] = []
lines.append(f"# {problem} Experiment Context")
lines.append("")

if state:
    initial_ms = state.get("initial_geomean_ms")
    current_ms = state.get("current_geomean_ms")
    total_improvement = state.get("total_improvement_pct")
    lines.append("## Current Loop State")
    lines.append(f"- Run tag: `{state.get('run_tag', '')}`")
    lines.append(f"- Loop status: `{state.get('loop_status', '')}`")
    lines.append(f"- Next iteration: `{state.get('next_iteration', '')}`")
    lines.append(f"- Kept iterations: `{state.get('kept_iterations', '')}`")
    lines.append(f"- Stall count: `{state.get('stall_count', '')}` / `{state.get('max_stall_experiments', '')}`")
    if initial_ms not in ("", None):
        lines.append(f"- Original baseline geomean: `{humanize_ms(initial_ms)}` (`{initial_ms} ms`)")
    if current_ms not in ("", None):
        lines.append(f"- Current best geomean: `{humanize_ms(current_ms)}` (`{current_ms} ms`)")
    if total_improvement not in ("", None):
        lines.append(f"- Total improvement vs original baseline: `{humanize_pct(total_improvement)}`")
    best_summary = (state.get("best_summary") or "").strip()
    if best_summary:
        lines.append(f"- Best iteration summary: {best_summary}")
    lines.append("")

rows: list[dict[str, str]] = []
results_path = Path(results_file)
if results_path.exists():
    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get("problem") == problem]

if rows:
    lines.append("## Recent Experiment Attempts")
    for row in rows[-results_limit:]:
        timestamp = row.get("timestamp", "")
        run_tag = row.get("run_tag", "")
        status = row.get("status", "")
        baseline_ms = row.get("baseline_geomean_ms", "")
        candidate_ms = row.get("candidate_geomean_ms", "")
        improvement_pct = row.get("improvement_pct", "")
        description = (row.get("description", "") or "").strip()
        branch = row.get("branch", "")
        artifact_dir = row.get("artifact_dir", "")
        parts = [
            f"{timestamp} [{status}]",
            f"run={run_tag}",
        ]
        if baseline_ms:
            parts.append(f"baseline={humanize_ms(baseline_ms)}")
        if candidate_ms:
            parts.append(f"candidate={humanize_ms(candidate_ms)}")
        if improvement_pct:
            parts.append(f"delta={humanize_pct(improvement_pct)}")
        if branch:
            parts.append(f"branch={branch}")
        if description:
            parts.append(f"note={description}")
        if artifact_dir:
            parts.append(f"artifacts={artifact_dir}")
        lines.append(f"- {' | '.join(parts)}")
    lines.append("")

changelog_path = Path(changelog_file)
if changelog_path.exists():
    text = changelog_path.read_text().strip()
    sections = [section.strip() for section in text.split("\n## ")[1:] if section.strip()]
    if sections:
        lines.append("## Recent Accepted Changes")
        for section in sections[-changelog_limit:]:
            first, *rest = section.splitlines()
            lines.append(f"- {first.strip()}")
            for line in rest:
                stripped = line.strip()
                if stripped:
                    lines.append(f"  {stripped}")
        lines.append("")

lines.append("## Guidance For The Next Iteration")
lines.append("- Reconstruct the current best submission from this history before choosing a new experiment.")
lines.append("- Avoid repeating discarded ideas unless you have a concrete new reason they should work now.")
lines.append("- Propose one concrete new experiment, execute it, and report the outcome in the JSON summary.")
lines.append("")

Path(out_file).write_text("\n".join(lines).rstrip() + "\n")
PY
}

ensure_problem_worktree() {
  local worktree_root="$1"
  local active_branch="$2"
  local problem="$3"
  local workspace_path

  workspace_path="$(problem_workspace_path "$worktree_root")"

  if [[ -d "$workspace_path/$problem" ]]; then
    printf '%s\n' "$workspace_path"
    return
  fi

  if [[ -e "$worktree_root" && ! -e "$worktree_root/.git" ]]; then
    echo "ERROR: Cannot resume problem worktree because $worktree_root exists but is not a git worktree" >&2
    exit 1
  fi

  if [[ ! -e "$worktree_root" ]]; then
    if git_branch_exists "$active_branch"; then
      git -C "$REPO_ROOT" worktree add "$worktree_root" "$active_branch" >/dev/null
    else
      git -C "$REPO_ROOT" worktree add -b "$active_branch" "$worktree_root" "$REPO_BASE_BRANCH" >/dev/null
    fi
  fi

  if [[ ! -d "$workspace_path/$problem" ]]; then
    prepare_problem_workspace "$worktree_root" "$problem"
  fi

  printf '%s\n' "$workspace_path"
}

write_branch_trace() {
  local branch_trace_dir="$1"
  local iterations_dir="$2"
  local bootstrap_dir="$3"
  local final_artifact_dir="$4"
  local state_file="$5"
  local best_submission_snapshot="$6"
  local initial_submission_snapshot="$7"
  local codex_bin="$8"
  local python_bin="$9"
  "$PYTHON_BIN" - \
    "$branch_trace_dir" \
    "$iterations_dir" \
    "$bootstrap_dir" \
    "$final_artifact_dir" \
    "$state_file" \
    "$best_submission_snapshot" \
    "$initial_submission_snapshot" \
    "$codex_bin" \
    "$python_bin" <<'PY'
import shutil
import sys
from pathlib import Path

(
    branch_trace_dir,
    iterations_dir,
    bootstrap_dir,
    final_artifact_dir,
    state_file,
    best_submission_snapshot,
    initial_submission_snapshot,
    codex_bin,
    python_bin,
) = sys.argv[1:]

trace_root = Path(branch_trace_dir)
if trace_root.exists():
    shutil.rmtree(trace_root)
trace_root.mkdir(parents=True, exist_ok=True)

files_to_copy = {
    Path(state_file): trace_root / "run-state.json",
    Path(best_submission_snapshot): trace_root / "current-best-submission.py",
    Path(initial_submission_snapshot): trace_root / "initial-submission.py",
}

for src, dst in files_to_copy.items():
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

for source_dir_name, source_dir in (
    ("bootstrap", Path(bootstrap_dir)),
    ("final", Path(final_artifact_dir)),
    ("iterations", Path(iterations_dir)),
):
    if source_dir.exists():
        destination = trace_root / source_dir_name
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)

readme = trace_root / "README.md"
readme.write_text(
    "\n".join(
        [
            "# Codex Trace Bundle",
            "",
            "This directory is committed with the branch so the Codex experiment trace",
            "travels with the winning submission.",
            "",
            "Contents:",
            "- `iterations/iter-*/codex.log`: full Codex stdout/stderr trace per iteration",
            "- `iterations/iter-*/prompt.md`: exact prompt fed to Codex",
            "- `iterations/iter-*/response.json`: structured final Codex response",
            "- `iterations/iter-*/history-context.md`: reconstructed experiment history seen by Codex",
            "- `run-state.json`: resumable loop state at commit time",
            "- `current-best-submission.py`: best submission snapshot chosen by the loop",
            "- `initial-submission.py`: starting submission snapshot for the run",
            "",
            f"- `codex_bin`: `{codex_bin}`",
            f"- `python_bin`: `{python_bin}`",
        ]
    )
    + "\n"
)
PY
}

write_prompt() {
  local prompt_file="$1"
  local problem="$2"
  local workspace_path="$3"
  local artifact_dir="$4"
  local autotune_search_acf="$5"
  local history_context_file="$6"
  local current_baseline_ms="$7"
  local current_total_improvement_pct="$8"
  local iteration_number="$9"
  local leaderboard="${problem%_py}"
  local current_baseline_human
  local current_total_improvement_human

  current_baseline_human="$(humanize_duration_ms "$current_baseline_ms")"
  current_total_improvement_human="$(humanize_percent "$current_total_improvement_pct")"
  cat > "$prompt_file" <<EOF
You are working on the Helion Kernel Challenge problem \`$problem\`.

Workspace root: \`$workspace_path\`
Active problem directory: \`$problem\`
Leaderboard name: \`$leaderboard\`
GPU target: \`B200_Nebius\`
Required reasoning effort: highest available (\`xhigh\`)
Current iteration: \`$iteration_number\`
Current best baseline for this iteration: \`${current_baseline_human}\` (\`${current_baseline_ms} ms\`)
Current total improvement vs the original baseline: \`${current_total_improvement_human}\`

The Codex session for this run has local autotuning forced on:
- \`HELION_FORCE_AUTOTUNE=1\`
- \`HELION_AUTOTUNE_EFFORT=full\`
EOF

  if [[ -n "$autotune_search_acf" ]]; then
    cat >> "$prompt_file" <<EOF
- \`HELION_AUTOTUNE_SEARCH_ACF=$autotune_search_acf\`
EOF
  fi

  cat >> "$prompt_file" <<EOF

Mandatory workflow:
1. Read \`$problem/task.yml\`, \`$problem/reference.py\`, \`$problem/task.py\`, and \`$problem/submission.py\`.
2. Use the repo skills \`arch-b200\` and \`helion-language\`.
3. If \`.agents/skills/helion-language/SKILL.md\` exists, use it too as problem-local guidance.
4. Reconstruct the context of past experiments from the history section below before deciding what to try next.
5. Propose one concrete next experiment that is meaningfully different from recent discarded attempts, then execute that experiment within \`$problem/submission.py\`.
6. Modify only \`$problem/submission.py\`.
7. Re-validate locally as needed with:
   - \`$PYTHON_BIN eval.py test $problem\`
   - \`$PYTHON_BIN eval.py benchmark $problem\`
8. During experiments, rely on the forced local autotune environment to search for fresh configs even if \`submission.py\` already has hardcoded configs.
9. When autotune reports a best config, patch \`$problem/submission.py\` so the relevant shape entry hardcodes that winning \`helion.Config(...)\`.
10. If the winning config includes an \`advanced_controls_file\`, preserve it in the final hardcoded config.
11. Before finishing, remove any temporary autotune-only code paths so the final \`submission.py\` no longer relies on runtime autotuning. The committed file must not depend on \`HELION_FORCE_AUTOTUNE\`, \`autotune_effort\`, or \`autotune_search_acf\`.
12. If you do not find a real improvement this iteration, explain why the proposed experiment failed or why it was not safe to keep.
13. Do not edit any file outside \`$problem/submission.py\`.

Artifacts for this iteration live in:
- \`$artifact_dir\`

History context to use for this iteration:

EOF

  cat "$history_context_file" >> "$prompt_file"

  cat >> "$prompt_file" <<EOF

Final response rules:
- Return JSON only.
- The JSON must match the provided schema exactly.
- \`status\` must be one of: \`improved\`, \`no_change\`, \`blocked\`, \`invalid\`.
- \`branch_memo\` must be a short phrase describing the change.
- \`commit_message\` must be a concise commit message for a successful candidate.
- \`baseline_geomean_ms\` and \`candidate_geomean_ms\` must be numbers.
- \`tests_passed\` and \`ready_to_submit\` must be booleans.
EOF
}

run_local_test() {
  local workspace_path="$1"
  local problem="$2"
  local log_file="$3"
  (
    cd "$workspace_path"
    export PYTHONDONTWRITEBYTECODE=1
    "$PYTHON_BIN" eval.py test "$problem"
  ) 2>&1 | tee "$log_file"
}

run_local_benchmark() {
  local workspace_path="$1"
  local problem="$2"
  local log_file="$3"
  (
    cd "$workspace_path"
    export PYTHONDONTWRITEBYTECODE=1
    "$PYTHON_BIN" eval.py benchmark "$problem"
  ) 2>&1 | tee "$log_file"
}

compute_improvement_pct() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
before = float(sys.argv[1])
after = float(sys.argv[2])
print(((before - after) / before) * 100.0)
PY
}

validate_submission_hardcodes_configs() {
  local submission_file="$1"
  "$PYTHON_BIN" - "$submission_file" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(), filename=str(path))

forbidden_env_vars = {
    "HELION_FORCE_AUTOTUNE",
    "HELION_AUTOTUNE_EFFORT",
    "HELION_AUTOTUNE_SEARCH_ACF",
}

issues: list[str] = []

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "autotune":
            issues.append(f"autotune() call at line {node.lineno}")

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value in forbidden_env_vars:
            issues.append(
                f"forbidden autotune environment reference {node.value!r} at line {node.lineno}"
            )

for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if not isinstance(decorator.func, ast.Attribute):
            continue
        if decorator.func.attr != "kernel":
            continue

        keyword_names = {keyword.arg for keyword in decorator.keywords if keyword.arg}
        bad_autotune_kwargs = sorted(
            name for name in keyword_names if name.startswith("autotune_")
        )
        if bad_autotune_kwargs:
            issues.append(
                "helion.kernel decorator at line "
                f"{decorator.lineno} uses autotune kwargs {bad_autotune_kwargs}"
            )
        if "config" not in keyword_names and "configs" not in keyword_names:
            issues.append(
                f"helion.kernel decorator at line {decorator.lineno} is missing config/configs"
            )

if issues:
    for issue in issues:
        print(issue)
    raise SystemExit(1)

print("ok")
PY
}

only_submission_changed() {
  local before_manifest="$1"
  local after_manifest="$2"
  local expected_path="$3"
  "$PYTHON_BIN" - "$before_manifest" "$after_manifest" "$expected_path" <<'PY'
import json
import sys
from pathlib import Path

before = json.loads(Path(sys.argv[1]).read_text())
after = json.loads(Path(sys.argv[2]).read_text())
expected = sys.argv[3]

changed = []
paths = sorted(set(before) | set(after))
for path in paths:
    if before.get(path) != after.get(path):
        changed.append(path)

if not changed:
    print("clean")
    raise SystemExit(0)

if changed == [expected]:
    print("ok")
    raise SystemExit(0)

for path in changed:
    print(path)
raise SystemExit(1)
PY
}

write_manifest() {
  local root_dir="$1"
  local out_file="$2"
  "$PYTHON_BIN" - "$root_dir" "$out_file" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
manifest = {}

for current_root, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for filename in filenames:
        path = Path(current_root) / filename
        rel = path.relative_to(root).as_posix()
        manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
PY
}

unique_branch_name() {
  local base="$1"
  local candidate="$base"
  local suffix=1
  while git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$candidate"; do
    candidate="${base}-${suffix}"
    suffix=$((suffix + 1))
  done
  printf '%s\n' "$candidate"
}

process_problem() {
  local problem="$1"
  local problem_run_dir="$ARTIFACT_ROOT/$problem"
  local iterations_dir="$problem_run_dir/iterations"
  local bootstrap_dir="$problem_run_dir/bootstrap"
  local final_artifact_dir="$problem_run_dir/final"
  local state_file="$problem_run_dir/run-state.json"
  local best_submission_snapshot="$problem_run_dir/current-best-submission.py"
  local initial_submission_snapshot="$problem_run_dir/initial-submission.py"
  local worktree_root="$WORKTREES_ROOT/$RUN_TAG-$problem"
  local branch_trace_rel="$WORKSPACE_REL/.branch-traces/$problem/$RUN_TAG"
  local branch_trace_dir="$worktree_root/$branch_trace_rel"
  local active_branch="codex/loop-$RUN_TAG-$problem"
  local history_dir="$STATE_ROOT/history/$problem"
  local changelog_file="$history_dir/CHANGELOG.md"
  local results_file="$history_dir/results.tsv"
  local expected_submission="$WORKSPACE_REL/$problem/submission.py"
  local history_context_file
  local prompt_file
  local schema_file
  local response_file
  local codex_log
  local baseline_test_log="$bootstrap_dir/baseline-test.log"
  local baseline_bench_log="$bootstrap_dir/baseline-benchmark.log"
  local candidate_test_log
  local candidate_bench_log
  local before_manifest
  local after_manifest
  local submit_test_log="$final_artifact_dir/popcorn-test.log"
  local submit_leaderboard_log="$final_artifact_dir/popcorn-leaderboard.log"
  local worktree_workspace
  local initial_geomean=""
  local initial_points=""
  local current_geomean=""
  local current_points=""
  local baseline_geomean=""
  local baseline_points=""
  local candidate_geomean=""
  local candidate_points=""
  local improvement_pct=""
  local total_improvement_pct=""
  local autotune_search_acf=""
  local best_branch_memo=""
  local best_commit_message=""
  local best_summary=""
  local best_ready_to_submit="false"
  local branch_memo=""
  local branch_slug=""
  local final_branch=""
  local commit_message=""
  local commit_hash=""
  local ready_to_submit=""
  local tests_passed=""
  local status=""
  local loop_status="active"
  local stop_reason=""
  local pause_reason=""
  local pause_stage=""
  local pending_iteration=""
  local pending_candidate_snapshot=""
  local resume_allowed="false"
  local next_iteration="1"
  local kept_iterations="0"
  local stall_count="0"
  local change_state=""
  local summary_text=""
  local leaderboard="${problem%_py}"
  local iteration_dir=""
  local result_artifact_dir="$problem_run_dir"
  local candidate_submission_snapshot=""
  local retry_pending_candidate="false"
  local state_exists="false"
  local run_finalized="false"
  local legacy_resumable="false"

  mkdir -p "$problem_run_dir" "$iterations_dir" "$bootstrap_dir" "$final_artifact_dir"
  ensure_problem_history_files "$problem"

  save_state() {
    if [[ -n "$initial_geomean" && -n "$current_geomean" ]]; then
      total_improvement_pct="$(compute_improvement_pct "$initial_geomean" "$current_geomean")"
    else
      total_improvement_pct=""
    fi

    "$PYTHON_BIN" - \
      "$state_file" \
      "$problem" \
      "$RUN_TAG" \
      "$loop_status" \
      "$stop_reason" \
      "$pause_reason" \
      "$pause_stage" \
      "$pending_iteration" \
      "$pending_candidate_snapshot" \
      "$resume_allowed" \
      "$worktree_root" \
      "$worktree_workspace" \
      "$active_branch" \
      "$final_branch" \
      "$initial_geomean" \
      "$initial_points" \
      "$current_geomean" \
      "$current_points" \
      "$total_improvement_pct" \
      "$next_iteration" \
      "$stall_count" \
      "$kept_iterations" \
      "$MAX_EXPERIMENTS_PER_PROBLEM" \
      "$MAX_STALL_EXPERIMENTS" \
      "$best_summary" \
      "$best_branch_memo" \
      "$best_commit_message" \
      "$best_ready_to_submit" \
      "$best_submission_snapshot" \
      "$initial_submission_snapshot" \
      "$commit_hash" <<'PY'
import json
import sys
from pathlib import Path

(
    state_file,
    problem,
    run_tag,
    loop_status,
    stop_reason,
    pause_reason,
    pause_stage,
    pending_iteration,
    pending_candidate_snapshot,
    resume_allowed,
    worktree_root,
    worktree_workspace,
    active_branch,
    final_branch,
    initial_geomean,
    initial_points,
    current_geomean,
    current_points,
    total_improvement_pct,
    next_iteration,
    stall_count,
    kept_iterations,
    max_experiments_per_problem,
    max_stall_experiments,
    best_summary,
    best_branch_memo,
    best_commit_message,
    best_ready_to_submit,
    best_submission_snapshot,
    initial_submission_snapshot,
    commit_hash,
) = sys.argv[1:]

def maybe_float(value: str):
    return float(value) if value not in {"", "None"} else None

def maybe_int(value: str):
    return int(value) if value not in {"", "None"} else None

data = {
    "problem": problem,
    "run_tag": run_tag,
    "loop_status": loop_status,
    "stop_reason": stop_reason,
    "pause_reason": pause_reason,
    "pause_stage": pause_stage,
    "pending_iteration": maybe_int(pending_iteration),
    "pending_candidate_snapshot": pending_candidate_snapshot,
    "resume_allowed": resume_allowed == "true",
    "worktree_root": worktree_root,
    "worktree_workspace": worktree_workspace,
    "active_branch": active_branch,
    "final_branch": final_branch,
    "initial_geomean_ms": maybe_float(initial_geomean),
    "initial_points_ms": initial_points,
    "current_geomean_ms": maybe_float(current_geomean),
    "current_points_ms": current_points,
    "total_improvement_pct": maybe_float(total_improvement_pct),
    "next_iteration": maybe_int(next_iteration),
    "stall_count": maybe_int(stall_count),
    "kept_iterations": maybe_int(kept_iterations),
    "max_experiments_per_problem": maybe_int(max_experiments_per_problem),
    "max_stall_experiments": maybe_int(max_stall_experiments),
    "best_summary": best_summary,
    "best_branch_memo": best_branch_memo,
    "best_commit_message": best_commit_message,
    "best_ready_to_submit": best_ready_to_submit == "true",
    "best_submission_snapshot": best_submission_snapshot,
    "initial_submission_snapshot": initial_submission_snapshot,
    "commit_hash": commit_hash,
}

Path(state_file).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  }

  record_result() {
    local result_status="$1"
    local description="$2"
    local row_artifact_dir="${3:-$result_artifact_dir}"
    local result_timestamp
    result_timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"

    append_results_row \
      "$results_file" \
      "$result_timestamp" \
      "$RUN_TAG" \
      "$problem" \
      "$result_status" \
      "${baseline_geomean:-}" \
      "${candidate_geomean:-}" \
      "${improvement_pct:-}" \
      "${tests_passed:-}" \
      "${ready_to_submit:-}" \
      "${final_branch:-}" \
      "${commit_hash:-}" \
      "$(sanitize_text "${commit_message:-}")" \
      "$(sanitize_text "$description")" \
      "${baseline_points:-}" \
      "${candidate_points:-}" \
      "$row_artifact_dir"
  }

  clear_pause_state() {
    pause_reason=""
    pause_stage=""
    resume_allowed="false"
  }

  pause_run() {
    local stage="$1"
    local reason="$2"
    local snapshot_path="${3:-}"

    loop_status="paused"
    stop_reason="$reason"
    pause_reason="$reason"
    pause_stage="$stage"
    resume_allowed="true"
    pending_iteration="${next_iteration:-}"
    pending_candidate_snapshot="$snapshot_path"
    stall_count="0"
    save_state
    record_result "PAUSED" "$reason" "$result_artifact_dir"
    log "PAUSED $problem: $reason"
    log "PAUSED $problem: rerun with --resume after fixing the environment/tooling issue"
  }

  restore_best_submission() {
    cp "$best_submission_snapshot" "$worktree_workspace/$problem/submission.py"
  }

  establish_baseline() {
    result_artifact_dir="$bootstrap_dir"

    if ! run_local_test "$worktree_workspace" "$problem" "$baseline_test_log" >/dev/null; then
      if log_indicates_resumable_failure "$baseline_test_log"; then
        pause_run "baseline_test" "Baseline test failed due to environment/tooling issue"
        return 1
      fi
      loop_status="completed"
      stop_reason="Baseline test failed"
      save_state
      record_result "BLOCKED" "Baseline test failed" "$bootstrap_dir"
      log "SKIP $problem: baseline test failed"
      return 1
    fi

    if ! run_local_benchmark "$worktree_workspace" "$problem" "$baseline_bench_log" >/dev/null; then
      if log_indicates_resumable_failure "$baseline_bench_log"; then
        pause_run "baseline_benchmark" "Baseline benchmark failed due to environment/tooling issue"
        return 1
      fi
      loop_status="completed"
      stop_reason="Baseline benchmark failed"
      save_state
      record_result "BLOCKED" "Baseline benchmark failed" "$bootstrap_dir"
      log "SKIP $problem: baseline benchmark failed"
      return 1
    fi

    initial_geomean="$(parse_benchmark_geomean "$baseline_bench_log")"
    initial_points="$(parse_benchmark_points "$baseline_bench_log")"
    current_geomean="$initial_geomean"
    current_points="$initial_points"
    cp "$worktree_workspace/$problem/submission.py" "$initial_submission_snapshot"
    cp "$worktree_workspace/$problem/submission.py" "$best_submission_snapshot"
    loop_status="active"
    stop_reason=""
    next_iteration="1"
    stall_count="0"
    kept_iterations="0"
    best_summary=""
    best_branch_memo=""
    best_commit_message=""
    best_ready_to_submit="false"
    clear_pause_state
    pending_iteration=""
    pending_candidate_snapshot=""
    retry_pending_candidate="false"
    save_state
    log "BASELINE $problem: $(humanize_duration_ms "$initial_geomean")"
    return 0
  }

  if [[ -f "$state_file" ]]; then
    state_exists="true"
    loop_status="$(json_field "$state_file" loop_status)"
    stop_reason="$(json_field "$state_file" stop_reason)"
    active_branch="$(json_field "$state_file" active_branch)"
    final_branch="$(json_field "$state_file" final_branch)"
    initial_geomean="$(json_field "$state_file" initial_geomean_ms)"
    initial_points="$(json_field "$state_file" initial_points_ms)"
    current_geomean="$(json_field "$state_file" current_geomean_ms)"
    current_points="$(json_field "$state_file" current_points_ms)"
    next_iteration="$(json_field "$state_file" next_iteration)"
    stall_count="$(json_field "$state_file" stall_count)"
    kept_iterations="$(json_field "$state_file" kept_iterations)"
    best_summary="$(json_field "$state_file" best_summary)"
    best_branch_memo="$(json_field "$state_file" best_branch_memo)"
    best_commit_message="$(json_field "$state_file" best_commit_message)"
    best_ready_to_submit="$(json_field "$state_file" best_ready_to_submit)"
    commit_hash="$(json_field "$state_file" commit_hash)"
    total_improvement_pct="$(json_field "$state_file" total_improvement_pct)"
    pause_reason="$(json_field_optional "$state_file" pause_reason)"
    pause_stage="$(json_field_optional "$state_file" pause_stage)"
    pending_iteration="$(json_field_optional "$state_file" pending_iteration)"
    pending_candidate_snapshot="$(json_field_optional "$state_file" pending_candidate_snapshot)"
    resume_allowed="$(json_field_optional "$state_file" resume_allowed)"
  fi

  if [[ -n "$commit_hash" || -n "$final_branch" ]]; then
    run_finalized="true"
  fi

  if [[ "$state_exists" != "true" && "$RESUME_RUN" == "true" ]]; then
    echo "ERROR: No prior run state found for run tag $RUN_TAG and problem $problem" >&2
    exit 2
  fi

  if [[ "$loop_status" == "completed" && "$run_finalized" != "true" ]]; then
    legacy_resumable="true"
  fi

  if [[ "$run_finalized" == "true" ]]; then
    if [[ "$RESUME_RUN" == "true" ]]; then
      echo "ERROR: Run $RUN_TAG for $problem is already finalized on branch ${final_branch:-<unknown>}" >&2
      exit 2
    fi
    if [[ "$loop_status" == "completed" ]]; then
      log "DONE $problem: loop already completed for run tag $RUN_TAG"
      return
    fi
  fi

  if [[ "$state_exists" == "true" && "$RESUME_RUN" != "true" ]]; then
    if [[ "$loop_status" == "paused" || "$legacy_resumable" == "true" ]]; then
      log "PAUSED $problem: ${pause_reason:-$stop_reason}"
      log "PAUSED $problem: rerun with --resume to continue run tag $RUN_TAG"
      return
    fi
  fi

  if [[ "$RESUME_RUN" == "true" && "$state_exists" == "true" ]]; then
    if [[ -n "$pending_iteration" ]]; then
      next_iteration="$pending_iteration"
    fi
    if [[ -n "$pending_candidate_snapshot" && -f "$pending_candidate_snapshot" ]]; then
      retry_pending_candidate="true"
    fi
  fi

  autotune_search_acf="$(problem_autotune_search_acf "$problem")"
  worktree_workspace="$(ensure_problem_worktree "$worktree_root" "$active_branch" "$problem")"

  if [[ "$state_exists" != "true" ]]; then
    log "START $problem"
    if ! establish_baseline; then
      return
    fi
  else
    log "RESUME $problem: continuing run tag $RUN_TAG from iteration $next_iteration with current best $(humanize_duration_ms "$current_geomean")"
    if [[ ! -f "$best_submission_snapshot" ]]; then
      cp "$worktree_workspace/$problem/submission.py" "$best_submission_snapshot"
    fi
    if [[ ! -f "$initial_submission_snapshot" ]]; then
      cp "$best_submission_snapshot" "$initial_submission_snapshot"
    fi
    if [[ "$RESUME_RUN" == "true" && ( "$pause_stage" == "baseline_test" || "$pause_stage" == "baseline_benchmark" || -z "$initial_geomean" ) ]]; then
      log "RESUME $problem: re-establishing baseline after paused baseline stage"
      if ! establish_baseline; then
        return
      fi
    elif [[ "$RESUME_RUN" == "true" ]]; then
      clear_pause_state
      loop_status="active"
      stop_reason=""
      save_state
    fi
  fi

  local -a codex_env=(
    PYTHONDONTWRITEBYTECODE=1
    HELION_FORCE_AUTOTUNE=1
    HELION_AUTOTUNE_EFFORT=full
    HELION_AUTOTUNE_PROGRESS_BAR=0
  )
  if [[ -n "$autotune_search_acf" ]]; then
    codex_env+=(HELION_AUTOTUNE_SEARCH_ACF="$autotune_search_acf")
  fi

  while :; do
    if (( next_iteration > MAX_EXPERIMENTS_PER_PROBLEM )); then
      stop_reason="Reached MAX_EXPERIMENTS_PER_PROBLEM=$MAX_EXPERIMENTS_PER_PROBLEM"
      break
    fi

    if (( stall_count >= MAX_STALL_EXPERIMENTS )); then
      stop_reason="Reached MAX_STALL_EXPERIMENTS=$MAX_STALL_EXPERIMENTS"
      break
    fi

    iteration_dir="$iterations_dir/iter-$(printf '%03d' "$next_iteration")"
    mkdir -p "$iteration_dir"
    result_artifact_dir="$iteration_dir"
    history_context_file="$iteration_dir/history-context.md"
    prompt_file="$iteration_dir/prompt.md"
    schema_file="$iteration_dir/output-schema.json"
    response_file="$iteration_dir/response.json"
    codex_log="$iteration_dir/codex.log"
    candidate_test_log="$iteration_dir/candidate-test.log"
    candidate_bench_log="$iteration_dir/candidate-benchmark.log"
    candidate_submission_snapshot="$iteration_dir/candidate-submission.py"
    before_manifest="$iteration_dir/workspace-before.json"
    after_manifest="$iteration_dir/workspace-after.json"

    baseline_geomean="$current_geomean"
    baseline_points="$current_points"
    candidate_geomean=""
    candidate_points=""
    improvement_pct=""
    summary_text=""
    status=""
    tests_passed=""
    ready_to_submit=""
    branch_memo=""
    commit_message=""

    if [[ "$retry_pending_candidate" == "true" ]]; then
      log "ITER $problem: retrying pending candidate for iteration $next_iteration after resume"
      cp "$pending_candidate_snapshot" "$worktree_workspace/$problem/submission.py"
      if [[ -f "$response_file" ]]; then
        summary_text="$(json_field_optional "$response_file" summary)"
        status="$(json_field_optional "$response_file" status)"
        tests_passed="$(json_field_optional "$response_file" tests_passed)"
        ready_to_submit="$(json_field_optional "$response_file" ready_to_submit)"
        branch_memo="$(json_field_optional "$response_file" branch_memo)"
        commit_message="$(json_field_optional "$response_file" commit_message)"
      fi
      change_state="ok"
    else
      restore_best_submission
      write_manifest "$worktree_workspace" "$before_manifest"
      write_history_context "$history_context_file" "$problem" "$state_file" "$changelog_file" "$results_file"
      write_output_schema "$schema_file"
      write_prompt \
        "$prompt_file" \
        "$problem" \
        "$worktree_workspace" \
        "$iteration_dir" \
        "$autotune_search_acf" \
        "$history_context_file" \
        "$baseline_geomean" \
        "${total_improvement_pct:-0}" \
        "$next_iteration"

      log "ITER $problem: starting iteration $next_iteration with baseline $(humanize_duration_ms "$baseline_geomean")"
      log "ITER $problem: streaming Codex trace to console and $codex_log"

      if ! env "${codex_env[@]}" "$CODEX_BIN" \
        -a never \
        exec \
        -C "$worktree_workspace" \
        -s workspace-write \
        -c 'model_reasoning_effort="xhigh"' \
        --output-schema "$schema_file" \
        --output-last-message "$response_file" \
        < "$prompt_file" 2>&1 | tee "$codex_log" | sed -u "s/^/[codex $problem iter $next_iteration] /"; then
        pause_run "codex_exec" "codex exec failed on iteration $next_iteration"
        return
      fi

      if [[ ! -s "$response_file" ]]; then
        pause_run "codex_exec" "Empty codex response on iteration $next_iteration"
        return
      fi

      if ! validate_json_response "$response_file"; then
        pause_run "codex_exec" "Invalid codex JSON response on iteration $next_iteration"
        return
      fi

      summary_text="$(json_field "$response_file" summary)"
      status="$(json_field "$response_file" status)"
      tests_passed="$(json_field "$response_file" tests_passed)"
      ready_to_submit="$(json_field "$response_file" ready_to_submit)"
      branch_memo="$(json_field "$response_file" branch_memo)"
      commit_message="$(json_field "$response_file" commit_message)"

      write_manifest "$worktree_workspace" "$after_manifest"

      if ! change_state="$(only_submission_changed "$before_manifest" "$after_manifest" "$problem/submission.py")"; then
        stall_count="$((stall_count + 1))"
        record_result "DISCARD" "${summary_text:-Files outside submission.py were modified}" "$iteration_dir"
        next_iteration="$((next_iteration + 1))"
        save_state
        log "ITER $problem: discarded iteration due to modifications outside submission.py"
        continue
      fi

      if [[ "$change_state" == "clean" ]]; then
        stall_count="$((stall_count + 1))"
        record_result "DISCARD" "${summary_text:-Codex made no submission.py changes}" "$iteration_dir"
        next_iteration="$((next_iteration + 1))"
        save_state
        log "ITER $problem: no submission.py changes"
        continue
      fi
    fi

    cp "$worktree_workspace/$problem/submission.py" "$candidate_submission_snapshot"

    if ! validate_submission_hardcodes_configs "$worktree_workspace/$problem/submission.py" >/dev/null; then
      stall_count="$((stall_count + 1))"
      record_result "DISCARD" "${summary_text:-Final submission.py still relies on runtime autotuning}" "$iteration_dir"
      next_iteration="$((next_iteration + 1))"
      pending_candidate_snapshot=""
      pending_iteration=""
      retry_pending_candidate="false"
      save_state
      log "ITER $problem: discarded runtime-autotuned final file"
      continue
    fi

    if ! run_local_test "$worktree_workspace" "$problem" "$candidate_test_log" >/dev/null; then
      if log_indicates_resumable_failure "$candidate_test_log"; then
        pause_run "candidate_test" "Candidate test failed due to environment/tooling issue on iteration $next_iteration" "$candidate_submission_snapshot"
        return
      fi
      stall_count="$((stall_count + 1))"
      record_result "DISCARD" "${summary_text:-Candidate test failed}" "$iteration_dir"
      next_iteration="$((next_iteration + 1))"
      pending_candidate_snapshot=""
      pending_iteration=""
      retry_pending_candidate="false"
      save_state
      log "ITER $problem: candidate test failed"
      continue
    fi

    if ! run_local_benchmark "$worktree_workspace" "$problem" "$candidate_bench_log" >/dev/null; then
      if log_indicates_resumable_failure "$candidate_bench_log"; then
        pause_run "candidate_benchmark" "Candidate benchmark failed due to environment/tooling issue on iteration $next_iteration" "$candidate_submission_snapshot"
        return
      fi
      stall_count="$((stall_count + 1))"
      record_result "DISCARD" "${summary_text:-Candidate benchmark failed}" "$iteration_dir"
      next_iteration="$((next_iteration + 1))"
      pending_candidate_snapshot=""
      pending_iteration=""
      retry_pending_candidate="false"
      save_state
      log "ITER $problem: candidate benchmark failed"
      continue
    fi

    candidate_geomean="$(parse_benchmark_geomean "$candidate_bench_log")"
    candidate_points="$(parse_benchmark_points "$candidate_bench_log")"
    improvement_pct="$(compute_improvement_pct "$baseline_geomean" "$candidate_geomean")"

    if [[ "$status" == "blocked" ]]; then
      if [[ "$retry_pending_candidate" == "true" ]]; then
        status="improved"
      else
        loop_status="completed"
        stop_reason="${summary_text:-Codex reported blocked}"
        save_state
        record_result "BLOCKED" "$stop_reason" "$iteration_dir"
        log "SKIP $problem: $stop_reason"
        return
      fi
    fi

    if [[ "$status" == "invalid" ]]; then
      if [[ "$retry_pending_candidate" == "true" ]]; then
        status="improved"
      else
        loop_status="completed"
        stop_reason="${summary_text:-Codex reported invalid}"
        save_state
        record_result "INVALID" "$stop_reason" "$iteration_dir"
        log "SKIP $problem: $stop_reason"
        return
      fi
    fi

    if [[ "$retry_pending_candidate" != "true" && "$tests_passed" != "true" ]]; then
      stall_count="$((stall_count + 1))"
      record_result "DISCARD" "${summary_text:-Codex reported tests_passed=false}" "$iteration_dir"
      next_iteration="$((next_iteration + 1))"
      pending_candidate_snapshot=""
      pending_iteration=""
      retry_pending_candidate="false"
      save_state
      log "ITER $problem: Codex reported tests_passed=false"
      continue
    fi

    if ! "$PYTHON_BIN" - "$candidate_geomean" "$baseline_geomean" <<'PY'
import sys
candidate = float(sys.argv[1])
baseline = float(sys.argv[2])
raise SystemExit(0 if candidate < baseline else 1)
PY
    then
      stall_count="$((stall_count + 1))"
      record_result "DISCARD" "${summary_text:-No benchmark improvement this iteration}" "$iteration_dir"
      next_iteration="$((next_iteration + 1))"
      pending_candidate_snapshot=""
      pending_iteration=""
      retry_pending_candidate="false"
      save_state
      log "ITER $problem: no improvement over current best ($(humanize_duration_ms "$candidate_geomean") vs $(humanize_duration_ms "$baseline_geomean"), delta $(humanize_percent "$improvement_pct"))"
      continue
    fi

    cp "$worktree_workspace/$problem/submission.py" "$best_submission_snapshot"
    current_geomean="$candidate_geomean"
    current_points="$candidate_points"
    best_summary="$summary_text"
    best_branch_memo="$branch_memo"
    best_commit_message="$commit_message"
    best_ready_to_submit="$ready_to_submit"
    kept_iterations="$((kept_iterations + 1))"
    stall_count="0"
    next_iteration="$((next_iteration + 1))"
    pending_candidate_snapshot=""
    pending_iteration=""
    retry_pending_candidate="false"
    save_state
    record_result "KEEP" "${summary_text:-Improved candidate kept as new baseline}" "$iteration_dir"
    log "ITER $problem: kept improved candidate at $(humanize_duration_ms "$current_geomean") (delta $(humanize_percent "$improvement_pct"))"
  done

  if (( kept_iterations == 0 )); then
    loop_status="completed"
    save_state
    log "DONE $problem: no improving candidate found (${stop_reason})"
    return
  fi

  total_improvement_pct="$(compute_improvement_pct "$initial_geomean" "$current_geomean")"
  if ! "$PYTHON_BIN" - "$total_improvement_pct" "$MIN_SPEEDUP_PCT" <<'PY'
import sys
actual = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if actual >= threshold else 1)
PY
  then
    loop_status="completed"
    save_state
    log "DONE $problem: best total improvement $(humanize_percent "$total_improvement_pct") is below threshold $(humanize_percent "$MIN_SPEEDUP_PCT")"
    return
  fi

  restore_best_submission
  commit_message="$best_commit_message"
  branch_memo="$best_branch_memo"
  ready_to_submit="$best_ready_to_submit"
  summary_text="$best_summary"
  baseline_geomean="$initial_geomean"
  baseline_points="$initial_points"
  candidate_geomean="$current_geomean"
  candidate_points="$current_points"
  improvement_pct="$total_improvement_pct"

  branch_slug="$(sanitize_branch_memo "${branch_memo:-candidate}")"
  final_branch="$(unique_branch_name "codex/autoresearch-$problem-$RUN_TAG-$branch_slug")"
  if [[ -z "$commit_message" ]]; then
    commit_message="helion/$problem: $branch_slug"
  fi

  git -C "$worktree_root" branch -m "$final_branch"
  save_state
  write_branch_trace \
    "$branch_trace_dir" \
    "$iterations_dir" \
    "$bootstrap_dir" \
    "$final_artifact_dir" \
    "$state_file" \
    "$best_submission_snapshot" \
    "$initial_submission_snapshot" \
    "$CODEX_BIN" \
    "$PYTHON_BIN"
  git -C "$worktree_root" add -f "$expected_submission" "$branch_trace_rel"
  git -C "$worktree_root" commit -m "$commit_message" >/dev/null
  commit_hash="$(git -C "$worktree_root" rev-parse HEAD)"
  loop_status="completed"
  save_state

  record_result "COMMIT" "${summary_text:-Committed best improvement after iterative search}" "$final_artifact_dir"
  append_changelog_entry \
    "$changelog_file" \
    "$(date '+%Y-%m-%d %H:%M:%S %z')" \
    "$RUN_TAG" \
    "$final_branch" \
    "$commit_hash" \
    "$initial_geomean" \
    "$current_geomean" \
    "$total_improvement_pct" \
    "${summary_text:-Committed best improvement after iterative search}" \
    "$commit_message"
  log "COMMIT $problem: $final_branch ($(humanize_duration_ms "$initial_geomean") -> $(humanize_duration_ms "$current_geomean"), delta $(humanize_percent "$total_improvement_pct"))"

  if [[ "$SKIP_SUBMIT" == "true" ]]; then
    log "DONE $problem: submit skipped"
    return
  fi

  if [[ "$ready_to_submit" != "true" ]]; then
    log "DONE $problem: candidate committed but ready_to_submit=false"
    return
  fi

  mkdir -p "$final_artifact_dir"

  if ! "$POPCORN_BIN" submit "$worktree_workspace/$problem/submission.py" \
    --gpu B200_Nebius \
    --leaderboard "$leaderboard" \
    --mode test \
    --no-tui >"$submit_test_log" 2>&1; then
    log "DONE $problem: popcorn test submission failed"
    return
  fi

  if ! "$POPCORN_BIN" submit "$worktree_workspace/$problem/submission.py" \
    --gpu B200_Nebius \
    --leaderboard "$leaderboard" \
    --mode leaderboard \
    --no-tui >"$submit_leaderboard_log" 2>&1; then
    log "DONE $problem: popcorn leaderboard submission failed"
    return
  fi

  log "DONE $problem: submitted successfully"
}

main() {
  local -a problems=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && problems+=("$line")
  done < <(discover_problems)

  if [[ -z "$PROBLEM_FILTER" ]]; then
    echo "ERROR: --problem is required. This launcher processes exactly one problem at a time." >&2
    echo "Available problems: ${problems[*]}" >&2
    exit 2
  fi

  if [[ "${#problems[@]}" -ne 1 ]]; then
    echo "ERROR: Expected exactly one problem after filtering." >&2
    exit 2
  fi

  mkdir -p "$WORKTREES_ROOT" "$ARTIFACT_ROOT"

  log "Run tag: $RUN_TAG"
  log "Workspace: $SCRIPT_DIR"
  log "Problem: ${problems[0]}"
  log "Artifacts: $ARTIFACT_ROOT"
  log "Worktrees: $WORKTREES_ROOT"
  log "Venv: ${VENV_PATH:-<none>}"
  log "Python: $(command -v "$PYTHON_BIN")"
  log "Submit: $([[ "$SKIP_SUBMIT" == "true" ]] && echo no || echo yes)"

  process_problem "${problems[0]}"
}

main "$@"
