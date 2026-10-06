#!/usr/bin/env bash
# =============================================================================
# run_judge_eval.sh — activity attribution (LLM-as-judge), Q3 of the MIAAM paper
#
# Runs llm_as_judge_eval_per_module.py once per entry of MODELS. For every
# module, the judge sees all objectives and activities (pedagogical intents and
# one example exercise per activity) and must place each test exercise in the
# right activity. The settings below are the ones reported in the paper
# (appendix "Details for Q3"):
#   - all nine modules of MIAAM V2 (544 activities);
#   - Gemma-4-31B on all 9,926 exercises;
#   - GPT-5 and GPT-5.5 on the same 10% of each activity's exercises (1,026),
#     which seed 1 selects;
#   - 3 trials per exercise with re-sampled examples, majority vote;
#   - temperature 0, low reasoning effort, through OpenRouter;
#   - a chance baseline (uniform activity within the module, no API calls).
#
# Usage:
#   bash evaluation_llm/run_judge_eval.sh              # run all models sequentially
#   bash evaluation_llm/run_judge_eval.sh --parallel   # run all in background
#   bash evaluation_llm/run_judge_eval.sh --dry-run    # print commands without executing
#   bash evaluation_llm/run_judge_eval.sh --force      # re-evaluate from scratch, discarding existing results
#
# Without --force a COMPLETE result file is skipped; a partial one (interrupted,
# or out of credit) is resumed by the python script, chunk by chunk.
#
# Results go to:  $OUTPUT_DIR/llm_as_judge_quality_per-module-<tag>.json
#                 (<tag> is the model id after the last "/")
# Logs go to:     $LOG_DIR/judge-eval_<tag>.log
#
# The paper counts "None" verdicts (abstentions, ties) as errors, but the JSON
# overall_accuracy excludes them: recompute accuracy from predicted_activities
# vs ground_truth_activities.
# =============================================================================

# ── Execution mode ─────────────────────────────────────────────────────────────
PARALLEL=false
DRY_RUN=false
FORCE=false
for arg in "$@"; do
  case "$arg" in
    --parallel) PARALLEL=true ;;
    --dry-run)  DRY_RUN=true  ;;
    --force)    FORCE=true    ;;
    *) echo "Unknown option: $arg (expected --parallel, --dry-run or --force)" >&2; exit 1 ;;
  esac
done

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="$SCRIPT_DIR/llm_as_judge_eval_per_module.py"
PYTHON="${PYTHON:-python3}"   # override to use another interpreter, e.g. PYTHON=.venv/bin/python
LOG_DIR="$REPO_DIR/logs"
OUTPUT_DIR="$REPO_DIR/results/llm_as_judge_eval/MIAAM_V2"

# MIAAM V2 snapshot. Set MIAAM_DATASET to point elsewhere; by default use the
# Hugging Face cache, where the files live in the snapshot refs/main points to.
HF_DATASET_CACHE="$HOME/.cache/huggingface/hub/datasets--GAIMHE--MIAAM-V2"
PATH_DATASET="${MIAAM_DATASET:-$HF_DATASET_CACHE/snapshots/$(cat "$HF_DATASET_CACHE/refs/main" 2>/dev/null)}"
if [[ ! -f "$PATH_DATASET/data/maths_exercises_table.parquet" ]]; then
  echo "Error: MIAAM-V2 not found at $PATH_DATASET" >&2
  echo "       run: hf download GAIMHE/MIAAM-V2 --repo-type dataset   (or set MIAAM_DATASET)" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# ── Load .env file ─────────────────────────────────────────────────────────────
# Expected content: OPENROUTER_API_KEY=sk-or-v1-...  (the chance baseline needs no key)
ENV_FILE="$REPO_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +o allexport
  echo "Loaded env vars from $ENV_FILE"
elif [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "Warning: no $ENV_FILE and OPENROUTER_API_KEY is unset — only the chance baseline can run"
fi

# =============================================================================
# Evaluation matrix
# =============================================================================

# ── Judge models ───────────────────────────────────────────────────────────────
# Fields: model_id | n_samples_per_activity | provider
#
# n_samples_per_activity is the fraction of each activity's exercises in the test
# set; 0 means all of them. With SEED=1, 0.1 selects exactly the 1,026 exercises
# the paper compares all models on (the GPT runs), and Gemma's full run contains
# them, so every model is comparable on that subset.
#
# provider (OpenRouter, optional) restricts the model to those providers,
# comma-separated, with no fallback; empty = OpenRouter picks. It is a cost and
# speed choice: gemma-4-31b-it is served from $0.09 to $0.75/M input and
# OpenRouter's pick drifts. Friendli measured on Oct 5: 25 concurrent judge calls
# in 8s, ~$0.0018/call on an 11k-token prompt, no failures. The provider behind
# the paper's Gemma file was not recorded.
#
# The pseudo-entry "chance/chance_baseline" runs the no-LLM floor: it makes no API
# calls, so give it the full set. Its tag already matches the file the script
# writes, which keeps the skip check honest.
MODELS=(
  "chance/chance_baseline|0|"
  "google/gemma-4-31b-it|0|"
  "openai/gpt-5|0.1|"
  "openai/gpt-5.5|0.1|"
)

# ── Evaluation parameters ──────────────────────────────────────────────────────
BASE_URL="https://openrouter.ai/api/v1"
SEED=1                      # fixes the sampled test set AND the in-prompt examples:
                            # keep it identical across models so the comparison is paired
TEMPERATURE=0.0             # greedy — a judge should not sample
N_CONTEXT_EXAMPLES=1        # example exercises per activity shown in the prompt
N_JUDGE_TRIALS=3            # judge trials per exercise, each re-sampling the examples;
                            # verdicts combined by majority. Multiplies the call count.
# OpenRouter reserves credit for the worst case a request could cost. With no cap
# the provider assumes the model ceiling (65536 tokens for gpt-5.5 = ~$2 held per
# call) and refuses with a 402 once the balance drops below that. The judge
# answers with one activity number — ~220 completion tokens including reasoning —
# so a small explicit cap removes the reservation problem without truncating.
MAX_TOKENS=4096
# Reasoning is billed as completion tokens. gpt-5.5 spent ~212 of 219 completion
# tokens reasoning to answer with one activity number; "low" cuts that.
REASONING_EFFORT=low

# OpenRouter throttles new accounts (e.g. 20 requests/min per model). Above 0 the
# client stops sending concurrently and paces itself with a rolling window; at 0
# each chunk is sent at once. Adding credits raises the limit.
RATE_LIMIT_CALLS=0          # max requests per window (0 = unlimited, send concurrently)
RATE_LIMIT_SLEEP=60         # window length in seconds
# Exercises judged per checkpoint inside a module (and the concurrency when
# RATE_LIMIT_CALLS=0). An interruption (out of credit, 429 storm, Ctrl-C) costs at
# most this many calls. Lower it when the key balance is tight.
CHUNK_SIZE=500

# =============================================================================
# Helpers — no need to edit below this line
# =============================================================================

# Mirrors the filename the python script builds (all objectives in context
# here, so no _ctxNobj suffix).
result_file_for() {
  local model_id="$1"
  echo "${OUTPUT_DIR%/}/llm_as_judge_quality_per-module-${model_id##*/}.json"
}

run_one() {
  local name="$1"; shift
  local log_file="$LOG_DIR/${name}.log"

  if $DRY_RUN; then
    echo "# ── $name"
    printf '%q ' "$@"     # one line, quoted so it can be pasted as is
    printf '\n\n'
    return 0
  fi

  echo "▶  $name  →  $log_file"
  if $PARALLEL; then
    "$@" > "$log_file" 2>&1 &
    echo "   PID $!  →  $log_file"
    PIDS+=("$!")
  else
    "$@" 2>&1 | tee "$log_file"
  fi
}

# =============================================================================
# Main loop
# =============================================================================
PIDS=()

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r model_id n_samples provider <<< "$model_entry"
  model_tag="${model_id##*/}"
  run_name="judge-eval_${model_tag}"

  result_file="$(result_file_for "$model_id")"
  if [[ -f "$result_file" ]] && ! $FORCE && ! $DRY_RUN; then
    # Skip only a finished run. A partial file (interrupted, out of credit) is
    # left for the python script, which resumes where it stopped.
    if "$PYTHON" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" \
         "$result_file" 2>/dev/null; then
      echo "⏭  $run_name  (already complete — $result_file)"
      continue
    fi
    done_mods=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('per_module',{})))" \
                  "$result_file" 2>/dev/null || echo "?")
    echo "↩  $run_name  (partial: $done_mods module(s) done — resuming)"
  fi

  # The chance baseline needs no model, endpoint or key.
  model_args=(--model "$model_id")
  if [[ "$model_id" == chance/* ]]; then
    model_args=(--chance-baseline)
  fi

  # --force must reach the python script too: it checkpoints and resumes from an
  # existing file, so skipping only this shell-level check would start a run that
  # immediately skips every module and writes nothing back.
  force_arg=()
  if $FORCE; then
    force_arg=(--force-restart)
  fi

  run_one "$run_name" \
    env PYTHONUNBUFFERED=1 "$PYTHON" "$EVAL_SCRIPT" \
    "${model_args[@]}" \
    --dataset                "$PATH_DATASET" \
    --output-dir             "$OUTPUT_DIR" \
    --base-url               "$BASE_URL" \
    --temperature            "$TEMPERATURE" \
    --max-tokens             "$MAX_TOKENS" \
    --reasoning-effort       "$REASONING_EFFORT" \
    --provider-only          "$provider" \
    --seed                   "$SEED" \
    --n-samples-per-activity "$n_samples" \
    --n-context-examples     "$N_CONTEXT_EXAMPLES" \
    --n-judge-trials         "$N_JUDGE_TRIALS" \
    --chunk-size             "$CHUNK_SIZE" \
    --rate-limit-calls       "$RATE_LIMIT_CALLS" \
    --rate-limit-sleep       "$RATE_LIMIT_SLEEP" \
    "${force_arg[@]}"
done

# ── Wait for parallel jobs ────────────────────────────────────────────────────
if $PARALLEL && (( ${#PIDS[@]} > 0 )); then
  echo ""
  echo "Waiting for ${#PIDS[@]} background job(s)…"
  for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
      echo "  ✓ PID $pid"
    else
      echo "  ✗ PID $pid (non-zero exit — check $LOG_DIR/)"
    fi
  done
  echo "All done."
fi
