# LLM/VLM Knowledge-Tracing eval

Zero-shot knowledge tracing with an LLM served via
[OpenRouter](https://openrouter.ai). For each evaluation window, the model
sees a student's last *n* attempts (text descriptions and/or screenshots,
plus the binary outcome of each) and emits either a self-reported
probability that the next attempt will be correct (**prob/KT task**) or,
on multiple-choice items, the option index the student will pick
(**answer task**). We report AUC / accuracy / Brier (prob) or top-1
accuracy (answer) alongside the pyKT baselines.

## Pipeline

```
interactions_test.parquet ──build_windows_{kt,answer}.py──▶ windows.parquet
                                                                │
                                                                ▼
                                run_eval_{kt,answer}.py ──▶ results.jsonl
                                                                │
                                                                ▼
                                  score_{kt,answer}.py ──▶ metrics.json
```

## Run it

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
jupyter notebook run_eval.ipynb
```

Edit the **Config** cell to pick task, model, modality, and history length;
re-run from there. All artifacts land under `<repo>/results/<run-tag>/`.

## Modes & knobs (Config cell)

- **`TASK`** — `"prob"` (KT) or `"answer"` (MC option index).
- **`MODEL`** — any OpenRouter chat model. Open-weight examples:
  `google/gemma-4-31b-it`, `mistralai/mistral-small-2603`,
  `qwen/qwen3.6-27b`. Closed: `openai/gpt-5-mini`, `anthropic/claude-…`,
  `google/gemini-…`.
- **`MODALITY`** — `text` uses `descriptions.json`; `vision` swaps in the
  compressed screenshot; `both` sends both.
- **`EXPERT_KNOWLEDGE`** — prepends `activity_name` /
  `activity_pedagogical_intent` and `objective_name` /
  `objective_pedagogical_intent` from `maths_exercises_table.parquet` to
  each exercise.
- **`N_HISTORY`** — how many past attempts the model sees per window.
  Watch vision per-request image caps (Mistral Small ≤ 8, Gemma-4 ~32,
  Qwen-3 ~50, Claude Sonnet ~100, GPT-5 / Gemini > 250).
- **`N_WINDOWS`** — windows sampled uniformly at random from the eligible
  `(student, target_idx)` pool. Controls wall-clock cost.
- **`REASONING`** — `"on"` (default) sends no chat-template kwargs; required
  for OpenAI / Anthropic / Google models. `"off"` sends
  `chat_template_kwargs={"enable_thinking": False}` for Qwen3 reasoning
  models.
- **`REASONING_EFFORT`** — `minimal/low/medium/high` for GPT-5 family;
  leave `None` otherwise.

## Output

`results.jsonl` — one window per line:

| key | prob (KT) | answer |
|---|---|---|
| `window_id` | ✓ | ✓ |
| `target_label` | 0/1 correctness | — |
| `p_correct` | float or `null` | — |
| `target_answer_idx` | — | int (ground truth pick) |
| `pred_answer_idx` | — | int or `null` |
| `pred_correct` | — | 0/1 if pick matches |
| `target_source` | "am" / "mia" | "am" / "mia" |
| `target_objective_id` | int | int |
| `target_n_options` | — | int |
| `raw_answer` | model text | model text |

`metrics.json` from `score_kt.py` reports overall AUC / Accuracy / Brier
plus per-label (recall on label=1, specificity on label=0, plus `mean_p`
for calibration), per-source, and per-objective breakdowns.
`score_answer.py` reports top-1 accuracy with random and "always-correct"
baselines per `n_options`.

## File layout

```
evaluation_llm/
  run_eval.ipynb            # entry point — drives the whole pipeline
  build_windows_kt.py       # prob task: interactions_test → windows.parquet
  build_windows_answer.py   # answer task (MC only)
  prompts_kt.py             # build_messages() for the prob task
  prompts_answer.py         # build_messages() for the answer task
  run_eval_kt.py            # async OpenRouter client → results.jsonl
  run_eval_answer.py        # answer task variant
  score_kt.py               # AUC / Accuracy / Brier + per-label/source/objective
  score_answer.py           # top-1 accuracy + random / always-correct baselines
  llm_as_judge_eval_per_module.py  # activity attribution (LLM-as-judge), standalone
  run_judge_eval.sh         # launches the paper's activity-attribution runs
  requirements.txt
  README.md
```

## Activity attribution (LLM-as-judge)

`llm_as_judge_eval_per_module.py` is a separate, standalone experiment: for
each module, the model sees every objective and activity (pedagogical
intents plus example exercises) and must say which activity a real exercise
from the dataset belongs to. Each exercise is judged `--n-judge-trials`
times with re-sampled examples and the verdicts are combined by majority.

The runs reported in the paper (Q3: Gemma-4-31B on all exercises, GPT-5 and
GPT-5.5 on the same 10% per activity, chance baseline) are launched by
`run_judge_eval.sh`, which reads `OPENROUTER_API_KEY` from a `.env` file at the
repo root and finds MIAAM-V2 in the Hugging Face cache (or `$MIAAM_DATASET`):

```bash
bash evaluation_llm/run_judge_eval.sh --dry-run    # print the commands first
bash evaluation_llm/run_judge_eval.sh              # skips complete runs, resumes partial ones
```

Single runs, directly:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
DATASET="$(hf download GAIMHE/MIAAM-V2 --repo-type dataset)"
python llm_as_judge_eval_per_module.py --dataset "$DATASET" \
    --model openai/gpt-5 --n-samples-per-activity 0 --n-judge-trials 3 --seed 1
python llm_as_judge_eval_per_module.py --dataset "$DATASET" \
    --chance-baseline --n-samples-per-activity 0 --seed 1   # no API calls
```

Results go to `results/llm_as_judge_eval/[MIAAM_V2/]llm_as_judge_quality_per-module-<model>.json`,
checkpointed after every chunk: re-running the same command resumes, and
`--force-restart` starts over. Keep `--seed` identical across models so the
test sets and in-prompt examples are paired. `overall_accuracy` in the JSON
excludes `None` verdicts (abstentions, ties); to count them as errors,
recompute from `predicted_activities` vs `ground_truth_activities`.

## Known sharp edges

- **Probability parsing.** The model's free-text output is scanned for the
  first float in `[0, 1]`. Long-form prose ("I'd say around 0.7") usually
  parses cleanly, but inspect `raw_answer` on a pilot run.
- **Reasoning models thinking budget.** OpenRouter routes per-provider, and
  some Qwen3 providers silently ignore `enable_thinking=false`. If you get
  > 50% non-parseable responses on a reasoning model, bump `MAX_TOKENS` to
  4096+ so the answer survives.
