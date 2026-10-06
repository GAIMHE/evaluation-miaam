"""Evaluate an LLM-as-judge at the activity level, using the full module as context.

For each module in the MIAAM dataset:
  - The judge is shown ALL objectives and their associated activities (with example
    exercises per activity).
  - The judge is asked: "which activity does this exercise belong to?" using a global
    activity numbering that spans all objectives in the module.
  - Test set: a fraction of the exercises of each activity.
  - Context shown to the judge: M example exercises per activity.

Modules with fewer than 2 activities are skipped.

Self-contained: talks to any OpenAI-compatible endpoint (OpenRouter by default)
and needs only the dataset directory (data/maths_exercises_table.parquet and
data/descriptions.json). Results are written to
<output-dir>/llm_as_judge_quality_per-module-<model>[_ctxNobj].json and
checkpointed as the run progresses, so re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import APITimeoutError, OpenAI

# ── Prompt ────────────────────────────────────────────────────────────────────
# Do not edit: any change, down to one character, makes new results incomparable
# with existing ones (and invalidates the provider's prompt cache).

# The space before the first newline is part of the prompt; it is kept out of
# the line end so editors that strip trailing whitespace cannot drop it.
BASE_PERSONA = ("You are a helpful assistant to a Professor teaching maths to French students using exercises. \n"
                "An exercise is defined by a question and its solution. All exercises match the same pedagogical "
                "intent targeted by the Professor.")

PROMPT_JUDGE_PER_MODULE = BASE_PERSONA + '''
You will help the Professor organize its exercises. All the exercises fall under the same global module:
Module : {module_name}

The module is organized into {n_objectives} objectives, each containing several activities. In total there are {n_activities} activities:
{activity_list}

Below are more details on each objective and its activities (pedagogical intents and example exercises).
{objectives_details}
Now here's a new exercise you must help the Professor select which activity it fits in. Answer only the activity number ({activity_numbers}).
If you think the exercise fits none of the activities: answer "None".
- Exercice:
{q}

Solution: {s}

Activity:
'''


def _strip_html(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text)
    return re.sub(r'<[^>]+>', '', text)


def build_prompt(exercise: dict, context: dict) -> str:
    objectives = context['objectives']

    activity_list      = ""
    objectives_details = ""
    activity_numbers   = []

    for obj in objectives:
        objectives_details += f"\n** Objectif : {obj['name']}\n"
        objectives_details += f"Intention pédagogique : {_strip_html(obj['pedagogical_intent'])}\n"
        for act in obj['activities']:
            activity_list    += f"- {act['global_id']}: {act['name']}\n"
            activity_numbers.append(str(act['global_id']))
            objectives_details += f"\n  * Activité {act['global_id']} : {act['name']}\n"
            objectives_details += f"    Intention pédagogique : {_strip_html(act['pedagogical_intent'])}\n"
            for j, ex in enumerate(act['exercises']):
                objectives_details += f"\n    - Exercice {j + 1}:\n{ex['question_str']}\n\nSolution: {ex['solution_str']}\n"

    n_activities = sum(len(obj['activities']) for obj in objectives)

    return PROMPT_JUDGE_PER_MODULE.format(
        module_name=context['module_name'],
        n_objectives=len(objectives),
        n_activities=n_activities,
        activity_list=activity_list,
        objectives_details=objectives_details,
        activity_numbers=", ".join(activity_numbers),
        q=exercise['question_str'],
        s=exercise['solution_str'],
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

def parse_exercise(raw_content, textual_description, gameplay_type: str = ""):
    """{question_str, solution_str} for one exercise, or None if it can't be rendered as text.

    Input-line exercises are rebuilt from their stored text; every other gameplay
    uses the screenshot description from descriptions.json. This is the
    representation behind the published results: switching input-line exercises to
    their descriptions changes their prompts and, through the exercises kept, the
    sampled test set, so the numbers would no longer be reproduced.
    """
    try:
        if gameplay_type == "INPUT_LINE_GLOBAL":
            if "[SÉLECTION parmi :" in raw_content["question"]:
                options = raw_content["question"].split("[SÉLECTION parmi :")[1].split("]")[0].split("?,")[
                    1].strip().split(", ")
                clean_question = (raw_content["question"].split("[SÉLECTION parmi :")[0]
                                  + f"[{'/'.join(options)}]"
                                  + raw_content["question"].split("]")[1])
                return {
                    "solution_str": next(_o for _o in options if _o in raw_content["correct_answer"]),
                    "question_str": raw_content["instruction"] + "\n" + clean_question,
                }
            elif "[CHAMP]" in raw_content["question"]:
                solution = re.findall(r'\d+(?:[,.]\d+)?', raw_content["correct_answer"])[0]
                return {
                    "question_str": f"{raw_content['instruction']}\n\n{raw_content['question'].replace('[CHAMP]', '_____')}",
                    "solution_str": solution,
                }
            else:
                return {
                    "question_str": f"{raw_content['instruction']}\n\n{raw_content['question']}",
                    "solution_str": raw_content["correct_answer"]
                }
        elif textual_description is not None:
            # No description (memory exercises have no screenshot): dropped.
            return {
                "question_str": textual_description,
                "solution_str": raw_content.get("correct_answer", ""),
            }
    except Exception:
        pass
    return None


def load_dataset(path_dataset: str) -> tuple[pd.DataFrame, dict]:
    parquet_path = os.path.join(path_dataset, "data", "maths_exercises_table.parquet")
    json_path    = os.path.join(path_dataset, "data", "descriptions.json")
    df_full = pd.read_parquet(parquet_path)
    with open(json_path) as f:
        descriptions = json.load(f)
    print(f"Loaded dataset: {len(df_full)} rows, "
          f"{df_full['module_name'].nunique()} modules, "
          f"{df_full['objective_name'].nunique()} objectives.")
    return df_full, descriptions


def load_module(df_full: pd.DataFrame, descriptions: dict, module_name: str) -> list[dict]:
    """Every exercise of the module that parses, in dataset order."""
    df = df_full[df_full['module_name'] == module_name]
    module_exercises = []
    for _, row in df.iterrows():
        try:
            raw_content = json.loads(row['content'])
        except Exception:
            try:
                raw_content = ast.literal_eval(row['content'])
            except Exception:
                continue
        textual_description = descriptions.get(row["exercise_id"])
        exercise = parse_exercise(raw_content, textual_description, gameplay_type=row["gameplay_type"])
        if not exercise:
            continue
        exercise['objective_name']              = row['objective_name']
        exercise['objective_pedagogical_intent'] = row['objective_pedagogical_intent']
        exercise['activity_name']               = row['activity_name']
        exercise['activity_pedagogical_intent'] = row['activity_pedagogical_intent']
        exercise['exercise_id']                 = row['exercise_id']
        module_exercises.append(exercise)
    return module_exercises


def build_full_hierarchy(module_exercises: list[dict]) -> tuple[list, dict]:
    """Build objective order and full hierarchy dict from loaded exercises."""
    obj_order = []
    hierarchy = {}
    for e in module_exercises:
        obj, act = e['objective_name'], e['activity_name']
        if obj not in hierarchy:
            obj_order.append(obj)
            hierarchy[obj] = {
                'pedagogical_intent': e['objective_pedagogical_intent'],
                'activity_order': [],
                'activities': {},
            }
        if act not in hierarchy[obj]['activities']:
            hierarchy[obj]['activity_order'].append(act)
            hierarchy[obj]['activities'][act] = {
                'pedagogical_intent': e['activity_pedagogical_intent'],
                'exercises': [],
            }
        hierarchy[obj]['activities'][act]['exercises'].append(e)
    return obj_order, hierarchy


def activities_in_order(module_exercises: list[dict]) -> list[tuple]:
    """Return (objective_name, activity_name) pairs in first-seen order."""
    seen, seen_set = [], set()
    for e in module_exercises:
        key = (e['objective_name'], e['activity_name'])
        if key not in seen_set:
            seen.append(key)
            seen_set.add(key)
    return seen


def build_context(module_name: str, obj_order: list, hierarchy: dict, act_global_id: dict,
                  rng: np.random.Generator, n_examples: int = 3,
                  n_objectives_context: int = None, required_objective: str = None,
                  exclude_exercise_id=None) -> tuple[dict, dict]:
    """Build context for the judge prompt.

    Args:
        n_examples: number of example exercises per activity shown in the prompt.
        n_objectives_context: total number of objectives to include in the prompt.
            The required_objective is always included; the rest are sampled randomly.
            None means include all objectives.
        required_objective: objective that must appear in the context (the GT objective
            of the exercises being judged). Ignored when n_objectives_context is None.
        act_global_id: {(obj_name, act_name): global_id}. Must use the same ordering
            as ground_truth so local→global stays consistent.

    Returns:
        (context dict for the prompt, local_to_global mapping {local_id: global_id}).
    """
    # Select which objectives to include
    if n_objectives_context is None:
        selected_objs = obj_order
    else:
        others = [o for o in obj_order if o != required_objective]
        n_others = min(n_objectives_context - 1, len(others))
        sampled = list(rng.choice(others, n_others, replace=False))
        # Preserve original ordering
        selected_objs = [o for o in obj_order
                         if o == required_objective or o in sampled]

    # Build context with local IDs (1-based within selected_objs)
    local_id = 1
    local_to_global = {}
    objectives_context = []
    for obj_name in selected_objs:
        obj_data = hierarchy[obj_name]
        activities_context = []
        for act_name in obj_data['activity_order']:
            act_data = obj_data['activities'][act_name]
            exs = act_data['exercises']
            if exclude_exercise_id is not None:
                # Never show an exercise as an example of its own classification.
                # Fall back to the full pool if it was the activity's only one.
                exs = [e for e in exs if e.get('exercise_id') != exclude_exercise_id] or exs
            indices = rng.choice(len(exs), min(n_examples, len(exs)), replace=False)
            activities_context.append({
                'name'              : act_name,
                'global_id'         : local_id,
                'pedagogical_intent': act_data['pedagogical_intent'],
                'exercises'         : [exs[i] for i in indices],
            })
            local_to_global[local_id] = act_global_id[(obj_name, act_name)]
            local_id += 1
        objectives_context.append({
            'name'              : obj_name,
            'pedagogical_intent': obj_data['pedagogical_intent'],
            'activities'        : activities_context,
        })

    context = {
        'module_name': module_name,
        'objectives' : objectives_context,
    }
    return context, local_to_global


# ── LLM client ────────────────────────────────────────────────────────────────

# Usage accounting. The provider reports tokens and cost on every response;
# without recording it there is no way to notice that, say, prompt caching has
# stopped working until the bill arrives.
_USAGE_LOCK = threading.Lock()
_USAGE = {"calls": 0, "failed": 0, "prompt_tokens": 0, "cached_tokens": 0,
          "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0}


def get_usage_totals() -> dict:
    """Snapshot of what has been spent so far this process."""
    with _USAGE_LOCK:
        return dict(_USAGE)


def _record_usage(completion=None, failed: bool = False) -> None:
    usage = getattr(completion, "usage", None)

    def _get(obj, *names):
        for nm in names:
            obj = getattr(obj, nm, None) if obj is not None else None
        return obj or 0

    with _USAGE_LOCK:
        _USAGE["calls"] += 1
        if failed:
            _USAGE["failed"] += 1
            return
        _USAGE["prompt_tokens"]     += getattr(usage, "prompt_tokens", 0) or 0
        _USAGE["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _USAGE["cached_tokens"]     += _get(usage, "prompt_tokens_details", "cached_tokens")
        _USAGE["reasoning_tokens"]  += _get(usage, "completion_tokens_details", "reasoning_tokens")
        _USAGE["cost"]              += float(getattr(usage, "cost", 0) or 0)


def format_usage(totals: dict) -> str:
    """One-line summary, including the cache hit rate that drives the cost."""
    n = max(1, totals["calls"] - totals["failed"])
    cached_pct = totals["cached_tokens"] / totals["prompt_tokens"] * 100 if totals["prompt_tokens"] else 0
    return (f"{totals['calls']} calls ({totals['failed']} failed) | "
            f"${totals['cost']:.4f} total, ${totals['cost'] / n:.4f}/call | "
            f"in {totals['prompt_tokens']:,} tok ({cached_pct:.0f}% cached) | "
            f"out {totals['completion_tokens']:,} tok "
            f"({totals['reasoning_tokens']:,} reasoning)")


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _status_of(exc):
    """HTTP status from an OpenAI-SDK error, if there is one."""
    if isinstance(exc, APITimeoutError):
        return 408      # no HTTP response at all; treat it as a request timeout so it is retried
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def _retry_after(exc):
    """Seconds the provider asked us to wait, if it said so."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _error_message(exc):
    """Provider message, tolerating errors whose body is absent or not a dict."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        msg = body.get("message")
        if isinstance(msg, str):
            return msg
    return str(exc)


class JudgeClient:
    """Minimal OpenAI-compatible chat client for single-answer judge calls."""

    THINK_STOP_TAG = "</think>"

    def __init__(self, model: str, base_url: str, api_key: str, temperature: float,
                 max_tokens: int, reasoning_effort: str, seed: int, provider_only: str = "",
                 max_retries: int = 8, retry_base_delay: float = 8.0,
                 rate_limit_calls: int = 0, rate_limit_sleep: float = 60.0,
                 max_workers: int = 90):
        # Per request. A provider can accept a call and never answer; with a long
        # timeout one such call froze a whole judge chunk for hours. Timed-out calls
        # are retried (see _status_of). 90 s fits a judge call that reasons up to its
        # 4096-token cap (~41 s on Friendli).
        self.client = OpenAI(base_url=base_url, api_key=api_key or None, timeout=90)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_sleep = rate_limit_sleep
        self.max_workers = max_workers
        self._api_call_timestamps = []

        # Sent with every request. The seed is honoured by some providers (e.g.
        # OpenAI) and ignored by others; it is harmless either way.
        self.request = {"model": model, "temperature": temperature, "seed": seed, "n": 1}
        if max_tokens > 0:
            # Without an explicit cap the provider reserves credit for the model's
            # ceiling (65536 tokens for gpt-5.5 = ~$2 held per call) and refuses with
            # a 402 once the balance drops below that. The judge answers with one
            # number, so a small cap costs nothing.
            self.request["max_tokens"] = max_tokens
        extra_body = {}
        anthropic_api = model.lower().startswith("anthropic/") or (api_key or "").startswith("sk-ant-")
        if not anthropic_api:
            # Hybrid models (e.g. Qwen3) answer directly instead of thinking first.
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        # OpenRouter otherwise picks among every provider of a model, whose prices
        # differ by up to ~8x (gemma-4-31b-it: $0.09–0.75/M input) and whose speed
        # and reliability vary just as much. Pinning providers makes cost and speed
        # reproducible.
        providers = [p.strip() for p in provider_only.split(",") if p.strip()]
        if providers and "openrouter" in base_url:
            extra_body["provider"] = {"only": providers, "allow_fallbacks": False}
        if reasoning_effort:
            # Reasoning is billed as completion tokens: gpt-5.5 spent ~212 of 219
            # completion tokens reasoning to output one activity number.
            extra_body["reasoning"] = {"effort": reasoning_effort}
        if extra_body:
            self.request["extra_body"] = extra_body

    def _complete_one(self, prompt: str):
        """Answer text, "" if the call landed without a usable answer, None if it failed.

        A response whose content is None is NOT a failure: the call landed and was
        billed, but the judge gave no answer — typically reasoning ran into
        max_tokens. At temperature 0 the same prompt does it again on every run, so
        retrying it only re-pays for it; "" records it as an unparseable verdict
        instead. The same holds for a prompt longer than the model's context.
        """
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(messages=messages, **self.request)
                break
            except Exception as e:
                status = _status_of(e)
                if status in RETRYABLE_STATUS and attempt < self.max_retries:
                    after = _retry_after(e)
                    if after is not None:
                        wait = after * (1.0 + random.random() * 0.5)   # never shorter than asked
                    else:
                        # Equal jitter: half the backoff plus a random half. With many
                        # workers failing together, a narrow band retried them in
                        # near-lockstep and re-saturated the pool on every wave; this
                        # spreads a wave across the whole interval.
                        full = min(self.retry_base_delay * (2 ** attempt), 120.0)
                        wait = full / 2.0 + random.random() * full / 2.0
                    print(f"  {status} from provider — retry {attempt + 1}/{self.max_retries} in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                print("completion problem: ", e)
                _record_usage(failed=True)
                if "longer than the model's context length" in _error_message(e):
                    return ""
                return None
        _record_usage(completion)
        if not completion.choices:
            return None
        return completion.choices[0].message.content or ""

    def complete(self, prompts: list[str]) -> list:
        """One answer per prompt (see _complete_one), in order.

        Concurrent by default. When rate_limit_calls > 0, requests are sent one at a
        time with a rolling-window check before each call, which avoids the burst
        limits of new OpenRouter accounts.
        """
        if not (self.rate_limit_calls > 0 and self.rate_limit_sleep > 0):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                return list(executor.map(self._complete_one, prompts))

        results = []
        stamps = self._api_call_timestamps
        for i, prompt in enumerate(prompts):
            now = time.time()
            stamps[:] = [t for t in stamps if now - t < self.rate_limit_sleep]
            if len(stamps) >= self.rate_limit_calls:
                wait = self.rate_limit_sleep - (now - stamps[0]) + 0.5
                if wait > 0:
                    print(f"Rate limit: {len(stamps)} calls in last {self.rate_limit_sleep}s, sleeping {wait:.1f}s")
                    time.sleep(wait)
                stamps[:] = [t for t in stamps if time.time() - t < self.rate_limit_sleep]
            stamps.append(time.time())
            results.append(self._complete_one(prompt))
            print(f"send {i + 1} / {len(prompts)} messages")
        return results


# Distinguishes "the call never landed" from "the judge answered, unparseably".
# Both look like None downstream, but only the first must be retried — recording
# it as a verdict would bake an API failure into the results permanently.
_CALL_FAILED = object()


def judge_exercises(llm: JudgeClient, exercises: list[dict], context_fn) -> tuple[list, int]:
    """Classify each exercise once.

    `context_fn(exercise_index)` returns ``(context, local_to_global)``. Callers
    should return the SAME context object for every exercise wherever possible:
    the context is ~99.9% of the prompt, so a shared prefix is prompt-cached by
    the provider and a per-exercise one is not — the difference is roughly an
    order of magnitude on the bill.

    Returns (one global activity id, None or _CALL_FAILED per exercise; number of
    failed calls).
    """
    prompts, mappings = [], []
    for i, ex in enumerate(exercises):
        context, local_to_global = context_fn(i)
        prompts.append(build_prompt(ex, context))
        mappings.append(local_to_global)

    verdicts, n_failed = [], 0
    for local_to_global, text in zip(mappings, llm.complete(prompts)):
        if text is None:
            n_failed += 1
            verdicts.append(_CALL_FAILED)
            continue
        text = text.strip()
        if JudgeClient.THINK_STOP_TAG in text:
            text = text.split(JudgeClient.THINK_STOP_TAG)[1].strip()
        match = re.search(r'\d+', text)
        local = int(match.group()) if match else None
        verdicts.append(local_to_global.get(local) if local is not None else None)
    return verdicts, n_failed


# ── Evaluation ────────────────────────────────────────────────────────────────

def _majority(verdicts: list):
    """Most frequent verdict; None on a tie or when nothing parsed."""
    counts = {}
    for v in verdicts:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else None


def _module_rng(seed: int, module_name: str) -> np.random.Generator:
    """Per-module generator, independent of which modules ran before.

    Resuming skips completed modules, so a single run-long generator would leave
    the remaining modules at a different state than an uninterrupted run — their
    test sets would differ, breaking both reproducibility and the nesting of
    samples drawn at different fractions. Seeding per module removes that
    coupling. hashlib (not hash()) because Python salts string hashing per
    process, which would make the seed differ between runs.
    """
    digest = int(hashlib.sha256(module_name.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([seed, digest])


def _save_results(output_path: str, all_results: dict, complete: bool = False,
                  partial: dict = None) -> None:
    """Write the results file atomically, recomputing the overall totals.

    Called after every chunk and module, so an interrupt costs at most the chunk
    in flight. Totals are derived from all_results rather than running counters,
    so a resumed file is identical to an uninterrupted one.
    """
    overall_correct = sum(v["n_correct"] for v in all_results.values())
    overall_total   = sum(v["n_valid"]   for v in all_results.values())
    overall_none    = sum(v["n_none"]    for v in all_results.values())
    payload = {
        # False while modules remain, so a launcher can tell a partial file from a
        # finished one and resume instead of skipping or restarting.
        "complete": complete,
        # Excludes None verdicts; count them as errors by recomputing from
        # predicted_activities vs ground_truth_activities.
        "overall_accuracy": overall_correct / overall_total * 100 if overall_total else None,
        "overall_correct" : overall_correct,
        "overall_total"   : overall_total,
        "overall_none"    : overall_none,
        "per_module"      : all_results,
        # Verdicts for the module currently being judged. A module is only written
        # to per_module once finished, so without this an interruption partway
        # through a 486-call module discards every call already paid for.
        "partial"         : partial,
    }
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)   # never leave a truncated file behind


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   help="MIAAM dataset directory, containing data/maths_exercises_table.parquet and "
                        "data/descriptions.json (e.g. the snapshot printed by "
                        "`hf download GAIMHE/MIAAM-V2 --repo-type dataset`)")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "results" / "llm_as_judge_eval"),
                   help="directory for the results JSON")

    g = p.add_argument_group("judge model")
    g.add_argument("--model", default="google/gemma-4-31b-it",
                   help="any chat model served by the endpoint; the part after the last '/' names the results file")
    g.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    g.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""),
                   help="defaults to $OPENROUTER_API_KEY")
    g.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; a judge should not sample")
    g.add_argument("--max-tokens", type=int, default=4096,
                   help="completion cap per call, reasoning included (<= 0 = provider default)")
    g.add_argument("--reasoning-effort", choices=["", "minimal", "low", "medium", "high"], default="low",
                   help="'' leaves the provider default")
    g.add_argument("--provider-only", default="",
                   help="OpenRouter only: comma-separated providers to restrict calls to, with no fallback "
                        "(e.g. 'Friendli'); empty = OpenRouter's routing")
    g.add_argument("--max-retries", type=int, default=8,
                   help="retries on a transient provider error (408/409/429/5xx); Retry-After is honoured, "
                        "otherwise exponential backoff with jitter")
    g.add_argument("--retry-base-delay", type=float, default=8.0,
                   help="first backoff delay in seconds; doubles per attempt, capped at 120s")
    g.add_argument("--rate-limit-calls", type=int, default=0,
                   help="> 0: send sequentially, at most this many calls per --rate-limit-sleep window; "
                        "0: send each chunk concurrently")
    g.add_argument("--rate-limit-sleep", type=float, default=60.0, help="rate-limit window in seconds")

    g = p.add_argument_group("evaluation")
    g.add_argument("--seed", type=int, default=1,
                   help="fixes the sampled test set AND the in-prompt examples: keep it identical across "
                        "models so the comparison is paired")
    g.add_argument("--n-samples-per-activity", type=float, default=0.25,
                   help="fraction (0.0-1.0) of exercises per activity to sample for the test set; <= 0 uses all")
    g.add_argument("--n-context-examples", type=int, default=1,
                   help="example exercises per activity shown to the judge in the prompt")
    g.add_argument("--n-objectives-context", type=int, default=0,
                   help="number of objectives shown to the judge (the ground-truth one is always included); "
                        "<= 0 shows all objectives in the module. When > 0 the output filename gets a "
                        "_ctxNobj suffix.")
    g.add_argument("--n-judge-trials", type=int, default=1,
                   help="judge trials per exercise, each re-sampling the in-prompt examples. Verdicts are "
                        "combined by majority (a tie gives None); the spread measures how sensitive the judge "
                        "is to the prompt. Multiplies the number of API calls.")
    g.add_argument("--chance-baseline", action="store_true",
                   help="no LLM: predict an activity uniformly at random within each module, as a floor for "
                        "the accuracy numbers. Uses the same test set as a real run and writes to the "
                        "chance_baseline results file.")
    g.add_argument("--chunk-size", type=int, default=50,
                   help="exercises judged per checkpoint within a module (and parallelism when "
                        "--rate-limit-calls is 0). An interruption costs at most this many calls.")
    g.add_argument("--force-restart", action="store_true",
                   help="ignore an existing results file and re-evaluate every module instead of resuming")
    return p.parse_args()


def main():
    args = parse_args()

    n_samples_per_activity = args.n_samples_per_activity if args.n_samples_per_activity > 0 else None
    n_context_examples     = args.n_context_examples
    n_objectives_context   = args.n_objectives_context if args.n_objectives_context > 0 else None
    n_judge_trials         = max(1, args.n_judge_trials)

    print(f"Judge model : {args.model}  (temperature {args.temperature})")
    print(f"Output dir  : {args.output_dir}")
    print(f"Test set    : {n_samples_per_activity if n_samples_per_activity else 'all'} per activity")
    print(f"Context     : {n_context_examples} example(s)/activity, "
          f"{n_objectives_context if n_objectives_context else 'all'} objective(s)")
    if args.chance_baseline:
        print("Trials      : 1 (a random guesser has no prompt; trials would only create ties)")
    else:
        print(f"Trials      : {n_judge_trials} per exercise (examples re-sampled each time, "
              f"exercise excluded from its own examples)")

    if args.chance_baseline:
        # No client is built, so the run needs no API key and costs nothing.
        print("Mode        : CHANCE BASELINE (uniform random activity, no LLM calls)")
        llm = None
    else:
        llm = JudgeClient(
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort, seed=args.seed,
            provider_only=args.provider_only, max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay, rate_limit_calls=args.rate_limit_calls,
            rate_limit_sleep=args.rate_limit_sleep,
        )

    df_full, descriptions = load_dataset(args.dataset)

    modules = sorted(df_full['module_name'].unique().tolist())

    model_tag   = "chance_baseline" if args.chance_baseline else args.model.split('/')[-1]
    context_tag = f"_ctx{n_objectives_context}obj" if n_objectives_context is not None else ""
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir,
                               f"llm_as_judge_quality_per-module-{model_tag}{context_tag}.json")

    all_results, saved_partial = {}, None
    if os.path.exists(output_path) and not args.force_restart:
        try:
            with open(output_path, encoding="utf-8") as f:
                _payload = json.load(f)
            all_results = _payload.get("per_module", {}) or {}
            saved_partial = _payload.get("partial")
        except json.JSONDecodeError:
            print(f"Existing {output_path} is not readable JSON — starting over.")
            all_results = {}
        if all_results:
            print(f"Resuming from {output_path}: "
                  f"{len(all_results)}/{len(modules)} module(s) already done.")

    print(f"\nEvaluating {len(modules)} modules.\n")

    for module_name in modules:
        print(f"\n{'='*70}")
        print(f"Module: {module_name}")

        if module_name in all_results:
            done = all_results[module_name]
            print(f"  Already evaluated ({done['n_correct']}/{done['n_valid']}) — skipping. "
                  f"Use --force-restart to redo it.")
            continue

        # Sampling must not depend on which modules preceded this one (see _module_rng).
        rng = _module_rng(args.seed, module_name)

        module_exercises = load_module(df_full, descriptions, module_name)
        if not module_exercises:
            print("  No valid exercises parsed, skipping.")
            continue

        ordered_activities = activities_in_order(module_exercises)

        if len(ordered_activities) < 2:
            obj, act = ordered_activities[0]
            print(f"  Only 1 activity (\"{act}\"), skipping.")
            continue

        # Global activity ID mapping: (obj_name, act_name) → 1-based int
        act_global_id = {key: i + 1 for i, key in enumerate(ordered_activities)}
        activity_names = [act for (_, act) in ordered_activities]
        obj_for_act    = {act: obj for (obj, act) in ordered_activities}

        print(f"  Activities ({len(ordered_activities)}):")
        for (obj, act), gid in act_global_id.items():
            print(f"    {gid}. [{obj}] {act}")

        # Build test set
        exercises    = module_exercises
        ground_truth = [act_global_id[(e['objective_name'], e['activity_name'])] for e in exercises]

        sampled_indices = None
        if n_samples_per_activity is not None:
            sampled_indices = []
            for act_id in range(1, len(ordered_activities) + 1):
                indices = [i for i, g in enumerate(ground_truth) if g == act_id]
                n_to_sample = max(1, round(len(indices) * n_samples_per_activity))
                if len(indices) > n_to_sample:
                    # Shuffle once, then truncate — do NOT use rng.choice(n).
                    # Two properties matter, and choice(n) has neither:
                    #  * nesting: with the same seed, the 10% sample is a prefix of
                    #    the 25% sample, so runs at different fractions share a
                    #    subset instead of overlapping only by chance;
                    #  * fixed rng consumption: a permutation of a fixed-length
                    #    array advances the generator by the same amount whatever
                    #    n_to_sample is, so the in-prompt context examples drawn
                    #    afterwards stay identical across fractions too.
                    order = rng.permutation(len(indices))
                    indices = [indices[i] for i in order[:n_to_sample]]
                sampled_indices.extend(indices)
            sampled_indices.sort()
            exercises    = [exercises[i]    for i in sampled_indices]
            ground_truth = [ground_truth[i] for i in sampled_indices]
        exercise_ids = [e['exercise_id'] for e in exercises]

        print(f"  Test set: {len(exercises)} exercises" +
              (f" ({n_samples_per_activity:.0%} sampled/activity)" if n_samples_per_activity else ""))

        # Build full hierarchy once for the module
        obj_order, hierarchy = build_full_hierarchy(module_exercises)

        # One context per trial, shared by every exercise in it, so the ~14k-token
        # prefix stays identical and gets prompt-cached. Only an exercise that
        # happens to have been drawn as one of its own examples needs a private
        # context — a few percent of cases, which pay full price.
        _shared_ctx = {}
        _rebuilds = [0]

        def _build(exclude_id, required_obj):
            return build_context(
                module_name, obj_order, hierarchy, act_global_id, rng,
                n_examples=n_context_examples,
                n_objectives_context=n_objectives_context,
                required_objective=required_obj,
                exclude_exercise_id=exclude_id,
            )

        def _contains(context, ex_id):
            return any(e.get("exercise_id") == ex_id
                       for o in context["objectives"]
                       for act in o["activities"]
                       for e in act["exercises"])

        def context_fn(i, trial):
            # The ground-truth objective only matters when the context is
            # restricted to a few objectives.
            required_obj = exercises[i]['objective_name'] if n_objectives_context is not None else None
            key = (trial, required_obj)
            if key not in _shared_ctx:
                _shared_ctx[key] = _build(None, required_obj)
            context, local_to_global = _shared_ctx[key]
            if _contains(context, exercise_ids[i]):
                _rebuilds[0] += 1
                return _build(exercise_ids[i], required_obj)
            return context, local_to_global

        if args.chance_baseline:
            # Uniform over this module's activities. Drawn after the test-set
            # sampling and from the same per-module generator, so sampled_indices
            # match a real run exactly and the files stay alignable.
            #
            # Always a single draw, whatever --n-judge-trials says: trials measure
            # sensitivity to the in-prompt examples, and a random guesser has no
            # prompt to be sensitive to. Combining several uniform draws by
            # majority leaves the expected accuracy at 1/n_act but makes almost
            # every exercise a tie — with ~57 activities, 3 draws are distinct
            # ~96% of the time — which would void the baseline instead of
            # sharpening it.
            n_act = len(ordered_activities)
            predicted_trials = [[int(rng.integers(1, n_act + 1))] for _ in exercises]
            print(f"  Chance baseline: uniform over {n_act} activities "
                  f"→ expected {100.0 / n_act:.2f}% (1 draw; --n-judge-trials ignored)")
        else:
            # Judge in chunks, saving after each, so an interruption costs a chunk
            # rather than the module. A chunk is one slice of exercises for one
            # trial; the per-trial context is shared across chunks, so the prompt
            # prefix stays cached throughout.
            chunk = max(1, args.chunk_size)
            signature = {"module": module_name, "n_exercises": len(exercises),
                         "n_trials": n_judge_trials, "seed": args.seed,
                         "first_ids": exercise_ids[:5]}
            predicted_trials = [[] for _ in exercises]
            if saved_partial and saved_partial.get("signature") == signature:
                for k, v in saved_partial.get("trials", {}).items():
                    predicted_trials[int(k)] = list(v)
                resumed = sum(len(t) for t in predicted_trials)
                if resumed:
                    print(f"  Resuming module: {resumed}/{len(exercises) * n_judge_trials} "
                          f"verdicts already collected")
            elif saved_partial and saved_partial.get("signature", {}).get("module") == module_name:
                print("  Ignoring stale partial progress (settings changed since it was written)")

            def _checkpoint():
                _save_results(output_path, all_results, partial={
                    "signature": signature,
                    "trials": {str(i): t for i, t in enumerate(predicted_trials) if t},
                })

            for trial in range(n_judge_trials):
                for start in range(0, len(exercises), chunk):
                    idxs = [i for i in range(start, min(start + chunk, len(exercises)))
                            if len(predicted_trials[i]) <= trial]
                    if not idxs:
                        continue
                    sub = [exercises[i] for i in idxs]

                    def chunk_ctx(j, _idxs=idxs, _trial=trial):
                        # Map the chunk-local index back, and pin the outer trial so
                        # context_fn returns that trial's shared (cached) context.
                        return context_fn(_idxs[j], _trial)

                    got, failed = judge_exercises(llm, sub, chunk_ctx)
                    if failed == len(sub):
                        # Every call in the chunk failed — almost always out of
                        # credit or rate-limited. Continuing would burn the rest of
                        # the module writing None verdicts, so stop while the
                        # already-paid-for work is safely checkpointed.
                        _checkpoint()
                        raise SystemExit(
                            f"\nAll {failed} calls in this chunk failed (see the errors above).\n"
                            f"Progress saved to {output_path}; re-run to resume from here.")
                    if failed:
                        print(f"    warning: {failed}/{len(sub)} calls failed in this chunk "
                              f"— they will be retried on the next run")
                    for i, verdict in zip(idxs, got):
                        # Keep only verdicts that came back. A failed slot is left
                        # empty so the resume filter picks the exercise up again.
                        if verdict is not _CALL_FAILED:
                            predicted_trials[i].append(verdict)

                    _checkpoint()
                    done = sum(len(t) for t in predicted_trials)
                    u = get_usage_totals()
                    n_ok = max(1, u["calls"] - u["failed"])
                    cached_pct = (u["cached_tokens"] / u["prompt_tokens"] * 100
                                  if u["prompt_tokens"] else 0)
                    print(f"    trial {trial + 1}/{n_judge_trials}  "
                          f"{done}/{len(exercises) * n_judge_trials} verdicts  (checkpointed)  "
                          f"${u['cost']:.3f} so far, ${u['cost'] / n_ok:.4f}/call, "
                          f"{cached_pct:.0f}% cached")

            # Never finalise a module with calls still missing: it would be stored
            # with fewer trials than requested and silently treated as finished.
            short = [i for i, t in enumerate(predicted_trials) if len(t) < n_judge_trials]
            if short:
                _checkpoint()
                raise SystemExit(
                    f"\n{len(short)} exercise(s) still short of {n_judge_trials} verdicts "
                    f"after all trials (failed calls).\n"
                    f"Progress saved to {output_path}; re-run to collect the rest.")

        predicted = [_majority(t) for t in predicted_trials]

        if not args.chance_baseline:
            n_prompts = len(exercises) * n_judge_trials
            print(f"  Prompt cache: {n_prompts - _rebuilds[0]}/{n_prompts} calls share a "
                  f"cached context ({_rebuilds[0]} rebuilt to exclude a self-example)")

        if n_judge_trials > 1:
            unanimous = sum(1 for t in predicted_trials if len(set(t)) == 1)
            print(f"  Prompt sensitivity: {unanimous}/{len(predicted_trials)} "
                  f"({unanimous / len(predicted_trials) * 100:.1f}%) unchanged across "
                  f"{n_judge_trials} example draws")

        n_none   = sum(1 for p in predicted if p is None)
        pairs    = [(p, g) for p, g in zip(predicted, ground_truth) if p is not None]
        n_valid  = len(pairs)
        n_correct = sum(1 for p, g in pairs if p == g)
        acc      = n_correct / n_valid * 100 if n_valid else None

        print(f"  Correct: {n_correct}/{n_valid}" + (f" ({acc:.1f}%)" if acc is not None else ""))
        if n_none:
            print(f"  Disagreement (None): {n_none}")

        per_activity = {}
        for act_id, (obj_name, act_name) in enumerate(ordered_activities, 1):
            act_pairs    = [(p, g) for p, g in pairs if g == act_id]
            n_act        = len(act_pairs)
            n_act_correct = sum(1 for p, g in act_pairs if p == g)
            act_acc      = n_act_correct / n_act * 100 if n_act else None
            per_activity[act_name] = {
                "activity_id"    : act_id,
                "objective_name" : obj_name,
                "n_exercises"    : n_act,
                "n_correct"      : n_act_correct,
                "accuracy"       : act_acc,
            }
            print(f"    Activity {act_id} ({act_name}): {n_act_correct}/{n_act}" +
                  (f" ({act_acc:.1f}%)" if act_acc is not None else ""))

        all_results[module_name] = {
            "module_name"          : module_name,
            "activity_names"       : activity_names,
            "activity_objectives"  : [obj for (obj, _) in ordered_activities],
            "objective_for_activity": obj_for_act,
            "sampled_indices"      : [int(i) for i in sampled_indices] if sampled_indices is not None else None,
            "n_exercises"          : len(exercises),
            "n_valid"              : n_valid,
            "n_none"               : n_none,
            "n_correct"            : n_correct,
            "accuracy"             : acc,
            "per_activity"         : per_activity,
            "predicted_activities" : predicted,
            "predicted_trials"     : predicted_trials,
            "n_judge_trials"       : 1 if args.chance_baseline else n_judge_trials,
            "ground_truth_activities": ground_truth,
            "exercises"            : [
                {"question_str": e["question_str"], "solution_str": e["solution_str"], "exercise_id": e["exercise_id"]}
                for e in exercises
            ],
        }
        print(f"  Usage so far: {format_usage(get_usage_totals())}")
        saved_partial = None            # this module is finished; nothing in flight
        _save_results(output_path, all_results)
        print(f"  Checkpointed ({len(all_results)}/{len(modules)} modules) → {output_path}")

    _save_results(output_path, all_results, complete=True)
    overall_correct = sum(v["n_correct"] for v in all_results.values())
    overall_total   = sum(v["n_valid"]   for v in all_results.values())
    overall_none    = sum(v["n_none"]    for v in all_results.values())
    overall_acc     = overall_correct / overall_total * 100 if overall_total else None
    print(f"\n{'='*70}")
    print(f"OVERALL: {overall_correct}/{overall_total}" +
          (f" ({overall_acc:.1f}%)" if overall_acc is not None else ""))
    if args.chance_baseline:
        # Exact expectation, free of the sampling noise a single draw carries.
        num = sum(v["n_exercises"] / len(v["activity_names"]) for v in all_results.values())
        den = sum(v["n_exercises"] for v in all_results.values())
        if den:
            print(f"Analytic expectation (uniform guessing): {num / den * 100:.2f}%")
    if overall_none:
        print(f"Total disagreements (None): {overall_none}")
    print(f"Results saved to {output_path}")
    if llm is not None:
        print(f"API usage this run: {format_usage(get_usage_totals())}")


if __name__ == "__main__":
    main()
