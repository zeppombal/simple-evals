"""Shared helpers for the LiveCodeBench (code generation) evals.

This module wraps the original LiveCodeBench (`lcb_runner`) code so the rest of
simple-evals can drive it through the standard sampler / EvalResult interfaces.
We import LCB's own dataset loader, prompt builder, code extractor and execution
harness unchanged so the *faithful* eval stays 100% faithful; the only thing this
file adds is uniform plumbing (record shapes, rubric rendering for the LLM-judge
variants).
"""

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Import the original LiveCodeBench package (`lcb_runner`).
#
# LiveCodeBench lives as a vendored directory inside this repo. It is a PEP-420
# namespace package (no top-level __init__.py), so we just need it on sys.path.
# The dataset loader, code extractor and execution harness all import cleanly
# regardless of the current working directory.
#
# We deliberately do NOT import the `lcb_runner.prompts` package: its __init__
# eagerly imports sibling modules that require `anthropic`, and
# `prompts/code_generation.py` runs a module-level `open(...)` with a path
# relative to the LiveCodeBench dir (the few-shot example JSONs, which are not
# always present). Instead we inline LCB's tiny generic code-generation prompt
# below, copied verbatim, so the prompt stays faithful without the fragile import.
# ---------------------------------------------------------------------------
_LCB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiveCodeBench")
if _LCB_DIR not in sys.path:
    sys.path.insert(0, _LCB_DIR)

from lcb_runner.lm_styles import LMStyle
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.benchmarks.code_generation import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

from typess import MessageList

# We build prompts in the standard OpenAI chat style (system + user), which is the
# `OpenAIChat` path in LiveCodeBench. `extract_code` is also keyed off this style.
LCB_LMSTYLE = LMStyle.OpenAIChat

# --- Generic code-generation prompt, copied verbatim from LiveCodeBench ---
# Source: LiveCodeBench/lcb_runner/prompts/code_generation.py
# (PromptConstants.SYSTEM_MESSAGE_GENERIC / FORMATTING_* and
#  get_generic_question_template_answer). Kept in sync by hand to avoid importing
# the prompts package; the wording must match LCB exactly to stay faithful.
_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)


def _generic_question_template_answer(question: CodeGenerationProblem) -> str:
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Format: {_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def build_prompt(problem: CodeGenerationProblem) -> MessageList:
    """Build the faithful LiveCodeBench code-generation prompt as a MessageList.

    This reproduces LCB's `OpenAIChat` prompt: the generic system message plus the
    generic question template (handles both starter-code / stdin formats).
    """
    return [
        {"role": "system", "content": _SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": _generic_question_template_answer(problem)},
    ]


def extract_solution(response_text: str) -> str:
    """Extract the submitted code from a model response (LCB's own extractor)."""
    return extract_code(response_text, LCB_LMSTYLE)


def _test_to_dict(test) -> dict:
    """Serialize an LCB `Test` to a plain dict (testtype as its string value)."""
    return {
        "input": test.input,
        "output": test.output,
        "testtype": test.testtype.value,
    }


def load_examples(
    release_version: str,
    num_examples: int | None,
    rng,
    n_repeats: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Load the LCB code-generation dataset and build self-contained example records.

    Each record carries everything needed by BOTH the faithful eval (the full
    `evaluation_sample` = all public+private tests) and the rubric evals (the
    individual test cases), so the generate -> evaluate split never re-downloads.
    """
    problems = load_code_generation_dataset(
        release_version=release_version, start_date=start_date, end_date=end_date
    )

    examples: list[dict] = []
    for problem in problems:
        examples.append(
            {
                "prompt_id": problem.question_id,
                "question_id": problem.question_id,
                "prompt": build_prompt(problem),
                # Faithful execution sample (all public + private tests, fn_name).
                "evaluation_sample": problem.get_evaluation_sample(),
                # Individual tests, used to build rubric items for the judge variants.
                "public_test_cases": [
                    _test_to_dict(t) for t in problem.public_test_cases
                ],
                "private_test_cases": [
                    _test_to_dict(t) for t in problem.private_test_cases
                ],
                "fn_name": problem.metadata.get("func_name", None),
                "example_tags": [problem.platform.value, problem.difficulty.value],
            }
        )

    if num_examples is not None and num_examples < len(examples):
        examples = rng.sample(examples, num_examples)

    return examples * n_repeats


# ---------------------------------------------------------------------------
# Rubric rendering for the LLM-as-a-judge variants.
#
# Each unit test becomes one rubric item: the judge is shown the problem, the
# extracted code, and a single test (input + expected output), and decides whether
# the code would produce the expected output for that input.
# ---------------------------------------------------------------------------
def _test_chars(test: dict) -> int:
    """Combined input+output size of a test (the dominant judge-prompt contributor)."""
    return len(test["input"]) + len(test["output"])


def render_test_criterion(test: dict, fn_name: str | None) -> str:
    """Render a single test case verbatim as a human-readable rubric criterion.

    No truncation: oversized tests are skipped upstream (`build_rubric_dicts`), so
    everything that reaches here fits the judge prompt in full.
    """
    inp = test["input"]
    out = test["output"]
    if fn_name:
        return (
            f"When `{fn_name}` is called with the following input arguments, it "
            f"returns the expected output.\nInput:\n{inp}\nExpected output:\n{out}"
        )
    return (
        "When the program is run on the following standard input, it prints exactly "
        f"the expected standard output (ignoring trailing whitespace).\n"
        f"Input (stdin):\n{inp}\nExpected output (stdout):\n{out}"
    )


def build_rubric_dicts(
    record: dict,
    max_public_tests: int | None = None,
    max_private_tests: int | None = None,
    max_test_chars: int | None = None,
) -> tuple[list[dict], dict]:
    """Build rubric-item dicts (one per unit test) from a generation/example record.

    For each kind (public, private), in original order:
      1. skip any test whose input+output exceeds `max_test_chars` (None/<=0 ⇒ no
         skip) — a test the judge can't see in full is not worth judging;
      2. apply the count cap (`max_*_tests`) to the SURVIVORS.

    Each surviving rubric keeps its original GLOBAL `test_index` (into the
    public+private concatenation) so it still joins to the faithful eval's
    `per_test_results[].index`.

    Returns `(rubrics, skip_summary)` where `skip_summary[kind]` =
    `{n_total, n_oversized_skipped, n_judged}`.
    """
    fn_name = record.get("fn_name", None)
    n_public = len(record.get("public_test_cases", []) or [])
    size_limited = max_test_chars is not None and max_test_chars > 0

    rubrics: list[dict] = []
    skip_summary: dict = {}

    for kind, key, cap, offset in (
        ("public", "public_test_cases", max_public_tests, 0),
        ("private", "private_test_cases", max_private_tests, n_public),
    ):
        tests = record.get(key, []) or []
        survivors: list[tuple[int, dict]] = []  # (global_index, test)
        n_oversized = 0
        for idx, test in enumerate(tests):
            if size_limited and _test_chars(test) > max_test_chars:
                n_oversized += 1
                continue
            survivors.append((offset + idx, test))
        if cap is not None:
            survivors = survivors[:cap]
        for global_index, test in survivors:
            rubrics.append(
                {
                    "criterion": render_test_criterion(test, fn_name),
                    "points": 1,
                    "tags": [kind],
                    "test_index": global_index,
                    "kind": kind,
                }
            )
        skip_summary[kind] = {
            "n_total": len(tests),
            "n_oversized_skipped": n_oversized,
            "n_judged": len(survivors),
        }
    return rubrics, skip_summary


def log_oversized_skip_summary(
    example_metadatas: list[dict], max_test_chars: int | None
) -> dict:
    """Loudly log how the test-size cap affected this rubric run.

    `example_metadatas` are the per-example dicts produced by the rubric evals (each
    carries `tests_skipped_oversized` and `fully_skipped`). Returns the aggregate
    counts so callers can stash them in metadata if desired.
    """
    n_problems = len(example_metadatas)
    n_problems_fully_skipped = sum(1 for m in example_metadatas if m.get("fully_skipped"))
    n_tests_total = n_tests_dropped = n_tests_judged = 0
    for m in example_metadatas:
        for s in (m.get("tests_skipped_oversized") or {}).values():
            n_tests_total += s.get("n_total", 0)
            n_tests_dropped += s.get("n_oversized_skipped", 0)
            n_tests_judged += s.get("n_judged", 0)

    if max_test_chars and max_test_chars > 0:
        cap_desc = f"{max_test_chars:,} chars (input+output)"
    else:
        cap_desc = "DISABLED (no size skipping)"
    pct = (100 * n_tests_dropped / n_tests_total) if n_tests_total else 0.0

    bar = "=" * 78
    print(
        "\n".join(
            [
                "",
                bar,
                f"  LiveCodeBench rubric eval | test-size cap = {cap_desc}",
                f"  Tests dropped as oversized: {n_tests_dropped:,} / {n_tests_total:,} "
                f"({pct:.1f}%)   |   judged: {n_tests_judged:,}",
                f"  Problems fully skipped (ALL tests oversized -> excluded from score): "
                f"{n_problems_fully_skipped} / {n_problems}",
                bar,
                "",
            ]
        ),
        flush=True,
    )
    return {
        "max_test_chars": max_test_chars,
        "n_problems": n_problems,
        "n_problems_fully_skipped": n_problems_fully_skipped,
        "n_tests_total": n_tests_total,
        "n_tests_dropped_oversized": n_tests_dropped,
        "n_tests_judged": n_tests_judged,
    }


_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(\{.*\})\s*```", re.DOTALL)


def parse_judge_json(text: str) -> dict:
    """Robustly extract the judge's JSON verdict from a (possibly chatty) response.

    Our judge prompt invites the model to reason, so capable models often emit prose
    (and/or a ```json fence) around the JSON object. The original IFEval parser only
    strips a fence at the exact string boundaries, so any leading prose makes
    `json.loads` fail "at char 0" and the verdict is silently lost. This tries, in
    order: a ```json/``` fenced object anywhere; the first-`{` .. last-`}` slice
    (handles nested objects like AR's `{"results":[...]}` plus surrounding prose); the
    raw string. Returns the first candidate that parses to a dict, else a clean
    fallback (never raises).
    """
    if not text or not text.strip():
        print("JSON decoding failed: empty response")
        return {"explanation": "<<PARSING ERROR>>", "criteria_met": False}

    s = text.strip()
    candidates: list[str] = []
    fenced = _JSON_FENCE_RE.findall(s)
    if fenced:
        candidates.append(fenced[-1])  # last fenced block = the final verdict
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidates.append(s[i : j + 1])
    candidates.append(s)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    snippet = s[:200].replace("\n", "\\n")
    print(f"JSON decoding failed: no JSON object found in response: {snippet!r}")
    return {"explanation": "<<PARSING ERROR>>", "criteria_met": False}
