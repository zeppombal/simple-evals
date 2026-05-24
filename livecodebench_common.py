"""Shared helpers for the LiveCodeBench (code generation) evals.

This module wraps the original LiveCodeBench (`lcb_runner`) code so the rest of
simple-evals can drive it through the standard sampler / EvalResult interfaces.
We import LCB's own dataset loader, prompt builder, code extractor and execution
harness unchanged so the *faithful* eval stays 100% faithful; the only thing this
file adds is uniform plumbing (record shapes, rubric rendering for the LLM-judge
variants).
"""

import os
import sys

# ---------------------------------------------------------------------------
# Import the original LiveCodeBench package (`lcb_runner`).
#
# LiveCodeBench lives as a vendored directory inside this repo. It is a PEP-420
# namespace package (no top-level __init__.py), so we just need it on sys.path.
# However, `lcb_runner/prompts/code_generation.py` runs a module-level
# `open("lcb_runner/prompts/few_shot_examples/.../*.json")` with a path relative
# to the LiveCodeBench directory, so we temporarily chdir there during import.
# ---------------------------------------------------------------------------
_LCB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiveCodeBench")
if _LCB_DIR not in sys.path:
    sys.path.insert(0, _LCB_DIR)

_cwd = os.getcwd()
try:
    os.chdir(_LCB_DIR)
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.code_generation import (
        PromptConstants,
        get_generic_question_template_answer,
    )
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.benchmarks.code_generation import (
        CodeGenerationProblem,
        load_code_generation_dataset,
    )
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
finally:
    os.chdir(_cwd)

from typess import MessageList

# We build prompts in the standard OpenAI chat style (system + user), which is the
# `OpenAIChat` path in LiveCodeBench. `extract_code` is also keyed off this style.
LCB_LMSTYLE = LMStyle.OpenAIChat


def build_prompt(problem: CodeGenerationProblem) -> MessageList:
    """Build the faithful LiveCodeBench code-generation prompt as a MessageList.

    This reproduces LCB's `OpenAIChat` prompt: the generic system message plus the
    generic question template (handles both starter-code / stdin formats).
    """
    return [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": get_generic_question_template_answer(problem)},
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
_MAX_RENDER_CHARS = 4000  # guard against pathologically large single tests


def _truncate(s: str) -> str:
    if s is not None and len(s) > _MAX_RENDER_CHARS:
        return s[:_MAX_RENDER_CHARS] + "\n...[truncated]..."
    return s


def render_test_criterion(test: dict, fn_name: str | None) -> str:
    """Render a single test case as a human-readable, objective rubric criterion."""
    inp = _truncate(test["input"])
    out = _truncate(test["output"])
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
) -> list[dict]:
    """Build rubric-item dicts (one per unit test) from a generation/example record.

    Caps are applied independently to public and private tests (private is the
    priority slice, matching LCB's reported pass@1). Returns dicts in the
    `RubricItem` schema so callers can `RubricItem.from_dict`.
    """
    fn_name = record.get("fn_name", None)
    n_public = len(record.get("public_test_cases", []) or [])
    rubrics: list[dict] = []

    for kind, key, cap, offset in (
        ("public", "public_test_cases", max_public_tests, 0),
        ("private", "private_test_cases", max_private_tests, n_public),
    ):
        tests = record.get(key, []) or []
        if cap is not None:
            tests = tests[:cap]
        for idx, test in enumerate(tests):
            rubrics.append(
                {
                    "criterion": render_test_criterion(test, fn_name),
                    "points": 1,
                    "tags": [kind],
                    # Global index into the public+private concatenation, matching
                    # the faithful eval's `per_test_results` ordering so judge vs
                    # execution can be joined per test.
                    "test_index": offset + idx,
                    "kind": kind,
                }
            )
    return rubrics
