"""LiveCodeBench code-generation rubric eval, aggregate (AR) variant.

Like `livecodebench_rubric` but all of a problem's unit tests are graded in a
SINGLE judge call (mirrors `ifeval_ar.py`). The judge returns a list of
`criteria_met` booleans, one per test, parsed with the same salvage/pad/truncate
robustness as IFEvalAR.
"""

import json
import random
from collections import defaultdict
from typing import Literal

import common
from ifeval import (
    HEALTHBENCH_HTML_JINJA,
    RubricItem,
    _aggregate_get_clipped_mean,
    calculate_score,
)
from ifeval_ar import _parse_ar_results, parse_json_to_dict
from livecodebench_common import (
    build_rubric_dicts,
    extract_solution,
    load_examples,
    log_oversized_skip_summary,
)
from livecodebench_eval import _make_generation_record
from livecodebench_rubric import _problem_text
from typess import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult

GRADER_TEMPLATE = """
Your job is to determine, for each of several unit tests, whether a candidate Python solution to a programming problem passes that test. You are given the problem statement, the candidate's submitted code, and a numbered list of unit tests (each an input and the expected output). Reason about what the code does when run on each input.

# Problem
<<problem>>

# Candidate code
```python
<<code>>
```

# Unit tests (rubric items)
<<rubric_items>>

# Instructions
Return a json object with a "results" field containing a list of objects, one per unit test, in the same order as the unit tests above. Each object should have a "criteria_met" field (boolean).
- "criteria_met" should be true only if running the candidate code on that test's input would produce exactly the expected output (ignoring trailing whitespace). A runtime error, a wrong answer, an infinite loop / timeout, or code that does not compile all mean the criterion is NOT met.

# Example
For 3 unit tests where the code passes the first and third but not the second:

```json
{
  "results": [
    {"criteria_met": true},
    {"criteria_met": false},
    {"criteria_met": true}
  ]
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class LiveCodeBenchRubricAR(Eval):
    def __init__(
        self,
        grader_model: SamplerBase | None,
        num_examples: int | None = None,
        n_repeats: int = 1,
        n_threads: int = 120,
        release_version: str = "release_v6",
        max_public_tests: int | None = None,
        max_private_tests: int | None = None,
        max_test_chars: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        mode: Literal["full", "generate", "evaluate"] = "full",
        generation_input_path: str | None = None,
    ):
        self.n_threads = n_threads
        self.grader_model = grader_model
        self.mode = mode
        self.generation_input_path = generation_input_path
        self.max_public_tests = max_public_tests
        self.max_private_tests = max_private_tests
        self.max_test_chars = max_test_chars

        if mode == "evaluate":
            assert (
                generation_input_path is not None
            ), "generation_input_path is required for mode='evaluate'"
            self.examples = []
        else:
            rng = random.Random(0)
            self.examples = load_examples(
                release_version=release_version,
                num_examples=num_examples,
                rng=rng,
                n_repeats=n_repeats,
                start_date=start_date,
                end_date=end_date,
            )

        if mode in ("evaluate", "full"):
            assert grader_model is not None, "grader_model is required for this mode"

    # ------------------------------------------------------------------
    # Grading: a single judge call for all unit tests of a problem.
    # ------------------------------------------------------------------
    def grade_sample(
        self,
        problem_text: str,
        code: str,
        rubric_dicts: list[dict],
        example_tags: list[str],
    ) -> tuple[dict, str, list[dict]]:
        rubric_items = [RubricItem.from_dict(d) for d in rubric_dicts]
        rubric_items_str = "\n".join(
            f"{i+1}. {item.criterion}" for i, item in enumerate(rubric_items)
        )
        grader_prompt = (
            GRADER_TEMPLATE.replace("<<problem>>", problem_text)
            .replace("<<code>>", code)
            .replace("<<rubric_items>>", rubric_items_str)
        )
        messages: MessageList = [dict(content=grader_prompt, role="user")]

        grading_response_list = None
        if rubric_items:
            tries, n_retries = 0, 2
            while tries < n_retries:
                sampler_response = self.grader_model(messages)
                parsed = _parse_ar_results(
                    parse_json_to_dict(sampler_response.response_text),
                    len(rubric_items),
                )
                if parsed is not None:
                    grading_response_list = parsed
                    break
                print("Grading failed due to bad JSON output, retrying...")
                tries += 1
        if grading_response_list is None:
            grading_response_list = [{"criteria_met": False}] * len(rubric_items)

        overall_score = calculate_score(rubric_items, grading_response_list)
        if overall_score is None:
            overall_score = 0.0
        metrics = {"overall_score": overall_score}

        example_tag_scores = {tag: overall_score for tag in example_tags}
        metrics.update(example_tag_scores)

        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append((rubric_item, grading_response))
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_score(list(items), list(grades))
            if score is not None:
                metrics[tag] = score

        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_dict, rubric_item, grading_response in zip(
            rubric_dicts, rubric_items, grading_response_list
        ):
            criteria_met = grading_response["criteria_met"]
            explanation = grading_response.get("explanation", "No explanation provided")
            readable_explanation_list.append(
                f"[{criteria_met}] ({rubric_dict.get('kind')}) {rubric_item}\n\tExplanation: {explanation}"
            )
            rubric_items_with_grades.append(
                {
                    **rubric_item.to_dict(),
                    "test_index": rubric_dict.get("test_index"),
                    "kind": rubric_dict.get("kind"),
                    "criteria_met": criteria_met,
                    "explanation": explanation,
                }
            )

        readable_explanation_list.sort(
            key=lambda x: x.startswith("[False]"), reverse=True
        )
        readable_explanation_str = "\n\n" + "\n\n".join(readable_explanation_list)
        return metrics, readable_explanation_str, rubric_items_with_grades

    def _grade_record(self, record: dict) -> SingleEvalResult:
        problem_text = _problem_text(record["prompt"])
        code = extract_solution(record["response_text"])
        rubric_dicts, skip_summary = build_rubric_dicts(
            record,
            max_public_tests=self.max_public_tests,
            max_private_tests=self.max_private_tests,
            max_test_chars=self.max_test_chars,
        )
        actual_msgs = record["actual_queried_message_list"]
        completion = [dict(content=record["response_text"], role="assistant")]
        convo = actual_msgs + completion

        # All tests oversized → nothing judgeable; exclude from the aggregate.
        if not rubric_dicts:
            html = common.jinja_env.from_string(
                HEALTHBENCH_HTML_JINJA.replace(
                    "{{ rubric_grades }}",
                    "All test cases skipped (exceeded max_test_chars).",
                )
            ).render(
                prompt_messages=actual_msgs,
                next_message=dict(content=record["response_text"], role="assistant"),
                score=0.0,
                extracted_answer=code,
            )
            return SingleEvalResult(
                html=html,
                score=None,
                convo=convo,
                metrics={},
                example_level_metadata={
                    "score": None,
                    "fully_skipped": True,
                    "tests_skipped_oversized": skip_summary,
                    "usage": record["response_metadata"]["usage"],
                    "extracted_code": code,
                    "rubric_items": [],
                    "prompt": actual_msgs,
                    "completion": completion,
                    "prompt_id": record["prompt_id"],
                    "question_id": record["question_id"],
                    "completion_id": record["completion_id"],
                },
            )

        metrics, readable_explanation_str, rubric_items_with_grades = self.grade_sample(
            problem_text=problem_text,
            code=code,
            rubric_dicts=rubric_dicts,
            example_tags=record["example_tags"],
        )
        score = metrics["overall_score"]
        html = common.jinja_env.from_string(
            HEALTHBENCH_HTML_JINJA.replace(
                "{{ rubric_grades }}", readable_explanation_str.replace("\n", "<br>")
            )
        ).render(
            prompt_messages=actual_msgs,
            next_message=dict(content=record["response_text"], role="assistant"),
            score=score,
            extracted_answer=code,
        )
        return SingleEvalResult(
            html=html,
            score=score,
            convo=convo,
            metrics=metrics,
            example_level_metadata={
                "score": score,
                "fully_skipped": False,
                "tests_skipped_oversized": skip_summary,
                "usage": record["response_metadata"]["usage"],
                "extracted_code": code,
                "rubric_items": rubric_items_with_grades,
                "prompt": actual_msgs,
                "completion": completion,
                "prompt_id": record["prompt_id"],
                "question_id": record["question_id"],
                "completion_id": record["completion_id"],
            },
        )

    # ------------------------------------------------------------------
    # Mode router (mirrors IFEval).
    # ------------------------------------------------------------------
    def __call__(self, sampler: SamplerBase | None) -> EvalResult:
        if self.mode == "generate":
            return self._run_generation(sampler)
        elif self.mode == "evaluate":
            return self._run_evaluation()
        else:  # "full"
            return self._run_full(sampler)

    def _aggregate_and_log(self, results: list[SingleEvalResult]) -> EvalResult:
        log_oversized_skip_summary(
            [r.example_level_metadata for r in results], self.max_test_chars
        )
        return _aggregate_get_clipped_mean(results)

    def _run_full(self, sampler: SamplerBase) -> EvalResult:
        def fn(example: dict) -> SingleEvalResult:
            sampler_response = sampler(example["prompt"])
            record = _make_generation_record(
                example=example,
                response_text=sampler_response.response_text,
                actual_queried_message_list=sampler_response.actual_queried_message_list,
                response_usage=sampler_response.response_metadata.get("usage", None),
            )
            return self._grade_record(record)

        results = common.map_with_progress(
            fn, self.examples, num_threads=self.n_threads, pbar=True
        )
        return self._aggregate_and_log(results)

    def _run_generation(self, sampler: SamplerBase) -> EvalResult:
        def fn(example: dict) -> dict:
            sampler_response = sampler(example["prompt"])
            return _make_generation_record(
                example=example,
                response_text=sampler_response.response_text,
                actual_queried_message_list=sampler_response.actual_queried_message_list,
                response_usage=sampler_response.response_metadata.get("usage", None),
            )

        records = common.map_with_progress(
            fn, self.examples, num_threads=self.n_threads, pbar=True
        )
        return EvalResult(
            score=None,
            metrics={},
            htmls=[],
            convos=[
                r["actual_queried_message_list"]
                + [dict(content=r["response_text"], role="assistant")]
                for r in records
            ],
            metadata={"example_level_metadata": records},
        )

    def _run_evaluation(self) -> EvalResult:
        with open(self.generation_input_path, "r") as f:
            records = [json.loads(line) for line in f]
        results = common.map_with_progress(
            self._grade_record, records, num_threads=self.n_threads, pbar=True
        )
        return self._aggregate_and_log(results)
