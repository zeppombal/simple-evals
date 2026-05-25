"""LiveCodeBench code-generation as a rubric / LLM-as-a-judge eval (per-item).

Instead of executing code, each unit test becomes a rubric item and an LLM judge
decides — one call per test — whether the candidate code would produce the expected
output for that test's input. This mirrors `ifeval.py` (single call per rubric item)
and lets us study judge-vs-execution agreement.

It reads the same generation JSONL produced by `livecodebench` (faithful) generate
mode; rubric caps (`max_public_tests` / `max_private_tests`) are applied at grade
time, so the judged test set can be reconfigured without regenerating.
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
from livecodebench_common import (
    build_rubric_dicts,
    extract_solution,
    load_examples,
    log_oversized_skip_summary,
    parse_judge_json,
)
from livecodebench_eval import _make_generation_record
from typess import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult

GRADER_TEMPLATE = """
Your job is to determine whether a candidate Python solution to a programming problem passes a specific unit test. You are given the problem statement, the candidate's submitted code, and one unit test (an input and the expected output). Reason about what the code does when run on the given input, and decide whether it produces the expected output.

# Problem
<<problem>>

# Candidate code
```python
<<code>>
```

# Unit test (rubric item)
<<rubric_item>>

# Instructions
Return a json object with a "criteria_met" field.
- "criteria_met" should be a boolean: true only if running the candidate code on the given input would produce exactly the expected output (ignoring trailing whitespace). A runtime error, a wrong answer, an infinite loop / timeout (30 seconds), or code that does not compile all mean the criterion is NOT met.

# Example
```json
{
  "criteria_met": false
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def _problem_text(prompt: MessageList) -> str:
    """The problem statement = the user turn(s) of the generation prompt."""
    return "\n\n".join(m["content"] for m in prompt if m.get("role") == "user")


class LiveCodeBenchRubric(Eval):
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
            assert generation_input_path is not None, (
                "generation_input_path is required for mode='evaluate'"
            )
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
    # Grading: one judge call per unit test (rubric item).
    # ------------------------------------------------------------------
    def _judge_one(
        self, problem_text: str, code: str, criterion: str, n_retries: int = 2
    ) -> dict:
        """Single judge call for one unit test (with retries on bad JSON)."""
        grader_prompt = (
            GRADER_TEMPLATE.replace("<<problem>>", problem_text)
            .replace("<<code>>", code)
            .replace("<<rubric_item>>", criterion)
        )
        messages: MessageList = [dict(content=grader_prompt, role="user")]
        tries = 0
        grading_response_dict = {"criteria_met": False}
        while tries < n_retries:
            sampler_response = self.grader_model(messages)
            grading_response_dict = parse_judge_json(sampler_response.response_text)
            if grading_response_dict.get("criteria_met") in (True, False):
                break
            print("Grading failed due to bad JSON output, retrying...")
            tries += 1
        return grading_response_dict

    def _aggregate_grades(
        self,
        rubric_dicts: list[dict],
        grading_response_list: list[dict],
        example_tags: list[str],
    ) -> tuple[dict, str, list[dict]]:
        """Turn per-test judge responses into per-problem metrics + grade records.

        Pure (no judging) — the judge calls are made separately so the progress bar
        can track individual tests across all problems.
        """
        rubric_items = [RubricItem.from_dict(d) for d in rubric_dicts]

        overall_score = calculate_score(rubric_items, grading_response_list)
        if overall_score is None:  # no test cases for this problem (shouldn't happen)
            overall_score = 0.0
        metrics = {"overall_score": overall_score}

        # example-level tags (platform, difficulty)
        example_tag_scores = {tag: overall_score for tag in example_tags}
        metrics.update(example_tag_scores)

        # rubric-level tags (public / private)
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append((rubric_item, grading_response))
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_score(list(items), list(grades))
            if score is not None:
                metrics[tag] = score

        # build per-test grade records (carry test_index/kind for join with execution)
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

    def _build_plan(self, record: dict) -> dict:
        """Precompute everything needed to judge a record's surviving tests."""
        rubric_dicts, skip_summary = build_rubric_dicts(
            record,
            max_public_tests=self.max_public_tests,
            max_private_tests=self.max_private_tests,
            max_test_chars=self.max_test_chars,
        )
        return {
            "record": record,
            "code": extract_solution(record["response_text"]),
            "problem_text": _problem_text(record["prompt"]),
            "rubric_dicts": rubric_dicts,
            "skip_summary": skip_summary,
        }

    def _score_record(
        self, plan: dict, grading_response_list: list[dict]
    ) -> SingleEvalResult:
        """Build the per-problem result from precomputed per-test judge responses."""
        record = plan["record"]
        code = plan["code"]
        rubric_dicts = plan["rubric_dicts"]
        skip_summary = plan["skip_summary"]
        actual_msgs = record["actual_queried_message_list"]
        completion = [dict(content=record["response_text"], role="assistant")]
        convo = actual_msgs + completion

        # All tests oversized → nothing judgeable; exclude from the aggregate
        # (score=None) instead of scoring it 0, but keep it in metadata for coverage.
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

        metrics, readable_explanation_str, rubric_items_with_grades = (
            self._aggregate_grades(
                rubric_dicts=rubric_dicts,
                grading_response_list=grading_response_list,
                example_tags=record["example_tags"],
            )
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
        metas = [r.example_level_metadata for r in results]
        log_oversized_skip_summary(metas, self.max_test_chars)
        result = _aggregate_get_clipped_mean(results)
        if result.metrics is None:
            result.metrics = {}
        # Per-test judge "pass" rates (criteria_met), named to match the faithful eval
        # (EvaluateLiveCodeBenchTrue) so judge vs execution can be compared directly.
        # micro = pooled over all judged tests; macro = mean of per-instance rates.
        n_met = sum(
            1 for m in metas for ri in m.get("rubric_items", []) if ri.get("criteria_met")
        )
        n_total = sum(len(m.get("rubric_items", [])) for m in metas)
        macro_rates = [m["score"] for m in metas if m.get("score") is not None]
        result.metrics["avg_test_pass_micro"] = n_met / n_total if n_total else 0.0
        result.metrics["avg_test_pass_macro"] = (
            sum(macro_rates) / len(macro_rates) if macro_rates else 0.0
        )
        return result

    def _generate_records(self, sampler: SamplerBase) -> list[dict]:
        def fn(example: dict) -> dict:
            sampler_response = sampler(example["prompt"])
            return _make_generation_record(
                example=example,
                response_text=sampler_response.response_text,
                actual_queried_message_list=sampler_response.actual_queried_message_list,
                response_usage=sampler_response.response_metadata.get("usage", None),
            )

        return common.map_with_progress(
            fn, self.examples, num_threads=self.n_threads, pbar=True
        )

    def _grade_and_aggregate(self, records: list[dict]) -> EvalResult:
        """Judge every surviving test with ONE flat progress bar over all tests.

        Flattening (vs nesting a per-test map inside a per-problem map) makes the bar
        reflect individual judge calls — like the faithful eval's per-test view — and
        keeps total threads at n_threads instead of n_problems x tests_per_problem.
        """
        plans = [self._build_plan(r) for r in records]
        grades: list[list[dict]] = [
            [None] * len(p["rubric_dicts"]) for p in plans  # type: ignore[list-item]
        ]
        tasks = [
            (pi, ri)
            for pi, plan in enumerate(plans)
            for ri in range(len(plan["rubric_dicts"]))
        ]

        def run_task(t: tuple[int, int]):
            pi, ri = t
            plan = plans[pi]
            grade = self._judge_one(
                plan["problem_text"], plan["code"], plan["rubric_dicts"][ri]["criterion"]
            )
            return pi, ri, grade

        flat = (
            common.map_with_progress(
                run_task, tasks, num_threads=self.n_threads, pbar=True
            )
            if tasks
            else []
        )
        for pi, ri, grade in flat:
            grades[pi][ri] = grade

        results = [self._score_record(plans[pi], grades[pi]) for pi in range(len(plans))]
        return self._aggregate_and_log(results)

    def _run_full(self, sampler: SamplerBase) -> EvalResult:
        records = self._generate_records(sampler)
        return self._grade_and_aggregate(records)

    def _run_generation(self, sampler: SamplerBase) -> EvalResult:
        records = self._generate_records(sampler)
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
        return self._grade_and_aggregate(records)
