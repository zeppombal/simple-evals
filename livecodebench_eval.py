"""Faithful LiveCodeBench code-generation eval.

This reproduces the original LiveCodeBench pass@1 by executing generated code
against the bundled unit tests via LCB's own `codegen_metrics` harness (no LLM
judge). It exposes the same generate / evaluate / full mode split as IFEval and
emits the uniform EvalResult / generation-JSONL format the rest of the repo parses.

The generation JSONL it writes is shared with the rubric variants
(`livecodebench_rubric`, `livecodebench_rubric_ar`): each record carries both the
faithful execution sample and the individual test cases.
"""

import hashlib
import json
import os
import random
import tempfile
from typing import Literal

import numpy as np

import common
from ifeval import (  # reuse the shared helpers / aggregation / HTML template
    HEALTHBENCH_HTML_JINJA,
    _aggregate_get_clipped_mean,
    get_usage_dict,
)
from livecodebench_common import codegen_metrics, extract_solution, load_examples
from typess import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult


def _local_exec_tmpdir() -> str:
    """Node-local temp dir for the code-execution harness.

    LCB's `check_correctness` spawns a `multiprocessing.Manager()` per problem,
    which opens an AF_UNIX socket under `$TMPDIR`. If `$TMPDIR` is on NFS (common on
    clusters), the Manager's socket cleanup hits ".nfs* Device or resource busy" and
    breaks the process pool, crashing the eval. Force a node-local dir (ext4 /tmp by
    default); override with LCB_EXEC_TMPDIR (e.g. /mnt/data/<user>/lcb_tmp).
    """
    d = os.environ.get("LCB_EXEC_TMPDIR", "/tmp")
    os.makedirs(d, exist_ok=True)
    return d


def _is_pass(code) -> bool:
    """LCB per-test result codes: True / positive == pass; negatives == fail."""
    return code is True or (isinstance(code, (int, float)) and code > 0)


def _make_generation_record(
    example: dict,
    response_text: str,
    actual_queried_message_list: MessageList,
    response_usage,
) -> dict:
    """Self-contained record consumed by the faithful AND rubric evaluate modes."""
    return {
        "prompt_id": example["prompt_id"],
        "question_id": example["question_id"],
        "prompt": example["prompt"],
        "response_text": response_text,
        "actual_queried_message_list": actual_queried_message_list,
        "response_metadata": {"usage": get_usage_dict(response_usage)},
        "evaluation_sample": example["evaluation_sample"],
        "public_test_cases": example["public_test_cases"],
        "private_test_cases": example["private_test_cases"],
        "fn_name": example["fn_name"],
        "example_tags": example["example_tags"],
        "completion_id": hashlib.sha256(
            (example["prompt_id"] + response_text).encode("utf-8")
        ).hexdigest(),
    }


class LiveCodeBenchEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase | None = None,  # accepted for API parity; unused
        num_examples: int | None = None,
        n_repeats: int = 1,
        n_threads: int = 120,
        release_version: str = "release_v6",
        timeout: int = 6,
        num_process_evaluate: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        mode: Literal["full", "generate", "evaluate"] = "full",
        generation_input_path: str | None = None,
    ):
        self.n_threads = n_threads
        self.timeout = timeout
        self.num_process_evaluate = num_process_evaluate or n_threads
        self.mode = mode
        self.generation_input_path = generation_input_path

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

    # ------------------------------------------------------------------
    # Grading via real code execution (faithful to LiveCodeBench).
    # ------------------------------------------------------------------
    def _grade_records(self, records: list[dict]) -> list[SingleEvalResult]:
        """Run LCB's `codegen_metrics` once over all records and build results."""
        eval_samples = [r["evaluation_sample"] for r in records]
        codes = [extract_solution(r["response_text"]) for r in records]
        generations = [[c] for c in codes]

        # One batched call to the unmodified LCB harness (manages its own pool).
        # Force a node-local TMPDIR so the per-problem multiprocessing.Manager sockets
        # don't land on NFS (which crashes the pool). Restored afterwards.
        _exec_tmp = _local_exec_tmpdir()
        _prev_env, _prev_tf = os.environ.get("TMPDIR"), tempfile.tempdir
        os.environ["TMPDIR"] = _exec_tmp
        tempfile.tempdir = _exec_tmp
        try:
            _, results, final_metadata = codegen_metrics(
                eval_samples,
                generations,
                k_list=[1],
                num_process_evaluate=self.num_process_evaluate,
                timeout=self.timeout,
            )
        finally:
            tempfile.tempdir = _prev_tf
            if _prev_env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = _prev_env

        single_results: list[SingleEvalResult] = []
        for i, record in enumerate(records):
            code = codes[i]
            res_list = results[i][0] if results.get(i) else []
            try:
                error_meta = json.loads(final_metadata[i][0])
            except Exception:
                error_meta = {}

            in_outs = json.loads(record["evaluation_sample"]["input_output"])
            num_total_tests = len(in_outs["inputs"])
            n_public = len(record.get("public_test_cases", []) or [])

            passed_count = sum(1 for c in res_list if _is_pass(c))
            solved = bool(num_total_tests > 0 and passed_count == num_total_tests)
            test_case_pass_rate = (
                passed_count / num_total_tests if num_total_tests else 0.0
            )

            per_test = []
            for idx in range(num_total_tests):
                passed = idx < len(res_list) and _is_pass(res_list[idx])
                per_test.append(
                    {
                        "index": idx,
                        "kind": "public" if idx < n_public else "private",
                        "passed": passed,
                    }
                )

            score = 1.0 if solved else 0.0
            platform, difficulty = record["example_tags"][:2]
            metrics = {
                "overall_score": score,
                "pass@1": score,
                "test_case_pass_rate": test_case_pass_rate,
                platform: score,
                difficulty: score,
            }

            readable = (
                f"pass@1={score} | tests passed {passed_count}/{num_total_tests}"
                + (
                    f"<br>error: {error_meta.get('error_message')}"
                    if not solved and error_meta.get("error_message")
                    else ""
                )
            )
            html = common.jinja_env.from_string(
                HEALTHBENCH_HTML_JINJA.replace("{{ rubric_grades }}", readable)
            ).render(
                prompt_messages=record["actual_queried_message_list"],
                next_message=dict(content=record["response_text"], role="assistant"),
                score=score,
                extracted_answer=code,
            )
            convo = record["actual_queried_message_list"] + [
                dict(content=record["response_text"], role="assistant")
            ]
            single_results.append(
                SingleEvalResult(
                    html=html,
                    score=score,
                    convo=convo,
                    metrics=metrics,
                    example_level_metadata={
                        "score": score,
                        "pass@1": score,
                        "test_case_pass_rate": test_case_pass_rate,
                        "num_total_tests": num_total_tests,
                        "num_passed_tests": passed_count,
                        "per_test_results": per_test,
                        "extracted_code": code,
                        "error_metadata": error_meta,
                        "usage": record["response_metadata"]["usage"],
                        "prompt": record["actual_queried_message_list"],
                        "completion": [
                            dict(content=record["response_text"], role="assistant")
                        ],
                        "prompt_id": record["prompt_id"],
                        "question_id": record["question_id"],
                        "completion_id": record["completion_id"],
                    },
                )
            )
        return single_results

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

    def _run_full(self, sampler: SamplerBase) -> EvalResult:
        records = self._generate_records(sampler)
        return _aggregate_get_clipped_mean(self._grade_records(records))

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
        return _aggregate_get_clipped_mean(self._grade_records(records))
