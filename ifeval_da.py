import argparse
import copy
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import common
from sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from typess import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult

INPUT_PATH = "ifeval/converted.jsonl"

GRADER_TEMPLATE = """
Your job is to look at a conversation and a set of rubric items, and provide an overall score for the last turn (i.e., the last assistant response, or the completion) based on how many of the rubric items are satisfied. Each rubric item is objective and binary: either the completion meets the criterion or it does not.

# Conversation
<<conversation>>

# Rubric items
<<rubric_items>>

# Instructions
Consider all rubric items above. Determine how many of the rubric items are met by the response.
Return a json object with a "score" field.
- The "score" field should be a string representing the fraction of rubric items met, always in the format "X/Y" where X is the number of met rubric items and Y is the total number of rubric items.
- If all rubric items are met, return "Y/Y" (e.g., "3/3").
- If no rubric items are met, return "0/Y" (e.g., "0/3").
- If some but not all rubric items are met, return the appropriate fraction (e.g., "2/3").

# Example
For example, if there are 3 rubric items and 2 are met, you should return:

```json
{
  "score": "2/3"
}
```

If all 3 are met:

```json
{
  "score": "3/3"
}
```

If none are met:

```json
{
  "score": "0/3"
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

HEALTHBENCH_HTML_JINJA = (
    common.HTML_JINJA.replace(
        "<p>Correct Answer: {{ correct_answer }}</p>\n",
        "",
    )
    + "<p>Rubrics with grades: {{ rubric_grades }}</p>"
)


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {
            "explanation": "<<PARSING ERROR>>",
            "score": "0/1",
        }


def parse_score(score_str: str) -> float:
    """Parse a score string like '2/3' into a float."""
    score_str = score_str.strip()
    if "/" in score_str:
        parts = score_str.split("/")
        return float(parts[0]) / float(parts[1])
    return float(score_str)


class RubricItem:
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"{self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }

    try:
        return {
            "input_tokens": response_usage.input_tokens,
            "input_cached_tokens": None,
            "output_tokens": response_usage.output_tokens,
            "output_reasoning_tokens": None,
            "total_tokens": response_usage.total_tokens,
        }
    except AttributeError:
        return {
            "input_tokens": response_usage.prompt_tokens,
            "input_cached_tokens": None,
            "output_tokens": response_usage.completion_tokens,
            "output_reasoning_tokens": None,
            "total_tokens": response_usage.total_tokens,
        }


PHYSICIAN_COMPLETION_MODES = {
    "Group 1": {
        "description": "No reference completions were provided to the physicians.",
        "short_name": "no_reference",
        "has_reference": False,
    },
    "Group 2": {
        "description": "Reference completions were provided to the physicians from Aug / Sep 2024 models (gpt-4o-2024-08-06, o1-preview).",
        "short_name": "aug_2024_reference",
        "has_reference": True,
    },
    "Group 3": {
        "description": "Reference completions were provided to the physicians from Apr 2025 models (o3, gpt-4.1).",
        "short_name": "apr_2025_reference",
        "has_reference": True,
    },
}


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and n_samples for final HealthBench scoring."""
    if stat == "mean":
        return np.clip(np.mean(values), 0, 1)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [
            _compute_clipped_stats(list(s), "mean") for s in bootstrap_samples
        ]
        return np.std(bootstrap_means)
    else:
        raise ValueError(f"Unknown {stat =}")


def _aggregate_get_clipped_mean(
    single_eval_results: list[SingleEvalResult],
) -> EvalResult:
    """
    Aggregate multiple SingleEvalResults into a single EvalResult for HealthBench.
    For each metric, returns the stats in _compute_clipped_stats.
    """
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        for stat in ["mean", "n_samples", "bootstrap_std"]:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_clipped_stats(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


class IFEvalDA(Eval):
    def __init__(
        self,
        grader_model: SamplerBase | None,
        num_examples: int | None = None,
        n_repeats: int = 1,
        # If set, evaluate human completions or reference completions instead of model completions.
        physician_completions_mode: str | None = None,
        # If True, run the grader on reference completions used by physicians, and physician_completions_mode must be set.
        run_reference_completions: bool = False,
        n_threads: int = 120,
        subset_name: Literal["hard", "consensus"] | None = None,
        # Mode: "full" (default), "generate" (only generate responses), "evaluate" (only grade pre-generated responses)
        mode: Literal["full", "generate", "evaluate"] = "full",
        # Path to generation JSONL file (required for mode="evaluate")
        generation_input_path: str | None = None,
    ):
        if run_reference_completions:
            assert (
                physician_completions_mode is not None
            ), "physician_completions_mode must be provided if run_reference_completions is True"
            assert PHYSICIAN_COMPLETION_MODES[physician_completions_mode][
                "has_reference"
            ], "physician_completions_mode must have reference completions if run_reference_completions is True"

        input_path = INPUT_PATH
        with open(input_path, "r") as f:
            examples = [json.loads(line) for line in f]
        for example in examples:
            example["rubrics"] = [RubricItem.from_dict(d) for d in example["rubrics"]]

        rng = random.Random(0)

        # physician completions mode
        self.physician_completions_mode = physician_completions_mode
        if self.physician_completions_mode is not None:
            assert (
                self.physician_completions_mode in PHYSICIAN_COMPLETION_MODES
            ), f"Invalid physician completions mode: {self.physician_completions_mode}; must be one of {PHYSICIAN_COMPLETION_MODES.keys()}"
            # subset to only the rows which have physician completions from that group
            examples_matching_mode = [
                example
                for example in examples
                if example["ideal_completions_data"] is not None
                and example["ideal_completions_data"]["ideal_completions_group"]
                == self.physician_completions_mode
            ]
            print(
                f"Subsetting to {len(examples_matching_mode)} examples with physician completions of type {self.physician_completions_mode} ({PHYSICIAN_COMPLETION_MODES[self.physician_completions_mode]['description']})"
            )

            examples = []
            if run_reference_completions:
                for example in examples_matching_mode:
                    for completion in example["ideal_completions_data"][
                        "ideal_completions_ref_completions"
                    ]:
                        new_example = copy.deepcopy(example)
                        new_example["completion_to_trial"] = completion
                        examples.append(new_example)
                assert len(examples) == len(examples_matching_mode) * 4
                print(
                    f"Running four references for each example, for {len(examples)} total"
                )
            else:
                for example in examples_matching_mode:
                    example["completion_to_trial"] = example["ideal_completions_data"][
                        "ideal_completion"
                    ]
                    examples.append(example)
                assert len(examples) == len(examples_matching_mode)

            if len(examples) == 0:
                raise ValueError(
                    f"No examples found matching mode {self.physician_completions_mode}"
                )

        if num_examples is not None and num_examples < len(examples):
            examples = rng.sample(
                examples,
                num_examples,
            )

        self.examples = examples * n_repeats
        self.n_threads = n_threads
        self.grader_model = grader_model
        self.mode = mode
        self.generation_input_path = generation_input_path

        # Validate mode-specific requirements
        if mode == "evaluate":
            assert (
                generation_input_path is not None
            ), "generation_input_path is required for mode='evaluate'"
        if mode == "generate":
            # grader_model is not needed for generation
            pass
        elif mode in ("evaluate", "full"):
            assert (
                grader_model is not None
            ), "grader_model is required for mode='evaluate' or 'full'"

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
    ) -> tuple[dict, str, list[dict]]:
        # construct conversation string
        convo_with_response = prompt + [dict(content=response_text, role="assistant")]
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response]
        )

        # format all rubrics as a numbered list
        rubric_items_str = "\n".join(
            f"{i+1}. {rubric_item}" for i, rubric_item in enumerate(rubric_items)
        )

        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
            "<<rubric_items>>", rubric_items_str
        )

        messages: MessageList = [dict(content=grader_prompt, role="user")]

        # Single LLM call with retries
        n_retries = 2
        tries = 0
        overall_score = 0.0
        score_raw = f"0/{len(rubric_items)}"
        while tries < n_retries:
            sampler_response = self.grader_model(messages)
            grading_response = sampler_response.response_text
            grading_response_dict = parse_json_to_dict(grading_response)
            if "score" in grading_response_dict:
                try:
                    score_raw = str(grading_response_dict["score"])
                    overall_score = parse_score(score_raw)
                    # Clamp to [0, 1]
                    overall_score = max(0.0, min(1.0, overall_score))
                    break
                except (ValueError, ZeroDivisionError):
                    pass
            print("Grading failed due to bad JSON output, retrying...")
            tries += 1

        metrics = {
            "overall_score": overall_score,
        }

        # Example-level tags get the same overall score
        example_tag_scores = {tag: overall_score for tag in example_tags}
        assert len(example_tag_scores) == len(example_tags)  # No duplicates.
        metrics.update(example_tag_scores)

        # No rubric-level tag scores in DA mode (no per-rubric breakdown available)

        # Readable explanation (simplified, no per-rubric detail)
        readable_explanation_str = (
            f"\n\nDirect assessment score: {score_raw} ({overall_score:.4f})"
        )

        # rubric_items_with_grades: store rubrics but without individual criteria_met
        rubric_items_with_grades = [
            {
                **rubric_item.to_dict(),
                "criteria_met": None,
                "explanation": "Direct assessment mode - no per-rubric grading",
            }
            for rubric_item in rubric_items
        ]

        return metrics, readable_explanation_str, rubric_items_with_grades, score_raw

    def __call__(self, sampler: SamplerBase | None) -> EvalResult:
        """Router that dispatches to the appropriate method based on mode."""
        if self.mode == "generate":
            return self._run_generation(sampler)
        elif self.mode == "evaluate":
            return self._run_evaluation()
        else:  # "full"
            return self._run_full(sampler)

    def _run_full(self, sampler: SamplerBase) -> EvalResult:
        """Run generation and evaluation together (original behavior)."""

        def fn(row: dict):
            prompt_messages = row["prompt"]

            if self.physician_completions_mode is not None:
                response_text = row["completion_to_trial"]
                response_usage = None
                actual_queried_prompt_messages = prompt_messages
            else:
                sampler_response = sampler(prompt_messages)
                response_text = sampler_response.response_text
                response_dict = sampler_response.response_metadata
                actual_queried_prompt_messages = (
                    sampler_response.actual_queried_message_list
                )
                response_usage = response_dict.get("usage", None)

            metrics, readable_explanation_str, rubric_items_with_grades, score_raw = (
                self.grade_sample(
                    prompt=actual_queried_prompt_messages,
                    response_text=response_text,
                    rubric_items=row["rubrics"],
                    example_tags=row["example_tags"],
                )
            )

            score = metrics["overall_score"]

            # Create HTML for each sample result
            html = common.jinja_env.from_string(
                HEALTHBENCH_HTML_JINJA.replace(
                    "{{ rubric_grades }}",
                    readable_explanation_str.replace("\n", "<br>"),
                )
            ).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=metrics["overall_score"],
                extracted_answer=response_text,
            )

            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata={
                    "score": score,
                    "score_raw": score_raw,
                    "usage": get_usage_dict(response_usage),
                    "rubric_items": rubric_items_with_grades,
                    "prompt": actual_queried_prompt_messages,
                    "completion": [dict(content=response_text, role="assistant")],
                    "prompt_id": row["prompt_id"],
                    "completion_id": hashlib.sha256(
                        (row["prompt_id"] + response_text).encode("utf-8")
                    ).hexdigest(),
                },
            )

        results = common.map_with_progress(
            fn,
            self.examples,
            num_threads=self.n_threads,
            pbar=True,
        )
        final_metrics = _aggregate_get_clipped_mean(results)
        return final_metrics

    def _run_generation(self, sampler: SamplerBase) -> EvalResult:
        """Generate responses only (no grading). Returns metadata for saving to JSONL."""

        def fn(row: dict):
            prompt_messages = row["prompt"]

            if self.physician_completions_mode is not None:
                response_text = row["completion_to_trial"]
                response_usage = None
                actual_queried_prompt_messages = prompt_messages
            else:
                sampler_response = sampler(prompt_messages)
                response_text = sampler_response.response_text
                response_dict = sampler_response.response_metadata
                actual_queried_prompt_messages = (
                    sampler_response.actual_queried_message_list
                )
                response_usage = response_dict.get("usage", None)

            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]

            # Build generation record with all info needed for later evaluation
            generation_record = {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "response_text": response_text,
                "actual_queried_message_list": actual_queried_prompt_messages,
                "response_metadata": {"usage": get_usage_dict(response_usage)},
                "example_tags": row["example_tags"],
                "rubrics": [rubric.to_dict() for rubric in row["rubrics"]],
                "completion_id": hashlib.sha256(
                    (row["prompt_id"] + response_text).encode("utf-8")
                ).hexdigest(),
            }

            return SingleEvalResult(
                score=None,
                metrics={},
                html=None,
                convo=convo,
                example_level_metadata=generation_record,
            )

        results = common.map_with_progress(
            fn,
            self.examples,
            num_threads=self.n_threads,
            pbar=True,
        )

        # Return minimal EvalResult (no scores, no HTML)
        return EvalResult(
            score=None,
            metrics={},
            htmls=[],
            convos=[r.convo for r in results],
            metadata={
                "example_level_metadata": [r.example_level_metadata for r in results]
            },
        )

    def _run_evaluation(self) -> EvalResult:
        """Load generations from JSONL and grade them."""
        # Load generation records
        with open(self.generation_input_path, "r") as f:
            generation_records = [json.loads(line) for line in f]

        def fn(gen_record: dict):
            # Reconstruct rubric items from dict
            rubric_items = [RubricItem.from_dict(d) for d in gen_record["rubrics"]]

            # Run grading
            metrics, readable_explanation_str, rubric_items_with_grades, score_raw = (
                self.grade_sample(
                    prompt=gen_record["actual_queried_message_list"],
                    response_text=gen_record["response_text"],
                    rubric_items=rubric_items,
                    example_tags=gen_record["example_tags"],
                )
            )

            score = metrics["overall_score"]

            # Create HTML
            html = common.jinja_env.from_string(
                HEALTHBENCH_HTML_JINJA.replace(
                    "{{ rubric_grades }}",
                    readable_explanation_str.replace("\n", "<br>"),
                )
            ).render(
                prompt_messages=gen_record["actual_queried_message_list"],
                next_message=dict(
                    content=gen_record["response_text"], role="assistant"
                ),
                score=score,
                extracted_answer=gen_record["response_text"],
            )

            convo = gen_record["actual_queried_message_list"] + [
                dict(content=gen_record["response_text"], role="assistant")
            ]

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata={
                    "score": score,
                    "score_raw": score_raw,
                    "usage": gen_record["response_metadata"]["usage"],
                    "rubric_items": rubric_items_with_grades,
                    "prompt": gen_record["actual_queried_message_list"],
                    "completion": [
                        dict(content=gen_record["response_text"], role="assistant")
                    ],
                    "prompt_id": gen_record["prompt_id"],
                    "completion_id": gen_record["completion_id"],
                },
            )

        results = common.map_with_progress(
            fn,
            generation_records,
            num_threads=self.n_threads,
            pbar=True,
        )
        final_metrics = _aggregate_get_clipped_mean(results)
        return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="IFEval Direct Assessment (DA) mode: all rubrics scored as a single fraction."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["physician_completions", "physician_completion_references"],
    )
    parser.add_argument("--examples", type=int, help="Number of examples to run")
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run",
    )
    args = parser.parse_args()

    if args.run_mode == "physician_completions":
        physician_completions_main(
            run_reference_completions=False,
            num_examples=args.examples,
            n_threads=args.n_threads or 1,
        )
    elif args.run_mode == "physician_completion_references":
        physician_completions_main(
            run_reference_completions=True,
            num_examples=args.examples,
            n_threads=args.n_threads or 1,
        )

    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")


def physician_completions_main(
    run_reference_completions: bool = False,
    num_examples: int | None = None,
    n_threads: int = 120,
):
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    dummy_sampler = SamplerBase()

    merge_metrics = []
    for pc_mode in PHYSICIAN_COMPLETION_MODES.keys():
        if (
            run_reference_completions
            and not PHYSICIAN_COMPLETION_MODES[pc_mode]["has_reference"]
        ):
            continue

        # run
        eval = IFEvalDA(
            grader_model=grading_sampler,
            physician_completions_mode=pc_mode,
            run_reference_completions=run_reference_completions,
            num_examples=num_examples,
            n_threads=n_threads,
        )
        result = eval(dummy_sampler)

        # report
        parsable_mode = PHYSICIAN_COMPLETION_MODES[pc_mode]["short_name"]
        if run_reference_completions:
            file_stem = f"ifeval_da_{parsable_mode}_referencecompletions_{date_str}"
        else:
            file_stem = f"ifeval_da_{parsable_mode}_humanbaseline_{date_str}"
        report_filename = Path(f"{file_stem}.html")
        report_filename.write_text(common.make_report(result))
        print(f"Report saved to {report_filename}")

        # metrics
        assert result.metrics is not None
        metrics = result.metrics
        result_filename = Path(f"{file_stem}.json")
        result_filename.write_text(json.dumps(metrics))
        print(f"Results saved to {result_filename}")

        full_result_dict = {
            "score": result.score,
            "metrics": result.metrics,
            "htmls": result.htmls,
            "convos": result.convos,
            "metadata": result.metadata,
        }
        full_result_filename = Path(f"{file_stem}_allresults.json")
        full_result_filename.write_text(json.dumps(full_result_dict, indent=2))
        print(f"All results saved to {full_result_filename}")

        # metrics df
        merge_metrics.append(
            {
                "eval_name": "ifeval_da",
                "model_name": f"{pc_mode} ({PHYSICIAN_COMPLETION_MODES[pc_mode]['description']})",
                "metric": metrics.get("overall_score", None),
            }
        )

    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
