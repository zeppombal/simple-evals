import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import common
from sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from typess import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult

GRADER_TEMPLATE = """
Your job is to compare two responses to a conversation, given a set of rubric items, and determine which response better satisfies the rubric items. Each rubric item is objective and binary: either the completion meets the criterion or it does not.

# Conversation
<<conversation>>

# Rubric items
<<rubric_items>>

<Response A>
<<response_a>>
</Response A>

<Response B>
<<response_b>>
</Response B>

# Instructions
Compare Response A and Response B based on how well each satisfies the rubric items above.
Return a json object with an "outcome" field.
- The "outcome" field should be "A is better" if Response A better satisfies the rubric items overall.
- The "outcome" field should be "B is better" if Response B better satisfies the rubric items overall.
- The "outcome" field should be "tie" if both responses satisfy the rubric items equally well (or equally poorly).

Consider ALL rubric items when making your judgment. A response that satisfies more rubric items should generally be preferred.

# Example 1

```json
{
  "outcome": "tie"
}
```

# Example 2

```json
{
  "outcome": "B is better"
}
```

# Example 3

```json
{
  "outcome": "A is better"
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

VALID_OUTCOMES = {"A is better", "B is better", "tie"}

PWC_HTML_JINJA = """
<h3>Prompt</h3>
{% for message in prompt_messages %}
<div class="message">
  <div class="role">{{ message.role }}</div>
  <div class="content"><pre>{{ message.content }}</pre></div>
</div>
{% endfor %}
<h3>Response A</h3>
<pre>{{ response_a }}</pre>
<h3>Response B</h3>
<pre>{{ response_b }}</pre>
<h3>Result</h3>
<p>Outcome: {{ outcome }}</p>
<p>Score (for A): {{ score }}</p>
"""


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {
            "explanation": "<<PARSING ERROR>>",
            "outcome": "tie",
        }


class RubricItem:
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"{self.criterion}"  # in healthbench there is an argument in favor of ablating including points in the prompt

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


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and n_samples."""
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
    Aggregate multiple SingleEvalResults into a single EvalResult.
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


class IFEvalPWC(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        num_examples: int | None = None,
        n_threads: int = 120,
        # Two generation input paths — required
        generation_input_path_a: str | None = None,
        generation_input_path_b: str | None = None,
    ):
        assert grader_model is not None, "grader_model is required for PWC"
        assert (
            generation_input_path_a is not None
        ), "generation_input_path_a is required"
        assert (
            generation_input_path_b is not None
        ), "generation_input_path_b is required"

        self.grader_model = grader_model
        self.n_threads = n_threads
        self.num_examples = num_examples
        self.generation_input_path_a = generation_input_path_a
        self.generation_input_path_b = generation_input_path_b

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text_a: str,
        response_text_b: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
    ) -> tuple[dict, str, dict]:
        # Format conversation (prompt only, without either response)
        convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in prompt])

        # Format all rubrics as a numbered list
        rubric_items_str = "\n".join(
            f"{i+1}. {rubric_item}" for i, rubric_item in enumerate(rubric_items)
        )

        grader_prompt = (
            GRADER_TEMPLATE.replace("<<conversation>>", convo_str)
            .replace("<<rubric_items>>", rubric_items_str)
            .replace("<<response_a>>", response_text_a)
            .replace("<<response_b>>", response_text_b)
        )

        messages: MessageList = [dict(content=grader_prompt, role="user")]

        # Single LLM call with retries
        n_retries = 2
        tries = 0
        outcome = "tie"  # Default fallback
        while tries < n_retries:
            sampler_response = self.grader_model(messages)
            grading_response = sampler_response.response_text
            grading_response_dict = parse_json_to_dict(grading_response)
            if "outcome" in grading_response_dict:
                o = grading_response_dict["outcome"]
                if o in VALID_OUTCOMES:
                    outcome = o
                    break
            print("Grading failed due to bad JSON output, retrying...")
            tries += 1

        # Score from perspective of model A
        if outcome == "A is better":
            overall_score = 1.0
        elif outcome == "B is better":
            overall_score = 0.0
        else:  # tie
            overall_score = 0.5

        metrics = {"overall_score": overall_score}

        # Example-level tags
        example_tag_scores = {tag: overall_score for tag in example_tags}
        metrics.update(example_tag_scores)

        # No rubric-level tags (instance-level comparison)

        readable_explanation_str = (
            f"\n\nPairwise comparison outcome: {outcome} (score for A: {overall_score})"
        )

        comparison_result = {"outcome": outcome}

        return metrics, readable_explanation_str, comparison_result

    def __call__(self, sampler: SamplerBase | None) -> EvalResult:
        """PWC only supports evaluate mode."""
        return self._run_evaluation()

    def _run_evaluation(self) -> EvalResult:
        """Load both generation files, pair by prompt_id, and grade comparisons."""
        # Load both generation files
        with open(self.generation_input_path_a, "r") as f:
            records_a = {}
            for line in f:
                r = json.loads(line)
                records_a[r["prompt_id"]] = r
        with open(self.generation_input_path_b, "r") as f:
            records_b = {}
            for line in f:
                r = json.loads(line)
                records_b[r["prompt_id"]] = r

        # Find common prompt_ids
        common_ids = [pid for pid in records_a.keys() if pid in records_b]
        if len(common_ids) < len(records_a) or len(common_ids) < len(records_b):
            print(
                f"Warning: {len(records_a)} records in A, {len(records_b)} in B, "
                f"{len(common_ids)} in common"
            )

        # Optionally subsample
        if self.num_examples is not None and self.num_examples < len(common_ids):
            rng = random.Random(0)
            common_ids = rng.sample(common_ids, self.num_examples)

        # Build paired records
        paired_records = [
            {
                "prompt_id": pid,
                "record_a": records_a[pid],
                "record_b": records_b[pid],
            }
            for pid in common_ids
        ]

        def fn(pair: dict):
            rec_a = pair["record_a"]
            rec_b = pair["record_b"]
            rubric_items = [RubricItem.from_dict(d) for d in rec_a["rubrics"]]

            metrics, readable_explanation_str, comparison_result = self.grade_sample(
                prompt=rec_a["actual_queried_message_list"],
                response_text_a=rec_a["response_text"],
                response_text_b=rec_b["response_text"],
                example_tags=rec_a["example_tags"],
                rubric_items=rubric_items,
            )

            score = metrics["overall_score"]

            html = common.jinja_env.from_string(PWC_HTML_JINJA).render(
                prompt_messages=rec_a["actual_queried_message_list"],
                response_a=rec_a["response_text"],
                response_b=rec_b["response_text"],
                outcome=comparison_result["outcome"],
                score=score,
            )

            convo = rec_a["actual_queried_message_list"]

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata={
                    "score": score,
                    "outcome": comparison_result["outcome"],
                    "prompt_id": pair["prompt_id"],
                    "response_text_a": rec_a["response_text"],
                    "response_text_b": rec_b["response_text"],
                    "completion_id_a": rec_a["completion_id"],
                    "completion_id_b": rec_b["completion_id"],
                },
            )

        results = common.map_with_progress(
            fn,
            paired_records,
            num_threads=self.n_threads,
            pbar=True,
        )
        final_metrics = _aggregate_get_clipped_mean(results)
        return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="IFEval Pairwise Comparison (PWC) mode: compare two models' responses side-by-side."
    )
    parser.add_argument(
        "--generation-input-a",
        type=str,
        required=True,
        help="Path to first generation JSONL file (Response A).",
    )
    parser.add_argument(
        "--generation-input-b",
        type=str,
        required=True,
        help="Path to second generation JSONL file (Response B).",
    )
    parser.add_argument("--examples", type=int, help="Number of examples to run")
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="Grader model name.",
    )
    parser.add_argument(
        "--base-url-grader",
        type=str,
        default=None,
        help="Base url for custom grader.",
    )
    parser.add_argument(
        "--grader-completion-args",
        type=str,
        default="{}",
        help="JSON string of completion args to pass to the grader.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output JSON.",
    )
    args = parser.parse_args()

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    if args.base_url_grader:
        grading_sampler = ChatCompletionSampler(
            model=args.grader_model,
            max_tokens=2048,
            base_url=args.base_url_grader,
            temperature=0.1,
            completion_args=json.loads(args.grader_completion_args.replace("'", '"')),
        )
    else:
        grading_sampler = ChatCompletionSampler(
            model=args.grader_model,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )

    eval_obj = IFEvalPWC(
        grader_model=grading_sampler,
        num_examples=args.examples,
        n_threads=args.n_threads or 1,
        generation_input_path_a=args.generation_input_a,
        generation_input_path_b=args.generation_input_b,
    )
    result = eval_obj(None)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stem = f"ifeval_pwc_{date_str}"

    report_filename = output_dir / f"{file_stem}.html"
    report_filename.write_text(common.make_report(result))
    print(f"Report saved to {report_filename}")

    assert result.metrics is not None
    metrics = result.metrics | {"score": result.score}
    metrics = dict(sorted(metrics.items()))
    result_filename = output_dir / f"{file_stem}.json"
    result_filename.write_text(json.dumps(metrics, indent=2))
    print(f"Results saved to {result_filename}")

    full_result_dict = {
        "score": result.score,
        "metrics": result.metrics,
        "htmls": result.htmls,
        "convos": result.convos,
        "metadata": result.metadata,
    }
    full_result_filename = output_dir / f"{file_stem}_allresults.json"
    full_result_filename.write_text(json.dumps(full_result_dict, indent=2))
    print(f"All results saved to {full_result_filename}")

    print(f"\nScore (A win rate): {result.score}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
