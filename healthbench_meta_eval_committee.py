"""
Committee-based meta-evaluation for HealthBench.

Takes pre-computed meta_evaluation_allresults.json files from multiple evaluators,
forms committee judgments via majority voting, and computes meta-evaluation metrics
in exactly the same way as healthbench_meta_eval.py.

Usage:
    python healthbench_meta_eval_committee.py \
        --input-files file1.json file2.json ... \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import common
from healthbench_meta_eval import compute_metrics_for_rater_by_class
from typess import SingleEvalResult

INPUT_PATH = "hb_data/meta_eval.jsonl"


def parse_criteria_met(response_text: str) -> bool:
    """Extract criteria_met boolean from grader response text.

    Reimplements the logic of healthbench_eval.parse_json_to_dict.
    """
    # Remove markdown-style ```json``` markers if present (same as original)
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
    try:
        parsed = json.loads(json_cleaned)
    except json.JSONDecodeError:
        # Same fallback as original parse_json_to_dict: treat parse failures as False
        return False
    label = parsed.get("criteria_met", False)
    if label is True or label is False:
        return label
    return False


def extract_labels_from_allresults(allresults_path: str) -> list[bool]:
    """Load an allresults.json file and extract per-rubric grader labels."""
    print(f"Loading {allresults_path}...")
    with open(allresults_path, "r") as f:
        data = json.load(f)
    convos = data["convos"]
    labels = []
    for i, convo in enumerate(convos):
        last_msg = convo[-1]["content"]
        try:
            label = parse_criteria_met(last_msg)
        except Exception as e:
            raise ValueError(
                f"Failed to parse label from convo {i} in {allresults_path}: {e}"
            )
        labels.append(label)
    print(f"  Extracted {len(labels)} labels from {allresults_path}")
    return labels


def compute_committee_judgments(
    all_labels: list[list[bool]],
) -> tuple[list[bool], list[float]]:
    """Compute committee labels via majority voting.

    Returns:
        committee_labels: majority vote (strict >0.5; ties resolve to False)
        pct_true_list: fraction of True votes per rubric
    """
    n_rubrics = len(all_labels[0])
    for labels in all_labels:
        assert (
            len(labels) == n_rubrics
        ), f"Label count mismatch: expected {n_rubrics}, got {len(labels)}"

    committee_labels = []
    pct_true_list = []
    for i in range(n_rubrics):
        votes = [labels[i] for labels in all_labels]
        pct_true = sum(votes) / len(votes)
        committee_label = pct_true > 0.5
        committee_labels.append(committee_label)
        pct_true_list.append(pct_true)

    return committee_labels, pct_true_list


def grade_sample(
    grader_label: bool,
    physician_labels: list[bool],
    category: str,
) -> dict:
    """Replicate HealthBenchMetaEval.grade_sample() logic to build per-sample metrics."""
    metrics = {
        "num_physician_labels": len(physician_labels),
        "percent_physician_pos": sum(physician_labels) / len(physician_labels),
    }
    metrics["model_predicted_positive"] = grader_label

    category_metrics = {f"{category}: {k}": v for k, v in metrics.items()}
    metrics = {**metrics, **category_metrics}
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Committee-based meta-evaluation for HealthBench."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="Paths to meta_evaluation_allresults.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output files.",
    )
    args = parser.parse_args()

    # Load ground-truth examples
    examples = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {INPUT_PATH}")

    # Extract per-rubric labels from each input file
    all_labels = []
    for input_file in args.input_files:
        labels = extract_labels_from_allresults(input_file)
        assert len(labels) == len(
            examples
        ), f"Label count {len(labels)} != example count {len(examples)} for {input_file}"
        all_labels.append(labels)
    print(f"Loaded labels from {len(all_labels)} evaluator(s)")

    # Compute committee judgments
    committee_labels, pct_true_list = compute_committee_judgments(all_labels)
    print(
        f"Committee: {sum(committee_labels)}/{len(committee_labels)} positive "
        f"({sum(committee_labels)/len(committee_labels):.4f})"
    )

    # Save per-rubric voting data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    votes_path = output_dir / "committee_votes.jsonl"
    with open(votes_path, "w") as f:
        for i in range(len(committee_labels)):
            record = {
                "idx": i,
                "committee_label": committee_labels[i],
                "pct_true": pct_true_list[i],
                "n_votes": len(all_labels),
                "votes": [labels[i] for labels in all_labels],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved voting data to {votes_path}")

    # Build SingleEvalResult list (replicating the original __call__ logic)
    results: list[SingleEvalResult] = []
    for idx, (example, committee_label) in enumerate(zip(examples, committee_labels)):
        metrics = grade_sample(
            grader_label=committee_label,
            physician_labels=example["binary_labels"],
            category=example["category"],
        )
        score = metrics["model_predicted_positive"]
        results.append(
            SingleEvalResult(html=None, score=score, convo=None, metrics=metrics)
        )

    # Model pairwise agreement metrics (same as original)
    model_agreement_metrics = compute_metrics_for_rater_by_class(
        self_pred_list=committee_labels,
        other_preds_list=[x["binary_labels"] for x in examples],
        cluster_list=[x["category"] for x in examples],
        model_or_physician="model",
    )

    # Physician agreement metrics (same loop as original)
    physician_rating_lists = defaultdict(lambda: ([], [], []))
    for example in examples:
        for i in range(len(example["binary_labels"])):
            physician_id = example["anonymized_physician_ids"][i]
            self_pred = example["binary_labels"][i]
            other_preds = (
                example["binary_labels"][:i] + example["binary_labels"][i + 1 :]
            )
            cluster = example["category"]
            physician_rating_lists[physician_id][0].append(self_pred)
            physician_rating_lists[physician_id][1].append(other_preds)
            physician_rating_lists[physician_id][2].append(cluster)

    physician_agreement_metric_lists = defaultdict(dict)
    for physician_id, (
        physician_rating_list,
        other_preds_list,
        cluster_list,
    ) in physician_rating_lists.items():
        physician_agreement_metrics = compute_metrics_for_rater_by_class(
            self_pred_list=physician_rating_list,
            other_preds_list=other_preds_list,
            cluster_list=cluster_list,
            model_or_physician="physician",
        )
        for k, v in physician_agreement_metrics.items():
            physician_agreement_metric_lists[k][physician_id] = v

    # Consolidate final metrics (same as original)
    final_metrics = common.aggregate_results(
        results, default_stats=("mean", "n_samples", "bootstrap_std")
    )
    model_agreement_metrics_condensed: dict[str, float] = {
        k: v["value"]
        for k, v in model_agreement_metrics.items()
        if v["value"] is not None
    }
    assert final_metrics.metrics is not None
    final_metrics.metrics.update(model_agreement_metrics_condensed)
    final_metrics.score = final_metrics.metrics["pairwise_model_f1_balanced"]

    final_metrics.metadata = {
        "model_agreement_metrics": model_agreement_metrics,
        "physician_agreement_metric_lists": physician_agreement_metric_lists,
    }

    # Save meta_evaluation.json (sorted metrics, same format as original)
    metrics_out = final_metrics.metrics | {"score": final_metrics.score}
    metrics_out = dict(sorted(metrics_out.items()))
    metrics_path = output_dir / "meta_evaluation.json"
    with open(metrics_path, "w") as f:
        f.write(json.dumps(metrics_out, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Score (pairwise_model_f1_balanced): {final_metrics.score}")

    # Save meta_evaluation_allresults.json (full result dict)
    allresults_path = output_dir / "meta_evaluation_allresults.json"
    result_dict = {
        "score": final_metrics.score,
        "metrics": final_metrics.metrics,
        "htmls": final_metrics.htmls,
        "convos": final_metrics.convos,
        "metadata": final_metrics.metadata,
    }
    with open(allresults_path, "w") as f:
        f.write(json.dumps(result_dict, indent=2))
    print(f"Saved full results to {allresults_path}")


if __name__ == "__main__":
    main()
