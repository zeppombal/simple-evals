#!/usr/bin/env python3
"""
Convert MultiChallenge raw.jsonl to HealthBench format.

Usage:
    python convert_to_hb_format.py [--input INPUT] [--output OUTPUT]
"""

import argparse
import json
from pathlib import Path


def convert_entry(entry: dict) -> dict:
    """
    Convert a single MultiChallenge entry to HealthBench format.
    """
    # Use QUESTION_ID directly as prompt_id
    prompt_id = entry["QUESTION_ID"]

    # Map CONVERSATION directly to prompt (already in correct format)
    prompt = entry["CONVERSATION"]

    # Create rubric from TARGET_QUESTION
    # Strip whitespace from criterion text
    criterion = entry["TARGET_QUESTION"].strip()
    rubrics = [{"criterion": criterion, "points": 1, "tags": []}]

    # Convert AXIS to example_tags
    axis = entry["AXIS"].lower()
    example_tags = [f"axis:{axis}"]

    return {
        "canary": "",
        "example_tags": example_tags,
        "ideal_completions_data": None,
        "prompt": prompt,
        "prompt_id": prompt_id,
        "rubrics": rubrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MultiChallenge raw.jsonl to HealthBench format"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=Path(__file__).parent / "raw.jsonl",
        type=Path,
        help="Input raw.jsonl file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=Path(__file__).parent / "converted.jsonl",
        type=Path,
        help="Output converted.jsonl file path",
    )
    args = parser.parse_args()

    # Read input
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {args.input}")

    # Convert entries
    converted = []
    for entry in entries:
        converted.append(convert_entry(entry))

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(converted)} entries to {args.output}")

    # Print summary of axis distribution
    axis_counts = {}
    for entry in entries:
        axis = entry["AXIS"]
        axis_counts[axis] = axis_counts.get(axis, 0) + 1

    print(f"\nAxis distribution:")
    for axis, count in sorted(axis_counts.items(), key=lambda x: -x[1]):
        print(f"  {axis}: {count}")


if __name__ == "__main__":
    main()
