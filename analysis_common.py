"""Shared helpers for the rebuttal analysis scripts (perplexity, self_recognition).

Both consume the generation JSONL produced by `simple_evals.py --mode=generate`. Every
benchmark (IFEval, HealthBench, LiveCodeBench) stores the exact context actually sent to
the generator (`actual_queried_message_list`) and the generator's completion
(`response_text`), so these analyses are benchmark-agnostic: we just reconstruct
`context_messages + [{"role": "assistant", "content": completion}]`.
"""

import json
from typing import Any


def load_generation_records(path: str) -> list[dict]:
    """Load a generation JSONL (one record per line)."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def reconstruct(record: dict) -> tuple[list[dict[str, Any]], str]:
    """Return (context_messages, completion_text) for a generation record.

    `actual_queried_message_list` is the exact list of {role, content} messages sent to
    the generator (including any system message); `response_text` is its completion.
    """
    context = record.get("actual_queried_message_list") or record.get("prompt") or []
    completion = record.get("response_text", "") or ""
    return context, completion


def parse_completion_args(raw: str) -> dict:
    """Parse the --model-completion-args JSON (tolerates single quotes, as ducttape uses)."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    return json.loads(raw.replace("'", '"'))
