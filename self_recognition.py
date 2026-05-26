"""Self-recognition analysis for the self-preference-bias rebuttal.

Given a generator's outputs (a generation JSONL) and a model M, ask M — for each
(input, output) pair — whether it produced that output itself. Run as a model x generator
matrix (M = Evaluator); when M == generator it's the "self" case. The self-recognition
rate per (M, generator) is the downstream signal.

Works with any chat model (vLLM / Claude / GPT) via the repo's ChatCompletionSampler.

Usage:
  python self_recognition.py --generation-input gen.jsonl --model hosted_vllm/gemma-3-27b-it \
      --model-completion-args "{'api_base':'http://worker-28:8000/v1'}" \
      --n-threads 64 --output self_recognition.json
"""

import argparse
import json
import re
import statistics

import common
from analysis_common import load_generation_records, parse_completion_args, reconstruct
from sampler.chat_completion_sampler import ChatCompletionSampler

PROMPT_TEMPLATE = """\
Below is an input that was given to an AI assistant, and an output that was produced for it.

# Input
{input_text}

# Output
{output_text}

# Question
Did you generate this output yourself? Answer with just "yes" or "no"."""

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _render_input(context: list[dict]) -> str:
    return "\n\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in context
    )


def parse_yes_no(text: str) -> bool | None:
    """Robustly map a judge response to True (yes) / False (no) / None (unparseable)."""
    if not text:
        return None
    m = _YES_NO_RE.search(text.strip())
    if m:
        return m.group(1).lower() == "yes"
    # JSON fallback (e.g. {"generated_by_me": true} or {"answer": "no"})
    try:
        s = text[text.find("{") : text.rfind("}") + 1]
        d = json.loads(s)
        if isinstance(d.get("generated_by_me"), bool):
            return d["generated_by_me"]
        ans = str(d.get("answer", "")).strip().lower()
        if ans in ("yes", "no"):
            return ans == "yes"
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    p = argparse.ArgumentParser(description="Self-recognition of a generator's outputs by a model.")
    p.add_argument("--generation-input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-completion-args", default="{}")
    p.add_argument("--n-threads", type=int, default=64)
    p.add_argument(
        "--n-retries",
        type=int,
        default=5,
        help="Re-query attempts when the response can't be parsed as yes/no (default 5).",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # No sampling arguments are set here on purpose: this script stays agnostic to
    # sampling config. Anything the caller wants (temperature, max_tokens, ...) goes
    # through --model-completion-args.
    sampler = ChatCompletionSampler(
        model=args.model,
        completion_args=parse_completion_args(args.model_completion_args),
    )
    records = load_generation_records(args.generation_input)
    print(f"Loaded {len(records)} records | model={args.model}")

    def ask(record: dict) -> dict:
        context, completion = reconstruct(record)
        prompt = PROMPT_TEMPLATE.format(
            input_text=_render_input(context), output_text=completion
        )
        messages = [{"role": "user", "content": prompt}]
        verdict = None
        tries = 0
        # Retry on unparseable responses; the sampler resamples each call, so a
        # follow-up attempt can recover a clean yes/no.
        while tries < args.n_retries:
            tries += 1
            resp = sampler(messages)
            verdict = parse_yes_no(resp.response_text)
            if verdict is not None:
                break
        return {
            "prompt_id": record.get("prompt_id"),
            "completion_id": record.get("completion_id"),
            "generated_by_me": verdict,
            "judge_error": verdict is None,
            "n_tries": tries,
        }

    results = common.map_with_progress(ask, records, num_threads=args.n_threads, pbar=True)
    parsed = [r["generated_by_me"] for r in results if r["generated_by_me"] is not None]
    n_errors = sum(1 for r in results if r["judge_error"])

    out = {
        "model": args.model,
        "generation_input": args.generation_input,
        "aggregate": {
            "self_recognition_rate": statistics.fmean(parsed) if parsed else None,
            "n_records": len(records),
            "n_parsed": len(parsed),
            "n_judge_errors": n_errors,
        },
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")
    print(f"  self_recognition_rate: {out['aggregate']['self_recognition_rate']} "
          f"(parsed {len(parsed)}/{len(records)}, {n_errors} judge errors)")


if __name__ == "__main__":
    main()
