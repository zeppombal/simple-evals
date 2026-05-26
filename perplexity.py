"""Perplexity analysis for the self-preference-bias rebuttal.

Given a generator's outputs (a generation JSONL) and a vLLM-served model M, compute M's
perplexity over each completion, scored *as if M had produced it* — the log-probs M
assigns to the completion tokens conditioned on the generator's context.

vLLM only: API models (Claude/GPT) don't expose prompt-token logprobs.

CORRECTNESS — why we frame the completion as a NON-FINAL assistant turn.
Chat templates scaffold the *final* assistant turn in model-specific, contaminating ways:
  - Qwen3 *Instruct* injects an empty `<think>\\n\\n</think>\\n\\n` block after the
    assistant header (verified: independent of `enable_thinking`);
  - reasoning models' generation prompts can force an opening `<think>` and expect a
    thinking trace before the answer — but our generation records only keep the final
    answer (the trace is stripped), so scoring the answer right after a forced `<think>`
    gives unrealistically high perplexity.
Templates render a *non-final* assistant turn (one followed by another turn) cleanly:
the thinking scaffolding is dropped and only the content remains. So we put the response
in an assistant message, append a throwaway user message as a boundary, and score only
the response's tokens — conditioned on `...<|im_start|>assistant\\n` with no trace, which
is the honest, uniform framing across instruct and reasoning models alike.

We isolate the response token span template-agnostically: render the same conversation
with empty assistant content and with the real content; the response tokens are exactly
the tokens by which the two renders differ (everything before is identical context +
assistant header; everything after is identical `<|im_end|>` + the boundary turn).

Per completion we report:
  - content:      the response tokens only (the faithful perplexity);
  - content_eot:  response tokens + the turn-ending token (P that the turn ends there).

Usage:
  python perplexity.py --generation-input gen.jsonl --model hosted_vllm/Qwen3-30B-A3B-Instruct-2507 \
      --model-completion-args "{'api_base':'http://worker-31:8122/v1'}" \
      --n-threads 64 --output perplexity.json
"""

import argparse
import json
import math
import statistics
import time

import requests
from openai import OpenAI

import common
from analysis_common import load_generation_records, parse_completion_args, reconstruct

# Throwaway user turn appended after the assistant turn so the response is rendered as a
# settled (non-final) turn. Its content is irrelevant: it lands entirely after the
# response, so it never affects the response tokens' log-probs.
BOUNDARY_USER = {"role": "user", "content": "Thank you."}


def _server_root(api_base: str) -> str:
    """vLLM mounts /tokenize at the server root, not under /v1."""
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root.rstrip("/")


def _tokenize(root, served, messages, n_retries=4) -> list[int]:
    payload = {"model": served, "messages": messages, "add_generation_prompt": False}
    last = None
    for attempt in range(n_retries):
        try:
            r = requests.post(f"{root}/tokenize", json=payload, timeout=120)
            r.raise_for_status()
            d = r.json()
            return d.get("tokens", d.get("token_ids", []))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2**attempt)
    raise RuntimeError(f"/tokenize failed after {n_retries} tries: {last}")


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def response_span(empty_ids: list[int], full_ids: list[int]) -> tuple[int, int]:
    """Token span [start, end) of the response within `full_ids`.

    `empty_ids` renders the same conversation with empty assistant content, so it equals
    `full_ids` minus the response tokens. start = common prefix; the trailing scaffold
    (everything after the assistant content: <|im_end|> + boundary turn) is identical in
    both, with length `len(empty) - start`, so end = len(full) - that.
    """
    start = _common_prefix_len(empty_ids, full_ids)
    trailing = len(empty_ids) - start
    end = len(full_ids) - trailing
    return start, end


def _span_stats(token_ids: list[int], prompt_logprobs: list, start: int, end: int) -> dict:
    logps: list[float] = []
    for i in range(start, min(end, len(token_ids))):
        entry = prompt_logprobs[i] if i < len(prompt_logprobs) else None
        if not entry:
            continue
        tid = token_ids[i]
        d = entry.get(str(tid)) or entry.get(tid)
        if d is None and len(entry) == 1:
            d = next(iter(entry.values()))
        if d is None:
            continue
        logps.append(d["logprob"] if isinstance(d, dict) else float(d))
    n = len(logps)
    s = sum(logps)
    avg = s / n if n else None
    return {
        "n_tokens": n,
        "avg_logprob": avg,
        "perplexity": math.exp(-avg) if n else None,
    }


def main():
    p = argparse.ArgumentParser(description="Perplexity of a vLLM model on a generator's outputs.")
    p.add_argument("--generation-input", required=True)
    p.add_argument("--model", required=True, help="litellm-style name, e.g. hosted_vllm/Qwen3-...")
    p.add_argument("--model-completion-args", default="{}", help="JSON; must contain api_base")
    p.add_argument("--n-threads", type=int, default=64)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cargs = parse_completion_args(args.model_completion_args)
    api_base = cargs.get("api_base")
    if not api_base:
        raise SystemExit("perplexity.py requires 'api_base' in --model-completion-args (vLLM only).")
    served = args.model.split("/")[-1]
    root = _server_root(api_base)
    client = OpenAI(base_url=api_base.rstrip("/"), api_key=cargs.get("api_key", "EMPTY"))

    records = load_generation_records(args.generation_input)
    print(f"Loaded {len(records)} records | model={served} | api_base={api_base}")

    def score(record: dict) -> dict | None:
        context, completion = reconstruct(record)
        if not completion.strip():
            return None
        msgs_empty = context + [{"role": "assistant", "content": ""}, BOUNDARY_USER]
        msgs_full = context + [{"role": "assistant", "content": completion}, BOUNDARY_USER]
        for attempt in range(4):
            try:
                empty_ids = _tokenize(root, served, msgs_empty)
                full_ids = _tokenize(root, served, msgs_full)
                start, end = response_span(empty_ids, full_ids)
                if end <= start:
                    return None
                resp = client.completions.create(
                    model=served, prompt=full_ids, max_tokens=1, temperature=0.0,
                    extra_body={"prompt_logprobs": 0},
                )
                plp = resp.choices[0].model_dump().get("prompt_logprobs")
                if plp is None:
                    raise RuntimeError("server returned no prompt_logprobs")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"perplexity failed for {record.get('prompt_id')}: {e}")
                    return None
                time.sleep(2**attempt)

        content = _span_stats(full_ids, plp, start, end)
        content_eot = _span_stats(full_ids, plp, start, end + 1)  # + turn-ending token
        return {
            "prompt_id": record.get("prompt_id"),
            "completion_id": record.get("completion_id"),
            "n_response_tokens": end - start,
            "content": content,
            "content_eot": content_eot,
        }

    results = common.map_with_progress(score, records, num_threads=args.n_threads, pbar=True)
    scored = [r for r in results if r is not None]

    def agg(kind: str) -> dict:
        ppls = [r[kind]["perplexity"] for r in scored if r[kind]["perplexity"] is not None]
        avglp = [r[kind]["avg_logprob"] for r in scored if r[kind]["avg_logprob"] is not None]
        return {
            "mean_perplexity": statistics.fmean(ppls) if ppls else None,
            "median_perplexity": statistics.median(ppls) if ppls else None,
            "mean_avg_logprob": statistics.fmean(avglp) if avglp else None,
            "n": len(ppls),
        }

    out = {
        "model": args.model,
        "generation_input": args.generation_input,
        "aggregate": {
            "n_records": len(records),
            "n_scored": len(scored),
            "n_skipped": len(results) - len(scored),
            "content": agg("content"),
            "content_eot": agg("content_eot"),
        },
        "results": scored,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")
    print(f"  content     mean PPL: {out['aggregate']['content']['mean_perplexity']}")
    print(f"  content_eot mean PPL: {out['aggregate']['content_eot']['mean_perplexity']}")
    print(f"  scored {len(scored)}/{len(records)}")


if __name__ == "__main__":
    main()
