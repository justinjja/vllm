# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure streaming generation and prefill with a user-supplied text corpus."""

import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL = "local-inference-lab/GLM-5.3-NVFP4"


def request(base, prompt, output_tokens, model, clock_origin=None):
    started = time.perf_counter()
    clock_origin = started if clock_origin is None else clock_origin
    first = last = None
    text = ""
    usage = None
    finish = None
    token_events = []
    with requests.post(
        base + "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": output_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        },
        stream=True,
        timeout=(10, 3600),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                token_ids = choice.get("token_ids")
                if token_ids:
                    token_events.append(
                        (time.perf_counter() - clock_origin, len(token_ids))
                    )
                piece = choice.get("text", "")
                if piece:
                    last = time.perf_counter()
                    first = first or last
                    text += piece
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - started
    assert usage is not None and first is not None, (usage, text)
    assert usage["completion_tokens"] == output_tokens, usage
    assert usage["prompt_tokens"] == len(prompt), usage
    assert finish == "length", finish
    tokens = usage["completion_tokens"]
    assert sum(count for _, count in token_events) == tokens
    return {
        "input_tokens": len(prompt),
        "output_tokens": tokens,
        "ttft_s": first - started,
        "elapsed_s": elapsed,
        "request_tokens_s": tokens / elapsed,
        "decode_tokens_s": (tokens - 1) / (last - first) if last > first else None,
        "prefill_tokens_s": len(prompt) / (first - started),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "output_text": text,
        "usage": usage,
        "finish_reason": finish,
        "token_events": token_events,
    }


def steady_decode(rows, trim_tokens):
    """Count accepted tokens during an interval where every request is decoding."""
    starts, ends = [], []
    for row in rows:
        count = 0
        start = end = None
        for timestamp, delta in row["token_events"]:
            count += delta
            if start is None and count >= trim_tokens:
                start = timestamp
            if end is None and count >= row["output_tokens"] - trim_tokens:
                end = timestamp
        if start is None or end is None:
            return None
        starts.append(start)
        ends.append(end)
    start, end = max(starts), min(ends)
    if end <= start:
        return None
    tokens = sum(
        delta
        for row in rows
        for timestamp, delta in row["token_events"]
        if start < timestamp <= end
    )
    return {
        "start_s": start,
        "end_s": end,
        "accepted_tokens": tokens,
        "aggregate_tokens_s": tokens / (end - start),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="glm-5.3-nvfp4")
    parser.add_argument("--tokenizer", default=MODEL)
    parser.add_argument(
        "--tokenizer-revision", default="b472e4ee53f6a9862da5486c56c6ca21be3dab70"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 30000])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--steady-trim-tokens", type=int, default=32)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    raw = args.corpus.read_text()
    ids = tokenizer.encode(raw, add_special_tokens=False, verbose=False)
    if len(ids) <= max(512, *args.contexts):
        parser.error("The corpus must contain more tokens than the longest prompt")
    if min(args.repeats, args.output_tokens, *args.concurrency, *args.contexts) < 1:
        parser.error("Repeat, token and concurrency counts must be positive")
    if args.steady_trim_tokens < 1:
        parser.error("Steady-state trimming must be positive")
    report = {
        "label": args.label,
        "started_at": time.time(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "corpus_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "method": "Unique-prefix corpus; fixed output length with ignore_eos; "
        "streaming first/last text timestamps; includes reasoning tokens. "
        "Steady decode counts returned accepted token IDs in the shared "
        "decoding interval after trimming each request's beginning and end. "
        "Performance measurement only; output quality is checked separately.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }

    def prompt(length, sequence):
        prefix = tokenizer.encode(
            f"Benchmark {args.label} request {sequence}.\n", add_special_tokens=False
        )
        start = (sequence * 7919) % (len(ids) - length)
        return (prefix + ids[start : start + length])[:length]

    request(args.base, prompt(128, 0), 8, args.model)
    sequence = 1
    specs = [("single", 512, 1, args.output_tokens)]
    specs += [("concurrent", 512, n, args.output_tokens) for n in args.concurrency]
    specs += [("prefill", n, 1, 1) for n in args.contexts]
    specs += [("context_decode", n, 1, 64) for n in args.contexts]
    for mode, length, concurrency, tokens in specs:
        for repeat in range(args.repeats):
            prompts = [prompt(length, sequence + i) for i in range(concurrency)]
            sequence += concurrency
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(
                    pool.map(
                        lambda p, n=tokens, origin=started: request(
                            args.base, p, n, args.model, origin
                        ),
                        prompts,
                    )
                )
            elapsed = time.perf_counter() - started
            row = {
                "mode": mode,
                "input_tokens": length,
                "concurrency": concurrency,
                "repeat": repeat,
                "elapsed_s": elapsed,
                "aggregate_tokens_s": sum(r["output_tokens"] for r in rows) / elapsed,
                "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
                "requests": rows,
                "steady_decode": steady_decode(rows, args.steady_trim_tokens),
            }
            report["cases"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print({k: v for k, v in row.items() if k != "requests"}, flush=True)
    report["finished_at"] = time.time()
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
