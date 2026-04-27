"""Per-scenario inference benchmark for the vLLM VLM endpoint.

Captures TTFT, end-to-end latency, output tokens/sec, and a snapshot of
vLLM's Prometheus metrics (prefix cache hit rate, KV cache utilization,
GPU cache usage) for every scenario in a scenarios JSON file.

Run at each optimization layer with a different --label, then diff the CSVs:

    # L0 baseline
    python vlm_serving/benchmark.py \\
        --scenarios scenarios/vlm_impeller_scenarios.json \\
        --label L0_fp16 \\
        --iters 3 --warmup 1 \\
        --out benchmarks/L0_fp16.csv

    # L1 AWQ (after relaunching the server against Qwen2.5-VL-7B-Instruct-AWQ)
    python vlm_serving/benchmark.py \\
        --scenarios scenarios/vlm_impeller_scenarios.json \\
        --label L1_awq_int4 \\
        --iters 3 --warmup 1 \\
        --out benchmarks/L1_awq_int4.csv

    # L2 serving-tuned (after relaunching with --enable-prefix-caching etc.)
    python vlm_serving/benchmark.py ... --label L2_tuned --out benchmarks/L2_tuned.csv

The output CSV has one row per (scenario_id, iteration). Aggregate it with
summarize_runs.py.

Env:
    VLLM_BASE_URL (default http://localhost:8000/v1)
    VLLM_MODEL    (default Qwen/Qwen2.5-VL-7B-Instruct)

Metrics captured per iteration:
    ttft_s              time from request-start to first non-empty content chunk
    e2e_s               total request time
    decode_s            e2e - ttft (approx LLM decode time; includes server scheduling jitter)
    prompt_tokens       from usage (requires stream_options.include_usage)
    completion_tokens   from usage
    decode_tps          completion_tokens / decode_s (output throughput)

vLLM /metrics scraped once per layer (i.e. one row in metrics_snapshot.json):
    gpu_cache_usage_perc, prefix_cache_hit_rate, num_requests_running, etc.
    (Raw Prometheus text is saved alongside so nothing is lost.)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from openai import OpenAI


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def one_streaming_call(
    client: OpenAI, model: str, image_path: Path, question: str, max_tokens: int
) -> dict:
    """Single streamed chat completion. Returns timing + token counts."""
    t_start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )

    ttft: Optional[float] = None
    text_chunks: list[str] = []
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    for chunk in stream:
        # Usage chunk arrives at the end when include_usage is set.
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            if ttft is None:
                ttft = time.perf_counter() - t_start
            text_chunks.append(content)

    e2e = time.perf_counter() - t_start
    text = "".join(text_chunks)
    decode_s = (e2e - ttft) if ttft is not None else None
    decode_tps = (
        completion_tokens / decode_s
        if (decode_s and decode_s > 0 and completion_tokens)
        else None
    )

    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "decode_s": decode_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tps": decode_tps,
        "response": text,
    }


def fetch_vllm_metrics(base_url: str) -> Optional[str]:
    """Scrape raw Prometheus text from vLLM's /metrics endpoint.

    base_url is .../v1; the metrics endpoint is at root, so we strip /v1.
    Returns None if unreachable.
    """
    root = re.sub(r"/v1/?$", "", base_url)
    url = f"{root}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"  (failed to fetch {url}: {e})", file=sys.stderr)
        return None


def parse_prometheus(text: str) -> dict:
    """Very small Prometheus text parser. Keeps the latest scalar value for
    each metric name; labels are folded into a '{..}' suffix so duplicates
    stay distinct."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # metric_name{labels} value  OR  metric_name value
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+\-.]+|NaN|Inf|-Inf)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            out[f"{name}{labels}"] = float(val)
        except ValueError:
            continue
    return out


def summarize_metric(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "n": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    p50 = statistics.median(vals_sorted)
    p95 = vals_sorted[min(n - 1, max(0, int(round(0.95 * (n - 1)))))]
    return {"mean": statistics.fmean(vals_sorted), "p50": p50, "p95": p95, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--label", required=True, help="Layer label, e.g. L0_fp16 / L1_awq / L2_tuned")
    ap.add_argument("--iters", type=int, default=3, help="Measured iterations per scenario")
    ap.add_argument("--warmup", type=int, default=1, help="Un-measured warmup iterations")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))["scenarios"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    print(f"Label:     {args.label}")
    print(f"Endpoint:  {base_url}")
    print(f"Model:     {model}")
    print(f"Scenarios: {len(scenarios)}   warmup={args.warmup}  iters={args.iters}")
    print()

    all_rows: list[dict] = []
    for s in scenarios:
        sid = s["id"]
        img = Path(s["image_path"])
        q = s["text"]
        if not img.exists():
            print(f"[{sid}] MISSING IMAGE {img}, skipping")
            continue

        print(f"[{sid}] {img.name}")
        # Warmup
        for _ in range(args.warmup):
            try:
                one_streaming_call(client, model, img, q, args.max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  warmup error: {e}", file=sys.stderr)

        # Measured iterations
        for it in range(args.iters):
            try:
                r = one_streaming_call(client, model, img, q, args.max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  iter {it} error: {e}", file=sys.stderr)
                continue
            row = {
                "label": args.label,
                "scenario_id": sid,
                "image": img.name,
                "iter": it,
                "ttft_s": r["ttft_s"],
                "e2e_s": r["e2e_s"],
                "decode_s": r["decode_s"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "decode_tps": r["decode_tps"],
            }
            all_rows.append(row)
            print(
                f"  iter {it}: ttft={_fmt(row['ttft_s'])}s  "
                f"e2e={_fmt(row['e2e_s'])}s  "
                f"tps={_fmt(row['decode_tps'])}  "
                f"tok(p/c)={row['prompt_tokens']}/{row['completion_tokens']}"
            )

    # CSV
    if all_rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote per-iteration CSV -> {out_path}")
    else:
        print("\nNo rows collected; is the server up?")
        return 2

    # Summary table
    print("\n=== Summary (across all iterations) ===")
    for key in ("ttft_s", "e2e_s", "decode_s", "decode_tps"):
        s = summarize_metric(all_rows, key)
        if s["n"]:
            print(f"  {key:10s}  mean={s['mean']:.3f}  p50={s['p50']:.3f}  p95={s['p95']:.3f}  n={s['n']}")
        else:
            print(f"  {key:10s}  (no data)")

    # vLLM /metrics snapshot
    metrics_text = fetch_vllm_metrics(base_url)
    if metrics_text:
        metrics_raw_path = out_path.with_suffix(".prom.txt")
        metrics_raw_path.write_text(metrics_text, encoding="utf-8")
        parsed = parse_prometheus(metrics_text)
        metrics_json_path = out_path.with_suffix(".metrics.json")
        metrics_json_path.write_text(
            json.dumps({"label": args.label, "metrics": parsed}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote vLLM metrics snapshot -> {metrics_raw_path} (+ {metrics_json_path})")
        for k in (
            "vllm:gpu_cache_usage_perc",
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_hits",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
        ):
            hits = [(name, v) for name, v in parsed.items() if name.startswith(k)]
            for name, v in hits:
                print(f"    {name} = {v}")

    return 0


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


if __name__ == "__main__":
    sys.exit(main())
