# NFR metrics: scrape vLLM /metrics, wrap agent runs, export CSV-friendly dict.

import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class NFRSnapshot:
    timestamp_s: float
    raw_text: str
    parsed: dict[str, float] = field(default_factory=dict)


# Wanted Prometheus metric prefixes. We grab the latest scalar per metric
# (vLLM exposes counters/gauges. for counters we diff start vs end).
_WANTED_PREFIXES = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
)

_PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+\-.]+|NaN|Inf|-Inf)$"
)


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text. Latest value per (name, labels) tuple."""
    out: dict[str, float] = {}
    # each pass handles the next item in the sequence
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if not any(name.startswith(p) for p in _WANTED_PREFIXES):
            continue
        try:
            out[f"{name}{labels}"] = float(val)
        except ValueError:
            continue
    return out


def _scrape(base_url: str, timeout_s: float = 3.0) -> Optional[NFRSnapshot]:
    """Best-effort GET on ``<base>/metrics`` (strips trailing /v1)."""
    root = re.sub(r"/v1/?$", "", base_url)
    url = f"{root}/metrics"
    # isolate errors so the rest of the call can bail cleanly
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    return NFRSnapshot(
        timestamp_s=time.perf_counter(), raw_text=text, parsed=_parse_prometheus(text)
    )


def _delta(a: dict[str, float], b: dict[str, float], key_prefix: str) -> float:
    """Sum of (b[k] - a[k]) for all keys whose name starts with key_prefix."""
    total = 0.0
    # each pass handles the next item in the sequence
    for k in b:
        if not k.startswith(key_prefix):
            continue
        total += b[k] - a.get(k, 0.0)
    return total


def _max(snapshot: Optional[NFRSnapshot], key_prefix: str) -> Optional[float]:
    """Max over all matching keys in one snapshot."""
    if snapshot is None:
        return None
    matches = [v for k, v in snapshot.parsed.items() if k.startswith(key_prefix)]
    return max(matches) if matches else None


# Collector
class NFRCollector:
    """Wrap an agent execution; emit a flat metrics dict at the end.

    The collector captures wall-clock around the agent run plus a vLLM
    Prometheus snapshot before and after so it can compute deltas
    (tokens emitted, prefix cache hits) attributable to this run.

    Note: deltas are approximate when other workloads share the vLLM
    instance. For a clean signal, run benchmarks one at a time.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        scrape_metrics: bool = True,
    ) -> None:
        self._base_url = base_url
        self._scrape = scrape_metrics
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._snap_before: Optional[NFRSnapshot] = None
        self._snap_after: Optional[NFRSnapshot] = None
        self._agent_metrics: dict[str, Any] = {}

    async def __aenter__(self) -> "NFRCollector":
        # Async work for `aenter`.
        self._t_start = time.perf_counter()
        if self._scrape:
            self._snap_before = _scrape(self._base_url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Async `aexit` isolated for readability.
        self._t_end = time.perf_counter()
        if self._scrape:
            self._snap_after = _scrape(self._base_url)

    # Attach per-agent metric blobs
    def attach_react_result(self, result) -> None:
        """Pull the per-step metrics out of a ReActResult and stash them."""
        self._agent_metrics = {
            "agent": "react",
            "agent_e2e_s": result.total_e2e_s,
            "num_iterations": result.num_iterations,
            "num_tool_calls": result.num_tool_calls,
            "total_llm_calls": result.total_llm_calls,
            "total_prompt_tokens": result.total_prompt_tokens,
            "total_completion_tokens": result.total_completion_tokens,
            "avg_ttft_s": result.avg_ttft_s,
            "total_tool_latency_s": result.total_tool_latency_s,
            "success": int(result.success),
            "error": result.error or "",
            "final_answer": (result.final_answer or "")[:500],
        }

    def attach_plan_execute_result(self, result, llm_metrics_list=None) -> None:
        """Pull metrics from a PlanExecuteRunner OrchestratorResult."""
        history = result.history or []
        n_steps = len(history)
        n_success = sum(1 for r in history if r.success)
        # Plan-Execute makes ~ (1 plan + N step args + 1 summary) LLM calls.
        # That's hard to attribute precisely without instrumenting the planner
        # - the deltas from Prometheus give us the aggregate.
        ttfts: list[float] = []
        prompt_tok = 0
        completion_tok = 0

        # runs when llm_metrics_list
        if llm_metrics_list:
            for m in llm_metrics_list:
                if m.ttft_s is not None:
                    ttfts.append(m.ttft_s)
                if m.prompt_tokens:
                    prompt_tok += m.prompt_tokens
                if m.completion_tokens:
                    completion_tok += m.completion_tokens
        self._agent_metrics = {
            "agent": "plan_execute",
            "num_iterations": n_steps,
            "num_tool_calls": n_steps,
            "total_llm_calls": (n_steps * 2) + 2,  # plan + per-step args + summary, approx
            "total_prompt_tokens": prompt_tok,
            "total_completion_tokens": completion_tok,
            "avg_ttft_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "total_tool_latency_s": None,
            "success": int(n_success == n_steps and n_steps > 0),
            "error": "" if n_steps > 0 else "empty_plan",
            "final_answer": (result.answer or "")[:500],
        }

    # Export
    def export_dict(self) -> dict[str, Any]:
        """Return one flat dict ready to be a CSV row."""

        # runs when self._t_start is None or self._t_end is None
        if self._t_start is None or self._t_end is None:
            raise RuntimeError("NFRCollector was not used as a context manager")

        out: dict[str, Any] = dict(self._agent_metrics)
        out["wall_e2e_s"] = round(self._t_end - self._t_start, 3)

        # Token-derived
        if self._agent_metrics:
            comp = self._agent_metrics.get("total_completion_tokens") or 0
            prom = self._agent_metrics.get("total_prompt_tokens") or 0
            out["prompt_completion_ratio"] = (
                round(prom / comp, 3) if comp else None
            )

        # vLLM Prometheus deltas
        before = self._snap_before
        after = self._snap_after

        # runs when before is None or after is None
        if before is None or after is None:
            out["vllm_metrics_available"] = 0
            return out

        out["vllm_metrics_available"] = 1
        out["vllm_prompt_tokens_delta"] = round(
            _delta(before.parsed, after.parsed, "vllm:prompt_tokens_total"), 1
        )
        out["vllm_generation_tokens_delta"] = round(
            _delta(before.parsed, after.parsed, "vllm:generation_tokens_total"), 1
        )
        out["vllm_prefix_cache_queries_delta"] = round(
            _delta(before.parsed, after.parsed, "vllm:prefix_cache_queries"), 1
        )
        out["vllm_prefix_cache_hits_delta"] = round(
            _delta(before.parsed, after.parsed, "vllm:prefix_cache_hits"), 1
        )

        # Rate computations
        q = out["vllm_prefix_cache_queries_delta"]
        h = out["vllm_prefix_cache_hits_delta"]
        out["vllm_prefix_cache_hit_rate"] = round(h / q, 4) if q > 0 else None

        # Gauges (max over the run window - read at end-of-run)
        out["vllm_gpu_cache_usage_max"] = _max(after, "vllm:gpu_cache_usage_perc")
        out["vllm_num_requests_running_max"] = _max(after, "vllm:num_requests_running")
        out["vllm_num_requests_waiting_max"] = _max(after, "vllm:num_requests_waiting")

        return out

