# ReAct agent on vision MCP tools. in-process tool calls for benchmark timing.

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# so `python -m agent.react...` finds src/
_REPO = Path(__file__).resolve().parent.parent.parent.parent

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from llm.base import LLMBackend  # noqa: E402
from llm.vllm import GenerationMetrics, VLLMBackend  # noqa: E402
from servers.vision import main as vision_main  # noqa: E402

_log = logging.getLogger(__name__)


# Tool registry
# Vision tools the agent can call. Names match the MCP server's @mcp.tool() decorations.
VISION_TOOLS: dict[str, dict[str, Any]] = {
    "vision.analyze_image": {
        "description": (
            "Free-form analysis: ask any question about the image. "
            "Use this for inspection / accept-reject / describe-defects queries."
        ),
        "args": {"image_ref": "string", "question": "string"},
        "args_required": ["image_ref", "question"],
    },
    "vision.classify_equipment": {
        "description": "Identify the primary piece of equipment shown in the image.",
        "args": {"image_ref": "string"},
        "args_required": ["image_ref"],
    },
    "vision.detect_visual_defects": {
        "description": "List any visible defects in the image (returns a list).",
        "args": {"image_ref": "string"},
        "args_required": ["image_ref"],
    },
    "vision.assess_condition": {
        "description": "Assess overall condition: good / degraded / critical.",
        "args": {"image_ref": "string"},
        "args_required": ["image_ref"],
    },
    "vision.read_gauge": {
        "description": "Read the numeric value off an analog gauge in the image.",
        "args": {"image_ref": "string"},
        "args_required": ["image_ref"],
    },
}


def _format_tools_for_prompt() -> str:
    # `format_tools_for_prompt` lives here. was getting too cramped inline.
    lines = []
    # each pass handles the next item in the sequence
    for name, spec in VISION_TOOLS.items():
        args_str = ", ".join(f"{k}: {v}" for k, v in spec["args"].items())
        lines.append(f"  - {name}({args_str}): {spec['description']}")
    return "\n".join(lines)


# Prompts
REACT_SYSTEM_PROMPT = """\
You are a visual inspection agent for industrial equipment.

CRITICAL: You CANNOT see images directly. You can only read what tools tell
you about an image. To answer any question about an image, you MUST first
call a vision tool, THEN reason over the tool's Observation.

Available tools:
{tools}

You respond in turns. Each turn MUST be exactly one of these two response shapes.

SHAPE A - call a tool (use this on the first turn, BEFORE any Observation):
Thought: <one sentence on what to do>
Action: <exactly one of the tool names above>
Action Input: <a single-line JSON object with the tool's arguments>

SHAPE B - final answer (use this on the second turn, AFTER an Observation has been returned to you):
Thought: I now have the inspection result.
Final Answer: <a complete answer to the user's question, grounded in the Observation>

Rules:
- One tool call is enough. Use vision.analyze_image for freeform inspection
  questions ("is this defective?", "describe the surface", "accept or reject?").
- After writing 'Action Input:', STOP. The system writes the Observation.
- After receiving an Observation, your next turn MUST be Final Answer
  (Shape B). Do NOT call another tool unless the previous one errored.
- Do NOT make up image paths or copy paths from instructions - use the
  exact image_ref given to you in the user prompt.
"""

REACT_USER_TEMPLATE = """\
Question: {question}

The image to analyze is at: {image_ref}

Begin.
"""


# Result dataclasses
@dataclass
class ReActStep:
    thought: str
    action: Optional[str]
    action_input: Optional[dict]
    observation: Optional[str]
    llm_metrics: Optional[GenerationMetrics] = None
    tool_latency_s: Optional[float] = None
    tool_error: Optional[str] = None


@dataclass
class ReActResult:
    question: str
    image_ref: str
    final_answer: str
    steps: list[ReActStep] = field(default_factory=list)
    total_e2e_s: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def num_iterations(self) -> int:
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        # walk through `num_tool_calls`, kept separate so the main flow stays readable.
        return sum(1 for s in self.steps if s.action is not None)

    @property
    def total_llm_calls(self) -> int:
        # `total_llm_calls` lives here. was getting too cramped inline.
        return sum(1 for s in self.steps if s.llm_metrics is not None)

    @property
    def total_prompt_tokens(self) -> int:
        # `total_prompt_tokens` lives here. was getting too cramped inline.
        return sum(
            s.llm_metrics.prompt_tokens or 0
            for s in self.steps
            if s.llm_metrics is not None
        )

    @property
    def total_completion_tokens(self) -> int:
        # `total_completion_tokens` lives here. was getting too cramped inline.
        return sum(
            s.llm_metrics.completion_tokens or 0
            for s in self.steps
            if s.llm_metrics is not None
        )

    @property
    def total_tool_latency_s(self) -> float:
        # `total_tool_latency_s` lives here. was getting too cramped inline.
        return sum(s.tool_latency_s or 0.0 for s in self.steps)

    @property
    def avg_ttft_s(self) -> Optional[float]:
        ttfts = [
            s.llm_metrics.ttft_s
            for s in self.steps
            if s.llm_metrics is not None and s.llm_metrics.ttft_s is not None
        ]
        return sum(ttfts) / len(ttfts) if ttfts else None


# Parsing helpers
_THOUGHT_RE = re.compile(
    r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Observation):|\Z)", re.DOTALL
)
_ACTION_RE = re.compile(
    r"\nAction:\s*(.+?)(?=\n(?:Action Input|Thought|Final Answer|Observation):|\Z)",
    re.DOTALL,
)
_ACTION_INPUT_RE = re.compile(
    r"\nAction Input:\s*(.+?)(?=\n(?:Thought|Action|Observation|Final Answer):|\Z)",
    re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(r"\nFinal Answer:\s*(.+?)\Z", re.DOTALL)


def _parse_action_input(raw: str) -> Optional[dict]:
    # `parse_action_input` lives here. was getting too cramped inline.
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).lstrip("json").strip()

    # wrap risky IO or RPC so we can surface a useful failure
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}") + 1

    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _parse_react_response(
    raw: str,
) -> tuple[str, Optional[str], Optional[dict], Optional[str]]:
    # drop fake Observation: lines. we inject real ones
    text = "\n" + raw  # leading newline so the (?<=\n) regexes catch line-start matches
    obs_idx = text.find("\nObservation:")

    if obs_idx != -1:
        text = text[:obs_idx]

    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    final_m = _FINAL_ANSWER_RE.search(text)

    if final_m:
        return thought, None, None, final_m.group(1).strip()

    action_m = _ACTION_RE.search(text)
    input_m = _ACTION_INPUT_RE.search(text)

    if action_m and input_m:
        action = action_m.group(1).strip().splitlines()[0].strip().rstrip(".")
        action_input = _parse_action_input(input_m.group(1))
        return thought, action, action_input, None

    return thought, None, None, None


# Tool invocation
async def _call_vision_tool(
    tool_name: str, args: dict
) -> tuple[str, Optional[str]]:
    # Async bit: `call_vision_tool` lives here. was getting too cramped inline.
    short_name = tool_name.split(".", 1)[1] if "." in tool_name else tool_name
    # wrap risky IO or RPC so we can surface a useful failure
    try:
        contents, _ = await vision_main.mcp.call_tool(short_name, args)
    except Exception as exc:
        return "", f"tool_call_failed: {exc}"

    if not contents:
        return "", "tool returned empty contents"

    text = contents[0].text or ""
    # If the tool returned an MCP error result, surface it
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "error" in payload:
            return text, payload["error"]
    except json.JSONDecodeError:
        pass
    return text, None


# ReAct runner
class ReActRunner:
    def __init__(
        self,
        llm: LLMBackend,
        max_iterations: int = 6,
        max_tokens_per_step: int = 1024,
    ) -> None:
        self._llm = llm
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens_per_step
        self._supports_metrics = hasattr(llm, "generate_with_metrics")

    def _llm_call(self, prompt: str) -> tuple[str, Optional[GenerationMetrics]]:
        # does `llm_call`, split out so we can reuse it from a few call sites.
        if self._supports_metrics:
            return self._llm.generate_with_metrics(prompt, max_tokens=self._max_tokens)  # type: ignore[attr-defined]
        return self._llm.generate(prompt), None

    async def run(self, question: str, image_ref: str) -> ReActResult:
        t_start = time.perf_counter()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools=_format_tools_for_prompt())
        user_prompt = REACT_USER_TEMPLATE.format(
            question=question, image_ref=image_ref
        )
        prompt_prefix = system_prompt + "\n\n" + user_prompt
        scratchpad = ""

        steps: list[ReActStep] = []
        final_answer: Optional[str] = None
        error: Optional[str] = None

        # each pass handles the next item in the sequence
        for iteration in range(self._max_iterations):
            full_prompt = prompt_prefix + scratchpad
            response_text, llm_metrics = self._llm_call(full_prompt)
            thought, action, action_input, fa = _parse_react_response(response_text)

            if fa is not None:
                steps.append(
                    ReActStep(
                        thought=thought,
                        action=None,
                        action_input=None,
                        observation=None,
                        llm_metrics=llm_metrics,
                    )
                )
                final_answer = fa
                break

            if action is None or action_input is None:
                # iter0 often forgets Action. default to analyze_image
                if iteration == 0 and not any(s.action for s in steps):
                    _log.warning(
                        "ReAct parse failed at iter 0 - falling back to "
                        "vision.analyze_image with the original question."
                    )
                    action = "vision.analyze_image"
                    action_input = {
                        "image_ref": image_ref,
                        "question": question,
                    }
                    if not thought:
                        thought = (
                            "(parse-fallback) I need to call vision.analyze_image "
                            "to inspect the image."
                        )
                else:
                    steps.append(
                        ReActStep(
                            thought=thought or response_text[:200],
                            action=None,
                            action_input=None,
                            observation=None,
                            llm_metrics=llm_metrics,
                            tool_error="parse_failure",
                        )
                    )
                    error = f"parse_failure at iteration {iteration}"
                    final_answer = (
                        thought.strip() or response_text.strip()[:500] or "(unparseable)"
                    )
                    break

            # only enter this block when the guard passes
            if action not in VISION_TOOLS:
                obs = (
                    f"Error: '{action}' is not a valid tool. "
                    f"Choose from: {list(VISION_TOOLS)}"
                )
                steps.append(
                    ReActStep(
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=obs,
                        llm_metrics=llm_metrics,
                        tool_error="unknown_tool",
                    )
                )
                scratchpad += (
                    f"\nThought: {thought}\nAction: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\nObservation: {obs}\n"
                )
                continue

            # pin image_ref / question for analyze_image (model drifts otherwise)
            if action == "vision.analyze_image":
                action_input["image_ref"] = image_ref
                if not action_input.get("question") or not isinstance(action_input.get("question"), str):
                    action_input["question"] = question
            elif "image_ref" not in action_input:
                action_input["image_ref"] = image_ref

            t_tool = time.perf_counter()
            obs_text, tool_err = await _call_vision_tool(action, action_input)
            tool_latency = time.perf_counter() - t_tool

            steps.append(
                ReActStep(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=obs_text,
                    llm_metrics=llm_metrics,
                    tool_latency_s=tool_latency,
                    tool_error=tool_err,
                )
            )
            obs_for_scratchpad = obs_text if len(obs_text) < 1500 else (obs_text[:1500] + "...[truncated]")
            scratchpad += (
                f"\nThought: {thought}\nAction: {action}\n"
                f"Action Input: {json.dumps(action_input)}\n"
                f"Observation: {obs_for_scratchpad}\n"
            )

        total_e2e = time.perf_counter() - t_start

        if final_answer is None:
            error = error or "max_iterations_reached"
            final_answer = "(no final answer - max iterations reached)"

        return ReActResult(
            question=question,
            image_ref=image_ref,
            final_answer=final_answer,
            steps=steps,
            total_e2e_s=total_e2e,
            success=error is None,
            error=error,
        )

    def run_sync(self, question: str, image_ref: str) -> ReActResult:
        # does `run_sync`, split out so we can reuse it from a few call sites.
        return asyncio.run(self.run(question, image_ref))

