# MCP step runner: LLM fills tool args per step, then stdio MCP call.

import json
import logging
import re
from pathlib import Path
from typing import Any

from llm import LLMBackend
from .models import Plan, PlanStep, StepResult

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# server name -> uv script name or path to python -m target
DEFAULT_SERVER_PATHS: dict[str, Path | str] = {
    "vision": "vision-mcp-server",
}

_PLACEHOLDER_RE = re.compile(r"\{step_(\d+)\}")

_ARG_RESOLUTION_PROMPT = """\
Generate the JSON arguments for the tool call below.

Question: {question}
Tool: {tool}
Tool parameters: {tool_schema}
Task: {task}

Prior step results:
{context}

YOUR RESPONSE MUST BE A SINGLE RAW JSON OBJECT AND NOTHING ELSE.
Do not write any explanation, reasoning, or prose - output only the JSON object.
Use EXACTLY the parameter names listed in "Tool parameters" above.
Use the task description and prior step results to determine the correct argument values.
If a value comes from a list, use the first relevant element.

JSON:"""


class Executor:
    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = (
            DEFAULT_SERVER_PATHS if server_paths is None else server_paths
        )

    async def get_server_descriptions(self) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        for name, path in self._server_paths.items():
            try:
                tools = await _list_tools(path)
                lines = []
                # each pass handles the next item in the sequence
                for t in tools:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    lines.append(f"  - {t['name']}({params}): {t['description']}")
                descriptions[name] = "\n".join(lines)
            except Exception as exc:
                descriptions[name] = f"  (unavailable: {exc})"
        return descriptions

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        # Async bit: `execute_plan` lives here. was getting too cramped inline.
        ordered = plan.resolved_order()
        total = len(ordered)

        server_names = {step.server for step in ordered}
        tool_schemas: dict[str, dict[str, str]] = {}
        # step through the batch one entry at a time
        for name in server_names:
            path = self._server_paths.get(name)
            if path is None:
                continue
            try:
                tools = await _list_tools(path)
                tool_schemas[name] = {
                    t["name"]: ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    for t in tools
                }
            except Exception:
                tool_schemas[name] = {}

        context: dict[int, StepResult] = {}
        results: list[StepResult] = []
        for step in ordered:
            _log.info(
                "Step %d/%d [%s]: %s",
                step.step_number,
                total,
                step.server,
                step.task,
            )
            schema = tool_schemas.get(step.server, {}).get(step.tool, "")
            result = await self.execute_step(step, context, question, tool_schema=schema)
            if result.success:
                _log.info("Step %d OK.", step.step_number)
            else:
                _log.warning("Step %d FAILED: %s", step.step_number, result.error)
            context[step.step_number] = result
            results.append(result)
        return results

    async def execute_step(
        self,
        step: PlanStep,
        context: dict[int, StepResult],
        question: str,
        tool_schema: str = "",
    ) -> StepResult:
        # Async bit: does `execute_step`, split out so we can reuse it from a few call sites.
        server_path = self._server_paths.get(step.server)
        if server_path is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=(
                    f"Unknown server '{step.server}'. "
                    f"Registered servers: {list(self._server_paths)}"
                ),
            )

        if not step.tool or step.tool.lower() in ("none", "null"):
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=step.expected_output,
                tool=step.tool,
                tool_args=step.tool_args,
            )

        # keep the happy path obvious by catching failures here
        try:
            _log.info("Step %d: calling LLM to resolve args.", step.step_number)
            resolved_args = await _resolve_args_with_llm(
                question, step.task, step.tool, tool_schema, context, self._llm
            )

            response = await _call_tool(server_path, step.tool, resolved_args)
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=response,
                tool=step.tool,
                tool_args=resolved_args,
            )
        except Exception as exc:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=str(exc),
                tool=step.tool,
                tool_args=step.tool_args,
            )


async def _resolve_args_with_llm(
    question: str,
    task: str,
    tool: str,
    tool_schema: str,
    context: dict[int, StepResult],
    llm: LLMBackend,
) -> dict:
    context_text = "\n".join(
        f"Step {n}: {r.response}" for n, r in sorted(context.items())
    )
    prompt = (
        _ARG_RESOLUTION_PROMPT
        .replace("{question}", question)
        .replace("{task}", task)
        .replace("{tool}", tool)
        .replace("{tool_schema}", tool_schema or "(unknown)")
        .replace("{context}", context_text or "(none)")
    )
    raw = llm.generate(prompt)
    resolved = _parse_json(raw)

    if resolved is None:
        _log.warning(
            "Tool '%s': arg resolution returned no parseable JSON (response: %r...)",
            tool, raw[:120],
        )
        return {}
    return resolved


def _parse_json(raw: str) -> dict | None:
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).lstrip("json").strip()
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
    _log.debug("_parse_json: could not extract a JSON object from: %r...", raw[:120])
    return None


def _make_stdio_params(server: Path | str) -> "StdioServerParameters":
    # `make_stdio_params` lives here. was getting too cramped inline.
    from mcp import StdioServerParameters

    if isinstance(server, str):
        return StdioServerParameters(
            command="uv",
            args=["run", server],
            cwd=str(_REPO_ROOT),
        )
    # wrap risky IO or RPC so we can surface a useful failure
    try:
        rel = server.relative_to(_REPO_ROOT)
        module = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        return StdioServerParameters(
            command="python",
            args=["-m", module],
            cwd=str(_REPO_ROOT),
        )
    except ValueError:
        return StdioServerParameters(command="python", args=[str(server)])


async def _list_tools(server_path: Path | str) -> list[dict]:
    # Async bit: walk through `list_tools`, kept separate so the main flow stays readable.
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ]
                tools.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": parameters,
                    }
                )
            return tools


async def _call_tool(server_path: Path | str, tool_name: str, args: dict) -> str:
    # Async bit: does `call_tool`, split out so we can reuse it from a few call sites.
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return _extract_content(result.content)


def _extract_content(content: list[Any]) -> str:
    return "\n".join(getattr(item, "text", str(item)) for item in content)


def _resolve_args(args: dict, context: dict[int, StepResult]) -> dict:
    # walk through `resolve_args`, kept separate so the main flow stays readable.
    resolved = {}
    # step through the batch one entry at a time
    for key, val in args.items():
        if isinstance(val, str):

            def _sub(m: re.Match) -> str:
                # walk through `sub`, kept separate so the main flow stays readable.
                n = int(m.group(1))
                return context[n].response if n in context else m.group(0)

            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        else:
            resolved[key] = val
    return resolved


def _parse_tool_call(raw: str) -> dict:
    # does `parse_tool_call`, split out so we can reuse it from a few call sites.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    # keep the happy path obvious by catching failures here
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"tool": None, "answer": text}

