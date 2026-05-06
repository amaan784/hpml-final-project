# LLM emits a fixed-format plan. tools named per step (args filled later by executor).

import logging
import re

from llm import LLMBackend
from .models import Plan, PlanStep

_log = logging.getLogger(__name__)

_PLAN_PROMPT = """\
You are a planning assistant for industrial asset operations and maintenance.

Decompose the question below into a sequence of subtasks. For each subtask,
assign a server and select the exact tool to call. Do NOT include tool arguments -
they will be resolved at execution time from the task description and prior results.

Available servers and tools:
{servers}

Output format - one block per step, exactly:

#Task1: <task description>
#Server1: <exact server name>
#Tool1: <exact tool name, or "none" if no tool call is needed>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Server2: <exact server name>
#Tool2: <exact tool name>
#Dependency2: #S1
#ExpectedOutput2: <what this step should produce>

Rules:
- Server and tool names must exactly match those listed above.
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if none.
- Keep tasks specific and actionable.

Question: {question}

Plan:
"""

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_SERVER_RE = re.compile(r"#Server(\d+):\s*(.+)")
_TOOL_RE = re.compile(r"#Tool(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


def parse_plan(raw: str) -> Plan:
    # `parse_plan` lives here. was getting too cramped inline.
    tasks = {int(m.group(1)): m.group(2).strip() for m in _TASK_RE.finditer(raw)}
    servers = {int(m.group(1)): m.group(2).strip() for m in _SERVER_RE.finditer(raw)}
    # drop "(args)" tail if the model copies signature text
    tools = {
        int(m.group(1)): m.group(2).strip().split("(")[0].strip()
        for m in _TOOL_RE.finditer(raw)
    }
    deps_raw = {int(m.group(1)): m.group(2).strip() for m in _DEP_RE.finditer(raw)}
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_RE.finditer(raw)}

    steps = []
    # repeat for every element we need to touch
    for n in sorted(tasks):
        raw_dep = deps_raw.get(n, "None").strip()

        if raw_dep.lower() == "none":
            dependencies = []
        else:
            dependencies = [int(x) for x in _DEP_NUM_RE.findall(raw_dep)]

            if not dependencies:
                raise ValueError(f"Invalid dependency format for step {n}: {raw_dep}")

            # each pass handles the next item in the sequence
            for dep in dependencies:
                if dep < 1 or dep >= n:
                    raise ValueError(
                        f"Invalid dependency reference for step {n}: #S{dep}"
                    )

        steps.append(
            PlanStep(
                step_number=n,
                task=tasks[n],
                server=servers.get(n, ""),
                tool=tools.get(n, ""),
                tool_args={},
                dependencies=dependencies,
                expected_output=outputs.get(n, ""),
            )
        )

    return Plan(steps=steps, raw=raw)


class Planner:
    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def generate_plan(
        self,
        question: str,
        server_descriptions: dict[str, str],
    ) -> Plan:
        # walk through `generate_plan`, kept separate so the main flow stays readable.
        servers_text = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in server_descriptions.items()
        )
        prompt = _PLAN_PROMPT.format(servers=servers_text, question=question)
        raw = self._llm.generate(prompt)
        return parse_plan(raw)
