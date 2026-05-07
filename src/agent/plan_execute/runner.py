# Plan-execute loop: discover MCP tools, plan, run steps, summarize.

import logging
from pathlib import Path

from llm import LLMBackend

from .executor import Executor
from .planner import Planner
from ..models import OrchestratorResult
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are summarizing the results of a multi-step task execution for an \
industrial asset operations system.

Original question: {question}

Step-by-step execution results:
{results}

Provide a concise, direct answer to the original question based on the results
above. Do not repeat the individual steps - just give the final answer.
"""


class PlanExecuteRunner(AgentRunner):
    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        self._planner = Planner(llm)
        self._executor = Executor(llm, server_paths)

    async def run(self, question: str) -> OrchestratorResult:
        _log.info("Discovering server capabilities...")
        server_descriptions = await self._executor.get_server_descriptions()

        _log.info("Planning...")
        plan = self._planner.generate_plan(question, server_descriptions)
        _log.info("Plan has %d step(s).", len(plan.steps))

        history = await self._executor.execute_plan(plan, question)

        _log.info("Summarising...")
        results_text = "\n\n".join(
            f"Step {r.step_number} - {r.task} (server: {r.server}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in history
        )
        answer = self._llm.generate(
            _SUMMARIZE_PROMPT.format(question=question, results=results_text)
        )

        return OrchestratorResult(
            question=question,
            answer=answer,
            plan=plan,
            history=history,
        )
