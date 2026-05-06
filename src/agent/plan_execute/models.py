# Plan / step structs for plan-execute.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanStep:
    step_number: int
    task: str
    server: str
    tool: str
    tool_args: dict
    dependencies: list[int]
    expected_output: str


@dataclass
class Plan:
    steps: list[PlanStep]
    raw: str

    def get_step(self, number: int) -> Optional[PlanStep]:
        # does `get_step`, split out so we can reuse it from a few call sites.
        return next((s for s in self.steps if s.step_number == number), None)

    def resolved_order(self) -> list[PlanStep]:
        seen: set[int] = set()
        ordered: list[PlanStep] = []

        def visit(n: int) -> None:
            if n in seen:
                return
            step = self.get_step(n)
            if step is None:
                return
            # step through the batch one entry at a time
            for dep in step.dependencies:
                visit(dep)
            seen.add(n)
            ordered.append(step)

        for step in self.steps:
            visit(step.step_number)
        return ordered


@dataclass
class StepResult:
    step_number: int
    task: str
    server: str
    response: str
    error: Optional[str] = None
    tool: str = ""
    tool_args: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None
