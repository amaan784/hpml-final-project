import argparse
import asyncio
import logging
import sys
from pathlib import Path

from llm.litellm import LiteLLMBackend

from .runner import PlanExecuteRunner


def _parse_server(values: list[str] | None) -> dict[str, Path | str] | None:
    if not values:
        return None

    servers: dict[str, Path | str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--server must be NAME=COMMAND_OR_PATH, got {item!r}")
        name, value = item.split("=", 1)
        servers[name.strip()] = value.strip()
    return servers


async def _run(args: argparse.Namespace) -> int:
    llm = LiteLLMBackend(args.model)
    runner = PlanExecuteRunner(llm, _parse_server(args.server))
    result = await runner.run(args.question)
    print(result.answer)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="plan-execute")
    parser.add_argument("question", help="Question or task for the plan-execute agent.")
    parser.add_argument(
        "--model",
        default="litellm_proxy/gpt-4o-mini",
        help="LiteLLM model id to use for planning and summarization.",
    )
    parser.add_argument(
        "--server",
        action="append",
        help="Override server command as NAME=COMMAND_OR_PATH. Repeat as needed.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
