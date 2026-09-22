from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import nemo_relay
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from agent_middleware_observability.managed_custom_state_graph import (
    build_live_subagents,
    build_offline_subagents,
    handoff,
)
from agent_middleware_observability.managed_scope import set_root
from agent_middleware_observability.relay_observer import JsonlTraceRecorder

DEFAULT_QUESTION = "Research the Relay handoff, then write a concise summary."

EXPECTED_EDGES = [
    ("orchestrator:request", "research_agent"),
    ("research_agent", "subagent:research"),
    ("subagent:research", "research.model"),
    ("subagent:research", "web_search"),
    ("orchestrator:request", "writer_agent"),
    ("writer_agent", "subagent:writer"),
    ("subagent:writer", "writer.model"),
    ("subagent:writer", "style_guide"),
]


async def run_episode(question: str, *, live: bool) -> dict[str, str]:
    subagents = build_live_subagents() if live else build_offline_subagents()

    with nemo_relay.scope.scope(
        "orchestrator:request",
        nemo_relay.ScopeType.Agent,
        input={"question": question},
        metadata={"mode": "live" if live else "offline", "architecture": "custom-state-graph"},
    ) as root:
        set_root(root)
        research = await nemo_relay.tools.execute(
            "research_agent",
            {"task": question},
            handoff("research", subagents["research"]),
            handle=root,
            tool_call_id="call_research_agent",
        )
        writer = await nemo_relay.tools.execute(
            "writer_agent",
            {"task": research.result["output"]},
            handoff("writer", subagents["writer"]),
            handle=root,
            tool_call_id="call_writer_agent",
        )
    return {
        "research": research.result["output"],
        "answer": writer.result["output"],
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Relay-managed custom StateGraph subagents.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--live", action="store_true", help="Use MODEL_NAME, OPENAI_BASE_URL, and OPENAI_API_KEY.")
    parser.add_argument("--trace-file", default="outputs/managed_custom_trace_events.jsonl")
    parser.add_argument("--summary-file", default="outputs/managed_custom_trace_summary.md")
    args = parser.parse_args()

    console = Console()
    trace_path = Path(args.trace_file)
    summary_path = Path(args.summary_file)

    with JsonlTraceRecorder(trace_path) as recorder:
        result = asyncio.run(run_episode(args.question, live=args.live))

    validation = recorder.write_summary(summary_path, result=result, required_edges=EXPECTED_EDGES)
    console.print(Panel.fit(result["answer"], title="Managed custom StateGraph result", border_style="green"))
    console.print(f"[bold]Trace file:[/bold] {trace_path}")
    console.print(f"[bold]Summary file:[/bold] {summary_path}")
    console.print(
        f"[bold]Hierarchy validation:[/bold] {'PASS' if validation.passed else 'FAIL'} "
        f"({validation.scope_count} scopes, {validation.mark_count} marks)"
    )
    if validation.missing_edges:
        for parent, child in validation.missing_edges:
            console.print(f"[red]Missing edge:[/red] {parent} -> {child}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
