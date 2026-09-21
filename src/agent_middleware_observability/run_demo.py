from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from agent_middleware_observability.agents import ReactLoopOrchestrator
from agent_middleware_observability.relay_observer import JsonlTraceRecorder

DEFAULT_REQUEST = """
Build a middleware-based observability approach for a custom Python React-loop
agent that invokes a LangGraph-based tool. The graph should expose deterministic
nodes and edges, preserve hierarchy for agent spawns and tool calls, and reduce
custom callback/exporter code by using NeMo Relay.
""".strip()

EXPECTED_EDGES = [
    ("agent:react-orchestrator", "tool:langgraph-observability-workflow"),
    ("tool:langgraph-observability-workflow", "node:plan"),
    ("tool:langgraph-observability-workflow", "node:map_workflow"),
    ("node:map_workflow", "subagent:workflow-mapper"),
    ("subagent:workflow-mapper", "tool:extract_state_graph"),
    ("tool:langgraph-observability-workflow", "node:assess_relay_fit"),
    ("node:assess_relay_fit", "subagent:trace-validator"),
    ("subagent:trace-validator", "tool:define_trace_checks"),
    ("tool:langgraph-observability-workflow", "node:recommend_next_steps"),
]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Relay-instrumented LangGraph agent demo.")
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="Agent task to run through the workflow.")
    parser.add_argument("--trace-file", default="outputs/trace_events.jsonl", help="Path for Relay ATOF JSONL events.")
    parser.add_argument("--summary-file", default="outputs/trace_summary.md", help="Path for a readable trace summary.")
    args = parser.parse_args()

    console = Console()
    trace_path = Path(args.trace_file)
    summary_path = Path(args.summary_file)

    agent = ReactLoopOrchestrator()
    with JsonlTraceRecorder(trace_path) as recorder:
        result = agent.invoke(args.request)

    validation = recorder.write_summary(summary_path, result=result, required_edges=EXPECTED_EDGES)

    console.print(Panel.fit(result["answer"], title="Agent result", border_style="green"))
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
