from __future__ import annotations

from functools import lru_cache, wraps
from typing import Any, Callable, TypedDict

import nemo_relay
from langgraph.graph import END, START, StateGraph

NodeResult = dict[str, Any]


class WorkflowMapperState(TypedDict, total=False):
    request: str
    run_id: str
    workflow_map: dict[str, Any]


class TraceValidatorState(TypedDict, total=False):
    run_id: str
    workflow_map: dict[str, Any]
    relay_fit: dict[str, Any]


@lru_cache(maxsize=1)
def get_workflow_mapper_graph():
    graph = StateGraph(WorkflowMapperState)
    graph.add_node(
        "extract_state_graph",
        relay_subagent_node("workflow-mapper", "extract_state_graph")(workflow_mapper_extract_node),
    )
    graph.add_edge(START, "extract_state_graph")
    graph.add_edge("extract_state_graph", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_trace_validator_graph():
    graph = StateGraph(TraceValidatorState)
    graph.add_node(
        "define_trace_checks",
        relay_subagent_node("trace-validator", "define_trace_checks")(trace_validator_checks_node),
    )
    graph.add_edge(START, "define_trace_checks")
    graph.add_edge("define_trace_checks", END)
    return graph.compile()


def relay_subagent_node(
    subagent_name: str,
    node_name: str,
) -> Callable[[Callable[[dict[str, Any]], NodeResult]], Callable[[dict[str, Any]], NodeResult]]:
    """Wrap one deterministic node inside a LangGraph subagent."""

    def decorator(func: Callable[[dict[str, Any]], NodeResult]) -> Callable[[dict[str, Any]], NodeResult]:
        @wraps(func)
        def wrapped(state: dict[str, Any]) -> NodeResult:
            scope_name = f"subagent-node:{subagent_name}.{node_name}"
            with nemo_relay.scope.scope(
                scope_name,
                nemo_relay.ScopeType.Function,
                metadata={
                    "component": "langgraph-subagent",
                    "subagent": subagent_name,
                    "langgraph.node": node_name,
                    "run_id": state.get("run_id"),
                },
                input=_summarize_state(state),
            ):
                nemo_relay.scope.event(
                    "subagent.node.enter",
                    data={"subagent": subagent_name, "node": node_name, "known_state_keys": sorted(state.keys())},
                    severity=nemo_relay.LogSeverity.Info,
                )
                updates = func(state)
                nemo_relay.scope.event(
                    "subagent.node.exit",
                    data={"subagent": subagent_name, "node": node_name, "updates": _summarize_state(updates)},
                    severity=nemo_relay.LogSeverity.Info,
                )
                return updates

        return wrapped

    return decorator


def workflow_mapper_extract_node(state: WorkflowMapperState) -> NodeResult:
    return {"workflow_map": extract_state_graph(state["request"])}


def trace_validator_checks_node(state: TraceValidatorState) -> NodeResult:
    return {"relay_fit": assess_expected_hierarchy(state.get("workflow_map", {}))}


def extract_state_graph(request: str) -> dict[str, Any]:
    with nemo_relay.scope.scope(
        "tool:extract_state_graph",
        nemo_relay.ScopeType.Tool,
        input={"request_chars": len(request)},
    ):
        nodes = [
            "plan",
            "map_workflow",
            "assess_relay_fit",
            "recommend_next_steps",
        ]
        edges = [
            ("START", "plan"),
            ("plan", "map_workflow"),
            ("map_workflow", "assess_relay_fit"),
            ("assess_relay_fit", "recommend_next_steps"),
            ("recommend_next_steps", "END"),
        ]
        nemo_relay.scope.event(
            "tool.output",
            data={"nodes": nodes, "edges": edges, "source": "reference-architecture"},
            severity=nemo_relay.LogSeverity.Info,
        )
        return {"nodes": nodes, "edges": edges}


def assess_expected_hierarchy(workflow_map: dict[str, Any]) -> dict[str, Any]:
    with nemo_relay.scope.scope(
        "tool:define_trace_checks",
        nemo_relay.ScopeType.Tool,
        input={"nodes": workflow_map.get("nodes", [])},
    ):
        result = {
            "target": "nested Relay scopes for orchestrator -> LangGraph tool -> node -> subagent graph -> tool",
            "must_not_be_flat": True,
            "expected_node_scopes": [f"node:{name}" for name in workflow_map.get("nodes", [])],
        }
        nemo_relay.scope.event("tool.output", data=result, severity=nemo_relay.LogSeverity.Info)
        return result


def _summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, str):
            summary[key] = value if len(value) < 160 else f"{value[:157]}..."
        elif isinstance(value, list):
            summary[key] = {"type": "list", "count": len(value)}
        elif isinstance(value, dict):
            summary[key] = {"type": "dict", "keys": sorted(value.keys())}
        else:
            summary[key] = value
    return summary
