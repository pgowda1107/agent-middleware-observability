from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Literal, TypedDict

import nemo_relay
from langgraph.graph import END, START, StateGraph

from agent_middleware_observability.subagents import get_trace_validator_graph, get_workflow_mapper_graph

NodeResult = dict[str, Any]


class ObservabilityState(TypedDict, total=False):
    request: str
    run_id: str
    plan: list[str]
    route: Literal["map_workflow", "recommend_next_steps"]
    workflow_map: dict[str, Any]
    relay_fit: dict[str, Any]
    answer: str


def build_observability_graph():
    graph = StateGraph(ObservabilityState)
    graph.add_node("plan", relay_node("plan")(plan_node))
    graph.add_node("map_workflow", relay_node("map_workflow")(map_workflow_node))
    graph.add_node("assess_relay_fit", relay_node("assess_relay_fit")(assess_relay_fit_node))
    graph.add_node("recommend_next_steps", relay_node("recommend_next_steps")(recommend_next_steps_node))

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "map_workflow": "map_workflow",
            "recommend_next_steps": "recommend_next_steps",
        },
    )
    graph.add_edge("map_workflow", "assess_relay_fit")
    graph.add_edge("assess_relay_fit", "recommend_next_steps")
    graph.add_edge("recommend_next_steps", END)
    return graph.compile()


def relay_node(name: str) -> Callable[[Callable[[ObservabilityState], NodeResult]], Callable[[ObservabilityState], NodeResult]]:
    """Wrap one deterministic LangGraph node in a Relay scope."""

    def decorator(func: Callable[[ObservabilityState], NodeResult]) -> Callable[[ObservabilityState], NodeResult]:
        @wraps(func)
        def wrapped(state: ObservabilityState) -> NodeResult:
            metadata = {
                "component": "langgraph-state-graph",
                "langgraph.node": name,
                "run_id": state.get("run_id"),
            }
            with nemo_relay.scope.scope(
                f"node:{name}",
                nemo_relay.ScopeType.Function,
                metadata=metadata,
                input=_summarize_state(state),
            ):
                nemo_relay.scope.event(
                    "node.enter",
                    data={"node": name, "known_state_keys": sorted(state.keys())},
                    severity=nemo_relay.LogSeverity.Info,
                )
                updates = func(state)
                nemo_relay.scope.event(
                    "node.exit",
                    data={"node": name, "updates": _summarize_state(updates)},
                    severity=nemo_relay.LogSeverity.Info,
                )
                return updates

        return wrapped

    return decorator


def plan_node(state: ObservabilityState) -> NodeResult:
    request = state["request"].lower()
    plan = [
        "identify deterministic LangGraph nodes and edges",
        "wrap each node with scoped Relay middleware",
        "spawn focused subagents for workflow mapping and trace validation",
        "compare middleware trace hierarchy with callback/exporter-only traces",
    ]
    route: Literal["map_workflow", "recommend_next_steps"] = (
        "map_workflow" if any(term in request for term in ("relay", "langgraph", "observability", "agent")) else "recommend_next_steps"
    )
    return {"plan": plan, "route": route}


def route_after_plan(state: ObservabilityState) -> str:
    return state.get("route", "recommend_next_steps")


def map_workflow_node(state: ObservabilityState) -> NodeResult:
    with nemo_relay.scope.scope(
        "subagent:workflow-mapper",
        nemo_relay.ScopeType.Agent,
        metadata={
            "responsibility": "derive nodes and edges from the reference architecture",
            "runtime": "compiled-langgraph-subgraph",
        },
    ):
        result = get_workflow_mapper_graph().invoke(
            {
                "request": state["request"],
                "run_id": state.get("run_id", ""),
            }
        )
        workflow_map = result["workflow_map"]
        nemo_relay.scope.event("subagent.result", data={"node_count": len(workflow_map["nodes"])})
    return {"workflow_map": workflow_map}


def assess_relay_fit_node(state: ObservabilityState) -> NodeResult:
    with nemo_relay.scope.scope(
        "subagent:trace-validator",
        nemo_relay.ScopeType.Agent,
        metadata={
            "responsibility": "define trace hierarchy validation criteria",
            "runtime": "compiled-langgraph-subgraph",
        },
    ):
        result = get_trace_validator_graph().invoke(
            {
                "workflow_map": state.get("workflow_map", {}),
                "run_id": state.get("run_id", ""),
            }
        )
        relay_fit = result["relay_fit"]
        nemo_relay.scope.event("subagent.result", data=relay_fit)
    return {"relay_fit": relay_fit}


def recommend_next_steps_node(state: ObservabilityState) -> NodeResult:
    workflow_map = state.get("workflow_map") or {"nodes": [], "edges": []}
    relay_fit = state.get("relay_fit") or {}
    answer = "\n".join(
        [
            "Middleware-based Relay instrumentation is the recommended path for this agent.",
            "",
            f"Mapped deterministic nodes: {', '.join(workflow_map.get('nodes', [])) or 'not available'}.",
            f"Mapped deterministic edges: {len(workflow_map.get('edges', []))}.",
            f"Trace validation target: {relay_fit.get('target', 'hierarchical node, subagent, and tool scopes')}.",
            "",
            "Implementation pattern:",
            "1. Keep the custom Python React-loop as the top-level agent scope.",
            "2. Treat the LangGraph workflow as a tool scope invoked by that orchestrator.",
            "3. Wrap every deterministic LangGraph node with Relay middleware.",
            "4. Invoke compiled LangGraph subgraphs for subagent work inside parent nodes.",
            "5. Emit tool scopes beneath the owning subagent or node so traces do not flatten.",
        ]
    )
    return {"answer": answer}


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
