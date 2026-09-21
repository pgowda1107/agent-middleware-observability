from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import nemo_relay
from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler

from agent_middleware_observability.graph import build_observability_graph


@dataclass
class ReactLoopOrchestrator:
    """Custom Python React-loop style orchestrator around a LangGraph tool."""

    graph: Any = field(default_factory=build_observability_graph)
    include_langgraph_callbacks: bool = False

    def invoke(self, request: str) -> dict[str, Any]:
        run_id = str(uuid4())
        initial_state = {"request": request, "run_id": run_id}
        with nemo_relay.scope.scope(
            "agent:react-orchestrator",
            nemo_relay.ScopeType.Agent,
            input={"request": request},
            metadata={"run_id": run_id, "architecture": "custom-python-react-loop"},
        ):
            nemo_relay.scope.event(
                "agent.spawn",
                data={"child": "tool:langgraph-observability-workflow", "run_id": run_id},
                severity=nemo_relay.LogSeverity.Info,
            )
            with nemo_relay.scope.scope(
                "tool:langgraph-observability-workflow",
                nemo_relay.ScopeType.Tool,
                input={"state_keys": sorted(initial_state.keys())},
                metadata={"run_id": run_id, "tool_kind": "compiled-langgraph"},
            ):
                config: dict[str, Any] = {"metadata": {"run_id": run_id}}
                if self.include_langgraph_callbacks:
                    config["callbacks"] = [NemoRelayCallbackHandler()]
                return self.graph.invoke(initial_state, config=config)
