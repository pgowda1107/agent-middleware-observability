from __future__ import annotations

import os
from typing import Annotated, Any, Sequence, TypedDict

import nemo_relay
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_middleware_observability.managed_relay_langgraph import relay_model_call, relay_tool_node
from agent_middleware_observability.managed_scope import subagent_scope


class ManagedAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def web_search(query: str) -> str:
    """Search the web for a query and return short findings."""

    return f"Findings for {query!r}: NeMo Relay captures hierarchical agent handoffs."


@tool
def style_guide(topic: str) -> str:
    """Return concise writing guidance for a topic."""

    return f"Style for {topic!r}: lead with the conclusion, name the handoff, keep it short."


class ScriptedChatModel:
    """Deterministic chat model used to validate Relay-managed execution without an API key."""

    def __init__(self, *, tool_name: str, tool_args: dict[str, Any], final_prefix: str) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.final_prefix = final_prefix

    def bind_tools(self, _: Sequence[Any]) -> ScriptedChatModel:
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        tool_messages = [message for message in messages if message.type == "tool"]
        if tool_messages:
            return AIMessage(content=f"{self.final_prefix}: {tool_messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": self.tool_args,
                    "id": f"call_{self.tool_name}",
                }
            ],
        )


def live_model():
    """Create a live OpenAI-compatible chat model from environment variables."""

    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ["NVIDIA_API_KEY"]
    return ChatOpenAI(
        model=os.environ["MODEL_NAME"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=api_key,
        temperature=0,
    )


def build_managed_react_graph(
    name: str,
    tools: Sequence[Any],
    system_prompt: str,
    *,
    model: Any,
    model_name: str,
):
    """Build a custom ReAct StateGraph with Relay-managed model and tool call sites."""

    chat_model = model.bind_tools(tools)

    async def model_node(state: ManagedAgentState) -> dict[str, list[Any]]:
        conversation = [SystemMessage(content=system_prompt), *state["messages"]]
        response = await relay_model_call(
            f"{name}.model",
            conversation,
            lambda messages: chat_model.ainvoke(messages),
            model_name=model_name,
        )
        return {"messages": [response]}

    def should_continue(state: ManagedAgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    builder = StateGraph(ManagedAgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", relay_tool_node(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    return builder.compile()


def build_offline_subagents() -> dict[str, Any]:
    """Build managed custom StateGraph subagents that run without external credentials."""

    research = build_managed_react_graph(
        "research",
        [web_search],
        "You research topics. Use web_search once, then answer briefly.",
        model=ScriptedChatModel(
            tool_name="web_search",
            tool_args={"query": "NeMo Relay custom StateGraph handoff"},
            final_prefix="Research",
        ),
        model_name="scripted-offline-model",
    )
    writer = build_managed_react_graph(
        "writer",
        [style_guide],
        "You polish text. Consult style_guide once, then rewrite briefly.",
        model=ScriptedChatModel(
            tool_name="style_guide",
            tool_args={"topic": "Relay handoff summary"},
            final_prefix="Writer",
        ),
        model_name="scripted-offline-model",
    )
    return {"research": research, "writer": writer}


def build_live_subagents() -> dict[str, Any]:
    """Build managed custom StateGraph subagents backed by a live OpenAI-compatible model."""

    model_name = os.environ["MODEL_NAME"]
    return {
        "research": build_managed_react_graph(
            "research",
            [web_search],
            "You research topics. Use web_search once, then answer briefly.",
            model=live_model(),
            model_name=model_name,
        ),
        "writer": build_managed_react_graph(
            "writer",
            [style_guide],
            "You polish text. Consult style_guide once, then rewrite briefly.",
            model=live_model(),
            model_name=model_name,
        ),
    }


def handoff(name: str, graph: Any):
    """Return a Relay tool handler that invokes one managed custom StateGraph subagent."""

    async def run(args: dict[str, Any]):
        task = args["task"]
        with subagent_scope(
            f"subagent:{name}",
            input={"task": task},
            metadata={"agent": name, "built_with": "custom-state-graph"},
        ):
            result = await graph.ainvoke({"messages": [HumanMessage(content=task)]})
        return nemo_relay.ToolExecutionResult({"output": result["messages"][-1].content})

    return run
