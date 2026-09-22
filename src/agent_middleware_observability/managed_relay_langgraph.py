from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import nemo_relay
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode


def relay_tool_node(tools: Sequence[Any], **kwargs: Any) -> ToolNode:
    """Create a LangGraph ToolNode whose tool calls use Relay managed execution."""

    async def awrap_tool_call(request: Any, execute: Callable[[Any], Awaitable[Any]]) -> Any:
        parent = nemo_relay.scope.get_handle()
        codec = nemo_relay.typed.BestEffortAnyCodec()
        tool_call = request.tool_call

        async def call(args: Any) -> nemo_relay.ToolExecutionResult[Any]:
            forwarded = request.override(tool_call={**tool_call, "args": args})
            return nemo_relay.ToolExecutionResult(await execute(forwarded))

        outcome = await nemo_relay.typed.tool_execute(
            name=tool_call["name"],
            args=tool_call.get("args") or {},
            func=call,
            args_codec=codec,
            result_codec=codec,
            handle=parent,
            tool_call_id=tool_call.get("id"),
        )
        return outcome.result

    return ToolNode(tools, awrap_tool_call=awrap_tool_call, **kwargs)


async def relay_model_call(
    name: str,
    messages: list[BaseMessage],
    invoke: Callable[[list[BaseMessage]], Awaitable[Any]],
    *,
    model_name: str,
) -> Any:
    """Run a custom StateGraph model node through Relay managed LLM execution."""

    codec = nemo_relay.typed.BestEffortAnyCodec()

    async def call(_: nemo_relay.LLMRequest) -> Any:
        return await invoke(messages)

    return await nemo_relay.typed.llm_execute(
        name,
        nemo_relay.LLMRequest(
            {},
            {"messages": [{"role": message.type, "content": str(message.content)} for message in messages]},
        ),
        call,
        codec,
        model_name=model_name,
    )
