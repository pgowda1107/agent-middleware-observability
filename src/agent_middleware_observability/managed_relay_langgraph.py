from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import nemo_relay
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode

JsonDict = dict[str, Any]


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
            {"messages": [relay_message_dict(message) for message in messages]},
        ),
        call,
        codec,
        model_name=model_name,
    )


def relay_message_dict(message: BaseMessage) -> JsonDict:
    """Convert LangChain messages into a JSON-safe chat request shape for Relay traces."""

    result: JsonDict = {
        "role": normalize_role(message.type),
        "content": json_safe(message.content),
    }

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        result["additional_kwargs"] = json_safe(additional_kwargs)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = json_safe(tool_calls)

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if invalid_tool_calls:
        result["invalid_tool_calls"] = json_safe(invalid_tool_calls)

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        result["tool_call_id"] = tool_call_id

    name = getattr(message, "name", None)
    if name:
        result["name"] = name

    status = getattr(message, "status", None)
    if status:
        result["status"] = status

    return result


def normalize_role(message_type: str) -> str:
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(message_type, message_type)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)
