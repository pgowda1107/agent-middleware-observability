from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_middleware_observability.rl_trace_pipeline import (
    JsonDict,
    ScopeRecord,
    build_scope_records,
    is_sensitive_key,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract faithful model-call samples from NeMo Relay LLM scopes."
    )
    parser.add_argument("--trace", default="outputs/managed_custom_trace_events.jsonl", help="Input Relay JSONL trace.")
    parser.add_argument("--output-dir", default="outputs/model_calls", help="Directory for model-call sample outputs.")
    parser.add_argument(
        "--no-redact-sensitive",
        action="store_true",
        help="Do not redact auth-like keys. Prompt and completion text is preserved either way.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_dir = Path(args.output_dir)
    events = read_jsonl(trace_path)
    scopes = build_scope_records(events)
    samples = extract_model_call_samples(
        scopes,
        redact_sensitive=not args.no_redact_sensitive,
        source_trace=trace_path,
    )
    write_outputs(trace_path=trace_path, output_dir=output_dir, samples=samples)
    print(f"Wrote {len(samples)} model-call sample(s) to {output_dir}")


def extract_model_call_samples(
    scopes_by_uuid: dict[str, ScopeRecord],
    *,
    redact_sensitive: bool,
    source_trace: Path | None = None,
) -> list[JsonDict]:
    samples: list[JsonDict] = []
    ordered_scopes = sorted(
        scopes_by_uuid.values(),
        key=lambda scope: ((scope.start_time.timestamp() if scope.start_time else float("inf")), scope.name),
    )
    per_episode_index: dict[str, int] = {}

    for scope in ordered_scopes:
        if scope.category != "llm" or scope.status != "OK":
            continue
        messages = extract_request_messages(scope.start_data, redact_sensitive=redact_sensitive)
        target = extract_assistant_target(scope.end_data, redact_sensitive=redact_sensitive)
        if not messages or not target:
            continue

        per_episode_index[scope.root_uuid] = per_episode_index.get(scope.root_uuid, 0) + 1
        sample_index = per_episode_index[scope.root_uuid]
        parent = scopes_by_uuid.get(scope.parent_uuid or "")
        model_name = None
        if isinstance(scope.category_profile, dict):
            model_name = scope.category_profile.get("model_name")

        sample = {
            "sample_id": f"{scope.root_uuid[:8]}-{sample_index:03d}-{safe_sample_name(scope.name)}",
            "source": "nemo-relay-atof",
            "source_trace": str(source_trace) if source_trace else None,
            "episode_id": scope.root_uuid,
            "scope_uuid": scope.uuid,
            "scope_name": scope.name,
            "parent_scope_name": parent.name if parent else None,
            "ancestor_path": ancestor_path(scope, scopes_by_uuid),
            "model_name": model_name,
            "target_kind": classify_target(target),
            "messages": messages,
            "target": target,
            "training_messages": [*messages, target],
            "metrics": {
                "input_message_count": len(messages),
                "target_tool_call_count": len(target.get("tool_calls") or []),
                "duration_ms": scope.duration_ms,
            },
        }
        samples.append(sample)

    return samples


def extract_request_messages(value: Any, *, redact_sensitive: bool) -> list[JsonDict]:
    if not isinstance(value, dict):
        return []
    content = value.get("content")
    if not isinstance(content, dict):
        return []
    messages = content.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        normalize_message(message, redact_sensitive=redact_sensitive)
        for message in messages
        if isinstance(message, dict)
    ]


def extract_assistant_target(value: Any, *, redact_sensitive: bool) -> JsonDict | None:
    payload = unwrap_message_payload(value)
    if not payload:
        return None

    target: JsonDict = {
        "role": "assistant",
        "content": normalize_content(payload.get("content")),
    }
    tool_calls = payload.get("tool_calls") or payload.get("additional_kwargs", {}).get("tool_calls")
    normalized_tool_calls = normalize_tool_calls(tool_calls, redact_sensitive=redact_sensitive)
    if normalized_tool_calls:
        target["tool_calls"] = normalized_tool_calls
    name = payload.get("name")
    if name:
        target["name"] = name
    return target


def unwrap_message_payload(value: Any) -> JsonDict | None:
    if not isinstance(value, dict):
        return None
    payload = value.get("data") if isinstance(value.get("data"), dict) else value
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "ai" or "tool_calls" in payload or "content" in payload:
        return payload
    return None


def normalize_message(message: JsonDict, *, redact_sensitive: bool) -> JsonDict:
    normalized: JsonDict = {
        "role": normalize_role(str(message.get("role") or message.get("type") or "")),
        "content": normalize_content(message.get("content")),
    }

    tool_calls = normalize_tool_calls(message.get("tool_calls"), redact_sensitive=redact_sensitive)
    if tool_calls:
        normalized["tool_calls"] = tool_calls

    tool_call_id = message.get("tool_call_id")
    if tool_call_id:
        normalized["tool_call_id"] = str(tool_call_id)

    name = message.get("name")
    if name:
        normalized["name"] = str(name)

    additional_kwargs = message.get("additional_kwargs")
    if isinstance(additional_kwargs, dict) and additional_kwargs:
        normalized["additional_kwargs"] = maybe_redact(additional_kwargs, redact_sensitive=redact_sensitive)

    return normalized


def normalize_role(role: str) -> str:
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(role, role)


def normalize_content(value: Any) -> str | list[Any]:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value
    return str(value)


def normalize_tool_calls(value: Any, *, redact_sensitive: bool) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    tool_calls: list[JsonDict] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = item.get("name") or function.get("name")
        if not name:
            continue

        if "arguments" in function:
            arguments = function["arguments"]
            if not isinstance(arguments, str):
                arguments = json.dumps(
                    maybe_redact(arguments, redact_sensitive=redact_sensitive),
                    sort_keys=True,
                )
        else:
            arguments = json.dumps(
                maybe_redact(item.get("args") or {}, redact_sensitive=redact_sensitive),
                sort_keys=True,
            )

        tool_calls.append(
            {
                "id": str(item.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": arguments,
                },
            }
        )
    return tool_calls


def maybe_redact(value: Any, *, redact_sensitive: bool) -> Any:
    if not redact_sensitive:
        return value
    if isinstance(value, dict):
        result: JsonDict = {}
        for key, nested in value.items():
            if is_sensitive_key(str(key)):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = maybe_redact(nested, redact_sensitive=redact_sensitive)
        return result
    if isinstance(value, list):
        return [maybe_redact(item, redact_sensitive=redact_sensitive) for item in value]
    return value


def classify_target(target: JsonDict) -> str:
    has_content = bool(target.get("content"))
    has_tools = bool(target.get("tool_calls"))
    if has_tools and has_content:
        return "content_and_tool_call"
    if has_tools:
        return "tool_call"
    return "content"


def ancestor_path(scope: ScopeRecord, scopes_by_uuid: dict[str, ScopeRecord]) -> list[str]:
    path: list[str] = []
    current = scope
    seen: set[str] = set()
    while current.parent_uuid and current.parent_uuid in scopes_by_uuid and current.uuid not in seen:
        seen.add(current.uuid)
        parent = scopes_by_uuid[current.parent_uuid]
        path.append(parent.name)
        current = parent
    return list(reversed(path))


def safe_sample_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-").lower()


def write_outputs(*, trace_path: Path, output_dir: Path, samples: list[JsonDict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_calls_path = output_dir / "model_call_samples.jsonl"
    sft_path = output_dir / "sft_messages.jsonl"
    summary_path = output_dir / "summary.md"

    with model_calls_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, sort_keys=True) + "\n")

    with sft_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(
                    {
                        "messages": sample["training_messages"],
                        "metadata": {
                            "sample_id": sample["sample_id"],
                            "episode_id": sample["episode_id"],
                            "scope_name": sample["scope_name"],
                            "parent_scope_name": sample["parent_scope_name"],
                            "target_kind": sample["target_kind"],
                            "model_name": sample["model_name"],
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    summary_path.write_text(render_summary(trace_path=trace_path, samples=samples), encoding="utf-8")


def render_summary(*, trace_path: Path, samples: list[JsonDict]) -> str:
    target_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    for sample in samples:
        target_counts[sample["target_kind"]] = target_counts.get(sample["target_kind"], 0) + 1
        scope_counts[sample["scope_name"]] = scope_counts.get(sample["scope_name"], 0) + 1

    lines = [
        "# Model-Call Sample Summary",
        "",
        f"- Input trace: `{trace_path}`",
        f"- Model-call samples: {len(samples)}",
        f"- Tool-call targets: {target_counts.get('tool_call', 0)}",
        f"- Content targets: {target_counts.get('content', 0)}",
        f"- Mixed content/tool targets: {target_counts.get('content_and_tool_call', 0)}",
        "",
        "These rows preserve prompt and assistant target text. Review them before sharing outside the project.",
        "",
    ]

    if scope_counts:
        lines.append("## Scope Counts")
        lines.append("")
        for scope_name, count in sorted(scope_counts.items()):
            lines.append(f"- `{scope_name}`: {count}")
        lines.append("")

    if samples:
        lines.append("## Samples")
        lines.append("")
        for sample in samples:
            target = sample["target"]
            tool_names = [
                tool_call["function"]["name"]
                for tool_call in target.get("tool_calls", [])
                if isinstance(tool_call, dict) and isinstance(tool_call.get("function"), dict)
            ]
            target_label = ", ".join(tool_names) if tool_names else truncate(str(target.get("content") or ""), 80)
            lines.append(
                f"- `{sample['sample_id']}` `{sample['scope_name']}` "
                f"{sample['target_kind']}: {target_label}"
            )
        lines.append("")

    return "\n".join(lines)


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


if __name__ == "__main__":
    main()
