from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_middleware_observability.rl_trace_pipeline import (
    Episode,
    build_episodes,
    find_handoffs,
    read_jsonl,
)

JsonDict = dict[str, Any]


SYSTEM_PROMPT = (
    "You are an orchestrator. Route work to the correct specialist subagent, "
    "use tools only when needed, and keep the handoff hierarchy clear."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create NeMo Gym task JSONL and a local verifier report from Relay traces."
    )
    parser.add_argument("--trace", default="outputs/managed_custom_trace_events.jsonl")
    parser.add_argument("--output-dir", default="outputs/nemo_gym")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows written to validation.jsonl when more than one row exists.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(trace_path)
    episodes = build_episodes(events)
    rows = [episode_to_nemo_gym_row(episode, trace_path=trace_path) for episode in episodes]

    train_rows, validation_rows = split_rows(rows, args.validation_ratio)
    write_jsonl(output_dir / "example.jsonl", rows)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)

    verification = [
        verify_trace_against_task(row, episode)
        for row, episode in zip(rows, episodes, strict=True)
    ]
    write_jsonl(output_dir / "verification.jsonl", verification)
    (output_dir / "summary.md").write_text(
        render_summary(trace_path=trace_path, rows=rows, verification=verification),
        encoding="utf-8",
    )

    passed = sum(1 for item in verification if item["passed"])
    print(
        f"Wrote {len(rows)} NeMo Gym row(s) to {output_dir}; "
        f"local verifier passed {passed}/{len(verification)}."
    )


def episode_to_nemo_gym_row(episode: Episode, *, trace_path: Path) -> JsonDict:
    root = find_root_scope(episode)
    prompt = extract_prompt(root.start_data if root else None)
    handoffs = find_handoffs(episode.scopes, episode.children_by_parent)
    scope_tree = [
        {
            "name": scope.name,
            "category": scope.category,
            "parent_name": parent_name(scope.uuid, episode),
            "status": scope.status,
            "depth": episode.depth_by_uuid.get(scope.uuid, 0),
        }
        for scope in episode.scopes
    ]

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": handoff_tools(),
            "parallel_tool_calls": False,
        },
        "task_data": {
            "episode_id": episode.episode_id,
            "source_trace": str(trace_path),
            "reward_spec": "relay-handoff-v1",
            "expected_handoffs": [
                {"from": item["from"], "to": item["to"]}
                for item in handoffs
            ],
            "expected_scope_tree": scope_tree,
            "expected_terminal_outcome": episode.outcome,
            "expected_terminal_reward": episode.terminal_reward,
        },
    }


def verify_trace_against_task(row: JsonDict, episode: Episode) -> JsonDict:
    task_data = row["task_data"]
    expected_handoffs = {
        (item["from"], item["to"])
        for item in task_data.get("expected_handoffs", [])
    }
    actual_handoffs = {
        (item["from"], item["to"])
        for item in find_handoffs(episode.scopes, episode.children_by_parent)
    }
    missing_handoffs = sorted(expected_handoffs - actual_handoffs)
    unexpected_handoffs = sorted(actual_handoffs - expected_handoffs)
    error_scopes = [
        {"name": scope.name, "error": scope.error}
        for scope in episode.scopes
        if scope.status == "ERROR"
    ]
    open_scopes = [scope.name for scope in episode.scopes if scope.status == "OPEN"]
    passed = not missing_handoffs and not unexpected_handoffs and not error_scopes and not open_scopes
    reward = 1.0 if passed else 0.0

    return {
        "episode_id": episode.episode_id,
        "passed": passed,
        "reward": reward,
        "missing_handoffs": [
            {"from": source, "to": target}
            for source, target in missing_handoffs
        ],
        "unexpected_handoffs": [
            {"from": source, "to": target}
            for source, target in unexpected_handoffs
        ],
        "error_scopes": error_scopes,
        "open_scopes": open_scopes,
    }


def split_rows(rows: list[JsonDict], validation_ratio: float) -> tuple[list[JsonDict], list[JsonDict]]:
    if len(rows) <= 1:
        return rows, rows
    validation_count = max(1, round(len(rows) * validation_ratio))
    split_at = max(1, len(rows) - validation_count)
    return rows[:split_at], rows[split_at:]


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def render_summary(*, trace_path: Path, rows: list[JsonDict], verification: list[JsonDict]) -> str:
    passed = sum(1 for item in verification if item["passed"])
    lines = [
        "# NeMo Gym Dataset Smoke Test",
        "",
        f"- Source trace: `{trace_path}`",
        f"- Rows: {len(rows)}",
        f"- Local verifier: {passed}/{len(verification)} passed",
        "",
        "## Files",
        "",
        "- `example.jsonl`",
        "- `train.jsonl`",
        "- `validation.jsonl`",
        "- `verification.jsonl`",
        "",
        "## Handoffs",
        "",
    ]
    for row in rows:
        task_data = row["task_data"]
        lines.append(f"### Episode `{task_data['episode_id']}`")
        lines.append("")
        for handoff in task_data["expected_handoffs"]:
            lines.append(f"- `{handoff['from']}` -> `{handoff['to']}`")
        lines.append("")
    return "\n".join(lines)


def find_root_scope(episode: Episode):
    roots = episode.children_by_parent.get(None, [])
    return roots[0] if roots else episode.scopes[0] if episode.scopes else None


def parent_name(scope_uuid: str, episode: Episode) -> str | None:
    by_uuid = {scope.uuid: scope for scope in episode.scopes}
    scope = by_uuid[scope_uuid]
    parent = by_uuid.get(scope.parent_uuid or "")
    return parent.name if parent else None


def extract_prompt(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("question", "request", "task", "prompt"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    return "Run the Relay handoff workflow and preserve the expected subagent hierarchy."


def handoff_tools() -> list[JsonDict]:
    return [
        {
            "type": "function",
            "name": "research_agent",
            "description": "Hand off research work to the research subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Research task."}
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "writer_agent",
            "description": "Hand off writing or summarization work to the writer subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Writing task."}
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


if __name__ == "__main__":
    main()
