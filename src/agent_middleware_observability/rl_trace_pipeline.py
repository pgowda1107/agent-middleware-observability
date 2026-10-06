from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

SENSITIVE_KEY_PARTS = (
    "authorization",
    "api-key",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "cookie",
    "bearer",
)


@dataclass(slots=True)
class ScopeRecord:
    uuid: str
    name: str
    category: str
    parent_uuid: str | None
    root_uuid: str
    start_time: datetime | None
    end_time: datetime | None = None
    start_data: Any = None
    end_data: Any = None
    start_metadata: JsonDict | None = None
    end_metadata: JsonDict | None = None
    category_profile: JsonDict | None = None
    marks: list[JsonDict] = field(default_factory=list)

    @property
    def metadata(self) -> JsonDict:
        merged: JsonDict = {}
        if isinstance(self.start_metadata, dict):
            merged.update(self.start_metadata)
        if isinstance(self.end_metadata, dict):
            merged.update(self.end_metadata)
        return merged

    @property
    def status(self) -> str:
        status = str(self.metadata.get("otel.status_code", "")).upper()
        if status == "ERROR":
            return "ERROR"
        if self.end_time is None:
            return "OPEN"
        return "OK"

    @property
    def error(self) -> str | None:
        metadata = self.metadata
        if self.status != "ERROR":
            return None
        return (
            metadata.get("otel.status_description")
            or metadata.get("exception.type")
            or metadata.get("error.type")
            or "unknown error"
        )

    @property
    def duration_ms(self) -> float | None:
        if not self.start_time or not self.end_time:
            return None
        return round((self.end_time - self.start_time).total_seconds() * 1000, 3)


@dataclass(slots=True)
class Episode:
    episode_id: str
    scopes: list[ScopeRecord]
    events: list[JsonDict]
    children_by_parent: dict[str | None, list[ScopeRecord]]
    depth_by_uuid: dict[str, int]
    step_rewards: dict[str, float]
    terminal_reward: float
    outcome: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NeMo Relay ATOF JSONL traces into sample RL episodes and transitions."
    )
    parser.add_argument("--input", default="outputs/managed_custom_trace_events.jsonl", help="Input Relay JSONL trace.")
    parser.add_argument("--output-dir", default="outputs/rl", help="Directory for RL dataset outputs.")
    parser.add_argument(
        "--include-payloads",
        action="store_true",
        help="Include truncated payload values. By default only payload shapes are exported.",
    )
    parser.add_argument("--max-string", type=int, default=160, help="Maximum exported string length.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    events = read_jsonl(input_path)
    episodes = build_episodes(events)
    write_outputs(
        input_path=input_path,
        output_dir=output_dir,
        raw_events=events,
        episodes=episodes,
        include_payloads=args.include_payloads,
        max_string=args.max_string,
    )

    transition_count = sum(len(episode.scopes) for episode in episodes)
    print(f"Wrote {len(episodes)} episode(s) and {transition_count} transition(s) to {output_dir}")


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    events: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return events


def build_episodes(events: list[JsonDict]) -> list[Episode]:
    scopes_by_uuid = build_scope_records(events)
    roots = sorted(
        {scope.root_uuid for scope in scopes_by_uuid.values()},
        key=lambda root_uuid: _first_scope_time(root_uuid, scopes_by_uuid),
    )

    episodes: list[Episode] = []
    for root_uuid in roots:
        episode_scopes = sorted(
            [scope for scope in scopes_by_uuid.values() if scope.root_uuid == root_uuid],
            key=lambda scope: (_sort_timestamp(scope.start_time), scope.name),
        )
        episode_events = [event for event in events if event.get("propagation_root_uuid") == root_uuid]
        children_by_parent = index_children(episode_scopes)
        depth_by_uuid = compute_depths(episode_scopes)
        step_rewards = {scope.uuid: score_scope(scope, children_by_parent, depth_by_uuid) for scope in episode_scopes}
        terminal_reward, outcome = score_episode(episode_scopes, children_by_parent, depth_by_uuid)
        episodes.append(
            Episode(
                episode_id=root_uuid,
                scopes=episode_scopes,
                events=episode_events,
                children_by_parent=children_by_parent,
                depth_by_uuid=depth_by_uuid,
                step_rewards=step_rewards,
                terminal_reward=terminal_reward,
                outcome=outcome,
            )
        )
    return episodes


def build_scope_records(events: list[JsonDict]) -> dict[str, ScopeRecord]:
    scopes: dict[str, ScopeRecord] = {}
    pending_marks: dict[str, list[JsonDict]] = defaultdict(list)

    for event in events:
        if event.get("kind") == "mark":
            parent_uuid = event.get("parent_uuid")
            if parent_uuid:
                if parent_uuid in scopes:
                    scopes[parent_uuid].marks.append(event)
                else:
                    pending_marks[parent_uuid].append(event)
            continue

        if event.get("kind") != "scope" or not event.get("uuid"):
            continue

        scope_uuid = event["uuid"]
        scope_category = event.get("scope_category")
        if scope_category == "start":
            scopes[scope_uuid] = ScopeRecord(
                uuid=scope_uuid,
                name=event.get("name") or "(unnamed)",
                category=infer_category(event),
                parent_uuid=event.get("parent_uuid"),
                root_uuid=event.get("propagation_root_uuid") or scope_uuid,
                start_time=parse_timestamp(event.get("timestamp")),
                start_data=event.get("data"),
                start_metadata=event.get("metadata"),
                category_profile=event.get("category_profile"),
                marks=pending_marks.pop(scope_uuid, []),
            )
        elif scope_category == "end":
            scope = scopes.get(scope_uuid)
            if not scope:
                scope = ScopeRecord(
                    uuid=scope_uuid,
                    name=event.get("name") or "(unnamed)",
                    category=infer_category(event),
                    parent_uuid=event.get("parent_uuid"),
                    root_uuid=event.get("propagation_root_uuid") or scope_uuid,
                    start_time=None,
                )
                scopes[scope_uuid] = scope
            scope.end_time = parse_timestamp(event.get("timestamp"))
            scope.end_data = event.get("data")
            scope.end_metadata = event.get("metadata")
            if event.get("category_profile") and not scope.category_profile:
                scope.category_profile = event.get("category_profile")

    for parent_uuid, marks in pending_marks.items():
        if parent_uuid in scopes:
            scopes[parent_uuid].marks.extend(marks)

    return scopes


def index_children(scopes: list[ScopeRecord]) -> dict[str | None, list[ScopeRecord]]:
    known = {scope.uuid for scope in scopes}
    children_by_parent: dict[str | None, list[ScopeRecord]] = defaultdict(list)
    for scope in scopes:
        parent_uuid = scope.parent_uuid if scope.parent_uuid in known else None
        children_by_parent[parent_uuid].append(scope)
    for children in children_by_parent.values():
        children.sort(key=lambda scope: (_sort_timestamp(scope.start_time), scope.name))
    return dict(children_by_parent)


def compute_depths(scopes: list[ScopeRecord]) -> dict[str, int]:
    by_uuid = {scope.uuid: scope for scope in scopes}
    depths: dict[str, int] = {}

    def depth(scope: ScopeRecord, seen: set[str]) -> int:
        if scope.uuid in depths:
            return depths[scope.uuid]
        if scope.uuid in seen:
            return 0
        parent = by_uuid.get(scope.parent_uuid or "")
        if not parent:
            depths[scope.uuid] = 0
            return 0
        result = depth(parent, {*seen, scope.uuid}) + 1
        depths[scope.uuid] = result
        return result

    for scope in scopes:
        depth(scope, set())
    return depths


def score_scope(
    scope: ScopeRecord,
    children_by_parent: dict[str | None, list[ScopeRecord]],
    depth_by_uuid: dict[str, int],
) -> float:
    if scope.status == "ERROR":
        return -3.0
    if scope.status == "OPEN":
        return -1.0

    category_scores = {
        "agent": 0.5,
        "tool": 0.4,
        "llm": 0.3,
        "function": 0.2,
        "unknown": 0.1,
    }
    score = category_scores.get(scope.category, 0.1)
    if scope.parent_uuid:
        score += 0.1
    if children_by_parent.get(scope.uuid):
        score += 0.1
    if is_handoff_scope(scope):
        score += 0.4
    if depth_by_uuid.get(scope.uuid, 0) > 0:
        score += 0.1
    return round(score, 3)


def score_episode(
    scopes: list[ScopeRecord],
    children_by_parent: dict[str | None, list[ScopeRecord]],
    depth_by_uuid: dict[str, int],
) -> tuple[float, str]:
    error_count = sum(1 for scope in scopes if scope.status == "ERROR")
    open_count = sum(1 for scope in scopes if scope.status == "OPEN")
    handoff_count = len(find_handoffs(scopes, children_by_parent))
    max_depth = max(depth_by_uuid.values(), default=0)

    if error_count:
        return round(-5.0 - error_count, 3), "error"
    if open_count:
        return round(-2.0 - open_count, 3), "incomplete"
    return round(5.0 + (handoff_count * 0.5) + min(max_depth, 5) * 0.2, 3), "success"


def write_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    raw_events: list[JsonDict],
    episodes: list[Episode],
    include_payloads: bool,
    max_string: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    transitions_path = output_dir / "transitions.jsonl"
    summary_path = output_dir / "summary.md"

    with episodes_path.open("w", encoding="utf-8") as episode_file:
        for episode in episodes:
            episode_file.write(
                json.dumps(
                    episode_to_record(episode, include_payloads=include_payloads, max_string=max_string),
                    sort_keys=True,
                )
                + "\n"
            )

    with transitions_path.open("w", encoding="utf-8") as transition_file:
        for episode in episodes:
            for transition in episode_to_transitions(
                episode,
                include_payloads=include_payloads,
                max_string=max_string,
            ):
                transition_file.write(json.dumps(transition, sort_keys=True) + "\n")

    summary_path.write_text(
        render_summary(input_path=input_path, raw_events=raw_events, episodes=episodes),
        encoding="utf-8",
    )


def episode_to_record(episode: Episode, *, include_payloads: bool, max_string: int) -> JsonDict:
    handoffs = find_handoffs(episode.scopes, episode.children_by_parent)
    return {
        "episode_id": episode.episode_id,
        "outcome": episode.outcome,
        "terminal_reward": episode.terminal_reward,
        "metrics": {
            "scope_count": len(episode.scopes),
            "mark_count": sum(len(scope.marks) for scope in episode.scopes),
            "handoff_count": len(handoffs),
            "error_count": sum(1 for scope in episode.scopes if scope.status == "ERROR"),
            "open_scope_count": sum(1 for scope in episode.scopes if scope.status == "OPEN"),
            "max_depth": max(episode.depth_by_uuid.values(), default=0),
        },
        "handoffs": handoffs,
        "trajectory": [
            scope_to_step(
                scope,
                episode,
                include_payloads=include_payloads,
                max_string=max_string,
            )
            for scope in episode.scopes
        ],
    }


def episode_to_transitions(episode: Episode, *, include_payloads: bool, max_string: int) -> list[JsonDict]:
    transitions: list[JsonDict] = []
    for index, scope in enumerate(episode.scopes):
        done = index == len(episode.scopes) - 1
        base_reward = episode.step_rewards[scope.uuid]
        reward = base_reward + (episode.terminal_reward if done else 0)
        parent = find_parent(scope, episode.scopes)
        children = episode.children_by_parent.get(scope.uuid, [])
        transitions.append(
            {
                "episode_id": episode.episode_id,
                "step_index": index,
                "scope_uuid": scope.uuid,
                "parent_uuid": parent.uuid if parent else None,
                "parent_name": parent.name if parent else None,
                "depth": episode.depth_by_uuid.get(scope.uuid, 0),
                "state": {
                    "active_scope": scope.name,
                    "ancestor_path": ancestor_path(scope, episode.scopes),
                    "child_count": len(children),
                },
                "action": {
                    "name": scope.name,
                    "type": scope.category,
                    "category_profile": scrub_payload(
                        scope.category_profile,
                        include_payloads=True,
                        max_string=max_string,
                    ),
                },
                "observation": {
                    "status": scope.status,
                    "duration_ms": scope.duration_ms,
                    "input": scrub_payload(
                        scope.start_data,
                        include_payloads=include_payloads,
                        max_string=max_string,
                    ),
                    "output": scrub_payload(
                        scope.end_data,
                        include_payloads=include_payloads,
                        max_string=max_string,
                    ),
                    "marks": [
                        {
                            "name": mark.get("name"),
                            "data": scrub_payload(
                                mark.get("data"),
                                include_payloads=include_payloads,
                                max_string=max_string,
                            ),
                        }
                        for mark in scope.marks
                    ],
                    "error": scrub_payload(scope.error, include_payloads=True, max_string=max_string),
                },
                "next_scope_names": [child.name for child in children],
                "base_reward": base_reward,
                "terminal_reward": episode.terminal_reward if done else 0.0,
                "reward": round(reward, 3),
                "done": done,
            }
        )
    return transitions


def scope_to_step(scope: ScopeRecord, episode: Episode, *, include_payloads: bool, max_string: int) -> JsonDict:
    parent = find_parent(scope, episode.scopes)
    children = episode.children_by_parent.get(scope.uuid, [])
    return {
        "uuid": scope.uuid,
        "name": scope.name,
        "category": scope.category,
        "parent_name": parent.name if parent else None,
        "depth": episode.depth_by_uuid.get(scope.uuid, 0),
        "status": scope.status,
        "duration_ms": scope.duration_ms,
        "reward": episode.step_rewards[scope.uuid],
        "child_names": [child.name for child in children],
        "metadata": scrub_payload(scope.metadata, include_payloads=True, max_string=max_string),
        "input": scrub_payload(scope.start_data, include_payloads=include_payloads, max_string=max_string),
        "output": scrub_payload(scope.end_data, include_payloads=include_payloads, max_string=max_string),
    }


def find_handoffs(scopes: list[ScopeRecord], children_by_parent: dict[str | None, list[ScopeRecord]]) -> list[JsonDict]:
    by_uuid = {scope.uuid: scope for scope in scopes}
    handoffs: list[JsonDict] = []
    for parent_uuid, children in children_by_parent.items():
        parent = by_uuid.get(parent_uuid or "")
        if not parent:
            continue
        for child in children:
            if is_handoff_scope(child):
                handoffs.append(
                    {
                        "from": parent.name,
                        "from_category": parent.category,
                        "to": child.name,
                        "to_category": child.category,
                    }
                )
    return handoffs


def is_handoff_scope(scope: ScopeRecord) -> bool:
    return scope.category == "agent" and scope.name.startswith("subagent:")


def find_parent(scope: ScopeRecord, scopes: list[ScopeRecord]) -> ScopeRecord | None:
    by_uuid = {candidate.uuid: candidate for candidate in scopes}
    return by_uuid.get(scope.parent_uuid or "")


def ancestor_path(scope: ScopeRecord, scopes: list[ScopeRecord]) -> list[str]:
    by_uuid = {candidate.uuid: candidate for candidate in scopes}
    path: list[str] = []
    current = scope
    seen: set[str] = set()
    while current.parent_uuid and current.parent_uuid in by_uuid and current.uuid not in seen:
        seen.add(current.uuid)
        parent = by_uuid[current.parent_uuid]
        path.append(parent.name)
        current = parent
    return list(reversed(path))


def render_summary(*, input_path: Path, raw_events: list[JsonDict], episodes: list[Episode]) -> str:
    transition_count = sum(len(episode.scopes) for episode in episodes)
    lines = [
        "# Sample RL Pipeline Summary",
        "",
        f"- Input trace: `{input_path}`",
        f"- Raw events: {len(raw_events)}",
        f"- Episodes: {len(episodes)}",
        f"- Transitions: {transition_count}",
        "",
    ]

    for episode_index, episode in enumerate(episodes, start=1):
        handoffs = find_handoffs(episode.scopes, episode.children_by_parent)
        lines.extend(
            [
                f"## Episode {episode_index}",
                "",
                f"- Episode id: `{episode.episode_id}`",
                f"- Outcome: `{episode.outcome}`",
                f"- Terminal reward: `{episode.terminal_reward}`",
                f"- Handoffs: {len(handoffs)}",
                "",
            ]
        )
        if handoffs:
            lines.append("### Handoff Edges")
            lines.append("")
            for handoff in handoffs:
                lines.append(f"- `{handoff['from']}` -> `{handoff['to']}`")
            lines.append("")
        lines.extend(
            [
                "### Trace Tree",
                "",
                "```text",
                render_episode_tree(episode),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def render_episode_tree(episode: Episode) -> str:
    def walk(parent_uuid: str | None, depth: int) -> list[str]:
        lines: list[str] = []
        for scope in episode.children_by_parent.get(parent_uuid, []):
            duration = f", {scope.duration_ms} ms" if scope.duration_ms is not None else ""
            reward = episode.step_rewards.get(scope.uuid, 0.0)
            lines.append(
                f"{'  ' * depth}- {scope.name} [{scope.category}, {scope.status}, reward={reward}{duration}]"
            )
            lines.extend(walk(scope.uuid, depth + 1))
        return lines

    return "\n".join(walk(None, 0))


def scrub_payload(value: Any, *, include_payloads: bool, max_string: int) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        result: JsonDict = {}
        for key, nested in value.items():
            if is_sensitive_key(str(key)):
                result[key] = "[REDACTED]"
            else:
                result[key] = scrub_payload(nested, include_payloads=include_payloads, max_string=max_string)
        return result
    if isinstance(value, list):
        if include_payloads:
            return [
                scrub_payload(item, include_payloads=include_payloads, max_string=max_string)
                for item in value[:10]
            ]
        return {"type": "list", "count": len(value)}
    if isinstance(value, tuple):
        return scrub_payload(list(value), include_payloads=include_payloads, max_string=max_string)
    if isinstance(value, str):
        if not include_payloads:
            return {"type": "str", "chars": len(value)}
        return truncate(value, max_string)
    if isinstance(value, (int, float, bool)):
        return value
    return truncate(str(value), max_string) if include_payloads else {"type": type(value).__name__}


def is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def truncate(value: str, max_string: int) -> str:
    if len(value) <= max_string:
        return value
    return value[: max(0, max_string - 3)] + "..."


def infer_category(event: JsonDict) -> str:
    category = event.get("category")
    if category:
        return str(category)
    name = str(event.get("name") or "")
    if name.startswith("subagent:") or name.startswith("agent:") or name.startswith("orchestrator:"):
        return "agent"
    if name.startswith("tool:"):
        return "tool"
    if name.endswith(".model"):
        return "llm"
    if name.startswith("node:") or name.startswith("subagent-node:"):
        return "function"
    return "unknown"


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _first_scope_time(root_uuid: str, scopes: dict[str, ScopeRecord]) -> float:
    times = [
        scope.start_time
        for scope in scopes.values()
        if scope.root_uuid == root_uuid and scope.start_time is not None
    ]
    return min(_sort_timestamp(timestamp) for timestamp in times) if times else float("inf")


def _sort_timestamp(value: datetime | None) -> float:
    return value.timestamp() if value else float("inf")


if __name__ == "__main__":
    main()
