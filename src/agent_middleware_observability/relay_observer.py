from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nemo_relay

JsonDict = dict[str, Any]


@dataclass(slots=True)
class TraceValidation:
    required_edges: list[tuple[str, str]]
    missing_edges: list[tuple[str, str]]
    scope_count: int
    mark_count: int

    @property
    def passed(self) -> bool:
        return not self.missing_edges


class JsonlTraceRecorder:
    """Direct Relay subscriber that records emitted ATOF events as JSONL."""

    def __init__(self, output_path: Path, subscriber_name: str = "local-jsonl-recorder") -> None:
        self.output_path = output_path
        self.subscriber_name = subscriber_name
        self.events: list[JsonDict] = []
        self._lock = threading.Lock()
        self._file = None

    def __enter__(self) -> JsonlTraceRecorder:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")
        nemo_relay.subscribers.register(self.subscriber_name, self._on_event)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        nemo_relay.subscribers.flush()
        nemo_relay.subscribers.deregister(self.subscriber_name)
        if self._file is not None:
            self._file.close()

    def _on_event(self, event: object) -> None:
        payload = event.to_dict()
        with self._lock:
            self.events.append(payload)
            if self._file is not None:
                self._file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def validation(self, required_edges: Iterable[tuple[str, str]]) -> TraceValidation:
        starts = _scope_starts(self.events)
        names_by_uuid = {event["uuid"]: event["name"] for event in starts}
        parent_edges = {
            (names_by_uuid.get(event.get("parent_uuid")), event["name"])
            for event in starts
            if event.get("parent_uuid") in names_by_uuid
        }
        required = list(required_edges)
        missing = [edge for edge in required if edge not in parent_edges]
        return TraceValidation(
            required_edges=required,
            missing_edges=missing,
            scope_count=len(starts),
            mark_count=sum(1 for event in self.events if event.get("kind") == "mark"),
        )

    def write_summary(
        self,
        output_path: Path,
        *,
        result: JsonDict,
        required_edges: Iterable[tuple[str, str]],
    ) -> TraceValidation:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validation = self.validation(required_edges)
        tree = render_trace_tree(self.events)
        lines = [
            "# Middleware Observability Trace Summary",
            "",
            f"- Trace events: {len(self.events)}",
            f"- Scope starts: {validation.scope_count}",
            f"- Marks/events: {validation.mark_count}",
            f"- Hierarchy validation: {'PASS' if validation.passed else 'FAIL'}",
            "",
            "## Expected Parent-Child Edges",
            "",
        ]
        for parent, child in validation.required_edges:
            status = "ok" if (parent, child) not in validation.missing_edges else "missing"
            lines.append(f"- {status}: `{parent}` -> `{child}`")

        lines.extend(
            [
                "",
                "## Final Agent Answer",
                "",
                result.get("answer", "(no answer produced)"),
                "",
                "## Trace Tree",
                "",
                "```text",
                tree,
                "```",
                "",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return validation


def render_trace_tree(events: list[JsonDict]) -> str:
    starts = _scope_starts(events)
    names_by_uuid = {event["uuid"]: event["name"] for event in starts}
    by_parent: dict[str | None, list[JsonDict]] = defaultdict(list)
    for event in starts:
        parent_uuid = event.get("parent_uuid")
        by_parent[parent_uuid if parent_uuid in names_by_uuid else None].append(event)

    def walk(parent_uuid: str | None, depth: int) -> list[str]:
        lines: list[str] = []
        for event in by_parent.get(parent_uuid, []):
            category = event.get("category") or "unknown"
            lines.append(f"{'  ' * depth}- {event['name']} [{category}]")
            lines.extend(walk(event["uuid"], depth + 1))
        return lines

    return "\n".join(walk(None, 0))


def _scope_starts(events: list[JsonDict]) -> list[JsonDict]:
    return [
        event
        for event in events
        if event.get("kind") == "scope" and event.get("scope_category") == "start" and event.get("uuid")
    ]
