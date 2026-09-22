from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

import nemo_relay

_root_uuid: contextvars.ContextVar[str | None] = contextvars.ContextVar("relay_root_uuid", default=None)


def set_root(handle: nemo_relay.ScopeHandle) -> None:
    """Remember the root request scope for managed tool-call handoffs."""

    _root_uuid.set(handle.uuid)


@contextmanager
def subagent_scope(name: str, **kwargs: object) -> Iterator[nemo_relay.ScopeHandle]:
    """Open a subagent scope under the managed tool call currently executing."""

    context = nemo_relay.capture_propagation_context_with_root(_root_uuid.get())
    stack = nemo_relay.create_scope_stack_from_propagation(context)
    with nemo_relay.use_scope_stack(stack):
        with nemo_relay.scope.scope(name, nemo_relay.ScopeType.Agent, **kwargs) as handle:
            yield handle
