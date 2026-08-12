"""Host-owned control plane for ending a model turn after a trusted tool handoff.

Tool result text is model-visible and may contain arbitrary JSON, so it cannot be
used as an authority for turn control.  Instead, ``model_tools`` binds the active
tool execution to a context variable.  A trusted in-process tool may request a
defer against that context, and the conversation loop consumes the request only
after the tool result has been persisted.
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


_MAX_ACKNOWLEDGEMENT_CHARS = 500
_MAX_REFERENCE_CHARS = 160
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TurnControlError(RuntimeError):
    """Raised when a defer request is invalid or outside tool execution."""


@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True)
class DeferredTurn:
    session_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    acknowledgement: str
    reference_id: str = ""


_CURRENT_TOOL_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "hermes_current_tool_execution_context", default=None
)
_PENDING: dict[tuple[str, str], DeferredTurn] = {}
_PENDING_LOCK = threading.Lock()


def _clean_bounded_text(value: str, *, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TurnControlError(f"{name} must be a string")
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    if not cleaned:
        raise TurnControlError(f"{name} must not be empty")
    if len(cleaned) > maximum:
        raise TurnControlError(f"{name} exceeds {maximum} characters")
    return cleaned


@contextmanager
def tool_execution_context(
    *,
    session_id: str | None,
    turn_id: str | None,
    tool_call_id: str | None,
    tool_name: str,
) -> Iterator[ToolExecutionContext | None]:
    """Bind host-authenticated identifiers while one registry tool executes."""

    if not session_id or not turn_id or not tool_call_id:
        yield None
        return
    context = ToolExecutionContext(
        session_id=str(session_id),
        turn_id=str(turn_id),
        tool_call_id=str(tool_call_id),
        tool_name=str(tool_name),
    )
    token = _CURRENT_TOOL_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_TOOL_CONTEXT.reset(token)


def current_tool_execution_context() -> ToolExecutionContext | None:
    """Return the active immutable context, if called from a registry tool."""

    return _CURRENT_TOOL_CONTEXT.get()


def request_current_turn_defer(
    *, acknowledgement: str, reference_id: str = ""
) -> DeferredTurn:
    """Request a deterministic end to the current turn after tool persistence.

    The first defer request for a turn wins.  Repeating the exact request is
    idempotent; a conflicting second request is rejected.
    """

    context = current_tool_execution_context()
    if context is None:
        raise TurnControlError("turn defer is only available during tool execution")
    acknowledgement = _clean_bounded_text(
        acknowledgement,
        name="acknowledgement",
        maximum=_MAX_ACKNOWLEDGEMENT_CHARS,
    )
    if reference_id:
        reference_id = _clean_bounded_text(
            reference_id,
            name="reference_id",
            maximum=_MAX_REFERENCE_CHARS,
        )
    decision = DeferredTurn(
        session_id=context.session_id,
        turn_id=context.turn_id,
        tool_call_id=context.tool_call_id,
        tool_name=context.tool_name,
        acknowledgement=acknowledgement,
        reference_id=reference_id,
    )
    key = (context.session_id, context.turn_id)
    with _PENDING_LOCK:
        previous = _PENDING.get(key)
        if previous is not None and previous != decision:
            raise TurnControlError("a different defer request already exists for this turn")
        _PENDING[key] = decision
    return decision


def peek_deferred_turn(*, session_id: str | None, turn_id: str | None) -> DeferredTurn | None:
    if not session_id or not turn_id:
        return None
    with _PENDING_LOCK:
        return _PENDING.get((str(session_id), str(turn_id)))


def consume_deferred_turn(*, session_id: str | None, turn_id: str | None) -> DeferredTurn | None:
    if not session_id or not turn_id:
        return None
    with _PENDING_LOCK:
        return _PENDING.pop((str(session_id), str(turn_id)), None)


def discard_deferred_turn(*, session_id: str | None, turn_id: str | None) -> None:
    if not session_id or not turn_id:
        return
    with _PENDING_LOCK:
        _PENDING.pop((str(session_id), str(turn_id)), None)


def discard_deferred_turn_for_tool(
    *,
    session_id: str | None,
    turn_id: str | None,
    tool_call_id: str | None,
) -> bool:
    """Discard only the provisional defer created by one failed tool call."""

    if not session_id or not turn_id or not tool_call_id:
        return False
    key = (str(session_id), str(turn_id))
    with _PENDING_LOCK:
        decision = _PENDING.get(key)
        if decision is None or decision.tool_call_id != str(tool_call_id):
            return False
        _PENDING.pop(key, None)
        return True


__all__ = [
    "DeferredTurn",
    "ToolExecutionContext",
    "TurnControlError",
    "consume_deferred_turn",
    "current_tool_execution_context",
    "discard_deferred_turn",
    "discard_deferred_turn_for_tool",
    "peek_deferred_turn",
    "request_current_turn_defer",
    "tool_execution_context",
]
