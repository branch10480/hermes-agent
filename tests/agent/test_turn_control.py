from __future__ import annotations

import pytest
from types import SimpleNamespace

import model_tools
from agent import tool_executor
from agent.turn_control import (
    TurnControlError,
    consume_deferred_turn,
    current_tool_execution_context,
    discard_deferred_turn,
    discard_deferred_turn_for_tool,
    peek_deferred_turn,
    request_current_turn_defer,
    tool_execution_context,
)


@pytest.fixture(autouse=True)
def _clear_test_turns():
    for turn_id in ("turn-1", "turn-2", "turn-model-tools"):
        discard_deferred_turn(session_id="session-1", turn_id=turn_id)
    yield
    for turn_id in ("turn-1", "turn-2", "turn-model-tools"):
        discard_deferred_turn(session_id="session-1", turn_id=turn_id)


def test_defer_requires_host_tool_execution_context():
    with pytest.raises(TurnControlError, match="only available during tool execution"):
        request_current_turn_defer(acknowledgement="queued")


def test_defer_is_scoped_to_exact_session_and_turn():
    with tool_execution_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="knowledge_start",
    ):
        context = current_tool_execution_context()
        assert context is not None
        assert context.tool_name == "knowledge_start"
        requested = request_current_turn_defer(
            acknowledgement="Knowledge job job_123 を受理しました。完了時に通知します。",
            reference_id="job_123",
        )

    assert current_tool_execution_context() is None
    assert peek_deferred_turn(session_id="session-1", turn_id="turn-2") is None
    assert peek_deferred_turn(session_id="session-1", turn_id="turn-1") == requested
    assert consume_deferred_turn(session_id="session-1", turn_id="turn-1") == requested
    assert peek_deferred_turn(session_id="session-1", turn_id="turn-1") is None


def test_identical_defer_is_idempotent_but_conflict_is_rejected():
    with tool_execution_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="knowledge_start",
    ):
        first = request_current_turn_defer(
            acknowledgement="queued", reference_id="job_123"
        )
        assert request_current_turn_defer(
            acknowledgement="queued", reference_id="job_123"
        ) == first
        with pytest.raises(TurnControlError, match="different defer request"):
            request_current_turn_defer(
                acknowledgement="different", reference_id="job_456"
            )


def test_failed_tool_discard_is_scoped_to_exact_tool_call():
    with tool_execution_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="knowledge_start",
    ):
        request_current_turn_defer(acknowledgement="queued")

    assert discard_deferred_turn_for_tool(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-other",
    ) is False
    assert peek_deferred_turn(session_id="session-1", turn_id="turn-1") is not None
    assert discard_deferred_turn_for_tool(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
    ) is True
    assert peek_deferred_turn(session_id="session-1", turn_id="turn-1") is None


def test_acknowledgement_is_bounded_and_strips_control_characters():
    with tool_execution_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="knowledge_start",
    ):
        decision = request_current_turn_defer(acknowledgement=" queued\x00 ")
        assert decision.acknowledgement == "queued"

    discard_deferred_turn(session_id="session-1", turn_id="turn-1")
    with tool_execution_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="knowledge_start",
    ):
        with pytest.raises(TurnControlError, match="exceeds"):
            request_current_turn_defer(acknowledgement="x" * 501)


def test_model_tools_binds_authenticated_context_around_registry_dispatch(monkeypatch):
    observed = {}

    def fake_dispatch(name, args, **kwargs):
        observed["context"] = current_tool_execution_context()
        request_current_turn_defer(
            acknowledgement="Knowledge job job_123 を受理しました。完了時に通知します。",
            reference_id="job_123",
        )
        return '{"ok": true}'

    monkeypatch.setattr(model_tools.registry, "dispatch", fake_dispatch)
    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", lambda **_: None)

    result = model_tools.handle_function_call(
        "knowledge_start",
        {},
        task_id="task-1",
        session_id="session-1",
        turn_id="turn-model-tools",
        tool_call_id="call-1",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert result == '{"ok": true}'
    assert observed["context"].session_id == "session-1"
    assert observed["context"].turn_id == "turn-model-tools"
    assert observed["context"].tool_call_id == "call-1"
    assert current_tool_execution_context() is None
    assert consume_deferred_turn(
        session_id="session-1", turn_id="turn-model-tools"
    ) is not None


def test_deferred_sibling_calls_receive_matching_non_effect_results(monkeypatch):
    messages = []
    emitted = []
    flushed = []
    agent = SimpleNamespace()
    calls = [
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="terminal"),
        ),
        SimpleNamespace(
            id="call-3",
            function=SimpleNamespace(name="write_file"),
        ),
    ]
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda *args, **kwargs: emitted.append(kwargs),
    )
    monkeypatch.setattr(
        tool_executor,
        "_flush_session_db_after_tool_progress",
        lambda *args, **kwargs: flushed.append(kwargs["stage"]) or True,
    )

    assert tool_executor._append_deferred_tool_results(
        agent,
        messages,
        calls,
        effective_task_id="task-1",
    ) is True

    assert [message["tool_call_id"] for message in messages] == ["call-2", "call-3"]
    assert all(message["effect_disposition"] == "none" for message in messages)
    assert [item["error_type"] for item in emitted] == [
        "turn_deferred",
        "turn_deferred",
    ]
    assert flushed == [
        "deferred tool result terminal",
        "deferred tool result write_file",
    ]
