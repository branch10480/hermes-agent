"""Behavioral tests for trusted tool-driven turn deferral."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_control import (
    discard_deferred_turn,
    peek_deferred_turn,
    request_current_turn_defer,
    tool_execution_context,
)
from agent.tool_executor import execute_tool_calls_segmented
from run_agent import AIAgent


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} test tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
        for name in ("knowledge_start", "terminal")
    ]


def _agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=5,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    agent.session_id = "session-defer-test"
    return agent


def _tool_response():
    call = SimpleNamespace(
        id="call-knowledge",
        type="function",
        function=SimpleNamespace(name="knowledge_start", arguments="{}"),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _deferred_tool(name, args, task_id, **kwargs):
    del args, task_id
    with tool_execution_context(
        session_id=kwargs["session_id"],
        turn_id=kwargs["turn_id"],
        tool_call_id=kwargs["tool_call_id"],
        tool_name=name,
    ):
        request_current_turn_defer(
            acknowledgement=(
                "Knowledgeジョブ job_20260812T010203Z_0123456789ab を受理しました。"
                "完了時に通知します。"
            ),
            reference_id="job_20260812T010203Z_0123456789ab",
        )
    return '{"ok":true,"status":"deferred"}'


def _run(agent: AIAgent, *, flush_side_effect=True):
    agent.client.chat.completions.create.return_value = _tool_response()
    flush_kwargs = (
        {"return_value": flush_side_effect}
        if isinstance(flush_side_effect, bool)
        else {"side_effect": flush_side_effect}
    )
    with (
        patch("run_agent.handle_function_call", side_effect=_deferred_tool),
        patch.object(
            agent,
            "_flush_messages_to_session_db",
            **flush_kwargs,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("Knowledgeを収集して")


def test_deferred_tool_ends_parent_turn_without_another_model_call():
    agent = _agent()
    try:
        result = _run(agent)
    finally:
        discard_deferred_turn(
            session_id=agent.session_id,
            turn_id=getattr(agent, "_current_turn_id", ""),
        )

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "tool_deferred_turn"
    assert "を受理しました" in result["final_response"]
    roles = [message["role"] for message in result["messages"][-3:]]
    assert roles == ["assistant", "tool", "assistant"]


def test_defer_does_not_claim_durable_success_when_ack_flush_fails():
    agent = _agent()
    try:
        result = _run(agent, flush_side_effect=[True, True, False])
    finally:
        discard_deferred_turn(
            session_id=agent.session_id,
            turn_id=getattr(agent, "_current_turn_id", ""),
        )

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "tool_deferred_ack_persistence_failed"
    assert "受理記録の保存に失敗" in result["final_response"]
    assert result["completed"] is False


def test_defer_skips_later_sibling_side_effect_and_closes_every_call_id():
    agent = _agent()
    agent._current_turn_id = "turn-sibling"
    first = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="knowledge_start", arguments="{}"),
    )
    second = SimpleNamespace(
        id="call-2",
        function=SimpleNamespace(name="terminal", arguments='{"command":"touch forbidden"}'),
    )
    message = SimpleNamespace(content="", tool_calls=[first, second])
    messages = []
    executed = []

    def dispatch(name, args, task_id, **kwargs):
        executed.append(name)
        if name == "knowledge_start":
            return _deferred_tool(name, args, task_id, **kwargs)
        return "SHOULD_NOT_RUN"

    try:
        with (
            patch("run_agent.handle_function_call", side_effect=dispatch),
            patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        ):
            agent._execute_tool_calls(message, messages, "task-1")
    finally:
        discard_deferred_turn(
            session_id=agent.session_id,
            turn_id="turn-sibling",
        )

    assert executed == ["knowledge_start"]
    assert [item["tool_call_id"] for item in messages] == ["call-1", "call-2"]
    assert messages[1]["effect_disposition"] == "none"
    assert "was not started" in messages[1]["content"]


def test_defer_closes_calls_across_segment_boundary_without_dispatching_them():
    agent = _agent()
    agent._current_turn_id = "turn-segmented-sibling"
    first = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="knowledge_start", arguments="{}"),
    )
    second = SimpleNamespace(
        id="call-2",
        function=SimpleNamespace(name="terminal", arguments='{"command":"touch forbidden"}'),
    )
    third = SimpleNamespace(
        id="call-3",
        function=SimpleNamespace(name="terminal", arguments='{"command":"touch also-forbidden"}'),
    )
    message = SimpleNamespace(content="", tool_calls=[first, second, third])
    segments = [
        ("sequential", [first]),
        ("parallel", [second]),
        ("sequential", [third]),
    ]
    messages = []
    executed = []

    def dispatch(name, args, task_id, **kwargs):
        executed.append(name)
        if name == "knowledge_start":
            return _deferred_tool(name, args, task_id, **kwargs)
        return "SHOULD_NOT_RUN"

    try:
        with (
            patch("run_agent.handle_function_call", side_effect=dispatch),
            patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        ):
            execute_tool_calls_segmented(
                agent,
                message,
                messages,
                "task-1",
                segments=segments,
            )
    finally:
        discard_deferred_turn(
            session_id=agent.session_id,
            turn_id="turn-segmented-sibling",
        )

    assert executed == ["knowledge_start"]
    assert [item["tool_call_id"] for item in messages] == [
        "call-1",
        "call-2",
        "call-3",
    ]
    assert all(item.get("effect_disposition") == "none" for item in messages[1:])


def test_failed_deferred_tool_discards_request_and_runs_later_sibling():
    agent = _agent()
    agent._current_turn_id = "turn-failed-defer"
    first = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="knowledge_start", arguments="{}"),
    )
    second = SimpleNamespace(
        id="call-2",
        function=SimpleNamespace(name="terminal", arguments='{"command":"true"}'),
    )
    message = SimpleNamespace(content="", tool_calls=[first, second])
    messages = []
    executed = []

    def dispatch(name, args, task_id, **kwargs):
        executed.append(name)
        if name == "knowledge_start":
            _deferred_tool(name, args, task_id, **kwargs)
            return '{"ok":false,"error":"runner failed"}'
        return "DONE"

    try:
        with (
            patch("run_agent.handle_function_call", side_effect=dispatch),
            patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        ):
            agent._execute_tool_calls(message, messages, "task-1")
        assert peek_deferred_turn(
            session_id=agent.session_id,
            turn_id="turn-failed-defer",
        ) is None
    finally:
        discard_deferred_turn(
            session_id=agent.session_id,
            turn_id="turn-failed-defer",
        )

    assert executed == ["knowledge_start", "terminal"]
    assert [item["tool_call_id"] for item in messages] == ["call-1", "call-2"]


def test_raised_deferred_tool_discards_request():
    agent = _agent()
    agent._current_turn_id = "turn-raised-defer"
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="knowledge_start", arguments="{}"),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    messages = []

    def dispatch(name, args, task_id, **kwargs):
        _deferred_tool(name, args, task_id, **kwargs)
        raise RuntimeError("runner exploded")

    try:
        with (
            patch("run_agent.handle_function_call", side_effect=dispatch),
            patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        ):
            agent._execute_tool_calls(message, messages, "task-1")
        assert peek_deferred_turn(
            session_id=agent.session_id,
            turn_id="turn-raised-defer",
        ) is None
    finally:
        discard_deferred_turn(
            session_id=agent.session_id,
            turn_id="turn-raised-defer",
        )

    assert len(messages) == 1
    assert "Error executing tool" in messages[0]["content"]
