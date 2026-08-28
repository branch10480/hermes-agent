"""Tests for the review fork's iteration budget and tool-loop breaker.

``auxiliary.background_review.max_iterations`` bounds the fork's loop, and
``_tighten_review_guardrails`` arms a strict hard-stop breaker so an
unattended review stuck re-sending a failing tool call gives up after a few
consecutive failures instead of burning the whole budget.
"""

from __future__ import annotations

import threading

import run_agent as run_agent_module
from run_agent import AIAgent
from agent.background_review import (
    _REVIEW_EXACT_FAILURE_BLOCK_AFTER,
    _REVIEW_MAX_ITERATIONS_DEFAULT,
    _REVIEW_SAME_TOOL_FAILURE_HALT_AFTER,
    _review_max_iterations,
    _tighten_review_guardrails,
)
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "discord"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.status_callback = None
    agent._background_review_agent = None
    agent._background_review_timer = None
    agent._background_review_cancel_event = None
    agent._background_review_closing = False
    agent._background_review_lock = threading.Lock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._printed = []
    agent._safe_print = lambda msg, *a, **kw: agent._printed.append(msg)
    agent._delivered = []
    agent.background_review_callback = agent._delivered.append
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


class RecordingReviewAgent:
    init_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self._session_messages = []
        self._tool_guardrails = ToolCallGuardrailController()

    def run_conversation(self, **kwargs):
        pass

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


def _spawn(agent):
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )


# ── max_iterations config ───────────────────────────────────────────────────


def test_review_max_iterations_default_is_16():
    # HERMES_HOME is sandboxed by conftest, so config carries the default.
    assert _review_max_iterations() == _REVIEW_MAX_ITERATIONS_DEFAULT == 16


def test_review_max_iterations_reads_config_and_clamps(monkeypatch):
    import hermes_cli.config as config_module

    def _fake_cfg(value):
        return {"auxiliary": {"background_review": {"max_iterations": value}}}

    monkeypatch.setattr(
        config_module, "load_config_readonly", lambda: _fake_cfg(8)
    )
    assert _review_max_iterations() == 8

    monkeypatch.setattr(
        config_module, "load_config_readonly", lambda: _fake_cfg(0)
    )
    assert _review_max_iterations() == 1

    monkeypatch.setattr(
        config_module, "load_config_readonly", lambda: _fake_cfg(500)
    )
    assert _review_max_iterations() == 100

    monkeypatch.setattr(
        config_module, "load_config_readonly", lambda: _fake_cfg("nonsense")
    )
    assert _review_max_iterations() == _REVIEW_MAX_ITERATIONS_DEFAULT


def test_fork_is_constructed_with_configured_budget(monkeypatch):
    import hermes_cli.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config_readonly",
        lambda: {"auxiliary": {"background_review": {"max_iterations": 5}}},
    )
    monkeypatch.setattr(run_agent_module, "AIAgent", RecordingReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    _spawn(_bare_agent())

    assert RecordingReviewAgent.init_kwargs.get("max_iterations") == 5


# ── tool-loop breaker ───────────────────────────────────────────────────────


def test_tighten_review_guardrails_arms_hard_stop():
    class Fork:
        _tool_guardrails = ToolCallGuardrailController()

    fork = Fork()
    _tighten_review_guardrails(fork)
    cfg = fork._tool_guardrails.config
    assert cfg.hard_stop_enabled is True
    assert cfg.same_tool_failure_halt_after == _REVIEW_SAME_TOOL_FAILURE_HALT_AFTER
    assert cfg.exact_failure_block_after == _REVIEW_EXACT_FAILURE_BLOCK_AFTER


def test_tighten_review_guardrails_keeps_stricter_user_thresholds():
    class Fork:
        _tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig(
                same_tool_failure_halt_after=2,
                exact_failure_block_after=2,
            )
        )

    fork = Fork()
    _tighten_review_guardrails(fork)
    cfg = fork._tool_guardrails.config
    assert cfg.same_tool_failure_halt_after == 2
    assert cfg.exact_failure_block_after == 2


def test_strict_breaker_halts_after_consecutive_failures():
    class Fork:
        _tool_guardrails = ToolCallGuardrailController()

    fork = Fork()
    _tighten_review_guardrails(fork)
    controller = fork._tool_guardrails

    # Varying args (not byte-identical) — the same-tool counter must trip.
    decision = None
    for i in range(_REVIEW_SAME_TOOL_FAILURE_HALT_AFTER):
        decision = controller.after_call(
            "skill_manage",
            {"action": "write_file", "name": f"skill-{i}"},
            '{"success": false, "error": "file_content is required"}',
        )
    assert decision is not None and decision.action == "halt"

    # A success in between resets the streak — no halt.
    controller2 = ToolCallGuardrailController(controller.config)
    for i in range(_REVIEW_SAME_TOOL_FAILURE_HALT_AFTER - 1):
        controller2.after_call(
            "skill_manage", {"n": i}, '{"success": false, "error": "x"}'
        )
    controller2.after_call(
        "skill_manage", {"n": 99}, '{"success": true, "message": "Patched"}'
    )
    late = controller2.after_call(
        "skill_manage", {"n": 100}, '{"success": false, "error": "x"}'
    )
    assert late.action != "halt"


def test_spawn_flow_tightens_fork_guardrails(monkeypatch):
    class GuardrailProbeAgent(RecordingReviewAgent):
        seen_config = None

        def run_conversation(self, **kwargs):
            type(self).seen_config = self._tool_guardrails.config

    monkeypatch.setattr(run_agent_module, "AIAgent", GuardrailProbeAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    _spawn(_bare_agent())

    cfg = GuardrailProbeAgent.seen_config
    assert cfg is not None
    assert cfg.hard_stop_enabled is True
    assert cfg.same_tool_failure_halt_after == _REVIEW_SAME_TOOL_FAILURE_HALT_AFTER


def test_halt_notice_posted_when_run_notices_enabled(monkeypatch):
    class HaltingReviewAgent(RecordingReviewAgent):
        def run_conversation(self, **kwargs):
            self._tool_guardrail_halt_decision = object()

    monkeypatch.setattr(run_agent_module, "AIAgent", HaltingReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.background_review_run_notifications = True

    _spawn(agent)

    assert any(
        "Self-improvement review aborted — repeated tool failures" in m
        for m in agent._delivered
    )
    assert not any("finished — nothing saved" in m for m in agent._delivered)
