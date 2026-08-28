"""Tests for background review run-visibility notices.

``display.background_review_run_notifications`` (or the per-agent attr pin)
posts lifecycle notices — started / finished-with-nothing-saved / failed — so
the user can tell WHEN the self-improvement review ran even on passes that
write nothing. Default off: only the existing action summary is shown.
"""

from __future__ import annotations

import json
import threading

import run_agent as run_agent_module
from run_agent import AIAgent
from agent.background_review import _run_notifications_enabled


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


class QuietReviewAgent:
    """Review fork that completes without saving anything."""

    def __init__(self, **kwargs):
        self._session_messages = []

    def run_conversation(self, **kwargs):
        pass

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


class SavingReviewAgent(QuietReviewAgent):
    """Review fork that produces one successful memory action."""

    def run_conversation(self, **kwargs):
        self._session_messages = [
            {
                "role": "tool",
                "tool_call_id": "call_new",
                "content": json.dumps(
                    {"success": True, "message": "Memory entry created."}
                ),
            }
        ]


def _spawn(agent):
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )


def test_run_notices_posted_when_enabled_and_nothing_saved(monkeypatch):
    monkeypatch.setattr(run_agent_module, "AIAgent", QuietReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.background_review_run_notifications = True

    _spawn(agent)

    assert any("Self-improvement review started" in m for m in agent._delivered)
    assert any(
        "Self-improvement review finished — nothing saved" in m
        for m in agent._delivered
    )
    # The stdout mirror carries the same notices.
    assert any("Self-improvement review started" in m for m in agent._printed)


def test_run_notices_absent_when_disabled(monkeypatch):
    monkeypatch.setattr(run_agent_module, "AIAgent", QuietReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.background_review_run_notifications = False

    _spawn(agent)

    assert agent._delivered == []
    assert agent._printed == []


def test_summary_gains_duration_suffix_when_enabled(monkeypatch):
    monkeypatch.setattr(run_agent_module, "AIAgent", SavingReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.background_review_run_notifications = True

    _spawn(agent)

    summaries = [m for m in agent._delivered if m.startswith("💾")]
    assert len(summaries) == 1
    assert "Memory entry created." in summaries[0]
    assert summaries[0].rstrip().endswith("s)")
    # No separate "finished" notice when the summary itself was posted.
    assert not any("finished — nothing saved" in m for m in agent._delivered)


def test_summary_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr(run_agent_module, "AIAgent", SavingReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.background_review_run_notifications = False

    _spawn(agent)

    assert agent._delivered == ["💾 Self-improvement review: Memory entry created."]


def test_failure_notice_posted_when_enabled(monkeypatch):
    class ExplodingReviewAgent(QuietReviewAgent):
        def run_conversation(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(run_agent_module, "AIAgent", ExplodingReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.background_review_run_notifications = True
    agent._emit_auxiliary_failure = lambda *a, **kw: None

    _spawn(agent)

    assert any("Self-improvement review failed" in m for m in agent._delivered)


def test_config_default_is_off_with_attr_unset():
    agent = _bare_agent()
    agent.background_review_run_notifications = None
    # HERMES_HOME is sandboxed by conftest, so config carries the default.
    assert _run_notifications_enabled(agent) is False
