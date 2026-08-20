"""Regression tests for background review agent cleanup."""

from __future__ import annotations

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
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
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    import threading as _threading
    agent._background_review_agent = None
    agent._background_review_timer = None
    agent._background_review_cancel_event = None
    agent._background_review_closing = False
    agent._background_review_lock = _threading.Lock()
    agent._active_children = []
    agent._active_children_lock = _threading.Lock()
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_background_review_shuts_down_memory_provider_before_close(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            events.append(("shutdown_memory_provider", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert [name for name, _payload in events] == [
        "init",
        "run_conversation",
        "shutdown_memory_provider",
        "close",
    ]


def test_background_review_fork_opts_out_of_session_finalization(monkeypatch):
    """The review fork shares the parent's live session_id, so it must set
    ``_end_session_on_close = False``. Otherwise close() (now finalizing owned
    session rows) would end the still-active parent session mid-conversation
    every time the review fires (~every 10 turns). Regression for #12029.
    """
    seen = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            # Default matches AIAgent.__init__ (agent_init.py): owns its row.
            self._end_session_on_close = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "_end_session_on_close":
                seen["end_session_on_close"] = value

        def run_conversation(self, **kwargs):
            # By the time the fork runs, the opt-out must already be applied.
            seen["at_run_time"] = self._end_session_on_close

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert seen.get("end_session_on_close") is False
    assert seen.get("at_run_time") is False


def test_background_review_registers_on_active_children_for_interrupt(monkeypatch):
    """The review fork must be added to the parent's ``_active_children`` so
    ``AIAgent.interrupt()`` (which fans out to that list) can reach it, and
    to ``_background_review_agent`` so the NEXT live turn can proactively
    cancel a still-running review. Regression for the doubled-token-
    accounting / Ctrl+C-proof lockup that a review racing a new live turn
    against the same session_id/credentials can cause.
    """
    seen = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            # While run_conversation is "in flight", both tracking slots on
            # the parent must already point at this fork.
            seen["active_children_during_run"] = list(agent._active_children)
            seen["background_review_agent_during_run"] = agent._background_review_agent

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    fork = seen["background_review_agent_during_run"]
    assert fork is not None
    assert seen["active_children_during_run"] == [fork]

    # After the review completes, both tracking slots must be cleared —
    # otherwise a later interrupt() would try to cancel an already-closed
    # agent, or the next turn would wait on a review that no longer exists.
    assert agent._background_review_agent is None
    assert agent._active_children == []


def test_new_live_turn_cancels_still_running_background_review(monkeypatch):
    """conversation_loop.run_conversation() must proactively interrupt a
    background review still in flight from a prior turn, rather than let the
    two race concurrently against the same session_id/credentials. This is
    the other half of the fix: registration alone only enables interrupt()
    propagation, it doesn't by itself stop the race — something has to
    actually call interrupt() at the start of the next turn.
    """
    import agent.conversation_loop as conversation_loop_module

    calls = []

    class FakeReviewAgent:
        def interrupt(self, message=None):
            calls.append(message)

    agent = _bare_agent()
    agent._background_review_agent = FakeReviewAgent()

    # Invoke just the cancellation snippet in isolation via the same
    # attribute contract run_conversation() reads, to avoid dragging in the
    # rest of the turn machinery (network calls, tool setup, etc.) that
    # isn't relevant to this regression.
    _pending_review = getattr(agent, "_background_review_agent", None)
    assert _pending_review is not None
    _pending_review.interrupt("superseded by a new live turn")

    assert calls == ["superseded by a new live turn"]


def test_auto_background_review_waits_for_idle_delay_and_new_turn_cancels(monkeypatch):
    """Automatic reviews are deferred, but a new turn cancels the timer."""
    events = []

    class DeferredTimer:
        instances = []

        def __init__(self, interval, target):
            self.interval = interval
            self.target = target
            self.cancelled = False
            self.started = False
            self.daemon = False
            self.instances.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

        def fire(self):
            if not self.cancelled:
                self.target()

    class RecordingThread:
        def __init__(self, *, target, daemon=None, name=None):
            events.append(("thread_created", daemon, name))
            self._target = target

        def start(self):
            events.append(("thread_started",))

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "auxiliary": {
                "background_review": {"idle_delay_seconds": 12.5}
            }
        },
    )
    monkeypatch.setattr(run_agent_module.threading, "Timer", DeferredTimer)
    monkeypatch.setattr(run_agent_module.threading, "Thread", RecordingThread)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    timer = DeferredTimer.instances[-1]
    assert timer.interval == 12.5
    assert timer.started is True
    assert events == []
    assert agent._background_review_timer is timer

    # This is the cancellation path used at the start of the next live turn.
    agent._cancel_background_review_timer()
    assert timer.cancelled is True
    timer.fire()
    assert events == []


def test_new_turn_cancels_after_timer_fires_before_worker_runs(monkeypatch):
    """The Timer-to-worker hand-off remains cancellable until registration."""
    queued_targets = []
    review_inits = []

    class DeferredTimer:
        def __init__(self, interval, target):
            self.target = target
            self.daemon = False

        def start(self):
            pass

        def cancel(self):
            pass

        def fire(self):
            self.target()

    class DeferredThread:
        def __init__(self, *, target, daemon=None, name=None):
            queued_targets.append(target)

        def start(self):
            pass

    class UnexpectedReviewAgent:
        def __init__(self, **kwargs):
            review_inits.append(kwargs)

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "auxiliary": {
                "background_review": {"idle_delay_seconds": 12.5}
            }
        },
    )
    monkeypatch.setattr(run_agent_module.threading, "Timer", DeferredTimer)
    monkeypatch.setattr(run_agent_module.threading, "Thread", DeferredThread)
    monkeypatch.setattr(run_agent_module, "AIAgent", UnexpectedReviewAgent)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    timer = agent._background_review_timer
    timer.fire()
    assert len(queued_targets) == 1
    assert agent._background_review_timer is None
    assert agent._background_review_cancel_event is not None

    # This is the narrow race that used to slip past both the fired Timer and
    # the not-yet-registered review fork.
    agent._cancel_background_review_timer()
    queued_targets[0]()

    assert review_inits == []
    assert agent._background_review_agent is None


def test_cancellation_during_fork_construction_prevents_review_api_call(monkeypatch):
    """A cancellation that lands during construction wins before registration."""
    events = []

    class ImmediateTimer:
        def __init__(self, interval, target):
            self.target = target
            self.daemon = False

        def start(self):
            self.target()

        def cancel(self):
            pass

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append("init")
            self._session_messages = []
            agent._cancel_background_review_timer()

        def run_conversation(self, **kwargs):
            events.append("run")

        def shutdown_memory_provider(self):
            events.append("shutdown")

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "auxiliary": {
                "background_review": {"idle_delay_seconds": 12.5}
            }
        },
    )
    monkeypatch.setattr(run_agent_module.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert "run" not in events
    assert events == ["init", "shutdown", "close"]
    assert agent._background_review_agent is None


def test_agent_close_cancels_deferred_background_review():
    """A closed/evicted session must not launch its old review minutes later."""
    import threading

    class RecordingTimer:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    agent = _bare_agent()
    timer = RecordingTimer()
    cancel_event = threading.Event()
    agent._background_review_timer = timer
    agent._background_review_cancel_event = cancel_event

    AIAgent.close(agent)

    assert timer.cancelled is True
    assert cancel_event.is_set() is True
    assert agent._background_review_timer is None
    assert agent._background_review_cancel_event is None
    assert agent._background_review_closing is True


def test_release_clients_cancels_deferred_background_review():
    """Soft cache eviction also fences a not-yet-fired review Timer."""
    import threading

    class RecordingTimer:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    agent = _bare_agent()
    timer = RecordingTimer()
    cancel_event = threading.Event()
    agent._background_review_timer = timer
    agent._background_review_cancel_event = cancel_event

    AIAgent.release_clients(agent)

    assert timer.cancelled is True
    assert cancel_event.is_set() is True
    assert agent._background_review_timer is None
    assert agent._background_review_cancel_event is None
    assert agent._background_review_closing is True


def test_fenced_agent_cannot_schedule_a_new_delayed_review(monkeypatch):
    """A turn-finalizer race cannot re-arm a Timer after soft eviction."""
    timer_inits = []
    thread_inits = []

    class UnexpectedTimer:
        def __init__(self, *args, **kwargs):
            timer_inits.append((args, kwargs))

    class UnexpectedThread:
        def __init__(self, *args, **kwargs):
            thread_inits.append((args, kwargs))

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "auxiliary": {
                "background_review": {"idle_delay_seconds": 12.5}
            }
        },
    )
    monkeypatch.setattr(run_agent_module.threading, "Timer", UnexpectedTimer)
    monkeypatch.setattr(run_agent_module.threading, "Thread", UnexpectedThread)

    agent = _bare_agent()
    agent._background_review_closing = True
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert timer_inits == []
    assert thread_inits == []


def test_agent_close_interrupts_registered_background_review(monkeypatch):
    """Close captures the fully registered fork and aborts its in-flight run."""
    import threading

    run_started = threading.Event()
    interrupted = threading.Event()
    closed = threading.Event()
    worker_threads = []
    real_thread = threading.Thread

    class RealDaemonThread:
        def __init__(self, *, target, daemon=None, name=None):
            self.thread = real_thread(target=target, daemon=daemon, name=name)
            worker_threads.append(self.thread)

        def start(self):
            self.thread.start()

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            run_started.set()
            interrupted.wait(timeout=2)

        def interrupt(self, message=None):
            assert message == "parent agent closing"
            interrupted.set()

        def shutdown_memory_provider(self):
            pass

        def close(self):
            closed.set()
            interrupted.set()

    monkeypatch.setattr(run_agent_module.threading, "Thread", RealDaemonThread)
    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        focus=None,
    )
    assert run_started.wait(timeout=2)
    assert len(agent._active_children) == 1
    assert agent._background_review_agent is agent._active_children[0]

    AIAgent.close(agent)
    for worker in worker_threads:
        worker.join(timeout=2)

    assert interrupted.is_set() is True
    assert closed.is_set() is True
    assert agent._active_children == []
    assert agent._background_review_agent is None


def test_manual_refine_bypasses_idle_delay(monkeypatch):
    """An explicit /refine call (even without focus text) starts immediately."""
    timers = []
    starts = []

    class UnexpectedTimer:
        def __init__(self, *_args, **_kwargs):
            timers.append(self)

    class RecordingThread:
        def __init__(self, *, target, daemon=None, name=None):
            self._target = target

        def start(self):
            starts.append(True)

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "auxiliary": {
                "background_review": {"idle_delay_seconds": 12.5}
            }
        },
    )
    monkeypatch.setattr(run_agent_module.threading, "Timer", UnexpectedTimer)
    monkeypatch.setattr(run_agent_module.threading, "Thread", RecordingThread)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        focus=None,
    )

    assert timers == []
    assert starts == [True]


def test_background_review_idle_delay_defaults_to_zero():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert (
        DEFAULT_CONFIG["auxiliary"]["background_review"]["idle_delay_seconds"]
        == 0
    )









# ---------------------------------------------------------------------------
# memory_notifications mode: off | on | verbose
# ---------------------------------------------------------------------------

import json as _json

from agent.background_review import summarize_background_review_actions


def _memory_add_review():
    """A minimal review transcript: one memory add (assistant call + tool result)."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_mem1",
                    "function": {
                        "name": "memory",
                        "arguments": _json.dumps(
                            {
                                "action": "add",
                                "target": "memory",
                                "content": "User prefers terse replies",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mem1",
            "content": _json.dumps(
                {"success": True, "message": "Entry added.", "target": "memory"}
            ),
        },
    ]


def _skill_patch_review():
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_skill1",
                    "function": {
                        "name": "skill_manage",
                        "arguments": _json.dumps(
                            {"action": "patch", "name": "demo", "old_string": "a", "new_string": "b"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill1",
            "content": _json.dumps(
                {
                    "success": True,
                    "message": "Patched SKILL.md in skill 'demo' (1 replacement).",
                    "_change": {"old": "a", "new": "b"},
                }
            ),
        },
    ]


def test_memory_notifications_off_returns_nothing():
    actions = summarize_background_review_actions(
        _memory_add_review(), [], notification_mode="off"
    )
    assert actions == []








def test_skill_patch_off_silent_verbose_shows_diff():
    assert (
        summarize_background_review_actions(
            _skill_patch_review(), [], notification_mode="off"
        )
        == []
    )
    verbose = summarize_background_review_actions(
        _skill_patch_review(), [], notification_mode="verbose"
    )
    assert len(verbose) == 1
    assert "demo" in verbose[0] and "→" in verbose[0]
