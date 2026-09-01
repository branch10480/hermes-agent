"""Regression tests for background review agent cleanup."""

from __future__ import annotations

import threading

import run_agent as run_agent_module
from run_agent import AIAgent


_REAL_THREAD = threading.Thread


class _TurnBoundaryReached(Exception):
    """Stop a live turn exactly when it reaches turn-context construction."""


class CapturingThread:
    targets = []

    def __init__(self, *, target, daemon=None, name=None):
        self.targets.append(target)

    def start(self):
        pass


class ObservedEvent:
    """A real Event that also exposes when a waiter starts waiting."""

    def __init__(self):
        self._event = threading.Event()
        self.wait_started = threading.Event()
        self.set_calls = 0

    def set(self):
        self.set_calls += 1
        self._event.set()

    def wait(self, timeout=None):
        self.wait_started.set()
        return self._event.wait(timeout)

    def is_set(self):
        return self._event.is_set()


class FakeReviewAgent:
    def __init__(self, **kwargs):
        self._session_messages = []

    def run_conversation(self, **kwargs):
        pass

    def interrupt(self, message=None):
        pass

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


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
    agent._background_review_run = None
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


def _install_live_turn_boundary(monkeypatch, on_boundary=None):
    import agent.conversation_loop as conversation_loop_module

    def stop_at_boundary(*args, **kwargs):
        if on_boundary is not None:
            on_boundary()
        raise _TurnBoundaryReached

    monkeypatch.setattr(
        conversation_loop_module,
        "build_turn_context",
        stop_at_boundary,
    )


def _run_wrapped_live_turn_to_boundary(agent, result):
    try:
        result["return"] = AIAgent.run_conversation(
            agent,
            "next turn",
            task_id="live-task",
        )
    except _TurnBoundaryReached:
        result["boundary_reached"] = True
    except BaseException as exc:  # surfaced in the test thread for a useful failure
        result["error"] = exc


def _install_relay_recorder(monkeypatch, review_run=None):
    from agent import relay_runtime
    from hermes_cli.observability import relay_shared_metrics

    calls = []

    def review_acknowledged():
        return bool(review_run and review_run.request_done.is_set())

    class RelayTurn:
        relay_enabled = True

    class RecordingCoordinator:
        def acquire_conversation(self, **kwargs):
            calls.append(("acquire", review_acknowledged()))
            return object()

        def begin_turn(self, lease, **kwargs):
            calls.append(("begin", review_acknowledged()))
            return RelayTurn()

        def finish_logical_calls(self, turn, **kwargs):
            pass

        def end_turn(self, turn, **kwargs):
            pass

        def release_conversation(self, lease):
            pass

    monkeypatch.setattr(
        relay_runtime,
        "SESSION_COORDINATOR",
        RecordingCoordinator(),
    )
    monkeypatch.setattr(
        relay_runtime,
        "current_profile_key",
        lambda: "/test-profile",
    )
    monkeypatch.setattr(
        relay_shared_metrics,
        "start_task_run",
        lambda **kwargs: calls.append(
            ("start_task_run", review_acknowledged())
        ),
    )
    monkeypatch.setattr(
        relay_shared_metrics,
        "finish_task_run",
        lambda **kwargs: None,
    )
    return calls


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


def test_background_review_skipped_in_delegation_subagent(monkeypatch):
    """The automatic post-turn review must NOT fire inside a delegation
    subagent (``_delegate_depth > 0``).

    Regression for #85859: the fork inherits the subagent's live model, so in
    a delegation subagent running a premium model it replayed the whole
    conversation at premium rates. Subagents are already barred from writing
    shared MEMORY.md, so there is nothing for the review to persist here.
    """
    forks = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            forks.append(kwargs)

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent._delegate_depth = 1  # this agent IS a delegation subagent

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        review_skills=True,
    )

    assert forks == [], "no review fork should be spawned inside a subagent"


def test_background_review_runs_at_top_level(monkeypatch):
    """Sibling guard for the subagent skip: at ``_delegate_depth == 0`` the
    review still fires exactly as before (the cost guard is subagent-only)."""
    forks = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            forks.append(kwargs)

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent._delegate_depth = 0  # top-level agent

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert len(forks) == 1, "top-level review must still spawn the fork"


def test_background_review_disabled_skips_automatic_spawn(monkeypatch):
    """``auxiliary.background_review.enabled: false`` must skip automatic
    post-turn forks while leaving ``/refine`` (focus set) working (#87250)."""
    from unittest.mock import patch

    forks = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            forks.append(kwargs)

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent._delegate_depth = 0
    cfg = {"auxiliary": {"background_review": {"enabled": False}}}

    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )
        assert forks == [], "automatic review must not spawn when disabled"

        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
            focus="save the deploy workflow",
        )
        assert len(forks) == 1, "/refine must still run when enabled=false"


def test_background_review_explicit_focus_runs_even_in_subagent(monkeypatch):
    """An explicit ``/refine`` (``focus`` set) is a deliberate user request and
    is honored regardless of depth — only the automatic post-turn review is
    suppressed in subagents."""
    forks = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            forks.append(kwargs)

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent._delegate_depth = 2

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_skills=True,
        focus="save the deploy workflow as a skill",
    )

    assert len(forks) == 1, "explicit focus review must run even in a subagent"


def test_background_review_registers_before_start_runs_and_cleans_up(monkeypatch):
    """The parent must own a unique review run before the worker can start."""
    seen = {}

    class RecordingReviewAgent(FakeReviewAgent):
        def run_conversation(self, **kwargs):
            seen["run"] = agent._background_review_run
            seen["active_children_during_run"] = list(agent._active_children)
            seen["background_review_agent_during_run"] = agent._background_review_agent

    monkeypatch.setattr(run_agent_module, "AIAgent", RecordingReviewAgent)
    CapturingThread.targets = []
    monkeypatch.setattr(run_agent_module.threading, "Thread", CapturingThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    run = agent._background_review_run
    assert run is not None
    assert len(CapturingThread.targets) == 1
    assert not run.request_done.is_set()

    observed_done = ObservedEvent()
    run.request_done = observed_done
    CapturingThread.targets[0]()

    fork = seen["background_review_agent_during_run"]
    assert fork is not None
    assert seen["run"] is run
    assert seen["active_children_during_run"] == [fork]
    assert observed_done.is_set()
    assert observed_done.set_calls == 1
    assert agent._background_review_run is None
    assert agent._background_review_agent is None
    assert agent._active_children == []


def test_live_turn_waits_for_review_exit_before_relay_and_turn_context(monkeypatch):
    """The outer production wrapper waits before same-session instrumentation."""
    review_entered = threading.Event()
    review_returned = threading.Event()
    allow_review_return = threading.Event()
    interrupted = threading.Event()
    boundary_reached = threading.Event()
    seen = {}

    class BlockingReviewAgent(FakeReviewAgent):
        def interrupt(self, message=None):
            seen["interrupt_message"] = message
            interrupted.set()

        def run_conversation(self, **kwargs):
            review_entered.set()
            assert allow_review_return.wait(2.0)
            review_returned.set()

    monkeypatch.setattr(run_agent_module, "AIAgent", BlockingReviewAgent)
    CapturingThread.targets = []
    monkeypatch.setattr(run_agent_module.threading, "Thread", CapturingThread)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )
    run = agent._background_review_run
    assert run is not None
    observed_done = ObservedEvent()
    run.request_done = observed_done

    monkeypatch.setattr(run_agent_module.threading, "Thread", _REAL_THREAD)
    worker = _REAL_THREAD(target=CapturingThread.targets[0], daemon=True)
    worker.start()
    assert review_entered.wait(2.0)

    def on_boundary():
        seen["review_returned_at_boundary"] = review_returned.is_set()
        boundary_reached.set()

    _install_live_turn_boundary(monkeypatch, on_boundary)
    relay_calls = _install_relay_recorder(monkeypatch, run)
    live_result = {}
    live = _REAL_THREAD(
        target=_run_wrapped_live_turn_to_boundary,
        args=(agent, live_result),
        daemon=True,
    )
    live.start()

    assert interrupted.wait(2.0)
    wait_started = observed_done.wait_started.wait(2.0)
    relay_calls_before_ack = list(relay_calls)
    allow_review_return.set()
    worker.join(timeout=2.0)
    live.join(timeout=2.0)

    assert not worker.is_alive()
    assert not live.is_alive()
    assert wait_started
    assert relay_calls_before_ack == []
    assert relay_calls == [
        ("acquire", True),
        ("begin", True),
        ("start_task_run", True),
    ]
    assert boundary_reached.is_set()
    assert seen["interrupt_message"] == "superseded by a new live turn"
    assert seen["review_returned_at_boundary"] is True
    assert live_result == {"boundary_reached": True}


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


def test_live_turn_cancels_review_during_startup_before_provider(monkeypatch):
    """A review cancelled before its worker runs must never call its provider."""
    provider_calls = []
    boundary_reached = threading.Event()

    class RecordingReviewAgent(FakeReviewAgent):
        def run_conversation(self, **kwargs):
            provider_calls.append(kwargs)

    monkeypatch.setattr(run_agent_module, "AIAgent", RecordingReviewAgent)
    CapturingThread.targets = []
    monkeypatch.setattr(run_agent_module.threading, "Thread", CapturingThread)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )
    run = agent._background_review_run
    assert run is not None

    _install_live_turn_boundary(monkeypatch, boundary_reached.set)
    relay_calls = _install_relay_recorder(monkeypatch, run)
    live_result = {}
    live = _REAL_THREAD(
        target=_run_wrapped_live_turn_to_boundary,
        args=(agent, live_result),
        daemon=True,
    )
    live.start()
    assert run.cancel_requested.wait(2.0)

    worker = _REAL_THREAD(target=CapturingThread.targets[0], daemon=True)
    worker.start()
    worker.join(timeout=2.0)
    live.join(timeout=2.0)

    assert not worker.is_alive()
    assert not live.is_alive()
    assert boundary_reached.is_set()
    assert provider_calls == []
    assert run.request_done.is_set()
    assert relay_calls == [
        ("acquire", True),
        ("begin", True),
        ("start_task_run", True),
    ]
    assert live_result == {"boundary_reached": True}


def test_live_turn_proceeds_when_review_acknowledgement_times_out(monkeypatch):
    """A broken review abort path must not block the foreground indefinitely.
    The live turn proceeds after the bounded wait, retaining foreground priority.
    """
    import time

    import agent.background_review as background_review_module

    review_entered = threading.Event()
    interrupt_entered = threading.Event()
    interrupt_returned = threading.Event()
    allow_interrupt_return = threading.Event()
    allow_review_return = threading.Event()

    class WedgedReviewAgent(FakeReviewAgent):
        def run_conversation(self, **kwargs):
            review_entered.set()
            allow_review_return.wait(5.0)

        def interrupt(self, message=None):
            interrupt_entered.set()
            allow_interrupt_return.wait(5.0)
            interrupt_returned.set()

    monkeypatch.setattr(run_agent_module, "AIAgent", WedgedReviewAgent)
    CapturingThread.targets = []
    monkeypatch.setattr(run_agent_module.threading, "Thread", CapturingThread)
    monkeypatch.setattr(
        background_review_module,
        "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )
    run = agent._background_review_run
    assert run is not None
    monkeypatch.setattr(run_agent_module.threading, "Thread", _REAL_THREAD)
    worker = _REAL_THREAD(target=CapturingThread.targets[0], daemon=True)
    worker.start()
    assert review_entered.wait(2.0)

    boundary_calls = []
    _install_live_turn_boundary(
        monkeypatch, lambda: boundary_calls.append(True)
    )
    relay_calls = _install_relay_recorder(monkeypatch, run)

    started = time.monotonic()
    live_result = {}
    live = _REAL_THREAD(
        target=_run_wrapped_live_turn_to_boundary,
        args=(agent, live_result),
        daemon=True,
    )
    live.start()
    live.join(timeout=5.0)

    elapsed = time.monotonic() - started

    allow_interrupt_return.set()
    allow_review_return.set()
    worker.join(timeout=2.0)

    assert elapsed < 2.0
    assert interrupt_entered.is_set()
    assert interrupt_returned.wait(2.0)
    assert not worker.is_alive()
    assert not live.is_alive()
    # Foreground retains priority: Relay/turn-context proceed even though
    # the review did not acknowledge within the bounded deadline.
    assert boundary_calls == [True]
    assert live_result == {"boundary_reached": True}
    assert relay_calls == [
        ("acquire", False),
        ("begin", False),
        ("start_task_run", False),
    ]
    assert agent.session_id == "test-session"


def test_live_turn_interrupts_legacy_review_but_keeps_foreground_priority(monkeypatch):
    """Legacy stubs are interrupted without turning review into a user blocker."""
    interrupts = []
    interrupt_called = threading.Event()

    class LegacyReviewAgent:
        def interrupt(self, message=None):
            interrupts.append(message)
            interrupt_called.set()

    agent = _bare_agent()
    del agent._background_review_run
    agent._background_review_agent = LegacyReviewAgent()
    boundary_calls = []
    _install_live_turn_boundary(
        monkeypatch, lambda: boundary_calls.append(True)
    )
    relay_calls = _install_relay_recorder(monkeypatch)

    live_result = {}
    live = _REAL_THREAD(
        target=_run_wrapped_live_turn_to_boundary,
        args=(agent, live_result),
        daemon=True,
    )
    live.start()
    live.join(timeout=5.0)

    assert interrupt_called.wait(2.0)
    assert interrupts == ["superseded by a new live turn"]
    assert not live.is_alive()
    assert boundary_calls == [True]
    assert live_result == {"boundary_reached": True}
    assert relay_calls == [
        ("acquire", False),
        ("begin", False),
        ("start_task_run", False),
    ]
    assert agent.session_id == "test-session"


def test_stale_review_cleanup_cannot_clear_or_signal_newer_review(monkeypatch):
    """A retired worker's late cleanup must be scoped to its own run identity."""
    first_cleanup_entered = threading.Event()
    allow_first_cleanup = threading.Event()
    instance_count = 0

    class BlockingCleanupReviewAgent(FakeReviewAgent):
        def __init__(self, **kwargs):
            nonlocal instance_count
            super().__init__(**kwargs)
            self.index = instance_count
            instance_count += 1

        def shutdown_memory_provider(self):
            if self.index == 0:
                first_cleanup_entered.set()
                assert allow_first_cleanup.wait(2.0)

    monkeypatch.setattr(run_agent_module, "AIAgent", BlockingCleanupReviewAgent)
    CapturingThread.targets = []
    monkeypatch.setattr(run_agent_module.threading, "Thread", CapturingThread)

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "first"}],
        review_memory=True,
    )
    first_run = agent._background_review_run
    first_worker = _REAL_THREAD(target=CapturingThread.targets[0], daemon=True)
    first_worker.start()
    assert first_cleanup_entered.wait(2.0)
    assert first_run.request_done.is_set()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "second"}],
        review_memory=True,
    )
    second_run = agent._background_review_run
    second_target = CapturingThread.targets[1]
    assert second_run is not first_run
    assert not second_run.request_done.is_set()

    allow_first_cleanup.set()
    first_worker.join(timeout=2.0)

    assert not first_worker.is_alive()
    assert agent._background_review_run is second_run
    assert not second_run.request_done.is_set()

    second_target()
    assert second_run.request_done.is_set()
    assert agent._background_review_run is None

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
