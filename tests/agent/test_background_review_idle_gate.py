"""Global idle gate for background review (H-034 / CR-008).

Observed failure this locks down: at 22:40 a skill-library review fired for a
session that had just finished, went to the same single local backend another
Discord session was still using, and both sides started failing with 503. The
review only ever watched its OWN agent — it neither waited for another
session's turn nor stopped when one started.

So the two behaviors under test are cross-session by construction: a review
owned by session 1 must not start while session 2 is mid-turn, and a review
already running for session 1 must be cancelled when session 2 begins a turn.
"""

from __future__ import annotations

import threading

import pytest

import run_agent as run_agent_module
from agent import live_turn_registry
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _clean_registry():
    live_turn_registry.reset_for_tests()
    yield
    live_turn_registry.reset_for_tests()


class _StubAgent:
    """Minimum surface begin_turn/end_turn and the review spawner read."""

    def __init__(self, session_id: str, platform: str = "discord"):
        self.session_id = session_id
        self.platform = platform


def _review_owner(session_id: str = "session-1") -> AIAgent:
    """A bare AIAgent shaped like the parent that spawns a review fork."""
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "discord"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = session_id
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "cached"
    import datetime as _dt

    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_a, **_kw: None
    agent._background_review_agent = None
    agent._background_review_timer = None
    agent._background_review_cancel_event = None
    agent._background_review_closing = False
    agent._background_review_lock = threading.Lock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._live_turn_tokens = set()
    agent._is_background_review_fork = False
    return agent


class _ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


class _RecordingTimer:
    """Collects re-arms instead of sleeping; ``fire()`` drives them by hand."""

    instances = []

    def __init__(self, interval, target):
        self.interval = interval
        self.target = target
        self.cancelled = False
        self.started = False
        self.daemon = False
        _RecordingTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.target()


def _use_recording_timer(monkeypatch):
    _RecordingTimer.instances = []
    monkeypatch.setattr(run_agent_module.threading, "Timer", _RecordingTimer)


def _gate_config(monkeypatch, **overrides):
    task = {"idle_delay_seconds": 0}
    task.update(overrides)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"background_review": task}},
    )


# ---------------------------------------------------------------------------
# The gate: another session's turn blocks the start
# ---------------------------------------------------------------------------

class TestIdleGateBlocksStart:
    def test_other_agent_busy_defers_the_review_instead_of_starting_it(
        self, monkeypatch
    ):
        """Session 2 mid-turn must keep session 1's review off the backend.

        The pre-gate code started immediately here — that is the 22:40
        priority inversion.
        """
        started = []
        _gate_config(monkeypatch)
        _use_recording_timer(monkeypatch)
        monkeypatch.setattr(
            run_agent_module.threading,
            "Thread",
            lambda **kw: started.append(kw) or _ImmediateThread(**kw),
        )

        other = _StubAgent("session-2")
        live_turn_registry.begin_turn(other, kind=live_turn_registry.TURN_KIND_LIVE)

        owner = _review_owner()
        AIAgent._spawn_background_review(
            owner,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )

        assert started == []
        assert _RecordingTimer.instances, "a busy backend must re-arm the timer"
        assert _RecordingTimer.instances[-1].interval == 5.0

    def test_review_starts_once_the_other_session_finishes(self, monkeypatch):
        """The deferred review is not lost — the next poll starts it."""
        started = []
        _gate_config(monkeypatch)
        _use_recording_timer(monkeypatch)
        monkeypatch.setattr(
            run_agent_module.threading,
            "Thread",
            lambda **kw: started.append(kw) or _ImmediateThread(target=lambda: None),
        )

        other = _StubAgent("session-2")
        token = live_turn_registry.begin_turn(other)

        owner = _review_owner()
        AIAgent._spawn_background_review(
            owner,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )
        assert started == []

        # Still busy on the first poll.
        _RecordingTimer.instances[-1].fire()
        assert started == []
        assert len(_RecordingTimer.instances) == 2

        live_turn_registry.end_turn(other, token)
        _RecordingTimer.instances[-1].fire()
        assert len(started) == 1
        assert started[0]["name"] == "bg-review"

    def test_owning_turn_does_not_block_its_own_review(self, monkeypatch):
        """The review is spawned from inside its own turn's finalizer.

        That turn is still registered at spawn time, so the gate must exclude
        it — otherwise no automatic review could ever start.
        """
        started = []
        _gate_config(monkeypatch)
        _use_recording_timer(monkeypatch)
        monkeypatch.setattr(
            run_agent_module.threading,
            "Thread",
            lambda **kw: started.append(kw) or _ImmediateThread(target=lambda: None),
        )

        owner = _review_owner()
        live_turn_registry.begin_turn(owner)

        AIAgent._spawn_background_review(
            owner,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )

        assert len(started) == 1
        assert _RecordingTimer.instances == []

    def test_manual_refine_is_never_idle_gated(self, monkeypatch):
        """/refine is a request the user is waiting on, not housekeeping."""
        started = []
        _gate_config(monkeypatch)
        _use_recording_timer(monkeypatch)
        monkeypatch.setattr(
            run_agent_module.threading,
            "Thread",
            lambda **kw: started.append(kw) or _ImmediateThread(target=lambda: None),
        )

        live_turn_registry.begin_turn(_StubAgent("session-2"))

        owner = _review_owner()
        AIAgent._spawn_background_review(
            owner,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
            focus=None,
        )

        assert len(started) == 1
        assert _RecordingTimer.instances == []

    def test_gate_can_be_disabled(self, monkeypatch):
        started = []
        _gate_config(monkeypatch, idle_gate=False)
        _use_recording_timer(monkeypatch)
        monkeypatch.setattr(
            run_agent_module.threading,
            "Thread",
            lambda **kw: started.append(kw) or _ImmediateThread(target=lambda: None),
        )

        live_turn_registry.begin_turn(_StubAgent("session-2"))

        owner = _review_owner()
        AIAgent._spawn_background_review(
            owner,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )

        assert len(started) == 1

    def test_review_gives_up_after_the_max_wait(self, monkeypatch):
        """A permanently busy backend must not leave a timer re-arming forever."""
        started = []
        _gate_config(monkeypatch, idle_gate_max_wait_seconds=60)
        _use_recording_timer(monkeypatch)
        monkeypatch.setattr(
            run_agent_module.threading,
            "Thread",
            lambda **kw: started.append(kw) or _ImmediateThread(target=lambda: None),
        )
        clock = [1000.0]
        monkeypatch.setattr(run_agent_module.time, "monotonic", lambda: clock[0])

        live_turn_registry.begin_turn(_StubAgent("session-2"))

        owner = _review_owner()
        AIAgent._spawn_background_review(
            owner,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_memory=True,
        )
        assert len(_RecordingTimer.instances) == 1

        # One poll inside the window re-arms; the next one is past the deadline.
        clock[0] += 30.0
        _RecordingTimer.instances[-1].fire()
        assert len(_RecordingTimer.instances) == 2

        clock[0] += 60.0
        _RecordingTimer.instances[-1].fire()
        assert started == []
        assert len(_RecordingTimer.instances) == 2, "must stop re-arming"
        assert owner._background_review_timer is None
        assert owner._background_review_cancel_event is None

    def test_worker_rechecks_the_gate_before_building_the_fork(self, monkeypatch):
        """Register-then-recheck closes the spawn-to-first-call window.

        A live turn that starts after the caller's gate check but before the
        fork exists has nothing to interrupt yet, so the worker itself must
        look again.
        """
        _gate_config(monkeypatch)
        review_inits = []

        class _UnexpectedReviewAgent:
            def __init__(self, **kwargs):
                review_inits.append(kwargs)

        monkeypatch.setattr(run_agent_module, "AIAgent", _UnexpectedReviewAgent)

        from agent.background_review import spawn_background_review_thread

        owner = _review_owner()
        target, _prompt = spawn_background_review_thread(
            owner,
            [{"role": "user", "content": "hi"}],
            review_memory=True,
            cancellation_event=threading.Event(),
            idle_gated=True,
        )

        live_turn_registry.begin_turn(_StubAgent("session-2"))
        target()

        assert review_inits == []


# ---------------------------------------------------------------------------
# Cross-session cancellation of a review already running
# ---------------------------------------------------------------------------

class TestCrossSessionCancellation:
    def test_live_turn_in_another_session_interrupts_a_running_review(
        self, monkeypatch
    ):
        """The 22:45 half of H-034: session 1's review was still on the backend
        when session 2's live turn started, and only session 1's own next turn
        would have stopped it. Any session's turn must now reach it."""
        interrupts = []

        class _RunningReviewFork:
            def interrupt(self, message=None):
                interrupts.append(message)

        cancel_event = threading.Event()
        handle = live_turn_registry.register_background_review(
            cancellation_event=cancel_event, session="owner"
        )
        assert handle.attach_agent(_RunningReviewFork()) is True

        import agent.conversation_loop as conversation_loop

        core_calls = []
        monkeypatch.setattr(
            conversation_loop,
            "_run_conversation_core",
            lambda *a, **kw: core_calls.append(a[0]) or {"final_response": "ok"},
        )

        # A completely unrelated session starts a turn.
        other = _StubAgent("session-2")
        result = conversation_loop.run_conversation(other, "hello")

        assert result == {"final_response": "ok"}
        assert core_calls == [other]
        assert interrupts == ["superseded by a new live turn"]
        assert cancel_event.is_set() is True
        assert handle.cancelled is True

    def test_review_fork_turn_does_not_cancel_reviews(self, monkeypatch):
        """The fork runs its own run_conversation. Cancelling there would make
        every review kill itself the moment it started."""
        interrupts = []

        class _RunningReviewFork:
            def interrupt(self, message=None):
                interrupts.append(message)

        handle = live_turn_registry.register_background_review(session="owner")
        handle.attach_agent(_RunningReviewFork())

        import agent.conversation_loop as conversation_loop

        monkeypatch.setattr(
            conversation_loop,
            "_run_conversation_core",
            lambda *a, **kw: {"final_response": "ok"},
        )

        fork = _StubAgent("session-1")
        fork._is_background_review_fork = True
        conversation_loop.run_conversation(fork, "review the conversation")

        assert interrupts == []
        assert handle.cancelled is False

    def test_turn_registration_is_released_even_when_the_turn_raises(
        self, monkeypatch
    ):
        """A turn that leaks its registration would wedge the gate closed for
        every later review in the process."""
        import agent.conversation_loop as conversation_loop

        def _boom(*_a, **_kw):
            raise RuntimeError("turn exploded")

        monkeypatch.setattr(conversation_loop, "_run_conversation_core", _boom)

        agent = _StubAgent("session-2")
        with pytest.raises(RuntimeError):
            conversation_loop.run_conversation(agent, "hello")

        assert live_turn_registry.backend_busy_reason() is None
        assert agent._live_turn_tokens == set()

    def test_review_fork_turn_still_holds_the_gate_closed(self, monkeypatch):
        """A running review occupies the backend, so a second review must wait
        even though the first one is not a live turn."""
        import agent.conversation_loop as conversation_loop

        seen = {}

        def _core(agent_arg, *_a, **_kw):
            seen["busy"] = live_turn_registry.backend_busy_reason()
            return {"final_response": "ok"}

        monkeypatch.setattr(conversation_loop, "_run_conversation_core", _core)

        fork = _StubAgent("session-1")
        fork._is_background_review_fork = True
        conversation_loop.run_conversation(fork, "review")

        assert seen["busy"] is not None
        assert "maintenance" in seen["busy"]


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

class TestRegistryContract:
    def test_busy_reason_never_contains_the_raw_session_id(self):
        """Gateway session keys embed Discord channel/thread/participant IDs
        and must not reach logs (H-014)."""
        secret = "discord:9876543210:thread-12345:user-777"
        live_turn_registry.begin_turn(_StubAgent(secret))

        reason = live_turn_registry.backend_busy_reason()
        assert reason is not None
        for fragment in ("9876543210", "thread-12345", "user-777", secret):
            assert fragment not in reason

    def test_attach_after_cancel_is_refused(self):
        """A cancel landing mid-construction must not be forgotten once the
        fork finally exists."""
        handle = live_turn_registry.register_background_review(session="owner")
        handle.cancel("superseded by a new live turn")

        assert handle.attach_agent(object()) is False
        assert handle.cancelled is True

    def test_unregister_is_idempotent(self):
        handle = live_turn_registry.register_background_review(session="owner")
        live_turn_registry.unregister_background_review(handle)
        live_turn_registry.unregister_background_review(handle)
        live_turn_registry.unregister_background_review(None)
        assert live_turn_registry.active_background_reviews() == []

    def test_cancel_reaches_every_session(self):
        handles = [
            live_turn_registry.register_background_review(session=f"s{i}")
            for i in range(3)
        ]
        assert live_turn_registry.cancel_background_reviews("stop") == 3
        assert all(h.cancelled for h in handles)

    def test_idle_gate_settings_fall_back_on_junk_values(self, monkeypatch):
        _gate_config(
            monkeypatch,
            idle_gate_poll_seconds="not-a-number",
            idle_gate_max_wait_seconds=None,
        )
        enabled, poll, max_wait = live_turn_registry.idle_gate_settings()
        assert enabled is True
        assert poll == 5.0
        assert max_wait == 300.0
