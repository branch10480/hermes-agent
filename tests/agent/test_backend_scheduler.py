"""Backend admission control (H-032 / H-033, CR-008).

The failure these lock down: one local backend, several Discord sessions plus
cron plus the background-review fork, and nothing deciding who goes next. All
of them submitted at once, the backend arbitrated by rejecting (12 × ``503
worker inference queue timed out`` in one window), and the fixed retry backoff
either re-submitted into a still-busy backend or slept through the moment it
went idle. Once, a background review took the slot ahead of a user who was
already waiting for it.

So the properties under test are the ones that failure violated:

* a live turn is admitted ahead of queued maintenance, whatever the order they
  arrived in;
* two live turns keep their arrival order, so a session's wait only shrinks;
* the next waiter starts on the *release* of the previous call, not on a poll
  tick or a backoff timer;
* nothing the scheduler can do fails a turn — a disengaged backend, an
  interrupt, or a wait past its deadline all end up submitting.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from agent import backend_scheduler, live_turn_registry


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_scheduler():
    backend_scheduler.reset_for_tests()
    live_turn_registry.reset_for_tests()
    yield
    backend_scheduler.reset_for_tests()
    live_turn_registry.reset_for_tests()


@pytest.fixture
def scheduler(monkeypatch):
    """Install a forced-on scheduler; returns a mutator for its settings."""

    def _apply(**overrides):
        cfg = backend_scheduler.SchedulerSettings(
            mode="on",
            max_concurrent_requests=1,
            queue_wait_seconds=10.0,
            poll_seconds=0.02,
            notify_after_seconds=0,
        )
        cfg = dataclasses.replace(cfg, **overrides)
        monkeypatch.setattr(backend_scheduler, "settings", lambda: cfg)
        return cfg

    _apply()
    return _apply


class _StubAgent:
    """The surface the scheduler reads: endpoint, session, maintenance flag."""

    def __init__(
        self,
        session_id: str = "session-1",
        *,
        maintenance: bool = False,
        base_url: str = "http://127.0.0.1:18088/v1",
    ):
        self.session_id = session_id
        self.platform = "discord"
        self.base_url = base_url
        self._is_background_review_fork = maintenance


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _queued_priorities() -> list:
    return [entry["priority"] for entry in backend_scheduler.snapshot()["waiting"]]


class _Claimant:
    """A thread that queues for the backend and records when it got in."""

    def __init__(self, agent, *, log: list, name: str, **acquire_kwargs):
        self.agent = agent
        self.name = name
        self.log = log
        self.ticket = None
        self.granted_at = None
        self.acquire_kwargs = acquire_kwargs
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.ticket = backend_scheduler.acquire(self.agent, **self.acquire_kwargs)
        self.granted_at = time.monotonic()
        self.log.append(self.name)

    def start(self):
        self.thread.start()
        return self

    def join(self, timeout: float = 5.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), f"{self.name} never finished acquiring"

    def release(self):
        backend_scheduler.release(self.ticket)


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------

def test_disengaged_scheduler_hands_back_no_ticket(scheduler):
    """mode=off is the pre-scheduler world: submit, do not queue."""
    scheduler(mode="off")
    assert backend_scheduler.acquire(_StubAgent()) is None
    # And a second caller is equally unimpeded — nothing is holding anything.
    assert backend_scheduler.acquire(_StubAgent("session-2")) is None
    assert backend_scheduler.snapshot()["active"] == []


def test_auto_mode_engages_only_for_a_local_backend(monkeypatch):
    """Hosted providers arbitrate their own concurrency; leave them alone."""
    cfg = backend_scheduler.SchedulerSettings(mode="auto")
    monkeypatch.setattr(backend_scheduler, "settings", lambda: cfg)

    assert backend_scheduler.engaged_for(
        _StubAgent(base_url="http://127.0.0.1:18088/v1")
    )
    assert backend_scheduler.engaged_for(
        _StubAgent(base_url="http://192.168.1.40:8000/v1")
    )
    assert not backend_scheduler.engaged_for(
        _StubAgent(base_url="https://api.anthropic.com")
    )
    assert not backend_scheduler.engaged_for(_StubAgent(base_url=""))


# ---------------------------------------------------------------------------
# (a) live overtakes maintenance
# ---------------------------------------------------------------------------

def test_live_turn_is_admitted_ahead_of_queued_maintenance(scheduler):
    """The priority inversion from H-034/CR-008, at the queue this time.

    Background review queued first. A user's turn arrives while it is still
    waiting. When the backend frees up, the user goes first — housekeeping
    that has been waiting longer does not get to keep the slot warm.
    """
    holder = backend_scheduler.acquire(_StubAgent("holder"))
    assert holder is not None

    order: list = []
    review = _Claimant(
        _StubAgent("review", maintenance=True), log=order, name="maintenance"
    ).start()
    assert _wait_until(lambda: _queued_priorities() == ["maintenance"])

    user = _Claimant(_StubAgent("user"), log=order, name="live").start()
    assert _wait_until(lambda: len(_queued_priorities()) == 2)

    backend_scheduler.release(holder)
    user.join()

    assert order == ["live"], "the live turn must not queue behind housekeeping"
    assert _queued_priorities() == ["maintenance"], "review still waits its turn"

    user.release()
    review.join()
    assert order == ["live", "maintenance"]
    review.release()


def test_maintenance_priority_comes_from_the_turn_registry(scheduler):
    """Classification is the registry's, not a second opinion (H-034)."""
    review = _StubAgent("review", maintenance=True)
    assert live_turn_registry.is_maintenance_agent(review)

    ticket = backend_scheduler.acquire(review)
    assert ticket is not None
    assert ticket.priority == backend_scheduler.PRIORITY_MAINTENANCE
    backend_scheduler.release(ticket)

    ticket = backend_scheduler.acquire(_StubAgent("user"))
    assert ticket.priority == backend_scheduler.PRIORITY_LIVE
    backend_scheduler.release(ticket)


# ---------------------------------------------------------------------------
# (b) arrival order is kept
# ---------------------------------------------------------------------------

def test_two_live_turns_are_served_in_arrival_order(scheduler):
    """A session's wait only ever shrinks; a newcomer cannot jump it."""
    holder = backend_scheduler.acquire(_StubAgent("holder"))
    order: list = []

    first = _Claimant(_StubAgent("session-a"), log=order, name="first").start()
    assert _wait_until(lambda: len(_queued_priorities()) == 1)
    second = _Claimant(_StubAgent("session-b"), log=order, name="second").start()
    assert _wait_until(lambda: len(_queued_priorities()) == 2)

    backend_scheduler.release(holder)
    first.join()
    assert order == ["first"]

    first.release()
    second.join()
    assert order == ["first", "second"]
    second.release()


def test_capacity_above_one_admits_that_many_at_once(scheduler):
    """A multi-slot local server is configured, not worked around.

    Each claimant runs on its own thread, the way turns do — two acquires on
    ONE thread are nesting, which is a different contract (see the re-entrancy
    test below).
    """
    scheduler(max_concurrent_requests=2)
    order: list = []
    a = _Claimant(_StubAgent("a"), log=order, name="a").start()
    b = _Claimant(_StubAgent("b"), log=order, name="b").start()
    a.join()
    b.join()
    assert len(backend_scheduler.snapshot()["active"]) == 2

    third = _Claimant(_StubAgent("c"), log=order, name="third").start()
    assert _wait_until(lambda: len(_queued_priorities()) == 1)

    a.release()
    third.join()
    assert order[-1] == "third"
    third.release()
    b.release()


# ---------------------------------------------------------------------------
# (c) the release is the wakeup
# ---------------------------------------------------------------------------

def test_next_caller_starts_on_release_not_on_a_poll_tick(scheduler):
    """The whole point of moving the wait here: no dead time on a free backend.

    The poll interval is set to five seconds — far longer than the assertion
    window — so a waiter that resumes promptly can only have been woken by the
    release event itself, not by the loop coming round again.
    """
    scheduler(poll_seconds=5.0, queue_wait_seconds=30.0)
    holder = backend_scheduler.acquire(_StubAgent("holder"))

    order: list = []
    waiter = _Claimant(_StubAgent("waiter"), log=order, name="waiter").start()
    assert _wait_until(lambda: len(_queued_priorities()) == 1)

    released_at = time.monotonic()
    backend_scheduler.release(holder)
    waiter.join()

    latency = waiter.granted_at - released_at
    assert latency < 1.0, f"waiter resumed {latency:.2f}s after release"
    waiter.release()


def test_released_permit_is_measured_and_feeds_the_queue_deadline(scheduler):
    """Item 3: the deadline tracks what calls on this backend actually cost."""
    scheduler(queue_wait_seconds=0, queue_wait_min_seconds=1, queue_wait_cap_seconds=1800)
    assert backend_scheduler.snapshot()["samples"] == 0

    ticket = backend_scheduler.acquire(_StubAgent("a"))
    time.sleep(0.05)
    backend_scheduler.release(ticket)

    snap = backend_scheduler.snapshot()
    assert snap["samples"] == 1
    assert snap["call_seconds_p90"] >= 0.04

    # Feed the statistic a realistic local-backend call and check the derived
    # deadline scales with it and with the number of claims ahead.
    for _ in range(8):
        backend_scheduler.record_call_duration(300.0)
    cfg = backend_scheduler.settings()
    one_ahead = backend_scheduler._wait_deadline_seconds(cfg, 1)
    two_ahead = backend_scheduler._wait_deadline_seconds(cfg, 2)
    assert one_ahead >= 300.0
    assert two_ahead > one_ahead
    assert two_ahead <= cfg.queue_wait_cap_seconds


def test_maintenance_waits_the_full_cap_rather_than_failing_open(scheduler):
    """Expiring means submitting anyway — which for housekeeping is H-034."""
    scheduler(queue_wait_seconds=0, queue_wait_cap_seconds=900)
    cfg = backend_scheduler.settings()
    backend_scheduler.record_call_duration(2.0)

    live = backend_scheduler._wait_deadline_seconds(
        cfg, 1, backend_scheduler.PRIORITY_LIVE
    )
    maintenance = backend_scheduler._wait_deadline_seconds(
        cfg, 1, backend_scheduler.PRIORITY_MAINTENANCE
    )
    assert maintenance == 900
    assert live < maintenance


def test_explicit_queue_wait_seconds_overrides_the_measurement(scheduler):
    scheduler(queue_wait_seconds=42.0)
    cfg = backend_scheduler.settings()
    backend_scheduler.record_call_duration(600.0)
    assert backend_scheduler._wait_deadline_seconds(cfg, 5) == 42.0


# ---------------------------------------------------------------------------
# Nothing here may fail a turn
# ---------------------------------------------------------------------------

def test_wait_past_the_deadline_submits_anyway(scheduler):
    """Fail open. A scheduling heuristic must never turn into a turn failure."""
    scheduler(queue_wait_seconds=0.2)
    holder = backend_scheduler.acquire(_StubAgent("holder"))

    order: list = []
    impatient = _Claimant(_StubAgent("impatient"), log=order, name="i").start()
    impatient.join()

    assert impatient.ticket is not None, "an expired wait still makes its call"
    assert impatient.ticket.over_capacity is True
    # Accounted for, so the queue does not promote a fresh waiter on top of it.
    assert len(backend_scheduler.snapshot()["active"]) == 2

    impatient.release()
    backend_scheduler.release(holder)
    assert backend_scheduler.snapshot()["active"] == []


def test_interrupted_caller_leaves_the_queue(scheduler):
    """An interrupted turn stops queueing instead of holding a place."""
    scheduler(queue_wait_seconds=30.0)
    holder = backend_scheduler.acquire(_StubAgent("holder"))

    aborting = {"value": False}

    def _abort():
        return aborting["value"]

    order: list = []
    waiter = _Claimant(
        _StubAgent("interrupted"), log=order, name="w", should_abort=_abort
    ).start()
    assert _wait_until(lambda: len(_queued_priorities()) == 1)

    aborting["value"] = True
    waiter.join()

    assert waiter.ticket is None, "an aborted wait holds no permit"
    assert backend_scheduler.snapshot()["waiting"] == []
    backend_scheduler.release(holder)


def test_a_raise_after_the_grant_still_hands_the_permit_back(scheduler):
    """The permit is the caller's before anything that can raise runs.

    Admission's bookkeeping — the log line saying who got in and how long they
    waited — sits after the grant. If that raises and the ticket is dropped,
    the permit stays counted in ``_active`` held by nobody: every later caller
    then queues behind it until its own deadline expires and it submits over
    capacity. Permanently, for the life of the process.
    """
    scheduler(queue_wait_seconds=0.2)

    class _ExplodingLogger:
        def warning(self, *_a, **_kw):
            raise RuntimeError("log sink is on fire")

        def info(self, *_a, **_kw):
            raise RuntimeError("log sink is on fire")

        def debug(self, *_a, **_kw):
            pass

    holder = backend_scheduler.acquire(_StubAgent("holder"))
    monkey = _ExplodingLogger()
    original = backend_scheduler.logger
    backend_scheduler.logger = monkey
    try:
        order: list = []
        waiter = _Claimant(_StubAgent("waiter"), log=order, name="w").start()
        waiter.join()
    finally:
        backend_scheduler.logger = original

    assert waiter.ticket is not None, "the caller must still get its permit"
    backend_scheduler.release(waiter.ticket)
    backend_scheduler.release(holder)
    assert backend_scheduler.snapshot()["active"] == []


def test_a_raise_before_the_grant_leaves_nothing_in_the_queue(scheduler):
    """The other half: a failure while queued must not wedge the queue.

    A ticket abandoned in ``_waiting`` (or granted by a concurrent release
    while admission was failing) holds a place nobody is coming back for.
    """
    scheduler(queue_wait_seconds=30.0)
    holder = backend_scheduler.acquire(_StubAgent("holder"))

    def _boom(*_a, **_kw):
        raise RuntimeError("deadline arithmetic failed")

    original = backend_scheduler._wait_deadline_seconds
    backend_scheduler._wait_deadline_seconds = _boom
    try:
        order: list = []
        doomed = _Claimant(_StubAgent("doomed"), log=order, name="d").start()
        doomed.join()
    finally:
        backend_scheduler._wait_deadline_seconds = original

    assert doomed.ticket is None, "a failed admission holds no permit"
    snapshot = backend_scheduler.snapshot()
    assert snapshot["waiting"] == []
    assert len(snapshot["active"]) == 1, "only the real holder is counted"

    # And the queue still works: the next caller is admitted on the release.
    backend_scheduler.release(holder)
    nxt = backend_scheduler.acquire(_StubAgent("next"))
    assert nxt is not None and nxt.waited_seconds < 1.0
    backend_scheduler.release(nxt)


def test_release_wakes_the_next_waiter_without_reading_config(scheduler):
    """The wakeup path must not depend on a config read.

    ``release`` sits between the response arriving and the next waiter waking.
    A config read there costs latency on every single call, and a failed one
    used to fall back to capacity 1 — silently under-promoting a multi-slot
    backend. The limit rides on the ticket instead, so a config layer that is
    broken (or just slow) cannot touch this path.
    """
    scheduler(max_concurrent_requests=2, queue_wait_seconds=30.0)
    order: list = []
    a = _Claimant(_StubAgent("a"), log=order, name="a").start()
    b = _Claimant(_StubAgent("b"), log=order, name="b").start()
    a.join()
    b.join()
    assert a.ticket.capacity == 2, "the limit is captured on the ticket"

    waiter = _Claimant(_StubAgent("c"), log=order, name="c").start()
    assert _wait_until(lambda: len(_queued_priorities()) == 1)

    def _boom():
        raise RuntimeError("config layer is down")

    original = backend_scheduler.settings
    backend_scheduler.settings = _boom
    try:
        a.release()
        waiter.join()
    finally:
        backend_scheduler.settings = original

    assert order[-1] == "c", "the waiter was woken without any config read"
    waiter.release()
    b.release()


def test_reentrant_acquire_on_one_thread_does_not_deadlock(scheduler):
    """A permit holder that re-enters must not wait on itself."""
    outer = backend_scheduler.acquire(_StubAgent("a"))
    inner = backend_scheduler.acquire(_StubAgent("a"))
    assert inner is not None and inner.reentrant is True
    assert len(backend_scheduler.snapshot()["active"]) == 1

    backend_scheduler.release(inner)
    assert len(backend_scheduler.snapshot()["active"]) == 1
    backend_scheduler.release(outer)
    assert backend_scheduler.snapshot()["active"] == []


def test_a_broken_config_read_disengages_rather_than_raising(monkeypatch):
    def _boom():
        raise RuntimeError("config on fire")

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", _boom, raising=False
    )
    # settings() swallows it and returns defaults; acquire() must not raise.
    assert backend_scheduler.settings().mode == "auto"
    assert backend_scheduler.acquire(_StubAgent(base_url="")) is None


def test_release_tolerates_none_and_double_release(scheduler):
    backend_scheduler.release(None)
    ticket = backend_scheduler.acquire(_StubAgent("a"))
    backend_scheduler.release(ticket)
    backend_scheduler.release(ticket)
    assert backend_scheduler.snapshot()["active"] == []


# ---------------------------------------------------------------------------
# (4) retries wait for the release, not for a timer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error,status,expected",
    [
        ("worker inference queue timed out", None, True),
        ("Service Unavailable", None, True),
        ("overloaded_error", None, True),
        ("upstream provider error", 503, True),
        ("upstream provider error", 529, True),
        ("invalid request: bad schema", 400, False),
        ("internal server error", 500, False),
        (None, None, False),
    ],
)
def test_backend_busy_classification(error, status, expected):
    assert (
        backend_scheduler.looks_like_backend_busy(error=error, status_code=status)
        is expected
    )


def test_busy_retry_collapses_to_the_queue_when_someone_else_holds_it(scheduler):
    """H-033: retrying on a timer re-enters a backend that is still busy.

    With the queue in place the retry's real wait is ``acquire`` at the top of
    the loop, which ends on the occupying call's release. Sleeping a fixed
    backoff first only adds latency, so it collapses to a floor.
    """
    other = backend_scheduler.acquire(_StubAgent("other-session"))
    assert other is not None

    override = backend_scheduler.retry_wait_override(
        _StubAgent("me"), 60.0, error="503 worker inference queue timed out"
    )
    assert override is not None and override <= 1.0
    backend_scheduler.release(other)


def test_busy_retry_keeps_its_backoff_when_the_contention_is_elsewhere(scheduler):
    """No release event to wait for: the rejection came from outside.

    Another Hermes process, or the server's own queue. Collapsing the backoff
    here would hot-loop through the retry budget in a couple of seconds.
    """
    assert not backend_scheduler.contended()
    assert (
        backend_scheduler.retry_wait_override(
            _StubAgent("me"), 60.0, error="503 worker inference queue timed out"
        )
        is None
    )


def test_non_busy_errors_keep_their_backoff(scheduler):
    other = backend_scheduler.acquire(_StubAgent("other"))
    assert (
        backend_scheduler.retry_wait_override(
            _StubAgent("me"), 60.0, error="400 invalid schema", status_code=400
        )
        is None
    )
    backend_scheduler.release(other)


def test_busy_rejections_are_counted_even_when_the_backoff_stands(scheduler):
    scheduler(mode="off")
    before = backend_scheduler.snapshot()["busy_signals"]
    backend_scheduler.retry_wait_override(
        _StubAgent("me"), 30.0, error="worker inference queue timed out"
    )
    assert backend_scheduler.snapshot()["busy_signals"] == before + 1


# ---------------------------------------------------------------------------
# (2) the waiting session is told where it stands
# ---------------------------------------------------------------------------

def test_queue_notice_reports_position_and_elapsed_wait(scheduler):
    """H-032's 147-second first response with no explanation."""
    from agent.conversation_loop import _backend_queue_notice

    statuses: list = []
    touches: list = []

    class _Agent(_StubAgent):
        def _emit_status(self, message):
            statuses.append(message)

        def _touch_activity(self, desc, **kwargs):
            touches.append(desc)

    notice = _backend_queue_notice(_Agent("waiting-session"))
    notice(2, 41.0)

    assert len(statuses) == 1
    assert "2 requests ahead" in statuses[0]
    assert "41s" in statuses[0]
    # The gateway's inactivity monitor measures silence — a legitimate queue
    # wait must not read as a stalled turn.
    assert touches and "queued for backend" in touches[0]


def test_queue_notice_stays_out_of_a_maintenance_surface(scheduler):
    """Nobody is watching a housekeeping queue; the log is enough."""
    from agent.conversation_loop import _backend_queue_notice

    statuses: list = []

    class _Agent(_StubAgent):
        def _emit_status(self, message):
            statuses.append(message)

        def _touch_activity(self, desc, **kwargs):
            pass

    notice = _backend_queue_notice(_Agent("review", maintenance=True))
    notice(1, 30.0)
    assert statuses == []


def test_queue_notice_never_names_a_raw_session(scheduler, caplog):
    """H-014: gateway session keys embed Discord IDs and never reach logs."""
    from agent.conversation_loop import _backend_queue_notice

    raw = "discord:1234567890:9876543210:5555555555"

    class _Agent(_StubAgent):
        def _emit_status(self, message):
            pass

        def _touch_activity(self, desc, **kwargs):
            pass

    with caplog.at_level("INFO"):
        _backend_queue_notice(_Agent(raw))(1, 25.0)

    assert "1234567890" not in caplog.text
    assert "9876543210" not in caplog.text
