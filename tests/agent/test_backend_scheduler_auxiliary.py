"""Auxiliary work on the shared backend (H-031, plan item 2-2).

``tests/agent/test_backend_scheduler.py`` pins the queue itself. This file pins
the half that used to bypass it: compression and title generation submit at the
same single-slot local backend a turn is using, and neither asked permission.
The observed cost was a 151-second summary occupying the slot a user was
waiting on, and a title call — the one request nobody is waiting for — landing
in the middle of someone's turn.

The two get opposite treatments, and the point of these tests is that the
difference is deliberate:

* compression **queues**. Somebody is blocked on it, so it takes a place in
  line and runs when the backend frees up.
* titling **yields**. The session already has a name derived from the user's
  own words, so a busy backend means the upgrade is skipped, not deferred
  forever.

Both stay fail-open: a disengaged queue, a foreign endpoint, or a wait past its
deadline all end up submitting exactly as they did before any of this existed.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from contextlib import contextmanager

import pytest

from agent import backend_scheduler, live_turn_registry, title_generator
from agent.context_compressor import ContextCompressor
from agent.title_generator import _auto_title_session
from hermes_state import SessionDB


LOCAL_ENDPOINT = "http://127.0.0.1:18088/v1"
HOSTED_ENDPOINT = "https://api.example.com/v1"


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


@pytest.fixture(autouse=True)
def _arbitration_always_allowed(monkeypatch):
    """Take the config file out of it; one test below puts it back."""
    monkeypatch.setattr(
        backend_scheduler, "_auxiliary_arbitration_allowed", lambda task: True
    )


@pytest.fixture
def scheduler(monkeypatch):
    """Install a fast, forced-on scheduler; returns a mutator for its settings."""

    def _apply(**overrides):
        cfg = backend_scheduler.SchedulerSettings(
            mode="auto",
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


@pytest.fixture
def aux_route(monkeypatch):
    """Pin where an auxiliary task resolves to, without touching config.yaml."""

    def _apply(provider="auto", base_url=None):
        def _resolve(task=None, provider_arg=None, model=None, *_args, **_kwargs):
            return provider, model, base_url, None, None

        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_task_provider_model", _resolve
        )

    _apply()
    return _apply


class _StubAgent:
    """The surface the registry reads off a turn owner."""

    def __init__(self, session_id: str = "other-session", *, maintenance: bool = False):
        self.session_id = session_id
        self.platform = "discord"
        self.base_url = LOCAL_ENDPOINT
        self._is_background_review_fork = maintenance


def _compressor(base_url: str = LOCAL_ENDPOINT) -> ContextCompressor:
    """A compressor stub carrying only what the admission path reads.

    ``__init__`` builds an LLM client and reads model metadata; the permit only
    needs to know which endpoint the summary is headed for.
    """
    compressor = object.__new__(ContextCompressor)
    compressor.summary_model = ""
    compressor.base_url = base_url
    return compressor


@contextmanager
def _permit_held_elsewhere(session: str = "holder"):
    """Occupy the backend from another thread for the body of the ``with``.

    Another thread, specifically: the re-entrancy guard is per thread, so a
    permit taken on *this* one would be handed straight back to the caller as a
    nested claim and prove nothing about queueing.
    """
    held = threading.Event()
    finish = threading.Event()

    def _hold():
        ticket = backend_scheduler.acquire(_StubAgent(session))
        held.set()
        finish.wait(timeout=30)
        backend_scheduler.release(ticket)

    thread = threading.Thread(target=_hold, daemon=True)
    thread.start()
    assert held.wait(timeout=5), "the holder never claimed the backend"
    try:
        yield
    finally:
        finish.set()
        thread.join(timeout=5)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------------------
# Endpoint resolution — which calls are ours to arbitrate
# ---------------------------------------------------------------------------

def test_auto_task_inherits_the_main_runtimes_endpoint(scheduler, aux_route):
    """``provider: auto`` is the case this exists for: the turn's own backend."""
    aux_route(provider="auto", base_url=None)
    assert (
        backend_scheduler.auxiliary_endpoint(
            "compression", main_base_url=LOCAL_ENDPOINT
        )
        == LOCAL_ENDPOINT
    )
    assert backend_scheduler.engaged_for_auxiliary(
        "compression", main_base_url=LOCAL_ENDPOINT
    )


def test_task_pinned_to_its_own_endpoint_is_judged_on_that_endpoint(
    scheduler, aux_route
):
    """An ``auxiliary.<task>.base_url`` override answers for itself."""
    aux_route(provider="custom", base_url=HOSTED_ENDPOINT)
    assert not backend_scheduler.engaged_for_auxiliary(
        "compression", main_base_url=LOCAL_ENDPOINT
    )

    aux_route(provider="custom", base_url=LOCAL_ENDPOINT)
    assert backend_scheduler.engaged_for_auxiliary(
        "compression", main_base_url=HOSTED_ENDPOINT
    )


def test_task_pinned_to_a_named_provider_is_left_alone(scheduler, aux_route):
    """A hosted provider brings its own concurrency; nothing here applies."""
    aux_route(provider="anthropic", base_url=None)
    assert (
        backend_scheduler.auxiliary_endpoint(
            "compression", main_base_url=LOCAL_ENDPOINT
        )
        == ""
    )
    assert not backend_scheduler.engaged_for_auxiliary(
        "compression", main_base_url=LOCAL_ENDPOINT
    )


def test_arbitration_can_be_switched_off_per_task(scheduler, aux_route, monkeypatch):
    """``auxiliary.<task>.scheduler_arbitration: false`` restores the bypass."""
    monkeypatch.undo()  # drop the autouse stub, exercise the real config read
    monkeypatch.setattr(
        backend_scheduler, "settings", lambda: backend_scheduler.SchedulerSettings(
            mode="auto", poll_seconds=0.02, notify_after_seconds=0
        )
    )
    aux_route(provider="auto", base_url=None)
    config = {"auxiliary": {"compression": {"scheduler_arbitration": False}}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)

    assert not backend_scheduler.engaged_for_auxiliary(
        "compression", main_base_url=LOCAL_ENDPOINT
    )
    assert (
        backend_scheduler.acquire_auxiliary(
            "compression", main_base_url=LOCAL_ENDPOINT
        )
        is None
    )


# ---------------------------------------------------------------------------
# Priority — who is waiting on this call
# ---------------------------------------------------------------------------

def test_a_summary_taken_during_a_live_turn_ranks_live():
    """Compaction fires mid-turn only as that turn's own preflight pass."""
    agent = _StubAgent()
    live_turn_registry.begin_turn(agent)
    assert backend_scheduler.auxiliary_priority() == backend_scheduler.PRIORITY_LIVE


def test_a_summary_with_nothing_in_flight_ranks_maintenance():
    """An idle-session compaction must not outrank a user's turn (H-034)."""
    assert (
        backend_scheduler.auxiliary_priority()
        == backend_scheduler.PRIORITY_MAINTENANCE
    )

    review = _StubAgent("review", maintenance=True)
    live_turn_registry.begin_turn(
        review, kind=live_turn_registry.TURN_KIND_MAINTENANCE
    )
    assert (
        backend_scheduler.auxiliary_priority()
        == backend_scheduler.PRIORITY_MAINTENANCE
    )


# ---------------------------------------------------------------------------
# Compression takes a permit
# ---------------------------------------------------------------------------

def test_compression_holds_a_permit_and_gives_it_back(scheduler, aux_route):
    compressor = _compressor()
    with compressor._arbitrated_backend_slot() as ticket:
        assert ticket is not None
        assert backend_scheduler.snapshot()["active"], "permit was never counted"
    assert backend_scheduler.snapshot()["active"] == [], "permit leaked"


def test_a_raising_summary_still_gives_the_permit_back(scheduler, aux_route):
    """A leaked permit wedges every other session, so the release is a finally."""
    compressor = _compressor()
    with pytest.raises(RuntimeError):
        with compressor._arbitrated_backend_slot():
            raise RuntimeError("summary provider fell over")
    assert backend_scheduler.snapshot()["active"] == []


def test_compression_on_a_foreign_endpoint_does_not_queue(scheduler, aux_route):
    """A summary pinned elsewhere shares nothing, so it waits for nothing."""
    aux_route(provider="custom", base_url=HOSTED_ENDPOINT)
    compressor = _compressor()
    with compressor._arbitrated_backend_slot() as ticket:
        assert ticket is None
        assert backend_scheduler.snapshot()["active"] == []


def test_compression_submits_when_the_queue_is_disengaged(scheduler, aux_route):
    """mode=off is the pre-queue world: no ticket, and the call still happens."""
    scheduler(mode="off")
    compressor = _compressor()
    ran = False
    with compressor._arbitrated_backend_slot() as ticket:
        assert ticket is None
        ran = True
    assert ran


def test_compression_waits_for_the_slot_then_takes_it(scheduler, aux_route):
    """The summary starts on the previous call's release, not on a timer."""
    holder = backend_scheduler.acquire(_StubAgent("holder"))
    assert holder is not None

    compressor = _compressor()
    started = threading.Event()
    admitted = threading.Event()

    def _summarize():
        started.set()
        with compressor._arbitrated_backend_slot() as ticket:
            if ticket is not None:
                admitted.set()

    worker = threading.Thread(target=_summarize, daemon=True)
    worker.start()

    assert started.wait(timeout=5)
    assert _wait_until(lambda: backend_scheduler.snapshot()["waiting"]), (
        "the summary never queued"
    )
    assert not admitted.is_set(), "the summary jumped an occupied backend"

    backend_scheduler.release(holder)
    assert admitted.wait(timeout=5), "the release never woke the summary"
    worker.join(timeout=5)
    assert backend_scheduler.snapshot()["active"] == []


def test_a_summary_past_its_deadline_submits_anyway(scheduler, aux_route):
    """Fail-open: admission is a heuristic and may never fail a compaction."""
    scheduler(queue_wait_seconds=0.05)
    with _permit_held_elsewhere():
        compressor = _compressor()
        with compressor._arbitrated_backend_slot() as ticket:
            assert ticket is not None
            assert ticket.over_capacity, "expected an over-capacity fail-open grant"


def test_a_summary_on_the_permit_holders_own_thread_does_not_wait(
    scheduler, aux_route
):
    """The nesting case, pinned: a same-thread summary re-enters, never queues.

    This is the shape the module docstring used to refuse the whole idea over.
    It is safe because the re-entrancy guard is per thread — and it stays safe
    only while that holds, which is why the cross-thread case is a separate
    test above rather than an assumption.
    """
    holder = backend_scheduler.acquire(_StubAgent("holder"))
    assert holder is not None
    try:
        compressor = _compressor()
        with compressor._arbitrated_backend_slot() as ticket:
            assert ticket is not None and ticket.reentrant
    finally:
        backend_scheduler.release(holder)
    assert backend_scheduler.snapshot()["active"] == []


# ---------------------------------------------------------------------------
# Titling yields instead
# ---------------------------------------------------------------------------

@pytest.fixture
def titled_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="sess-1", source="cli")
    return db


@pytest.fixture
def fast_title_polling(monkeypatch):
    monkeypatch.setattr(title_generator, "_BUSY_POLL_SECONDS", 0.02)


@pytest.fixture
def title_llm(monkeypatch):
    """Stand in for the title model; records whether it was ever called."""
    calls = []

    def _generate(user_message, **kwargs):
        calls.append(user_message)
        return "Model Written Title"

    monkeypatch.setattr(title_generator, "generate_title", _generate)
    return calls


def _run_titler(db, *, exclude=frozenset(), base_url=LOCAL_ENDPOINT):
    _auto_title_session(
        db,
        "sess-1",
        "fix the flaky auth test",
        main_runtime={"base_url": base_url},
        exclude_turn_tokens=exclude,
    )


def test_title_waits_out_a_busy_backend_then_upgrades(
    scheduler, aux_route, fast_title_polling, title_llm, titled_db, monkeypatch
):
    monkeypatch.setattr(title_generator, "_busy_defer_seconds", lambda: 10.0)
    other = _StubAgent()
    token = live_turn_registry.begin_turn(other)

    done = threading.Event()
    worker = threading.Thread(
        target=lambda: (_run_titler(titled_db), done.set()), daemon=True
    )
    worker.start()

    # Nothing may be sent while someone else's turn owns the backend.
    time.sleep(0.2)
    assert not title_llm, "the title call ignored a busy backend"

    live_turn_registry.end_turn(other, token)
    assert done.wait(timeout=10), "the titler never resumed"
    worker.join(timeout=5)

    assert title_llm == ["fix the flaky auth test"]
    assert titled_db.get_session_title("sess-1") == "Model Written Title"
    assert titled_db.get_session_title_source("sess-1") == "llm"


def test_title_is_skipped_when_the_backend_stays_busy(
    scheduler, aux_route, fast_title_polling, title_llm, titled_db, monkeypatch
):
    """The derived name stands rather than a user's turn queuing behind it."""
    monkeypatch.setattr(title_generator, "_busy_defer_seconds", lambda: 0.15)
    live_turn_registry.begin_turn(_StubAgent())

    _run_titler(titled_db)

    assert title_llm == [], "a busy backend still got a title call"
    assert titled_db.get_session_title("sess-1") == "fix the flaky auth test"
    assert titled_db.get_session_title_source("sess-1") == "derived"


def test_busy_defer_zero_restores_the_pre_gate_behaviour(
    scheduler, aux_route, fast_title_polling, title_llm, titled_db, monkeypatch
):
    monkeypatch.setattr(title_generator, "_busy_defer_seconds", lambda: 0.0)
    live_turn_registry.begin_turn(_StubAgent())

    _run_titler(titled_db)

    assert title_llm == ["fix the flaky auth test"]
    assert titled_db.get_session_title_source("sess-1") == "llm"


def test_the_titler_does_not_stand_down_for_its_own_turn(
    scheduler, aux_route, fast_title_polling, title_llm, titled_db, monkeypatch
):
    """Titling is forked from the turn prologue, so its own turn is registered.

    Reading that registration as "the backend is busy" would mean no session on
    an arbitrated backend was ever titled — the wait can only ever end after
    the turn it belongs to.
    """
    monkeypatch.setattr(title_generator, "_busy_defer_seconds", lambda: 30.0)
    owner = _StubAgent("own-session")
    token = live_turn_registry.begin_turn(owner)

    _run_titler(titled_db, exclude=frozenset({token}))

    assert title_llm == ["fix the flaky auth test"]
    assert titled_db.get_session_title_source("sess-1") == "llm"


def test_an_in_flight_call_still_holds_the_titler_back(
    scheduler, aux_route, fast_title_polling, title_llm, titled_db, monkeypatch
):
    """The excluded turn is still covered — by the permit it is holding."""
    monkeypatch.setattr(title_generator, "_busy_defer_seconds", lambda: 0.15)
    owner = _StubAgent("own-session")
    token = live_turn_registry.begin_turn(owner)
    ticket = backend_scheduler.acquire(owner)
    assert ticket is not None

    _run_titler(titled_db, exclude=frozenset({token}))

    assert title_llm == [], "the title call landed on top of a live request"
    assert titled_db.get_session_title_source("sess-1") == "derived"
    backend_scheduler.release(ticket)


def test_a_hosted_title_endpoint_is_never_deferred(
    scheduler, aux_route, fast_title_polling, title_llm, titled_db, monkeypatch
):
    """Nothing is shared with the local backend, so nothing is owed to it."""
    monkeypatch.setattr(title_generator, "_busy_defer_seconds", lambda: 30.0)
    aux_route(provider="custom", base_url=HOSTED_ENDPOINT)
    live_turn_registry.begin_turn(_StubAgent())

    _run_titler(titled_db)

    assert title_llm == ["fix the flaky auth test"]
    assert titled_db.get_session_title_source("sess-1") == "llm"
