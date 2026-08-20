"""Tests for the consecutive-denial circuit breaker in smart approvals.

After ``approvals.denial_breaker_threshold`` consecutive guardian DENY
verdicts in one session, the deny message returned to the model escalates
from "Do NOT retry" to a hard-stop CIRCUIT BREAKER instruction. Any
approval resets the tally. State is per-session and capped in size.

Follows the existing smart-approval mocking patterns from
tests/tools/test_execute_code_approval_cluster.py: monkeypatch
``_smart_approve`` / ``_get_approval_mode`` on the module and drive the
public guard entry points.
"""

from __future__ import annotations

import threading

import pytest

from tools import approval as A

BREAKER_MARKER = "CIRCUIT BREAKER:"


@pytest.fixture
def breaker_session(monkeypatch):
    """A clean gateway smart-mode session with the guardian forced to DENY.

    Uses the gateway path with a notify callback that resolves 'deny'
    (user denies the smart-DENY override) so the guard returns a definitive
    BLOCKED message — the channel the breaker text rides on.
    """
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 3)
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, "breaker-test-danger", f"risk:{command}"),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )

    session_key = "breaker-test-session"
    token = A.set_current_session_key(session_key)
    A.clear_session(session_key)
    with A._lock:
        A._permanent_approved.discard("breaker-test-danger")
        A._permanent_approved.discard("execute_code")
        A._session_approved.get(session_key, set()).discard("breaker-test-danger")
        A._session_approved.get(session_key, set()).discard("execute_code")
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)
    try:
        yield session_key
    finally:
        A.reset_current_session_key(token)
        A.clear_session(session_key)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)


def _register_resolver(session_key: str, result):
    """Notify callback resolving the newest queued approval with *result*."""
    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = result
                entries[-1].event.set()
    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


def _denied_terminal(command="dangerous thing"):
    return A.check_all_command_guards(command, "local")


def _denied_execute_code(code="print('x')"):
    return A.check_execute_code_guard(code, "local")


# ---------------------------------------------------------------------------
# (a) Two denials -> normal message; third -> breaker text present
# ---------------------------------------------------------------------------

def test_breaker_trips_on_third_consecutive_denial(breaker_session):
    _register_resolver(breaker_session, "deny")

    first = _denied_terminal("dangerous one")
    second = _denied_terminal("dangerous two")
    third = _denied_terminal("dangerous three")

    assert first["approved"] is False
    assert BREAKER_MARKER not in first["message"]
    assert second["approved"] is False
    assert BREAKER_MARKER not in second["message"]
    assert third["approved"] is False
    assert BREAKER_MARKER in third["message"]
    assert "3 consecutive commands were blocked" in third["message"]
    assert "STOP attempting variations" in third["message"]
    assert third[A.APPROVAL_BREAKER_METADATA_KEY] == {
        "version": 1,
        "type": "consecutive_smart_denials",
        "tripped": True,
        "count": 3,
        "threshold": 3,
    }


# ---------------------------------------------------------------------------
# (b) An approval resets the tally
# ---------------------------------------------------------------------------

def test_approval_resets_tally(breaker_session, monkeypatch):
    _register_resolver(breaker_session, "deny")
    _denied_terminal("dangerous one")
    _denied_terminal("dangerous two")
    A._record_approval_breaker_trip(
        breaker_session,
        {
            "version": 1,
            "type": "consecutive_smart_denials",
            "tripped": True,
            "count": 3,
            "threshold": 3,
        },
    )
    assert A.peek_approval_breaker_trip() is not None

    # Guardian approves the next command → tally resets.
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "approve")
    ok = _denied_terminal("benign command")
    assert ok["approved"] is True and ok.get("smart_approved") is True
    # Approval resets only the future denial tally; a trip already observed
    # by an active outer call remains pending for its exact executor consume.
    assert A.peek_approval_breaker_trip() is not None

    # Back to denials: the count restarts, so the next deny is #1, not #3.
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    after = _denied_terminal("dangerous again")
    assert after["approved"] is False
    assert BREAKER_MARKER not in after["message"]


def test_human_approval_resets_tally(breaker_session):
    _register_resolver(breaker_session, "deny")
    _denied_terminal("dangerous one")
    _denied_terminal("dangerous two")
    A._record_approval_breaker_trip(
        breaker_session,
        {
            "version": 1,
            "type": "consecutive_smart_denials",
            "tripped": True,
            "count": 3,
            "threshold": 3,
        },
    )
    assert A.peek_approval_breaker_trip() is not None

    # User overrides the smart DENY (one-operation approval) → tally resets.
    _register_resolver(breaker_session, "once")
    ok = _denied_terminal("dangerous but user says yes")
    assert ok["approved"] is True and ok.get("user_approved") is True
    assert A.peek_approval_breaker_trip() is not None

    _register_resolver(breaker_session, "deny")
    after = _denied_terminal("dangerous again")
    assert after["approved"] is False
    assert BREAKER_MARKER not in after["message"]


# ---------------------------------------------------------------------------
# (c) Threshold 0 disables the breaker
# ---------------------------------------------------------------------------

def test_threshold_zero_disables_breaker_and_side_channel(breaker_session, monkeypatch):
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 0)
    _register_resolver(breaker_session, "deny")

    results = [
        _denied_terminal("threshold disabled one"),
        _denied_terminal("threshold disabled two"),
        _denied_terminal("threshold disabled three"),
    ]

    assert all(result["approved"] is False for result in results)
    assert all(BREAKER_MARKER not in result["message"] for result in results)
    assert all(A.APPROVAL_BREAKER_METADATA_KEY not in result for result in results)
    assert A.peek_approval_breaker_trip() is None


# ---------------------------------------------------------------------------
# (d) Tally is per-session — two session keys are independent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (e) BOTH call paths increment: terminal guard and execute_code guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Headless hard-deny path (no cli/gateway/ask override) also increments
# ---------------------------------------------------------------------------

def test_headless_smart_deny_increments_and_trips(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "0")
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 3)
    monkeypatch.setattr(A, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, "headless-breaker-danger", f"risk:{command}"),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )
    # CLI-interactive path: the owner denies via the prompt callback.
    monkeypatch.setattr(A, "prompt_dangerous_approval",
                        lambda *args, **kwargs: "deny")

    session_key = "headless-breaker-session"
    token = A.set_current_session_key(session_key)
    A.clear_session(session_key)
    with A._lock:
        A._permanent_approved.discard("headless-breaker-danger")
        A._session_approved.get(session_key, set()).discard(
            "headless-breaker-danger")
    try:
        first = A.check_all_command_guards("dangerous h1", "local")
        second = A.check_all_command_guards("dangerous h2", "local")
        third = A.check_all_command_guards("dangerous h3", "local")
        assert BREAKER_MARKER not in first["message"]
        assert BREAKER_MARKER not in second["message"]
        assert BREAKER_MARKER in third["message"]
    finally:
        A.reset_current_session_key(token)
        A.clear_session(session_key)


# ---------------------------------------------------------------------------
# Eviction cap: the tally dict never grows past _DENIAL_TALLY_MAX_SESSIONS
# ---------------------------------------------------------------------------

def test_tally_evicts_oldest_sessions():
    with A._lock:
        saved = dict(A._denial_tally)
        A._denial_tally.clear()
    try:
        for i in range(A._DENIAL_TALLY_MAX_SESSIONS + 10):
            A._record_denial(f"evict-session-{i}")
        with A._lock:
            assert len(A._denial_tally) == A._DENIAL_TALLY_MAX_SESSIONS
            # Oldest entries were evicted, newest survive.
            assert "evict-session-0" not in A._denial_tally
            assert (
                f"evict-session-{A._DENIAL_TALLY_MAX_SESSIONS + 9}"
                in A._denial_tally
            )
    finally:
        with A._lock:
            A._denial_tally.clear()
            A._denial_tally.update(saved)


def test_clear_session_resets_denial_tally(breaker_session):
    _register_resolver(breaker_session, "deny")
    _denied_terminal("dangerous one")
    _denied_terminal("dangerous two")
    A._record_approval_breaker_trip(
        breaker_session,
        {
            "version": 1,
            "type": "consecutive_smart_denials",
            "tripped": True,
            "count": 3,
            "threshold": 3,
        },
    )
    assert A.peek_approval_breaker_trip() is not None

    A.clear_session(breaker_session)
    assert A.peek_approval_breaker_trip() is None

    _register_resolver(breaker_session, "deny")
    after = _denied_terminal("dangerous after clear")
    assert BREAKER_MARKER not in after["message"]


def test_pending_trip_is_scoped_to_exact_turn_and_tool_call(breaker_session):
    """An old turn's refreshed trip cannot stop a later user turn."""
    metadata = {
        "version": 1,
        "type": "consecutive_smart_denials",
        "tripped": True,
        "count": 3,
        "threshold": 3,
    }
    A._record_approval_breaker_trip(
        breaker_session,
        metadata,
        turn_id="old-turn",
        tool_call_id="old-call",
    )

    assert A.consume_approval_breaker_trip(
        breaker_session,
        turn_id="fresh-turn",
        tool_call_id="fresh-call",
    ) is None
    assert A.consume_approval_breaker_trip(
        breaker_session,
        turn_id="old-turn",
        tool_call_id="old-call",
    ) == metadata


def test_approval_reset_race_keeps_pending_trip():
    """A concurrent approval reset cannot erase an already recorded trip."""
    session_key = "breaker-reset-race"
    turn_id = "turn-reset-race"
    tool_call_id = "execute-reset-race"
    metadata = {
        "version": 1,
        "type": "consecutive_smart_denials",
        "tripped": True,
        "count": 3,
        "threshold": 3,
    }
    A.clear_session(session_key)
    barrier = threading.Barrier(2)

    def record_trip():
        barrier.wait()
        A._record_approval_breaker_trip(
            session_key,
            metadata,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
        )

    def reset_approval_tally():
        barrier.wait()
        A._reset_denials(session_key)

    threads = [
        threading.Thread(target=record_trip),
        threading.Thread(target=reset_approval_tally),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert A.peek_approval_breaker_trip(
            session_key, turn_id=turn_id, tool_call_id=tool_call_id
        ) == metadata
    finally:
        A.clear_session(session_key)
