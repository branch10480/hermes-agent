"""Tests for the consecutive-denial circuit breaker in smart approvals.

The breaker has two stages. After ``approvals.denial_breaker_threshold``
consecutive guardian DENY verdicts in one user turn, dangerous operations are
LOCKED OUT for the rest of that turn — further candidates are refused without
a guardian call, safe work continues, and the turn does not end. Only after
``approvals.denial_breaker_lockout_attempts`` more dangerous-operation
attempts does the breaker HARD STOP the turn. Any approval resets the tally.
State is per-session/turn and capped in size.

Follows the existing smart-approval mocking patterns from
tests/tools/test_execute_code_approval_cluster.py: monkeypatch
``_smart_approve`` / ``_get_approval_mode`` on the module and drive the
public guard entry points.
"""

from __future__ import annotations

import logging
import threading

import pytest

from tools import approval as A

BREAKER_MARKER = "CIRCUIT BREAKER:"
LOCKOUT_MARKER = "DANGEROUS-OPERATION LOCKOUT:"


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
    monkeypatch.setattr(A, "_get_denial_breaker_lockout_attempts", lambda: 3)
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
# (a) Two denials -> normal message; third -> lockout, turn continues
# ---------------------------------------------------------------------------

def test_breaker_locks_out_on_third_consecutive_denial(breaker_session):
    _register_resolver(breaker_session, "deny")

    first = _denied_terminal("dangerous one")
    second = _denied_terminal("dangerous two")
    third = _denied_terminal("dangerous three")

    assert first["approved"] is False
    assert LOCKOUT_MARKER not in first["message"]
    assert second["approved"] is False
    assert LOCKOUT_MARKER not in second["message"]
    assert third["approved"] is False
    assert LOCKOUT_MARKER in third["message"]
    # The turn-ending instruction must NOT be there yet.
    assert BREAKER_MARKER not in third["message"]
    assert "3 consecutive commands were blocked" in third["message"]
    assert "locked out for the rest of this turn" in third["message"]
    assert "3 more dangerous-operation attempt(s) will end the turn" in third[
        "message"
    ]
    assert third[A.APPROVAL_BREAKER_METADATA_KEY] == {
        "version": 3,
        "type": "consecutive_smart_denials",
        "tripped": True,
        "count": 3,
        "threshold": 3,
        "hard_stop_threshold": 6,
        "mode": "lockout",
        # v2 carries the redacted denial detail across the side channel so the
        # conversation loop can name the refused class without the command.
        "reason_code": "breaker_test_danger",
        "effect_class": "unclassified",
        "safe_alternative": A._safe_alternative_for("unclassified"),
    }
    # No structural trip: the executor has nothing to end the turn on.
    assert A.peek_approval_breaker_trip() is None


def test_lockout_refuses_without_asking_the_guardian(breaker_session, monkeypatch):
    """Post-lockout candidates must not burn another guardian LLM call."""
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        _denied_terminal(f"dangerous {index}")

    guardian_calls: list = []

    def _counting_guardian(command, description):
        guardian_calls.append(command)
        return "deny"

    monkeypatch.setattr(A, "_smart_approve", _counting_guardian)
    fourth = _denied_terminal("dangerous four")

    assert guardian_calls == []
    assert fourth["approved"] is False
    assert fourth["denial_lockout"] is True
    assert "refused without being assessed" in fourth["message"]
    assert "nobody has refused it" in fourth["message"]
    assert LOCKOUT_MARKER in fourth["message"]
    assert fourth[A.APPROVAL_BREAKER_METADATA_KEY]["mode"] == "lockout"
    assert A.peek_approval_breaker_trip() is None


def test_lockout_guidance_names_the_tools_that_still_work(breaker_session):
    """"Stop" alone is what produced the retry loop — name the way forward."""
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        blocked = _denied_terminal(f"dangerous {index}")

    assert A._DENIAL_LOCKOUT_GUIDANCE in blocked["message"]
    assert "Do not retry blocked commands or variants" in blocked["message"]
    assert "report your findings" in blocked["message"]
    assert "write_file" in blocked["message"]


def test_lockout_guidance_forbids_reproducing_the_refused_effect(breaker_session):
    """Naming a route that still works must not read as a way around the deny.

    The saved-script hint is the exact shape a model would re-use to smuggle
    the refused operation through, so the same sentence has to close that door.
    """
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        blocked = _denied_terminal(f"dangerous {index}")

    assert (
        "Never use these to reproduce a refused operation" in blocked["message"]
    )
    assert (
        "a refusal applies to the effect, not the phrasing" in blocked["message"]
    )


def test_session_approved_pattern_still_runs_during_the_lockout(breaker_session):
    """is_approved runs first, so an approved pattern is never a candidate."""
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        _denied_terminal(f"dangerous {index}")
    assert A._denial_lockout_is_active(breaker_session) is True

    # The user approved this exact detector key earlier in the session.
    A.approve_session(breaker_session, "breaker-test-danger")
    approved = _denied_terminal("dangerous but already approved")

    assert approved["approved"] is True
    assert approved["message"] is None


def test_session_approved_execute_code_still_runs_during_the_lockout(
    breaker_session,
):
    """Same ordering guarantee on the execute_code guard."""
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        _denied_terminal(f"dangerous {index}")
    assert A._denial_lockout_is_active(breaker_session) is True

    A.approve_session(breaker_session, "execute_code")
    approved = _denied_execute_code("print('already approved')")

    assert approved["approved"] is True
    assert approved["message"] is None


def test_breaker_stage_warning_is_logged_once_per_transition(
    breaker_session, caplog
):
    """Every retry re-entering the same stage must not re-log the warning.

    Alerting built on warning volume would otherwise scale with the number of
    retries — exactly what the lockout exists to absorb quietly.
    """
    caplog.set_level(logging.WARNING, logger="tools.approval")
    _register_resolver(breaker_session, "deny")
    for index in range(6):
        _denied_terminal(f"dangerous {index}")

    stage_lines = [
        record.getMessage() for record in caplog.records
        if "circuit breaker stage" in record.getMessage()
    ]
    assert len(stage_lines) == 2
    assert "lockout" in stage_lines[0]
    assert "hard_stop" in stage_lines[1]
    tripped_lines = [
        record.getMessage() for record in caplog.records
        if "circuit breaker tripped" in record.getMessage()
    ]
    assert len(tripped_lines) == 2


def test_safe_commands_still_run_during_the_lockout(breaker_session, monkeypatch):
    """The whole point of the lockout: the turn keeps doing useful work."""
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        _denied_terminal(f"dangerous {index}")

    # A command neither the pattern detector nor tirith flags never becomes a
    # candidate, so it is approved before the lockout is even consulted.
    monkeypatch.setattr(
        A, "detect_dangerous_command", lambda command: (False, None, None)
    )
    safe = A.check_all_command_guards("python3 /tmp/analysis.py", "local")

    assert safe["approved"] is True
    assert safe["message"] is None


def test_hard_stop_after_the_post_lockout_attempts(breaker_session):
    """Three more dangerous attempts after the lockout end the turn."""
    _register_resolver(breaker_session, "deny")
    results = [_denied_terminal(f"dangerous {index}") for index in range(6)]

    assert LOCKOUT_MARKER in results[2]["message"]
    assert all(
        LOCKOUT_MARKER in result["message"] and BREAKER_MARKER not in result["message"]
        for result in results[2:5]
    )
    sixth = results[5]
    assert BREAKER_MARKER in sixth["message"]
    assert "kept being attempted after the lockout" in sixth["message"]
    assert "STOP attempting variations" in sixth["message"]
    metadata = sixth[A.APPROVAL_BREAKER_METADATA_KEY]
    assert metadata["mode"] == "hard_stop"
    assert metadata["count"] == 6
    assert metadata["hard_stop_threshold"] == 6
    # Only the hard stop writes the side channel the executor halts on.
    assert A.peek_approval_breaker_trip() == metadata


def test_zero_lockout_attempts_restores_the_immediate_hard_stop(
    breaker_session, monkeypatch
):
    monkeypatch.setattr(A, "_get_denial_breaker_lockout_attempts", lambda: 0)
    _register_resolver(breaker_session, "deny")

    _denied_terminal("dangerous one")
    _denied_terminal("dangerous two")
    third = _denied_terminal("dangerous three")

    assert BREAKER_MARKER in third["message"]
    assert third[A.APPROVAL_BREAKER_METADATA_KEY]["mode"] == "hard_stop"
    assert A.peek_approval_breaker_trip() is not None


def test_execute_code_is_locked_out_too(breaker_session, monkeypatch):
    """The lockout covers the execute_code guard, not just terminal()."""
    _register_resolver(breaker_session, "deny")
    for index in range(3):
        _denied_terminal(f"dangerous {index}")

    guardian_calls: list = []
    monkeypatch.setattr(
        A, "_smart_approve",
        lambda command, description: guardian_calls.append(command) or "deny",
    )
    blocked = _denied_execute_code("print('analysis')")

    assert guardian_calls == []
    assert blocked["approved"] is False
    assert blocked["denial_lockout"] is True
    assert "execute_code script" in blocked["message"]
    assert LOCKOUT_MARKER in blocked["message"]


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
    assert LOCKOUT_MARKER not in after["message"]


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
    assert LOCKOUT_MARKER not in after["message"]


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
    assert all(LOCKOUT_MARKER not in result["message"] for result in results)
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

def test_headless_smart_deny_increments_and_locks_out(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "0")
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 3)
    monkeypatch.setattr(A, "_get_denial_breaker_lockout_attempts", lambda: 3)
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
        assert LOCKOUT_MARKER not in first["message"]
        assert LOCKOUT_MARKER not in second["message"]
        assert LOCKOUT_MARKER in third["message"]
        assert BREAKER_MARKER not in third["message"]
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
            assert ("evict-session-0", "") not in A._denial_tally
            assert (
                (f"evict-session-{A._DENIAL_TALLY_MAX_SESSIONS + 9}", "")
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
    assert LOCKOUT_MARKER not in after["message"]


def test_denials_do_not_carry_into_a_new_turn(breaker_session):
    """Old denied work must not make a fresh user request stop early."""
    _register_resolver(breaker_session, "deny")

    old_tokens = A.set_current_observability_context(turn_id="old-turn")
    try:
        first = _denied_terminal("dangerous old one")
        second = _denied_terminal("dangerous old two")
    finally:
        A.reset_current_observability_context(old_tokens)

    fresh_tokens = A.set_current_observability_context(turn_id="fresh-turn")
    try:
        fresh_first = _denied_terminal("dangerous fresh one")
        fresh_second = _denied_terminal("dangerous fresh two")
        fresh_third = _denied_terminal("dangerous fresh three")
    finally:
        A.reset_current_observability_context(fresh_tokens)

    assert LOCKOUT_MARKER not in first["message"]
    assert LOCKOUT_MARKER not in second["message"]
    assert LOCKOUT_MARKER not in fresh_first["message"]
    assert LOCKOUT_MARKER not in fresh_second["message"]
    assert LOCKOUT_MARKER in fresh_third["message"]


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
