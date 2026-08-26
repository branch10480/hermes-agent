"""A fail-closed Tirith verdict is a hard deny, never an approvable warning.

F-2: on a remote surface (gateway/cron) where security.tirith_fail_open_gateway
is false, check_command_security returns ``{"action": "block", "fail_closed":
True}`` when it could not scan the command at all — tirith missing, a scan
timeout, or the crash breaker being open. "Could not scan" is categorically
different from "scanned and found a risk": there is no finding for a human, or
the Smart Approval guardian, to inspect and knowingly approve. The approval
layer must therefore route it straight to a hard deny, so it never reaches the
smart auto-approve path (guardian verdict == "approve") or the gateway approval
prompt where an owner could wave it through.
"""

from __future__ import annotations

import pytest

from tools import approval as A


def _fail_closed_verdict(reason_code="tirith_missing"):
    return {
        "action": "block",
        "fail_closed": True,
        "reason_code": reason_code,
        "findings": [
            {
                "rule_id": f"{reason_code}:deadbeef",
                "severity": "HIGH",
                "title": "Tirith security scanner unavailable",
                "description": "This command could not be security-scanned.",
                "reason_code": reason_code,
            }
        ],
        "summary": f"tirith unavailable ({reason_code}) — fail closed",
    }


@pytest.fixture
def gateway_smart_session(monkeypatch):
    """Gateway smart-mode session whose guardian would APPROVE and whose owner
    would APPROVE at the prompt — so only the fail-closed short-circuit can
    produce a deny."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    # Guardian would auto-approve, and any dangerous-pattern detection is off,
    # so nothing but the fail-closed branch can deny.
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "approve")
    monkeypatch.setattr(
        A, "detect_dangerous_command", lambda command: (False, None, None)
    )

    session_key = "failclosed-test-session"
    token = A.set_current_session_key(session_key)
    A.clear_session(session_key)
    # An owner resolver that would APPROVE the gateway prompt if we ever reached
    # it — the deny must come from the short-circuit, not from a refusal here.
    with A._lock:
        A._gateway_queues.pop(session_key, None)

        def _cb(_approval_data):
            with A._lock:
                entries = A._gateway_queues.get(session_key, [])
                if entries:
                    entries[-1].result = "approve"
                    entries[-1].event.set()

        A._gateway_notify_cbs[session_key] = _cb
    try:
        yield session_key
    finally:
        A.reset_current_session_key(token)
        A.clear_session(session_key)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)


@pytest.mark.parametrize(
    "reason_code",
    ["tirith_missing", "tirith_timeout", "tirith_circuit_open"],
)
def test_fail_closed_block_is_denied_even_in_smart_mode(
    monkeypatch, gateway_smart_session, reason_code
):
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: _fail_closed_verdict(reason_code),
        raising=False,
    )

    result = A.check_all_command_guards("cat /etc/passwd", "local")

    assert result["approved"] is False, result
    # The guardian said APPROVE and the owner resolver said approve, so a
    # smart auto-approve or a prompt approval would both have flipped this.
    assert not result.get("smart_approved")
    assert result.get("tirith_fail_closed") is True
    assert "BLOCKED by security policy" in result["message"]
    # No user-attribution: nobody was asked.
    assert "denied by user" not in result["message"].lower()


def test_ordinary_tirith_block_still_reaches_smart_approval(
    monkeypatch, gateway_smart_session
):
    """A normal (non-fail-closed) block is a real finding a human can inspect,
    so smart mode may still auto-approve it — the short-circuit must not swallow
    ordinary blocks."""
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {
            "action": "block",
            "findings": [
                {"rule_id": "homograph-url", "severity": "HIGH",
                 "title": "Homograph URL", "description": "suspicious host"}
            ],
            "summary": "homograph",
        },
        raising=False,
    )

    result = A.check_all_command_guards("curl http://xn--e1afmkfd.example", "local")

    assert result["approved"] is True
    assert result.get("smart_approved") is True
