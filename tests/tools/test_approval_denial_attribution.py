"""Denial attribution, side-channel redaction, and log hygiene.

Three separate defects lived in the same code path (see the harness
observations H-011 / H-013 / H-014 / H-015):

  * A guardian/policy denial that no human ever saw was reported to the model
    as "Command denied by user … The user has NOT consented", which is a false
    statement about the user's position and pushes the model toward asking for
    a bypass instead of finding a safe route.
  * The circuit breaker's closing guidance told the user to run the refused
    command manually or approve it with ``/approve`` — neither is possible on
    a surface where a policy denial resolves itself and queues nothing.
  * The breaker warning logged the raw gateway session key, which on Discord
    embeds channel, thread, and participant IDs, into persisted log files.

Mocking follows tests/tools/test_denial_circuit_breaker.py: force the
guardian to DENY, drive the public guard entry points, and patch the module
globals the guards resolve at call time.
"""

from __future__ import annotations

import logging

import pytest

from agent.conversation_loop import (
    _SAFE_ALTERNATIVE_JA_BY_EFFECT_CLASS,
    approval_breaker_final_message,
)
from tools import approval as A

# A session key shaped like the gateway's Discord keys: every segment is an
# identifier that must never reach a log file.
DISCORD_CHANNEL_ID = "112233445566778899"
DISCORD_THREAD_ID = "998877665544332211"
DISCORD_USER_ID = "123456789012345678"
DISCORD_SESSION_KEY = (
    f"discord:{DISCORD_CHANNEL_ID}:{DISCORD_THREAD_ID}:{DISCORD_USER_ID}"
)
DISCORD_ID_FRAGMENTS = (
    DISCORD_CHANNEL_ID,
    DISCORD_THREAD_ID,
    DISCORD_USER_ID,
)

# Phrases that assert something about the user's position. None of them may
# appear in a denial the user was never asked about.
USER_ATTRIBUTION_PHRASES = (
    "denied by user",
    "User denied",
    "has NOT consented",
    "wait for the user to respond",
)

# Routes to a human bypass. The breaker must never offer these.
MANUAL_BYPASS_PHRASES = (
    "/approve",
    "run it manually",
    "run the command",
    "手元の端末",
    "明示承認",
)

PATTERN_KEY = "recursive delete"


@pytest.fixture
def smart_deny_session(monkeypatch):
    """Gateway smart-mode session whose guardian always returns DENY."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 3)
    monkeypatch.setattr(A, "_get_denial_breaker_lockout_attempts", lambda: 3)
    monkeypatch.setattr(A, "_smart_deny_is_final", lambda: False)
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, PATTERN_KEY, PATTERN_KEY),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )

    token = A.set_current_session_key(DISCORD_SESSION_KEY)
    A.clear_session(DISCORD_SESSION_KEY)
    with A._lock:
        A._permanent_approved.discard(PATTERN_KEY)
        A._permanent_approved.discard("execute_code")
        A._gateway_queues.pop(DISCORD_SESSION_KEY, None)
        A._gateway_notify_cbs.pop(DISCORD_SESSION_KEY, None)
    try:
        yield DISCORD_SESSION_KEY
    finally:
        A.reset_current_session_key(token)
        A.clear_session(DISCORD_SESSION_KEY)
        with A._lock:
            A._gateway_queues.pop(DISCORD_SESSION_KEY, None)
            A._gateway_notify_cbs.pop(DISCORD_SESSION_KEY, None)


def _register_notify(session_key: str, calls: list) -> None:
    """Notify callback that only records that it was invoked."""
    def cb(approval_data):
        calls.append(approval_data)
    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


def _register_resolver(session_key: str, result: str) -> None:
    """Notify callback resolving the newest queued approval with *result*."""
    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = result
                entries[-1].event.set()
    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


def _install_policy_auto_deny(monkeypatch, session_key: str) -> list:
    """Mimic the deployed policy plugin that resolves a smart DENY itself.

    It wraps ``_await_gateway_decision`` and short-circuits any request
    carrying ``smart_denied``, returning the legacy two-key dict with no
    ``human_responded`` — exactly the shape core must read as "nobody was
    asked".
    """
    calls: list = []
    _register_notify(session_key, calls)
    original = A._await_gateway_decision

    def wrapper(key, notify_cb, approval_data, *, surface="gateway"):
        if approval_data.get("smart_denied") is True:
            return {"resolved": True, "choice": "deny"}
        return original(key, notify_cb, approval_data, surface=surface)

    monkeypatch.setattr(A, "_await_gateway_decision", wrapper)
    return calls


# ---------------------------------------------------------------------------
# (a) Automatic denials carry no user attribution
# ---------------------------------------------------------------------------

def test_policy_auto_deny_is_not_attributed_to_the_user(
    smart_deny_session, monkeypatch
):
    _install_policy_auto_deny(monkeypatch, smart_deny_session)

    result = A.check_all_command_guards("rm -rf /srv/data", "local")

    assert result["approved"] is False
    assert result["approval_outcome"] == A.APPROVAL_OUTCOME_GUARDIAN_DENIED
    # The coarse legacy field keeps its historical value for old consumers.
    assert result["outcome"] == "denied"
    for phrase in USER_ATTRIBUTION_PHRASES:
        assert phrase not in result["message"], phrase


def test_policy_auto_deny_reports_reason_code_and_alternative(
    smart_deny_session, monkeypatch
):
    _install_policy_auto_deny(monkeypatch, smart_deny_session)

    result = A.check_all_command_guards("rm -rf /srv/data", "local")

    assert result["reason_code"] == "recursive_delete"
    assert result["effect_class"] == "destructive_filesystem"
    assert result["safe_alternative"]
    assert "recursive_delete" in result["message"]
    assert "destructive_filesystem" in result["message"]
    assert "Safe alternative" in result["message"]
    # Rewording is refused identically — but the model is not told the user
    # refused, and not told to stop working.
    assert "denied the same way" in result["message"]


def test_cli_smart_deny_final_never_reaches_the_prompt(
    smart_deny_session, monkeypatch
):
    """On the CLI surface too, a final guardian DENY skips the human prompt.

    Without ``smart_deny_final`` the CLI shows a one-operation override
    prompt; a deployment that treats Smart Approval as the whole decision
    must not render an override it will refuse anyway, and must not report
    the result as the owner's refusal.
    """
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(A, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(A, "_smart_deny_is_final", lambda: True)
    prompts: list = []

    def _prompt(*args, **kwargs):
        prompts.append(kwargs)
        return "deny"

    monkeypatch.setattr(A, "prompt_dangerous_approval", _prompt)

    result = A.check_all_command_guards("rm -rf /srv/data", "local")

    assert prompts == []
    assert result["approved"] is False
    assert result["approval_outcome"] == A.APPROVAL_OUTCOME_GUARDIAN_DENIED
    for phrase in USER_ATTRIBUTION_PHRASES:
        assert phrase not in result["message"], phrase


def test_execute_code_policy_auto_deny_is_not_attributed_to_the_user(
    smart_deny_session, monkeypatch
):
    _install_policy_auto_deny(monkeypatch, smart_deny_session)

    result = A.check_execute_code_guard("import os; os.system('rm -rf /')",
                                        "local")

    assert result["approved"] is False
    assert result["approval_outcome"] == A.APPROVAL_OUTCOME_GUARDIAN_DENIED
    assert result["effect_class"] == "arbitrary_code_execution"
    for phrase in USER_ATTRIBUTION_PHRASES:
        assert phrase not in result["message"], phrase


def test_smart_deny_final_never_asks_anyone(smart_deny_session, monkeypatch):
    """``approvals.smart_deny_final`` decides in core, with no prompt at all."""
    calls: list = []
    _register_notify(smart_deny_session, calls)
    monkeypatch.setattr(A, "_smart_deny_is_final", lambda: True)

    result = A.check_all_command_guards("rm -rf /srv/data", "local")

    assert calls == []
    assert result["approval_outcome"] == A.APPROVAL_OUTCOME_GUARDIAN_DENIED


# ---------------------------------------------------------------------------
# (b) A real human decision keeps its existing wording
# ---------------------------------------------------------------------------

def test_explicit_user_deny_keeps_user_attribution(smart_deny_session):
    _register_resolver(smart_deny_session, "deny")

    result = A.check_all_command_guards("rm -rf /srv/data", "local")

    assert result["approved"] is False
    assert result["approval_outcome"] == A.APPROVAL_OUTCOME_USER_DENIED
    assert result["outcome"] == "denied"
    assert result["user_consent"] is False
    assert "denied by user" in result["message"]
    assert "has NOT consented" in result["message"]


def test_unanswered_request_is_timed_out_not_denied(
    smart_deny_session, monkeypatch
):
    calls: list = []
    _register_notify(smart_deny_session, calls)
    monkeypatch.setattr(A, "_get_approval_timeout", lambda: 0)

    result = A.check_all_command_guards("rm -rf /srv/data", "local")

    assert calls, "the user must actually have been asked"
    assert result["approval_outcome"] == A.APPROVAL_OUTCOME_TIMED_OUT
    assert result["outcome"] == "timeout"
    assert "Silence is not consent" in result["message"]


# ---------------------------------------------------------------------------
# (c) Breaker guidance offers no manual bypass and no raw command
# ---------------------------------------------------------------------------

def _trip_breaker(monkeypatch, session_key) -> dict:
    """Drive the breaker to its first stage: the dangerous-operation lockout."""
    _install_policy_auto_deny(monkeypatch, session_key)
    A.check_all_command_guards("rm -rf /srv/one", "local")
    A.check_all_command_guards("rm -rf /srv/two", "local")
    return A.check_all_command_guards("rm -rf /srv/secret-three", "local")


def _hard_stop_breaker(monkeypatch, session_key) -> dict:
    """Keep attempting dangerous work through the lockout until the turn ends."""
    _trip_breaker(monkeypatch, session_key)
    A.check_all_command_guards("rm -rf /srv/four", "local")
    A.check_all_command_guards("rm -rf /srv/five", "local")
    return A.check_all_command_guards("rm -rf /srv/secret-six", "local")


def test_lockout_message_offers_no_manual_bypass(
    smart_deny_session, monkeypatch
):
    third = _trip_breaker(monkeypatch, smart_deny_session)

    assert "DANGEROUS-OPERATION LOCKOUT:" in third["message"]
    for phrase in MANUAL_BYPASS_PHRASES:
        assert phrase not in third["message"], phrase


def test_breaker_message_offers_no_manual_bypass(
    smart_deny_session, monkeypatch
):
    sixth = _hard_stop_breaker(monkeypatch, smart_deny_session)

    assert "CIRCUIT BREAKER:" in sixth["message"]
    for phrase in MANUAL_BYPASS_PHRASES:
        assert phrase not in sixth["message"], phrase


def test_lockout_refusal_never_claims_a_human_refused_it(
    smart_deny_session, monkeypatch
):
    """A candidate refused without assessment is nobody's decision but policy's."""
    _trip_breaker(monkeypatch, smart_deny_session)
    fourth = A.check_all_command_guards("rm -rf /srv/four", "local")

    assert fourth["denial_lockout"] is True
    assert fourth["user_consent"] is None
    assert fourth["approval_outcome"] == A.APPROVAL_OUTCOME_GUARDIAN_DENIED
    for phrase in USER_ATTRIBUTION_PHRASES:
        assert phrase not in fourth["message"], phrase


def test_breaker_side_channel_carries_reason_not_command(
    smart_deny_session, monkeypatch
):
    sixth = _hard_stop_breaker(monkeypatch, smart_deny_session)
    metadata = sixth[A.APPROVAL_BREAKER_METADATA_KEY]

    assert metadata["reason_code"] == "recursive_delete"
    assert metadata["effect_class"] == "destructive_filesystem"
    assert metadata["safe_alternative"]
    # The blocked command never enters the side channel or the model text.
    for value in metadata.values():
        assert "secret-six" not in str(value)
    assert "secret-six" not in sixth["message"]


def test_final_turn_message_offers_no_manual_bypass():
    message = approval_breaker_final_message({
        "tool_name": "terminal",
        "count": 3,
        "threshold": 3,
        "reason_code": "recursive_delete",
        "effect_class": "destructive_filesystem",
        "safe_alternative": A._safe_alternative_for("destructive_filesystem"),
    })

    for phrase in MANUAL_BYPASS_PHRASES:
        assert phrase not in message, phrase
    assert "destructive_filesystem" in message
    assert "recursive_delete" in message


def test_final_turn_message_survives_a_v1_side_channel_record():
    """A record written before the redacted detail existed still renders."""
    message = approval_breaker_final_message({
        "tool_name": "terminal",
        "count": 3,
        "threshold": 3,
    })

    assert "terminal" in message
    for phrase in MANUAL_BYPASS_PHRASES:
        assert phrase not in message, phrase


# ---------------------------------------------------------------------------
# (d) Logs never carry the raw session key
# ---------------------------------------------------------------------------

def test_breaker_warning_does_not_log_the_discord_session_key(
    smart_deny_session, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="tools.approval")

    sixth = _hard_stop_breaker(monkeypatch, smart_deny_session)
    assert "CIRCUIT BREAKER:" in sixth["message"]

    assert "Smart-approval circuit breaker tripped" in caplog.text
    # The stage line is what tells lockout apart from a turn-ending stop.
    assert "Smart-approval circuit breaker stage" in caplog.text
    assert "lockout" in caplog.text
    assert "hard_stop" in caplog.text
    assert DISCORD_SESSION_KEY not in caplog.text
    for fragment in DISCORD_ID_FRAGMENTS:
        assert fragment not in caplog.text, fragment


def test_session_fingerprint_is_stable_and_not_reversible():
    first = A._log_session_key(DISCORD_SESSION_KEY)
    second = A._log_session_key(DISCORD_SESSION_KEY)

    assert first == second
    assert first != A._log_session_key(DISCORD_SESSION_KEY + "x")
    assert A._log_session_key("") == "none"
    for fragment in DISCORD_ID_FRAGMENTS:
        assert fragment not in first


# ---------------------------------------------------------------------------
# Reason code / effect class derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern_key,expected_class", [
    ("recursive delete", "destructive_filesystem"),
    ("SQL DROP", "destructive_data"),
    ("execute_code", "arbitrary_code_execution"),
    ("force kill processes", "process_control"),
    ("world/other-writable permissions", "permission_change"),
    ("something nobody classified", "unclassified"),
    # These three fell through to "unclassified" and handed the model the
    # generic advice, which is what let it keep re-trying the same inline
    # script under a different syntax until the denial breaker tripped.
    ("script execution via heredoc", "arbitrary_code_execution"),
    ("script execution via -e/-c flag", "arbitrary_code_execution"),
    ("tirith:lookalike_tld", "network_egress"),
])
def test_effect_class_derivation(pattern_key, expected_class):
    context = A._deny_context([pattern_key])
    assert context["effect_class"] == expected_class
    assert context["safe_alternative"]


def test_script_denial_keeps_native_alternatives_without_saved_code_bypass():
    """Actionable alternatives preserve the original refused-effect boundary."""
    alternative = A._deny_context(["script execution via heredoc"])["safe_alternative"]
    assert "read_file / write_file" in alternative
    assert "verify that its effects are authorized" in alternative
    assert "Saving refused code to a file does not make its execution safe" in alternative
    assert "bash /path/script.sh" not in alternative


def test_every_effect_class_has_a_japanese_alternative():
    """The user-facing breaker message must not fall back to English.

    ``approval_breaker_final_message`` writes Japanese, so a new effect class
    added to tools.approval without a counterpart here would silently drop an
    English sentence into the middle of it.
    """
    # A class added to _EFFECT_CLASS_RULES but not to the English table falls
    # back to the generic advice — the same silent hole that left inline-script
    # denials unactionable. _UNCLASSIFIED_EFFECT is table-only, hence subset.
    assert {c for c, _ in A._EFFECT_CLASS_RULES} <= set(
        A._SAFE_ALTERNATIVES_BY_EFFECT_CLASS
    )
    assert set(_SAFE_ALTERNATIVE_JA_BY_EFFECT_CLASS) == set(
        A._SAFE_ALTERNATIVES_BY_EFFECT_CLASS
    )


def test_final_turn_message_localizes_the_safe_alternative():
    message = approval_breaker_final_message({
        "tool_name": "terminal",
        "count": 3,
        "threshold": 3,
        "reason_code": "script_execution_via_heredoc",
        "effect_class": "arbitrary_code_execution",
        # The English text the model saw travels on the side channel; the
        # user-facing message renders the Japanese counterpart instead.
        "safe_alternative": A._safe_alternative_for("arbitrary_code_execution"),
    })

    assert _SAFE_ALTERNATIVE_JA_BY_EFFECT_CLASS["arbitrary_code_execution"] in message
    assert "use the dedicated built-in tools" not in message


def test_final_turn_message_keeps_an_unknown_class_alternative():
    """An effect class this build has no translation for still says something."""
    message = approval_breaker_final_message({
        "tool_name": "terminal",
        "count": 3,
        "threshold": 3,
        "effect_class": "some_future_class",
        "safe_alternative": "do the safe thing instead",
    })

    assert "do the safe thing instead" in message


def test_reason_code_is_bounded_and_alphanumeric():
    code = A._normalize_reason_code("rm -rf /home/someone/secret dir; " * 20)
    assert len(code) <= A._REASON_CODE_MAX_LEN
    assert all(character.isalnum() or character == "_" for character in code)
    assert A._normalize_reason_code("") == "unspecified"
    assert A._normalize_reason_code("!!!") == "unspecified"
