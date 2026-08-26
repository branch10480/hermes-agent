"""Structured deny logging (H-006): one INFO line per denial, no raw command.

Every denial layer -- the hardline floor, the sudo-stdin guard, user
``approvals.deny`` rules, the Smart Approval guardian, an explicit user deny,
and an approval timeout -- now emits the same ``approval_deny`` line with the
same fields. Before this, the three pre-approval block paths logged the raw
command into the persisted agent.log and carried no reason code at all.

Three properties are load-bearing here:

1. Every origin emits exactly one structured line.
2. The line never contains the raw command, the raw session key, or the raw
   user deny glob -- only run-local fingerprints.
3. The circuit breaker is untouched. Its tally counts guardian DANGEROUS
   denials only; the observation counter added for this logging is separate,
   and folding the two would trip the breaker on commands the guardian never
   assessed.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import tools.approval as mod
from agent.redact import command_fingerprint, session_key_fingerprint

# Distinctive enough that a substring check is meaningful.
SESSION_KEY = "discord:112233445566778899:998877665544332211:123456789012345678"
CANARY = "CANARYCMD9f3a"


def _deny_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("approval_deny ")]


def _fields(line: str) -> dict[str, str]:
    """Parse one ``approval_deny k=v k=v`` line into a dict."""
    return dict(part.split("=", 1) for part in line.split()[1:])


@pytest.fixture
def deny_env(monkeypatch, caplog):
    """Clean, non-interactive approval state with INFO capture on."""
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION",
                "HERMES_CRON_SESSION", "HERMES_INTERACTIVE",
                "HERMES_EXEC_ASK", "SUDO_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(mod, "_get_approval_config",
                        lambda: {"mode": "manual", "deny": []})
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _: {"action": "allow", "findings": [], "summary": ""},
    )
    token = mod.set_current_session_key(SESSION_KEY)
    mod._denial_tally.clear()
    mod._deny_event_tally.clear()
    caplog.set_level(logging.INFO, logger="tools.approval")
    try:
        yield
    finally:
        mod._denial_tally.clear()
        mod._deny_event_tally.clear()
        try:
            mod._approval_session_key.reset(token)
        except Exception:
            pass
        mod.clear_session(SESSION_KEY)


class TestCommandFingerprint:
    def test_is_opaque_and_bounded(self):
        fp = command_fingerprint(f"rm -rf /tmp/{CANARY}")
        assert CANARY not in fp
        # blake2s digest_size=6 → 12 hex chars.
        assert len(fp) == 12
        assert int(fp, 16) >= 0

    def test_empty_command_is_none(self):
        assert command_fingerprint("") == command_fingerprint(None) == "none"

    def test_stable_within_a_run(self):
        assert command_fingerprint("ls -la") == command_fingerprint("ls -la")

    def test_distinct_commands_get_distinct_fingerprints(self):
        assert command_fingerprint("ls -la") != command_fingerprint("ls -l")

    def test_uses_a_different_salt_than_session_keys(self):
        """Sharing the session salt would link a string logged as a command to
        the same string logged as a session key."""
        assert command_fingerprint(SESSION_KEY) != session_key_fingerprint(SESSION_KEY)


class TestPolicyBlockOrigins:
    """The three paths that block before the approval layer is reached."""

    def test_hardline_block_logs_one_structured_line(self, deny_env, caplog):
        result = mod.check_all_command_guards(f"rm -rf / # {CANARY}", "local")
        assert result["approved"] is False
        assert result["hardline"] is True

        lines = _deny_lines(caplog)
        assert len(lines) == 1
        fields = _fields(lines[0])
        assert fields["origin"] == mod.APPROVAL_OUTCOME_HARDLINE
        assert fields["reason_code"] == "recursive_delete_of_root_filesystem"
        assert fields["effect_class"] == "destructive_filesystem"
        assert fields["turn_deny_count"] == "1"

    def test_sudo_stdin_block_logs_one_structured_line(self, deny_env, caplog):
        result = mod.check_all_command_guards(
            f"echo hunter2 | sudo -S rm /tmp/{CANARY}", "local")
        assert result["approved"] is False

        lines = _deny_lines(caplog)
        assert len(lines) == 1
        fields = _fields(lines[0])
        assert fields["origin"] == mod.APPROVAL_OUTCOME_SUDO_STDIN
        assert fields["effect_class"] == "permission_change"

    def test_user_deny_rule_logs_a_glob_fingerprint_not_the_glob(
            self, deny_env, caplog, monkeypatch):
        glob = f"*{CANARY}*"
        monkeypatch.setattr(mod, "_get_approval_config",
                            lambda: {"mode": "manual", "deny": [glob]})

        result = mod.check_all_command_guards(
            f"curl https://{CANARY}.example.test", "local")
        assert result["approved"] is False
        assert result["user_deny"] is True

        lines = _deny_lines(caplog)
        assert len(lines) == 1
        fields = _fields(lines[0])
        assert fields["origin"] == mod.APPROVAL_OUTCOME_USER_DENY_RULE
        assert fields["reason_code"] == "user_deny_rule"
        assert fields["deny_rule_id"] == f"glob:{command_fingerprint(glob)}"

    def test_check_dangerous_command_uses_the_same_paths(self, deny_env, caplog):
        """The older entry point must not drift from check_all_command_guards."""
        result = mod.check_dangerous_command(f"rm -rf / # {CANARY}", "local")
        assert result["approved"] is False
        fields = _fields(_deny_lines(caplog)[0])
        assert fields["origin"] == mod.APPROVAL_OUTCOME_HARDLINE

    @pytest.mark.parametrize("command,extra_config", [
        (f"rm -rf / # {CANARY}", {}),
        (f"echo hunter2 | sudo -S rm /tmp/{CANARY}", {}),
        (f"curl https://{CANARY}.example.test", {"deny": [f"*{CANARY}*"]}),
    ])
    def test_no_raw_command_session_key_or_glob_in_any_log_record(
            self, deny_env, caplog, monkeypatch, command, extra_config):
        if extra_config:
            monkeypatch.setattr(
                mod, "_get_approval_config",
                lambda: {"mode": "manual", "deny": [], **extra_config})

        mod.check_all_command_guards(command, "local")

        emitted = "\n".join(r.getMessage() for r in caplog.records)
        assert emitted, "expected at least one log record"
        assert CANARY not in emitted
        assert SESSION_KEY not in emitted
        assert "112233445566778899" not in emitted

    def test_block_result_carries_the_denial_detail(self, deny_env):
        result = mod.check_all_command_guards(f"rm -rf / # {CANARY}", "local")
        assert result["approval_outcome"] == mod.APPROVAL_OUTCOME_HARDLINE
        assert result["outcome"] == "blocked"
        assert result["reason_code"] == "recursive_delete_of_root_filesystem"
        assert result["effect_class"] == "destructive_filesystem"
        assert result["safe_alternative"]
        # Nobody was asked and nobody can override the hardline floor.
        assert result["user_consent"] is None

    def test_user_deny_rule_result_reports_no_user_consent(
            self, deny_env, monkeypatch):
        monkeypatch.setattr(mod, "_get_approval_config",
                            lambda: {"mode": "manual", "deny": [f"*{CANARY}*"]})
        result = mod.check_all_command_guards(
            f"curl https://{CANARY}.example.test", "local")
        assert result["approval_outcome"] == mod.APPROVAL_OUTCOME_USER_DENY_RULE
        assert result["user_consent"] is False
        assert result["reason_code"] == "user_deny_rule"

    def test_hardline_message_and_flags_are_unchanged(self, deny_env):
        """Additive only: the model-facing contract must not move."""
        result = mod.check_all_command_guards("rm -rf /", "local")
        assert result["approved"] is False
        assert result["hardline"] is True
        assert result["message"].startswith("BLOCKED (hardline):")


class TestPolicyBlockObserverHook:
    def _capture(self, command, config=None):
        captured = []
        with patch("hermes_cli.plugins.invoke_hook",
                   side_effect=lambda name, **kw: captured.append((name, kw))):
            result = mod.check_all_command_guards(command, "local")
        return result, captured

    def test_post_hook_fires_with_the_new_fields(self, deny_env):
        _result, captured = self._capture(f"rm -rf / # {CANARY}")

        # No pre_approval_request: nobody was asked.
        assert [name for name, _ in captured] == ["post_approval_response"]
        payload = captured[0][1]
        assert payload["origin"] == mod.APPROVAL_OUTCOME_HARDLINE
        assert payload["approval_outcome"] == mod.APPROVAL_OUTCOME_HARDLINE
        assert payload["reason_code"] == "recursive_delete_of_root_filesystem"
        assert payload["effect_class"] == "destructive_filesystem"
        assert payload["deny_count"] == 1
        assert payload["decided_by"] == "policy"
        assert payload["surface"] == "policy"
        # Bounded vocabulary existing metrics consumers already understand.
        assert payload["choice"] == "deny"
        assert payload["command_fingerprint"] == command_fingerprint(
            f"rm -rf / # {CANARY}")
        assert payload["safe_alternative"]

    def test_deny_count_climbs_within_a_turn(self, deny_env):
        counts = []
        with patch("hermes_cli.plugins.invoke_hook",
                   side_effect=lambda name, **kw: counts.append(kw["deny_count"])):
            mod.check_all_command_guards("rm -rf /", "local")
            mod.check_all_command_guards("mkfs.ext4 /dev/sda1", "local")
        assert counts == [1, 2]

    def test_user_deny_rule_hook_never_carries_the_glob(self, deny_env, monkeypatch):
        glob = f"*{CANARY}*"
        monkeypatch.setattr(mod, "_get_approval_config",
                            lambda: {"mode": "manual", "deny": [glob]})
        _result, captured = self._capture(f"curl https://{CANARY}.example.test")
        payload = captured[0][1]
        assert glob not in str(payload)
        assert payload["deny_rule_id"] == f"glob:{command_fingerprint(glob)}"
        assert payload["description"] == mod._USER_DENY_RULE_DESCRIPTION

    def test_a_crashing_observer_never_changes_the_block(self, deny_env):
        with patch("hermes_cli.plugins.invoke_hook",
                   side_effect=RuntimeError("observer failed")):
            result = mod.check_all_command_guards("rm -rf /", "local")
        assert result["approved"] is False
        assert result["hardline"] is True

    def test_a_crashing_redactor_drops_the_command_but_still_reports(self, deny_env):
        captured = []
        with (
            patch("agent.redact.redact_sensitive_text",
                  side_effect=RuntimeError("redactor failed")),
            patch("hermes_cli.plugins.invoke_hook",
                  side_effect=lambda name, **kw: captured.append(kw)),
        ):
            result = mod.check_all_command_guards(f"rm -rf / # {CANARY}", "local")
        assert result["approved"] is False
        assert len(captured) == 1
        assert "command" not in captured[0]
        assert "command_truncated" not in captured[0]
        assert captured[0]["command_fingerprint"]

    def test_oversized_payloads_are_bounded_and_flagged(self, deny_env):
        # A giant inline payload trips the parser-limit hardline block, which
        # is exactly where an unbounded command would reach observers.
        payload = "echo " + ("A" * 200_000)
        _result, captured = self._capture(payload)
        hook = captured[0][1]
        assert hook["command_truncated"] is True
        assert len(hook["command"]) <= mod._POLICY_BLOCK_HOOK_COMMAND_CHARS
        assert hook["command_fingerprint"] == command_fingerprint(payload)

    def test_ordinary_commands_are_not_flagged_as_truncated(self, deny_env):
        _result, captured = self._capture("rm -rf /")
        assert captured[0][1]["command_truncated"] is False

    def test_a_secret_straddling_the_preview_cut_is_fully_redacted(
            self, deny_env):
        # Redaction must run before the 4096-char cut: cutting first splits
        # this key at the boundary into a 12-char head fragment — too short
        # for the ``sk-`` vendor pattern (10+ chars after the prefix) — which
        # then survives redaction verbatim. Redacting first, the redactor
        # sees the whole key and collapses it to its head/tail marker, so the
        # key body never reaches the payload.
        secret = "sk-ant-api03-" + "Zq7" * 30
        # "echo " is 5 chars and the key needs a word boundary before it;
        # place it so exactly 12 of its chars fall before the cut.
        head_len = mod._POLICY_BLOCK_HOOK_COMMAND_CHARS - 12 - 6
        payload = "echo " + ("A" * head_len) + " " + secret + (" B" * 100_000)
        _result, captured = self._capture(payload)
        command = captured[0][1]["command"]
        assert "api03" not in command
        assert "Zq7Zq7" not in command
        assert len(command) <= mod._POLICY_BLOCK_HOOK_COMMAND_CHARS


class TestApprovalFlowOrigins:
    """guardian_denied / user_denied / timed_out already had a deny context;
    they now emit the same structured line as everything else."""

    def _smart_deny(self, monkeypatch, *, final=True):
        # An approval surface has to exist at all, or both guards return
        # approved=True long before the guardian is consulted.
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(mod, "_smart_approve", lambda *_: "deny")
        monkeypatch.setattr(mod, "_smart_deny_is_final", lambda: final)

    def test_guardian_deny_logs_the_structured_line(
            self, deny_env, caplog, monkeypatch):
        self._smart_deny(monkeypatch)
        result = mod.check_all_command_guards(f"rm -rf /tmp/{CANARY}", "local")
        assert result["approved"] is False
        assert result["smart_denied"] is True

        fields = _fields(_deny_lines(caplog)[0])
        assert fields["origin"] == mod.APPROVAL_OUTCOME_GUARDIAN_DENIED
        assert fields["effect_class"] == "destructive_filesystem"
        assert CANARY not in "\n".join(r.getMessage() for r in caplog.records)

    def test_guardian_deny_on_execute_code_logs_the_structured_line(
            self, deny_env, caplog, monkeypatch):
        self._smart_deny(monkeypatch)
        result = mod.check_execute_code_guard(f"print('{CANARY}')", "local")
        assert result["approved"] is False
        fields = _fields(_deny_lines(caplog)[0])
        assert fields["origin"] == mod.APPROVAL_OUTCOME_GUARDIAN_DENIED

    @pytest.mark.parametrize("choice,origin", [
        ("deny", mod.APPROVAL_OUTCOME_USER_DENIED),
        ("timeout", mod.APPROVAL_OUTCOME_TIMED_OUT),
    ])
    def test_cli_deny_and_timeout_log_distinct_origins(
            self, deny_env, caplog, monkeypatch, choice, origin):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "manual")

        def cb(command, description, *, allow_permanent=True, **kwargs):
            return choice

        result = mod.check_all_command_guards(
            f"rm -rf /tmp/{CANARY}", "local", approval_callback=cb)
        assert result["approved"] is False

        lines = _deny_lines(caplog)
        assert len(lines) == 1
        assert _fields(lines[0])["origin"] == origin

    def test_an_approved_command_logs_nothing(self, deny_env, caplog, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "manual")

        def cb(command, description, *, allow_permanent=True, **kwargs):
            return "once"

        result = mod.check_all_command_guards(
            "rm -rf /tmp/approved", "local", approval_callback=cb)
        assert result["approved"] is True
        assert _deny_lines(caplog) == []

    def test_every_documented_origin_is_reachable(self):
        """The vocabulary the docs promise is the vocabulary the code emits."""
        assert set(mod._DENY_ORIGINS) == {
            "guardian_denied", "user_denied", "timed_out",
            "hardline", "sudo_stdin", "user_deny_rule",
        }
        for origin in mod._DENY_ORIGINS:
            assert origin in mod._LEGACY_OUTCOME_BY_APPROVAL_OUTCOME


class TestBreakerIsUnaffected:
    """The breaker's tally must keep counting guardian DANGEROUS denials only."""

    def test_policy_blocks_never_touch_the_breaker_tally(self, deny_env, monkeypatch):
        monkeypatch.setattr(mod, "_get_denial_breaker_threshold", lambda: 1)
        for _ in range(5):
            result = mod.check_all_command_guards("rm -rf /", "local")
            assert mod.APPROVAL_BREAKER_METADATA_KEY not in result
        assert mod._denial_tally == {}
        # The observation counter did move -- that is the whole point of it
        # being a separate dict.
        assert sum(mod._deny_event_tally.values()) == 5

    def test_guardian_denials_still_trip_the_breaker(self, deny_env, monkeypatch):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(mod, "_smart_approve", lambda *_: "deny")
        monkeypatch.setattr(mod, "_smart_deny_is_final", lambda: True)
        monkeypatch.setattr(mod, "_get_denial_breaker_threshold", lambda: 2)

        first = mod.check_all_command_guards("rm -rf /tmp/a", "local")
        second = mod.check_all_command_guards("rm -rf /tmp/b", "local")

        assert mod.APPROVAL_BREAKER_METADATA_KEY not in first
        metadata = second[mod.APPROVAL_BREAKER_METADATA_KEY]
        assert metadata["tripped"] is True
        assert metadata["count"] == 2

    def test_interleaved_policy_blocks_do_not_advance_the_breaker(
            self, deny_env, monkeypatch):
        """A hardline block between two guardian denials must not count as a
        third one -- that would trip the breaker a denial early."""
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(mod, "_smart_deny_is_final", lambda: True)
        monkeypatch.setattr(mod, "_get_denial_breaker_threshold", lambda: 3)
        monkeypatch.setattr(mod, "_smart_approve", lambda *_: "deny")

        mod.check_all_command_guards("rm -rf /tmp/a", "local")
        mod.check_all_command_guards("rm -rf /", "local")  # hardline, not guardian
        second = mod.check_all_command_guards("rm -rf /tmp/b", "local")

        assert mod.APPROVAL_BREAKER_METADATA_KEY not in second
        assert mod._denial_tally[mod._denial_tally_key(SESSION_KEY)] == 2

    def test_observation_counter_is_bounded(self, deny_env, monkeypatch):
        monkeypatch.setattr(mod, "_DENY_EVENT_TALLY_MAX_SESSIONS", 4)
        for i in range(20):
            mod._record_deny_event(f"session-{i}")
        assert len(mod._deny_event_tally) == 4
