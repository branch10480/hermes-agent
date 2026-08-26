"""The harness refuses to let an agent turn rewrite its own safety settings.

Observation H-018: the terminal guard checked ordinary commands but nothing
stopped ``hermes config set approvals.mode off`` from a Discord turn, so a
single turn could permanently widen every gate it would meet afterwards. The
config-file write guard did not cover it either — the CLI reaches the same
values without touching the file directly.

Two boundaries are covered here: the ``terminal`` command check that runs
before a shell ever sees the command, and the in-process config writers that a
tool implementation could reach without a shell.
"""

import json

import pytest

import tools.safety_config_guard as guard_module
import tools.terminal_tool as terminal_tool_module
from tools.safety_config_guard import (
    SAFETY_CONFIG_DENIED_REASON_CODE,
    SafetyConfigChangeDenied,
    changed_protected_config_keys,
    find_safety_config_change_in_command,
    matched_protected_config_key,
    owner_initiated_safety_config_change,
)

SESSION_KEY = "agent:main:discord:group:chat-1:thread-1"


def _as_session(monkeypatch, *, gateway=True, cron=False):
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    if gateway:
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    if cron:
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")


def _denial(command):
    raw = terminal_tool_module._safety_config_change_denial(command)
    return None if raw is None else json.loads(raw)


# --------------------------------------------------------------------------
# Which keys count as safety settings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "approvals.mode",
        "approvals.cron_mode",
        "approvals.denial_breaker_threshold",
        "approvals.smart_policy",
        "command_allowlist",
        "security.tirith_enabled",
        "security.tirith_path",
        "security.tirith_fail_open",
        "security.tirith_fail_open_gateway",
        "security.redact_secrets",
        "toolsets",
        "platform_toolsets",
        "agent.disabled_toolsets",
        "hooks_auto_accept",
        "skills.inline_shell",
        "delegation.subagent_auto_approve",
        "terminal.backend",
        "proxy.enforce_on_docker",
        "secrets.onepassword.binary_path",
        "discord.allowed_channels",
        "platforms.discord.allow_admin_from",
        "dashboard.basic_auth.password_hash",
    ],
)
def test_safety_keys_are_protected(key):
    assert matched_protected_config_key(key) is not None


@pytest.mark.parametrize(
    "key",
    [
        "display.skin",
        "display.timestamps",
        "model",
        "model.default",
        "agent.max_turns",
        "terminal.timeout",
        "compression.threshold",
        "auxiliary.free_only",
        "tts.provider",
    ],
)
def test_ordinary_keys_are_not_protected(key):
    assert matched_protected_config_key(key) is None


def test_overwriting_a_parent_section_is_protected_too():
    """`config set agent '{}'` would take agent.disabled_toolsets with it."""
    assert matched_protected_config_key("agent") == "agent.disabled_toolsets"
    assert matched_protected_config_key("security") == "security"


def test_key_matching_ignores_case():
    assert matched_protected_config_key("APPROVALS.Mode") == "approvals"


@pytest.mark.parametrize(
    "key",
    [
        # F-8(a): the owner's recovery / audit channel is protected as its own
        # class. No code path writes it, but an agent that could must not.
        "updates",
        "updates.channel",
        "updates.auto",
        "updates.pin",
    ],
)
def test_updates_subtree_is_protected(key):
    assert matched_protected_config_key(key) == "updates"


# --------------------------------------------------------------------------
# Terminal commands on the messaging / cron surfaces
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "hermes config set security.tirith_fail_open_gateway true",
        "hermes config set approvals.mode off",
        "hermes config unset approvals.denial_breaker_threshold",
        "hermes config set security.tirith_path /tmp/not-tirith",
        "hermes config edit",
        "hermes approvals suggest --apply 1,2",
        "hermes tools disable web --platform discord",
    ],
)
def test_direct_safety_config_change_is_denied_on_gateway(monkeypatch, command):
    _as_session(monkeypatch)

    result = _denial(command)
    assert result is not None
    assert result["status"] == "blocked"
    assert result["reason_code"] == SAFETY_CONFIG_DENIED_REASON_CODE


def test_safety_config_change_is_denied_on_cron(monkeypatch):
    _as_session(monkeypatch, gateway=False, cron=True)

    result = _denial("hermes config set approvals.mode off")
    assert result is not None
    assert result["surface"] == "cron"


@pytest.mark.parametrize(
    "command",
    [
        "env FOO=1 hermes config set approvals.mode off",
        "command hermes config set approvals.mode off",
        "sudo hermes config set approvals.mode off",
        "timeout 30 hermes config set approvals.mode off",
        "env -u HERMES_GATEWAY_SESSION hermes config set approvals.mode off",
        "/usr/local/bin/hermes config set approvals.mode off",
        "./venv/bin/hermes config set approvals.mode off",
        "bash -lc 'hermes config set approvals.mode off'",
        "sh -c \"cd /tmp && hermes config set approvals.mode off\"",
        "zsh --command 'hermes config set approvals.mode off'",
        "echo start && hermes config set approvals.mode off",
        "echo start; hermes config set approvals.mode off",
        "true | hermes config set approvals.mode off",
        "OUT=$(hermes config set approvals.mode off)",
        "echo `hermes config set approvals.mode off`",
        "python3 -m hermes_cli.main config set approvals.mode off",
        "python3 -c 'import os; os.system(\"hermes config set approvals.mode off\")'",
        "perl -e 'system(q{hermes config set approvals.mode off})'",
    ],
)
def test_wrapped_forms_are_denied(monkeypatch, command):
    _as_session(monkeypatch)

    assert _denial(command) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        # F-1: a value-taking global flag before the subcommand used to make its
        # *value* look like the subcommand, so the whole check fell through.
        "hermes -m foo config set security.tirith_enabled false",
        "hermes --model anthropic/claude-sonnet-4 config set approvals.mode off",
        "hermes --provider openrouter config set security.tirith_path /tmp/x",
        "hermes --reasoning high config set approvals.mode off",
        "hermes -t web config unset approvals.denial_breaker_threshold",
        "hermes --resume latest config set approvals.mode off",
        "hermes -r latest config set approvals.mode off",
        "hermes --in /tmp config set approvals.mode off",
        "hermes -z 'do a thing' config set security.tirith_enabled false",
        "hermes --usage-file /tmp/u.json config set approvals.mode off",
        "hermes -c mysession config set approvals.mode off",
        "hermes -s myskill config set approvals.mode off",
        # The flag value literally repeats the subcommand word.
        "hermes -m config config set security.tirith_enabled false",
        "hermes config edit config",  # config edit is refused wholesale
        "hermes -m foo approvals suggest --apply 1,2",
        "hermes -m foo tools disable web --platform discord",
        # Wrapper + global flag combined.
        "env A=1 hermes -m foo config set approvals.mode off",
        "bash -lc 'hermes -m foo config set approvals.mode off'",
    ],
)
def test_global_flags_before_subcommand_are_denied(monkeypatch, command):
    _as_session(monkeypatch)

    assert _denial(command) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        # A flag whose value equals a guarded subcommand, but with no protected
        # write following, must NOT trip the guard.
        "hermes -m config chat",
        "hermes --resume config chat -q hi",
        # Non-protected key stays on the ordinary path even behind a global flag.
        "hermes -m foo config set model anthropic/claude-sonnet-4",
        "hermes --reasoning high config get approvals.mode",
    ],
)
def test_global_flags_do_not_over_block_ordinary_commands(monkeypatch, command):
    _as_session(monkeypatch)

    assert _denial(command) is None, command


@pytest.mark.parametrize(
    "command",
    [
        # Non-protected keys keep taking the ordinary approval path.
        "hermes config set display.skin synthwave",
        "hermes config set model anthropic/claude-sonnet-4",
        "hermes config set agent.max_turns 200",
        "hermes config unset display.timestamps",
        # Reading is always fine.
        "hermes config get approvals.mode",
        "hermes config show",
        "hermes approvals suggest",
        "hermes tools list",
        # Ordinary work must not trip the textual fallback.
        "git commit -m 'document why hermes config set approvals.mode is refused'",
        "echo 'hermes config set approvals.mode off'",
        "rg -n 'hermes config set approvals' docs/",
        "pytest -q tests/tools",
        "ls -la",
    ],
)
def test_ordinary_commands_are_untouched_on_gateway(monkeypatch, command):
    _as_session(monkeypatch)

    assert _denial(command) is None, command


@pytest.mark.parametrize(
    "command",
    [
        "hermes config set approvals.mode off",
        "hermes config set security.tirith_enabled false",
        "hermes config edit",
    ],
)
def test_local_surface_keeps_its_current_behavior(command):
    """CLI / TUI / desktop: a human owns the terminal, so nothing changes."""
    assert _denial(command) is None


def test_denial_names_the_setting_and_a_non_bypassing_alternative(monkeypatch):
    _as_session(monkeypatch)

    result = _denial("hermes config set security.tirith_fail_open_gateway true")

    assert result["reason_code"] == SAFETY_CONFIG_DENIED_REASON_CODE
    assert result["effect_class"] == "safety_control_modification"
    assert result["protected_setting"] == "security"
    alternative = result["safe_alternative"]
    assert "owner" in alternative
    # The alternative must not send the model looking for a way around the
    # boundary, and must not hand it a host path or a value to try.
    for bypass in ("/approve", "yolo", "sudo", "~/.hermes", "config.yaml"):
        assert bypass not in alternative
    assert "true" not in result["protected_setting"]


def test_denial_log_line_carries_no_command_text(monkeypatch, caplog):
    _as_session(monkeypatch)

    with caplog.at_level("WARNING", logger=terminal_tool_module.logger.name):
        _denial("hermes config set security.tirith_path /tmp/attacker/tirith")

    assert caplog.records, "the denial should be observable in the log"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "/tmp/attacker" not in logged
    assert SESSION_KEY not in logged


# --------------------------------------------------------------------------
# In-process config writers (defense in depth)
# --------------------------------------------------------------------------


def test_set_config_value_refuses_a_protected_key_on_gateway(monkeypatch):
    from hermes_cli.config import set_config_value

    _as_session(monkeypatch)

    with pytest.raises(SafetyConfigChangeDenied) as excinfo:
        set_config_value("approvals.mode", "off")

    assert excinfo.value.reason_code == SAFETY_CONFIG_DENIED_REASON_CODE
    assert excinfo.value.surface == "gateway"


def test_unset_config_value_refuses_a_protected_key_on_gateway(monkeypatch):
    from hermes_cli.config import unset_config_value

    _as_session(monkeypatch)

    with pytest.raises(SafetyConfigChangeDenied):
        unset_config_value("security.tirith_enabled")


def test_set_config_value_still_writes_ordinary_keys_on_gateway(monkeypatch):
    from hermes_cli.config import get_config_path, set_config_value
    from utils import fast_safe_load

    _as_session(monkeypatch)

    set_config_value("display.skin", "synthwave")

    saved = fast_safe_load(get_config_path().read_text(encoding="utf-8"))
    assert saved["display"]["skin"] == "synthwave"


def test_set_config_value_is_unchanged_on_a_local_surface():
    from hermes_cli.config import get_config_path, set_config_value
    from utils import fast_safe_load

    set_config_value("approvals.mode", "off")

    saved = fast_safe_load(get_config_path().read_text(encoding="utf-8"))
    assert saved["approvals"]["mode"] == "off"


def test_owner_initiated_writes_are_allowed_through(monkeypatch):
    """The /approvals slash command runs on the gateway but is owner-driven."""
    from hermes_cli.config import get_config_path, set_config_value
    from utils import fast_safe_load

    _as_session(monkeypatch)

    with owner_initiated_safety_config_change():
        set_config_value("approvals.mode", "manual")

    saved = fast_safe_load(get_config_path().read_text(encoding="utf-8"))
    assert saved["approvals"]["mode"] == "manual"


def test_save_config_refuses_a_bulk_write_that_moves_a_safety_setting(monkeypatch):
    from hermes_cli.config import load_config, save_config

    save_config({"approvals": {"mode": "smart"}, "display": {"skin": "default"}})
    _as_session(monkeypatch)

    config = load_config()
    config["approvals"]["mode"] = "off"

    with pytest.raises(SafetyConfigChangeDenied) as excinfo:
        save_config(config)

    # Reported at the granularity of the protected pattern, so the message
    # never echoes a sub-key that came in with the caller's config.
    assert "approvals" in excinfo.value.keys


def test_save_config_still_persists_unrelated_edits_on_gateway(monkeypatch):
    from hermes_cli.config import get_config_path, load_config, save_config
    from utils import fast_safe_load

    save_config({"approvals": {"mode": "manual"}})
    _as_session(monkeypatch)

    config = load_config()
    config.setdefault("display", {})["skin"] = "mono"
    save_config(config)

    saved = fast_safe_load(get_config_path().read_text(encoding="utf-8"))
    assert saved["display"]["skin"] == "mono"
    assert saved["approvals"]["mode"] == "manual"


def test_defaulted_safety_keys_do_not_look_like_a_change():
    """A merged config vs a raw file must not read as "every default changed"."""
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge

    import copy

    merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), {"display": {"skin": "mono"}})
    raw_side = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), {})

    assert changed_protected_config_keys(merged, raw_side) == []


def test_changed_protected_keys_ignores_unrelated_edits():
    before = {"security": {"tirith_enabled": True}, "display": {"skin": "a"}}
    after = {"security": {"tirith_enabled": False}, "display": {"skin": "b"}}

    changed = changed_protected_config_keys(after, before)

    # The whole `security` subtree is protected, so that is the reported
    # granularity; `display.skin` moved too and is deliberately absent.
    assert changed == ["security"]


def test_cli_save_config_value_refuses_a_protected_key_on_gateway(monkeypatch):
    from cli import save_config_value

    _as_session(monkeypatch)

    with pytest.raises(SafetyConfigChangeDenied):
        save_config_value("approvals.destructive_slash_confirm", False)


def test_cli_save_config_value_allows_the_owner_opt_out(monkeypatch):
    from cli import save_config_value

    _as_session(monkeypatch)

    with owner_initiated_safety_config_change():
        assert save_config_value("approvals.destructive_slash_confirm", False)


# --------------------------------------------------------------------------
# The boundary has no runtime off switch
# --------------------------------------------------------------------------


def test_the_guard_itself_cannot_be_disabled_from_config():
    """Any future switch for this boundary belongs under the protected subtree."""
    assert matched_protected_config_key("security.safety_config_guard") == "security"
    assert matched_protected_config_key("security.allow_self_modification") == "security"


def test_owner_flag_does_not_leak_out_of_its_block(monkeypatch):
    _as_session(monkeypatch)

    with owner_initiated_safety_config_change():
        assert guard_module.current_guarded_surface() is None

    assert guard_module.current_guarded_surface() == "gateway"
