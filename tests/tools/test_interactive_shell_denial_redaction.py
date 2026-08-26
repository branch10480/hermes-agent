"""The interactive-shell denial log must not leak inline secrets (F-4, H-014).

``_interactive_shell_denial`` warns to the persisted gateway log when it
refuses a PTY shell/REPL launch on a remote surface. The launch line can carry
an inline secret (``env OPENAI_API_KEY=sk-... bash``); the warning previously
previewed the raw command, so the key landed in the log. It must be redacted.
"""

from __future__ import annotations

import json
import logging

import tools.terminal_tool as terminal_tool

# A key with a distinctive middle segment: redaction preserves only head/tail,
# so the middle must be gone from any log line.
_SECRET = "sk-proj0000THISMIDDLEISSECRET9999zzzz"


def test_denial_log_does_not_leak_inline_key(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    command = f"env OPENAI_API_KEY={_SECRET} bash"
    with caplog.at_level(logging.WARNING, logger=terminal_tool.logger.name):
        raw = terminal_tool._interactive_shell_denial(command, pty=True)

    # The launch is refused on the gateway surface.
    assert raw is not None
    payload = json.loads(raw)
    assert payload["status"] == "blocked"

    # The warning fired, but the secret's body is not in any log record.
    log_blob = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Blocked interactive shell/REPL launch" in log_blob
    assert "THISMIDDLEISSECRET" not in log_blob
    assert _SECRET not in log_blob


def test_local_surface_is_not_denied(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    # A human owns the terminal locally, so the launch is allowed (None).
    assert terminal_tool._interactive_shell_denial("bash", pty=True) is None
