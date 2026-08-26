"""Authorization around background processes (observations H-017 / H-020).

Two boundaries are covered here:

* ``process(action='write'/'submit')`` and PTY shell/REPL launches are disabled
  on the messaging and cron surfaces, because input pushed into a running
  process skips every guard the original ``terminal`` call passed.
* Every ID-addressed process action is bound to the session that started the
  process, so knowing a process ID is not enough to read its output, write to
  it, or kill it.
"""

import json

import pytest

import tools.process_registry as process_registry_module
import tools.terminal_tool as terminal_tool_module
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides

OWNER_KEY = "agent:main:discord:group:chat-1:thread-1"
OTHER_KEY = "agent:main:discord:group:chat-2:thread-2"

# Task ids only stay distinct when the task owns its own sandbox; every other
# id collapses onto the shared "default" container.
ISOLATED_TASK = "benchmark-42"


@pytest.fixture()
def registry(monkeypatch):
    """Swap the module singleton for a fresh registry."""
    fresh = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", fresh)
    return fresh


@pytest.fixture()
def isolated_task():
    register_task_env_overrides(ISOLATED_TASK, {"docker_image": "python:3.11"})
    yield ISOLATED_TASK
    clear_task_env_overrides(ISOLATED_TASK)


@pytest.fixture()
def owned_process(registry):
    session = ProcessSession(
        id="proc_owned0000",
        command="bash",
        task_id="default",
        session_key=OWNER_KEY,
    )
    registry._running[session.id] = session
    return session


def _as_session(monkeypatch, session_key, *, gateway=True, cron=False):
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    if gateway:
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    if cron:
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")


def _call(action, **args):
    return json.loads(
        process_registry_module._handle_process({"action": action, **args})
    )


# --------------------------------------------------------------------------
# H-017 — stdin actions on remote surfaces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["write", "submit"])
def test_stdin_denied_on_gateway_surface(monkeypatch, owned_process, action):
    _as_session(monkeypatch, OWNER_KEY)

    result = _call(action, session_id=owned_process.id, data="rm -rf /")

    assert result["reason_code"] == "process_stdin_disabled_on_remote_surface"
    assert result["surface"] == "gateway"
    assert result["alternative"]["tool"] == "terminal"
    assert result["retryable"] is False


@pytest.mark.parametrize("action", ["write", "submit"])
def test_stdin_denied_on_cron_surface(monkeypatch, owned_process, action):
    _as_session(monkeypatch, OWNER_KEY, gateway=False, cron=True)

    result = _call(action, session_id=owned_process.id, data="whoami")

    assert result["reason_code"] == "process_stdin_disabled_on_remote_surface"
    assert result["surface"] == "cron"


def test_stdin_denial_precedes_lookup(monkeypatch, registry):
    """The denial must not depend on the ID resolving — no existence oracle."""
    _as_session(monkeypatch, OWNER_KEY)

    result = _call("write", session_id="proc_does_not_exist", data="x")

    assert result["reason_code"] == "process_stdin_disabled_on_remote_surface"


@pytest.mark.parametrize("action", ["write", "submit"])
def test_stdin_allowed_on_local_surface(monkeypatch, registry, action):
    """CLI/TUI keep the old behaviour: the call reaches the registry."""
    session = ProcessSession(id="proc_local0000", command="bash", task_id="default")
    registry._running[session.id] = session

    result = _call(action, session_id=session.id, data="hello")

    # No stdin pipe is attached to this synthetic session, so the registry's
    # own error is the proof that policy let the call through.
    assert "reason_code" not in result
    assert result["status"] == "error"
    assert "stdin not available" in result["error"]


@pytest.mark.parametrize("action", ["poll", "log", "wait", "kill", "close"])
def test_read_and_lifecycle_actions_survive_on_gateway(
    monkeypatch, owned_process, action
):
    _as_session(monkeypatch, OWNER_KEY)

    result = _call(action, session_id=owned_process.id, timeout=1)

    assert "reason_code" not in result
    assert result.get("status") != "not_found"


def test_list_still_scopes_to_the_calling_session(monkeypatch, registry):
    mine = ProcessSession(id="proc_mine00000", command="sleep 1", session_key=OWNER_KEY)
    theirs = ProcessSession(
        id="proc_theirs000", command="sleep 1", session_key=OTHER_KEY
    )
    registry._running[mine.id] = mine
    registry._running[theirs.id] = theirs
    _as_session(monkeypatch, OWNER_KEY)

    listed = {p["session_id"] for p in _call("list")["processes"]}

    assert listed == {mine.id}


# --------------------------------------------------------------------------
# H-020 — process IDs are bound to their session
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["poll", "log", "wait", "kill", "close"])
def test_other_session_cannot_touch_a_process(monkeypatch, owned_process, action):
    _as_session(monkeypatch, OTHER_KEY)

    result = _call(action, session_id=owned_process.id, timeout=1)

    assert result == {
        "status": "not_found",
        "error": f"No process with ID {owned_process.id}",
    }


@pytest.mark.parametrize("action", ["write", "submit"])
def test_other_task_cannot_write_to_a_process(registry, isolated_task, action):
    """Ownership also binds stdin on the surfaces where stdin is still allowed."""
    theirs = ProcessSession(id="proc_theirs000", command="bash", task_id=isolated_task)
    registry._running[theirs.id] = theirs

    result = _call(action, session_id=theirs.id, data="x")

    assert result == {
        "status": "not_found",
        "error": f"No process with ID {theirs.id}",
    }


def test_ownership_denial_is_indistinguishable_from_a_missing_process(
    monkeypatch, owned_process
):
    _as_session(monkeypatch, OTHER_KEY)

    denied = _call("poll", session_id=owned_process.id)
    missing = _call("poll", session_id="proc_never_existed")

    assert denied["status"] == missing["status"]
    assert denied["error"].replace(owned_process.id, "X") == missing[
        "error"
    ].replace("proc_never_existed", "X")


@pytest.mark.parametrize("action", ["poll", "log", "kill", "close"])
def test_owning_session_keeps_access(monkeypatch, owned_process, action):
    _as_session(monkeypatch, OWNER_KEY)

    result = _call(action, session_id=owned_process.id)

    assert result.get("status") != "not_found"


def test_cli_without_a_session_key_falls_back_to_the_task(registry, isolated_task):
    """A plain CLI run has no session identity; the sandbox still binds it."""
    mine = ProcessSession(id="proc_cli000000", command="sleep 1", task_id="default")
    other_task = ProcessSession(
        id="proc_othertask", command="sleep 1", task_id=isolated_task
    )
    registry._running[mine.id] = mine
    registry._running[other_task.id] = other_task

    assert _call("poll", session_id=mine.id)["status"] != "not_found"
    assert _call("poll", session_id=other_task.id)["status"] == "not_found"


def test_per_turn_task_ids_do_not_orphan_a_process(registry):
    """The one-shot/goal-loop CLI mints a fresh task id every turn.

    A process started in one of those turns must stay reachable from the next
    one, so an unbound caller falls back to the sandbox rather than to a task
    id that has already been replaced.
    """
    spawned_last_turn = ProcessSession(
        id="proc_lastturn0", command="sleep 1", task_id="default", session_key=""
    )
    registry._running[spawned_last_turn.id] = spawned_last_turn

    result = json.loads(
        process_registry_module._handle_process(
            {"action": "poll", "session_id": spawned_last_turn.id},
            task_id="4f2c1e70-this-turns-uuid",
        )
    )

    assert result["status"] != "not_found"


def test_gateway_refuses_a_process_with_no_session_identity(monkeypatch, registry):
    """Half-identified pairs fail closed where conversations share a process."""
    unbound = ProcessSession(id="proc_unbound00", command="sleep 1", task_id="default")
    registry._running[unbound.id] = unbound
    _as_session(monkeypatch, OWNER_KEY)

    assert _call("poll", session_id=unbound.id)["status"] == "not_found"


def test_subagent_task_ids_still_share_the_parents_processes(registry):
    """delegate_task children collapse onto the parent container by design."""
    spawned_by_parent = ProcessSession(
        id="proc_shared0000", command="sleep 1", task_id="default"
    )
    registry._running[spawned_by_parent.id] = spawned_by_parent

    result = json.loads(
        process_registry_module._handle_process(
            {"action": "poll", "session_id": spawned_by_parent.id},
            task_id="subagent-7",
        )
    )

    assert result["status"] != "not_found"


def test_denial_logging_never_prints_a_raw_session_key(
    monkeypatch, owned_process, caplog
):
    _as_session(monkeypatch, OTHER_KEY)

    with caplog.at_level("WARNING", logger=process_registry_module.logger.name):
        _call("poll", session_id=owned_process.id)

    assert caplog.records, "the denial should be observable in the log"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert OWNER_KEY not in logged
    assert OTHER_KEY not in logged
    assert "chat-1" not in logged and "thread-2" not in logged


# --------------------------------------------------------------------------
# H-017 — interactive shell / REPL launches
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "bash",
        "zsh -l",
        "sh -s",
        "python",
        "python3 -",
        "node",
        "env FOO=1 bash",
        "cd /tmp && bash",
    ],
)
def test_interactive_launches_are_detected(command):
    assert terminal_tool_module._starts_interactive_shell(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'echo hi'",
        "bash -lc 'pytest -q'",
        "python3 manage.py runserver",
        "python -m http.server",
        "node app.js",
        "npm run dev",
        "echo data | python",
        "python < script.py",
        "git rebase -i",
    ],
)
def test_non_interactive_commands_are_left_alone(command):
    assert terminal_tool_module._starts_interactive_shell(command) is False


def test_pty_shell_blocked_on_gateway_surface(monkeypatch):
    _as_session(monkeypatch, OWNER_KEY)

    result = json.loads(terminal_tool_module._interactive_shell_denial("bash", True))

    assert result["status"] == "blocked"
    assert result["reason_code"] == "interactive_shell_disabled_on_remote_surface"
    assert result["surface"] == "gateway"
    assert result["alternative"]["tool"] == "terminal"


def test_pty_shell_blocked_on_cron_surface(monkeypatch):
    _as_session(monkeypatch, OWNER_KEY, gateway=False, cron=True)

    result = json.loads(terminal_tool_module._interactive_shell_denial("python", True))

    assert result["surface"] == "cron"


def test_pty_shell_allowed_on_local_surface():
    assert terminal_tool_module._interactive_shell_denial("bash", True) is None


def test_non_pty_and_non_shell_commands_are_not_blocked(monkeypatch):
    _as_session(monkeypatch, OWNER_KEY)

    assert terminal_tool_module._interactive_shell_denial("bash", False) is None
    assert terminal_tool_module._interactive_shell_denial("pytest -q", True) is None
