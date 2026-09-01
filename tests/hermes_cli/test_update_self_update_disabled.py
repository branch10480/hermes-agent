"""``updates.self_update_enabled`` — the owner opt-out for externally-managed
checkouts.

The setting exists for installations whose checkout belongs to something
outside Hermes (a Nix flake pinning a revision, config management). There, a
self-update rewrites a tree the managing system believes it owns, and any
uncommitted work in it gets swept into an autostash nobody asked for. So the
contract under test is not merely "update exits non-zero" — it is that the
refusal lands *before* the first mutation: no pre-update backup, no autostash,
no git.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main


class _Sentinel(Exception):
    """Raised by a mutating step to prove the guard let execution through."""


@pytest.fixture
def mutation_tripwires(monkeypatch):
    """Make every tree-mutating step _cmd_update_impl can reach explode.

    Returns the list that records which one fired, so a test can assert both
    "nothing ran" and "the first thing that would have run is X".

    ``subprocess.run`` is wired too, but on its own it is not proof of a tree
    mutation. Since v2026.8.31 the apply path runs a read-only "plan phase"
    before the first mutating step: ``update_inventory.collect_runtime_inventory``
    shells out (on macOS, ``launchctl print``) to survey the running fleet,
    and upstream wraps the whole phase in ``except Exception``, so the
    tripwire's ``_Sentinel`` merely makes the survey come back empty. Ordering
    assertions therefore go through :func:`_mutating_only`; the "nothing ran
    at all" assertions keep ``subprocess.run`` and stay exact.
    """
    fired = []

    def _trip(name):
        def _boom(*args, **kwargs):
            fired.append(name)
            raise _Sentinel(name)

        return _boom

    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", _trip("pre_update_backup"))
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", _trip("stash")
    )
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", _trip("restore"))
    monkeypatch.setattr(hermes_main, "_discard_stashed_changes", _trip("discard"))
    monkeypatch.setattr(hermes_main.subprocess, "run", _trip("subprocess.run"))
    monkeypatch.setattr(hermes_main.subprocess, "Popen", _trip("subprocess.Popen"))
    return fired


def _mutating_only(fired):
    """Drop the read-only pre-update probe from a tripwire sequence.

    Verified caller as of v2026.8.31: ``_cmd_update_impl`` -> plan phase ->
    ``update_inventory.collect_runtime_inventory`` -> ``gateway._get_service_pids``
    -> ``_launchd_print_service_pid`` -> ``subprocess.run(["launchctl", ...])``.
    That path only surveys the running fleet; it never touches the checkout.

    Residual gap this accepts: a *mutating* ``subprocess.run`` slipping in
    ahead of ``_run_pre_update_backup`` would no longer show up in the two
    "guard is fail-open" assertions. The refusal contract itself — the thing
    this module exists to pin — is unaffected: the tests that carry it assert
    ``mutation_tripwires == []`` with ``subprocess.run`` still armed.
    """
    return [name for name in fired if name != "subprocess.run"]


def _set_updates_config(monkeypatch, updates):
    monkeypatch.setattr(
        hermes_config, "load_config_readonly", lambda: {"updates": updates}
    )


# ---------------------------------------------------------------------------
# self_update_enabled() — the shared predicate
# ---------------------------------------------------------------------------


def test_self_update_enabled_defaults_to_true_when_key_absent(monkeypatch):
    _set_updates_config(monkeypatch, {"non_interactive_local_changes": "stash"})
    assert hermes_config.self_update_enabled() is True


def test_self_update_enabled_false_disables(monkeypatch):
    _set_updates_config(monkeypatch, {"self_update_enabled": False})
    assert hermes_config.self_update_enabled() is False


def test_self_update_enabled_honors_quoted_false(monkeypatch):
    """A quoted "false" in YAML must disable, not read as a truthy string."""
    _set_updates_config(monkeypatch, {"self_update_enabled": "false"})
    assert hermes_config.self_update_enabled() is False


def test_self_update_enabled_treats_a_valueless_key_as_absent(monkeypatch):
    """``self_update_enabled:`` with nothing after it parses to None.

    That is an unfinished line, not an opt-out. Reading it as ``bool(None)``
    would block every update — the exact opposite of the fail-safe the rest of
    this function is built around.
    """
    _set_updates_config(monkeypatch, {"self_update_enabled": None})
    assert hermes_config.self_update_enabled() is True


def test_self_update_enabled_fails_safe_when_config_unreadable(monkeypatch):
    """A broken config must not be what stops a user from updating."""

    def _explode():
        raise OSError("config.yaml is unreadable")

    monkeypatch.setattr(hermes_config, "load_config_readonly", _explode)
    assert hermes_config.self_update_enabled() is True


def test_self_update_enabled_tolerates_non_dict_updates_section(monkeypatch):
    monkeypatch.setattr(
        hermes_config, "load_config_readonly", lambda: {"updates": "nonsense"}
    )
    assert hermes_config.self_update_enabled() is True


def test_default_config_ships_self_update_enabled_true():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["updates"]["self_update_enabled"] is True


# ---------------------------------------------------------------------------
# `hermes update` (apply path)
# ---------------------------------------------------------------------------


def test_update_refuses_before_any_mutation_when_disabled(
    monkeypatch, mutation_tripwires, capsys
):
    _set_updates_config(monkeypatch, {"self_update_enabled": False})

    with pytest.raises(SystemExit) as excinfo:
        hermes_main._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert excinfo.value.code == 1
    assert mutation_tripwires == []

    out = capsys.readouterr().out
    assert "updates.self_update_enabled=false" in out
    # The banner belongs to the update that never started.
    assert "Updating Hermes Agent" not in out


def test_update_refuses_in_gateway_mode_too(monkeypatch, mutation_tripwires, capsys):
    """The gateway spawns `hermes update --gateway`; that path must refuse as
    well, otherwise a detached process would still touch the tree."""
    _set_updates_config(monkeypatch, {"self_update_enabled": False})

    with pytest.raises(SystemExit) as excinfo:
        hermes_main._cmd_update_impl(SimpleNamespace(), gateway_mode=True)

    assert excinfo.value.code == 1
    assert mutation_tripwires == []
    assert "self_update_enabled" in capsys.readouterr().out


def test_update_proceeds_when_key_absent(monkeypatch, mutation_tripwires):
    """Default (key unset) leaves the existing flow untouched: execution
    reaches the pre-update backup, which is the first mutating step."""
    _set_updates_config(monkeypatch, {"non_interactive_local_changes": "stash"})
    monkeypatch.setattr(
        hermes_config, "load_config", lambda: {"updates": {}}
    )

    with pytest.raises(_Sentinel):
        hermes_main._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert _mutating_only(mutation_tripwires) == ["pre_update_backup"]


def test_update_proceeds_when_config_read_fails(monkeypatch, mutation_tripwires):
    """Fail-safe end to end: an unreadable config does not block the update."""

    def _explode():
        raise OSError("config.yaml is unreadable")

    monkeypatch.setattr(hermes_config, "load_config_readonly", _explode)
    monkeypatch.setattr(hermes_config, "load_config", lambda: {})

    with pytest.raises(_Sentinel):
        hermes_main._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert _mutating_only(mutation_tripwires) == ["pre_update_backup"]


def test_check_path_is_unaffected_by_the_guard(monkeypatch, mutation_tripwires):
    """``--check`` only fetches and reports, so the setting must not close it.

    It never reaches the guard because ``cmd_update`` branches to
    ``_cmd_update_check`` first; this pins that ordering so a future
    refactor cannot slide the read-only path behind the refusal.
    """
    _set_updates_config(monkeypatch, {"self_update_enabled": False})
    monkeypatch.setattr(hermes_config, "is_managed", lambda: False)
    monkeypatch.setattr(hermes_config, "detect_install_method", lambda root: "source")

    checked = []
    monkeypatch.setattr(
        hermes_main, "_cmd_update_check",
        lambda **kwargs: checked.append(kwargs) or None,
    )

    def _impl_must_not_run(*args, **kwargs):
        raise AssertionError("--check must not reach the apply path")

    monkeypatch.setattr(hermes_main, "_cmd_update_impl", _impl_must_not_run)

    hermes_main.cmd_update(SimpleNamespace(check=True, branch=None))

    assert len(checked) == 1
    assert mutation_tripwires == []


# ---------------------------------------------------------------------------
# Dashboard (`POST /api/hermes/update`)
# ---------------------------------------------------------------------------


def test_dashboard_update_endpoint_refuses_when_disabled(monkeypatch):
    """The dashboard button spawns the same pipeline, so it needs the same gate.

    Without this the System page is a way around the setting on any host whose
    dashboard is reachable.
    """
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_cli.web_server as ws

    monkeypatch.setattr(ws, "self_update_enabled", lambda: False)

    def _must_not_spawn(*args, **kwargs):
        raise AssertionError("the update pipeline must not start")

    monkeypatch.setattr(ws.subprocess, "Popen", _must_not_spawn)

    client = TestClient(ws.app)
    client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    body = client.post("/api/hermes/update").json()

    assert body["ok"] is False
    assert body["error"] == "self_update_disabled"
    assert body["pid"] is None
    assert "self_update_enabled" in body["message"]


def test_updates_subtree_is_protected_from_the_agent_surface():
    """The agent must not be able to re-enable self-update on itself."""
    from tools.safety_config_guard import matched_protected_config_key

    assert (
        matched_protected_config_key("updates.self_update_enabled") is not None
    )
