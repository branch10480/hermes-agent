"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import contextlib
import sys

import pytest


def _is_hermes_runtime_module(name: str) -> bool:
    return (
        name in ("hermes_constants", "hermes_cli", "hermes_state")
        or name.startswith("hermes_cli.")
        or name.startswith("hermes_state.")
    )


@contextlib.contextmanager
def fresh_hermes_module_imports():
    """Drop cached ``hermes_cli`` / ``hermes_state`` modules, then put them back.

    Several kanban fixtures point ``HERMES_HOME`` at a throwaway directory and
    then evict these modules so the next import re-reads the env var. Evicting
    is fine; *not restoring* is what caused incident H-035.

    ``hermes_cli.main`` is imported at module scope by many other test files
    (``from hermes_cli import main as cli_main``). Once it has been dropped from
    ``sys.modules``, those files keep a reference to the now-orphaned module
    object while the production code re-imports a *fresh* one at call time
    (``hermes_cli.update_cmd._m()``). Every ``patch.object(cli_main, ...)`` in
    those files then silently patches the orphan and the real code path runs
    completely unmocked — which is how ``test_update_venv_health.py`` and
    ``test_update_orphan_backend_reap.py`` came to run ``hermes update``'s real
    git sequence (fetch → autostash → checkout main → reset → pull) against the
    developer's own checkout.

    Restoring the original module objects afterwards keeps the eviction local to
    the test that asked for it. ``importlib.reload`` needs no such treatment: it
    mutates the module in place, so identities never change.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if _is_hermes_runtime_module(name)
    }
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if _is_hermes_runtime_module(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
