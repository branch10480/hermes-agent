"""A running turn keeps its sandbox out of the terminal-tool idle sweep (H-026).

``_cleanup_inactive_envs`` measures idleness by the last *tool* call, so a turn
that spends minutes inside one model API call looked abandoned and had its
environment retired mid-turn — the next tool call then rebuilt it without the
session cwd or shell state. ``tools.env_activity`` records the turns that are
still running so the sweep skips them.

Scope: these cover the environment only. Nothing here claims anything about a
background process whose tracked wrapper exited early (H-007) — that handle is
lost with or without this retention.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.env_activity as env_activity
import tools.terminal_tool as tt


IDLE_SECONDS = 300
LONG_AGO = 10_000  # comfortably past any lifetime_seconds under test


class _FakeEnv:
    """Minimal environment double that records its own teardown."""

    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global; no test may inherit or leak an entry."""
    env_activity.reset()
    yield
    env_activity.reset()


@pytest.fixture
def env_registry():
    """Register/withdraw a fake env under an isolated task id.

    The task carries an ``env_type`` override so it keeps its own container key
    instead of collapsing onto the process-wide ``"default"`` sandbox that other
    tests share.
    """
    created = []

    def make(task_id, *, idle_for=LONG_AGO, isolated=True):
        if isolated:
            tt.register_task_env_overrides(task_id, {"env_type": "local"})
        key = tt._resolve_container_task_id(task_id)
        env = _FakeEnv()
        with tt._env_lock:
            tt._active_environments[key] = env
            tt._last_activity[key] = time.time() - idle_for
        created.append((task_id, key))
        return key, env

    yield make

    for task_id, key in created:
        tt.clear_task_env_overrides(task_id)
        with tt._env_lock:
            tt._active_environments.pop(key, None)
            tt._last_activity.pop(key, None)


# ── the sweep ────────────────────────────────────────────────────────────────


def test_idle_env_is_retired_when_no_turn_is_running(env_registry):
    """Baseline: with nothing registered the sweep behaves as it always has."""
    key, env = env_registry("envact-idle")

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert env.cleaned is True
    assert key not in tt._active_environments
    assert key not in tt._last_activity


def test_running_turn_keeps_its_env(env_registry):
    """A turn in flight is not idle, however long since its last tool call."""
    task_id = "envact-busy"
    key, env = env_registry(task_id)
    env_activity.register_active_turn("turn-busy", task_id)

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert env.cleaned is False
    assert tt._active_environments[key] is env
    # The retention also refreshes the clock, so the environment gets a full
    # idle window after the turn ends instead of being retired immediately.
    assert tt._last_activity[key] == pytest.approx(time.time(), abs=5)


def test_env_is_retired_again_once_the_turn_releases(env_registry):
    """Release restores ordinary idle behavior — retention is not permanent."""
    task_id = "envact-release"
    key, env = env_registry(task_id)
    env_activity.register_active_turn("turn-release", task_id)
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is False

    env_activity.release_active_turn("turn-release")
    # Re-stale the entry: the protected sweep above refreshed it.
    with tt._env_lock:
        tt._last_activity[key] = time.time() - LONG_AGO

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert env.cleaned is True
    assert key not in tt._active_environments


def test_collapsed_container_id_is_protected_too(env_registry):
    """Top-level turns pass a per-turn task id but share the ``default`` env.

    The registry holds the raw id, so the sweep has to resolve it to the
    container key before comparing.
    """
    task_id = "envact-collapsed"
    key, env = env_registry(task_id, isolated=False)
    assert key == "default"
    env_activity.register_active_turn("turn-collapsed", task_id)

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert env.cleaned is False
    assert tt._active_environments[key] is env


def test_other_tasks_are_untouched_by_a_running_turn(env_registry):
    """Retention is per task — one busy turn does not pin every sandbox."""
    busy_key, busy_env = env_registry("envact-multi-busy")
    idle_key, idle_env = env_registry("envact-multi-idle")
    env_activity.register_active_turn("turn-multi", "envact-multi-busy")

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert busy_env.cleaned is False
    assert idle_env.cleaned is True
    assert idle_key not in tt._active_environments
    assert busy_key in tt._active_environments


# ── the registry ─────────────────────────────────────────────────────────────


def test_release_is_idempotent_and_tolerates_unknown_turns():
    env_activity.register_active_turn("turn-idem", "task-idem")
    env_activity.release_active_turn("turn-idem")
    env_activity.release_active_turn("turn-idem")
    env_activity.release_active_turn("never-registered")
    assert env_activity.active_task_ids() == set()


def test_blank_identifiers_never_half_register():
    """A caller that cannot name its turn must not pin a task forever."""
    env_activity.register_active_turn("", "task-blank")
    env_activity.register_active_turn("turn-blank", "")
    env_activity.register_active_turn("turn-blank", None)
    assert env_activity.active_task_ids() == set()


def test_stale_registration_expires_and_is_reported(caplog, env_registry):
    """A turn that dies between hooks must not pin its sandbox for the process."""
    task_id = "envact-stale"
    key, env = env_registry(task_id)
    env_activity.register_active_turn("turn-stale", task_id)

    with caplog.at_level("WARNING", logger="tools.env_activity"):
        alive = env_activity.active_task_ids(
            now=time.monotonic() + env_activity.MAX_TURN_AGE_SECONDS + 1
        )

    assert alive == set()
    assert "stale active-turn registration" in caplog.text
    # Dropped, not just filtered: the next sweep retires the environment.
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is True


# ── through the real turn entry point ────────────────────────────────────────


def _mock_response(content="done"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=MagicMock(),
            session_id="envact-session",
            platform="telegram",
        )
    agent.client = MagicMock()
    return agent


def test_env_survives_an_api_call_longer_than_the_idle_timer(env_registry):
    """The H-026 shape, driven through ``AIAgent.run_conversation``.

    The model call stands in for a multi-minute prefill: while it is in flight
    the environment has been untouched for far longer than ``lifetime_seconds``,
    yet the sweep must leave it alone. Once the turn returns, the same sweep
    retires it as before.
    """
    task_id = "envact-e2e"
    key, env = env_registry(task_id)
    agent = _make_agent()
    observed = {}

    def slow_call(*args, **kwargs):
        # Stand in for the minutes spent inside one API call.
        with tt._env_lock:
            tt._last_activity[key] = time.time() - LONG_AGO
        tt._cleanup_inactive_envs(IDLE_SECONDS)
        observed["cleaned_mid_turn"] = env.cleaned
        observed["present_mid_turn"] = tt._active_environments.get(key) is env
        observed["registered"] = task_id in env_activity.active_task_ids()
        return _mock_response()

    agent.client.chat.completions.create.side_effect = slow_call

    result = agent.run_conversation("hello", task_id=task_id)

    assert result["final_response"] == "done"
    assert observed["registered"] is True
    assert observed["cleaned_mid_turn"] is False
    assert observed["present_mid_turn"] is True

    # Turn over: the registration is gone and the environment is idle again.
    assert env_activity.active_task_ids() == set()
    with tt._env_lock:
        tt._last_activity[key] = time.time() - LONG_AGO
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is True
    assert key not in tt._active_environments


def test_registration_is_released_when_the_turn_raises(env_registry):
    """An exception mid-turn must not leave the sandbox pinned."""
    task_id = "envact-raise"
    key, env = env_registry(task_id)
    agent = _make_agent()

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            agent.run_conversation("hello", task_id=task_id)

    assert env_activity.active_task_ids() == set()
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is True
