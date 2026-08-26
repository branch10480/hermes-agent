"""A running turn keeps its sandbox out of the terminal-tool idle sweep (H-026).

``_cleanup_inactive_envs`` measures idleness by the last *tool* call, so a turn
that spends minutes inside one model API call looked abandoned and had its
environment retired mid-turn — the next tool call then rebuilt it without the
session cwd or shell state. ``agent.live_turn_registry`` records the turns that
are still running, and their task ids, so the sweep skips them.

That registry is shared with the background-review idle gate: one record per
turn serves both consumers, so a turn is never in flight for one of them and
finished for the other.

Scope: these cover the environment only. Nothing here claims anything about a
background process whose tracked wrapper exited early (H-007) — that handle is
lost with or without this retention.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.live_turn_registry as live_turn_registry
import tools.terminal_tool as tt


IDLE_SECONDS = 300
LONG_AGO = 10_000  # comfortably past any lifetime_seconds under test


class _FakeEnv:
    """Minimal environment double that records its own teardown."""

    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def _turn_agent(session_id="envact-session"):
    """The surface ``begin_turn`` reads off the agent that owns a turn."""
    return SimpleNamespace(session_id=session_id, platform="telegram")


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global; no test may inherit or leak an entry."""
    live_turn_registry.reset_for_tests()
    yield
    live_turn_registry.reset_for_tests()


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
    live_turn_registry.begin_turn(_turn_agent(), task_id=task_id)

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
    owner = _turn_agent()
    token = live_turn_registry.begin_turn(owner, task_id=task_id)
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is False

    live_turn_registry.end_turn(owner, token)
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
    live_turn_registry.begin_turn(_turn_agent(), task_id=task_id)

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert env.cleaned is False
    assert tt._active_environments[key] is env


def test_other_tasks_are_untouched_by_a_running_turn(env_registry):
    """Retention is per task — one busy turn does not pin every sandbox."""
    busy_key, busy_env = env_registry("envact-multi-busy")
    idle_key, idle_env = env_registry("envact-multi-idle")
    live_turn_registry.begin_turn(_turn_agent(), task_id="envact-multi-busy")

    tt._cleanup_inactive_envs(IDLE_SECONDS)

    assert busy_env.cleaned is False
    assert idle_env.cleaned is True
    assert idle_key not in tt._active_environments
    assert busy_key in tt._active_environments


# ── the registry ─────────────────────────────────────────────────────────────


def test_release_is_idempotent_and_tolerates_unknown_turns():
    owner = _turn_agent()
    token = live_turn_registry.begin_turn(owner, task_id="task-idem")
    live_turn_registry.end_turn(owner, token)
    live_turn_registry.end_turn(owner, token)
    live_turn_registry.end_turn(owner, 10_000_000)
    live_turn_registry.end_turn(owner, None)
    assert live_turn_registry.active_task_ids() == set()


def test_a_turn_without_a_task_id_pins_nothing():
    """A caller that cannot name its sandbox must not pin one.

    It still registers for the idle gate — occupying the backend is a fact
    about the turn, not about its environment.
    """
    live_turn_registry.begin_turn(_turn_agent(), task_id="")
    live_turn_registry.begin_turn(_turn_agent(), task_id=None)

    assert live_turn_registry.active_task_ids() == set()
    assert live_turn_registry.backend_busy_reason() is not None


def test_one_record_serves_the_sweep_and_the_idle_gate(env_registry):
    """The two consumers must never disagree about a turn being in flight.

    They read one registration; this pins that, so a future change cannot
    quietly reintroduce a second registry that goes out of step.
    """
    task_id = "envact-shared"
    _key, env = env_registry(task_id)
    owner = _turn_agent()
    token = live_turn_registry.begin_turn(owner, task_id=task_id)

    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is False
    assert live_turn_registry.backend_busy_reason() is not None

    live_turn_registry.end_turn(owner, token)

    assert live_turn_registry.active_task_ids() == set()
    assert live_turn_registry.backend_busy_reason() is None


def test_the_registered_task_id_is_the_one_the_turn_will_run_under():
    """The wrapper reads task_id off its own call, so it must read it right.

    ``AIAgent.run_conversation`` passes it by keyword, but the positional
    fallback is pinned to the real signature here: reordering
    ``_run_conversation_core``'s parameters would otherwise silently register
    the wrong id and stop protecting the sandbox.
    """
    import inspect

    from agent.conversation_loop import _run_conversation_core, _turn_task_id

    # ``agent`` is the wrapper's own first argument, not part of *args.
    params = list(inspect.signature(_run_conversation_core).parameters)[1:]
    ahead = params.index("task_id")
    positional = tuple(f"arg{i}" for i in range(ahead)) + ("task-positional",)

    assert _turn_task_id(positional, {}) == "task-positional"
    assert _turn_task_id((), {"task_id": "task-keyword"}) == "task-keyword"
    assert _turn_task_id(("just-a-message",), {}) is None


def test_stale_registration_expires_and_is_reported(caplog, env_registry):
    """A turn that dies between hooks must not pin its sandbox for the process."""
    task_id = "envact-stale"
    key, env = env_registry(task_id)
    live_turn_registry.begin_turn(_turn_agent(), task_id=task_id)

    with caplog.at_level("WARNING", logger="agent.live_turn_registry"):
        alive = live_turn_registry.active_task_ids(
            now=time.monotonic() + live_turn_registry.MAX_TURN_AGE_SECONDS + 1
        )

    assert alive == set()
    assert "stale active-turn registration" in caplog.text
    # Dropped, not just filtered: the next sweep retires the environment, and
    # the idle gate stops seeing a turn that will never end.
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is True
    assert live_turn_registry.backend_busy_reason() is None


def test_stale_report_never_leaks_the_session_key(caplog, env_registry):
    """Gateway session keys embed Discord channel/participant ids (H-014)."""
    secret = "discord:9876543210:thread-12345:user-777"
    task_id = "envact-stale-secret-suffix"
    env_registry(task_id)
    live_turn_registry.begin_turn(_turn_agent(secret), task_id=task_id)

    with caplog.at_level("WARNING", logger="agent.live_turn_registry"):
        live_turn_registry.active_task_ids(
            now=time.monotonic() + live_turn_registry.MAX_TURN_AGE_SECONDS + 1
        )

    for fragment in ("9876543210", "thread-12345", "user-777", secret):
        assert fragment not in caplog.text
    # Only the task prefix is identifying enough to log.
    assert "envact-s" in caplog.text
    assert task_id not in caplog.text


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
        observed["registered"] = task_id in live_turn_registry.active_task_ids()
        return _mock_response()

    agent.client.chat.completions.create.side_effect = slow_call

    result = agent.run_conversation("hello", task_id=task_id)

    assert result["final_response"] == "done"
    assert observed["registered"] is True
    assert observed["cleaned_mid_turn"] is False
    assert observed["present_mid_turn"] is True

    # Turn over: the registration is gone and the environment is idle again.
    assert live_turn_registry.active_task_ids() == set()
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
        "agent.conversation_loop._run_conversation_core",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            agent.run_conversation("hello", task_id=task_id)

    assert live_turn_registry.active_task_ids() == set()
    tt._cleanup_inactive_envs(IDLE_SECONDS)
    assert env.cleaned is True
