"""Which agent tasks have a turn in flight, so their sandbox is not reclaimed.

``tools.terminal_tool`` retires a task's environment once it has gone
``lifetime_seconds`` without a *tool* call.  That measure reads a turn parked
inside a single model API call as idle: local backends routinely spend five to
eight minutes prefilling a large context, so the sweep tore the environment
down mid-turn and the next tool call rebuilt it — losing the session cwd and
whatever shell state the turn had established (H-026).

This registry records the task ids of turns that are still running so the sweep
can leave their environments alone.  Scope note: it protects the *environment*.
A background process whose tracked wrapper exited early still loses its handle;
that is a separate defect with a separate fix, and nothing here addresses it.

Entries are keyed per turn and carry a start timestamp, so a turn that dies
between its register and release hooks expires instead of pinning an
environment for the life of the process.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Safety valve for a turn that never reaches its release hook. Well above the
# longest turns seen in practice (multi-hour agentic sessions): this is a leak
# stop, not a turn timeout, and expiring sooner would reintroduce the mid-turn
# teardown the registry exists to prevent.
MAX_TURN_AGE_SECONDS = 12 * 3600.0

_lock = threading.Lock()
# turn_key -> (task_id, started_at)
_active_turns: Dict[str, Tuple[str, float]] = {}


def register_active_turn(turn_key: str, task_id: Optional[str]) -> None:
    """Mark *task_id* as having work in flight for the turn *turn_key*.

    Both arguments must be non-empty strings; anything else is ignored so a
    caller that cannot name its turn never half-registers a task that will
    then never be released.
    """
    if not isinstance(turn_key, str) or not turn_key:
        return
    if not isinstance(task_id, str) or not task_id:
        return
    with _lock:
        # Monotonic: the timestamp is only ever read as an age, and a wall-clock
        # step (NTP, sleep/wake) must not expire a live turn's registration.
        _active_turns[turn_key] = (task_id, time.monotonic())


def release_active_turn(turn_key: str) -> None:
    """Drop the registration made by :func:`register_active_turn`.

    Idempotent: releasing an unknown or already-released turn is a no-op, so
    the caller can put this in a ``finally`` without tracking whether the
    matching register succeeded.
    """
    if not isinstance(turn_key, str) or not turn_key:
        return
    with _lock:
        _active_turns.pop(turn_key, None)


def active_task_ids(
    *,
    now: Optional[float] = None,
    max_age_seconds: Optional[float] = None,
) -> Set[str]:
    """Return the task ids whose turn is still running.

    Registrations older than *max_age_seconds* are dropped and reported — a
    turn that outlives the cap has lost its release hook, and leaving it in
    place would keep its environment alive for the rest of the process.

    *now* is a ``time.monotonic()`` reading; tests pass one to age entries out
    without waiting.
    """
    clock = time.monotonic() if now is None else float(now)
    cap = MAX_TURN_AGE_SECONDS if max_age_seconds is None else float(max_age_seconds)

    expired: list[Tuple[str, float]] = []
    with _lock:
        for turn_key, (task_id, started_at) in list(_active_turns.items()):
            if clock - started_at > cap:
                _active_turns.pop(turn_key, None)
                expired.append((task_id, started_at))
        live = {task_id for task_id, _started in _active_turns.values()}

    # The turn key embeds the session id, which never goes to a log (H-014);
    # the task prefix is enough to say which environment was affected.
    for task_id, started_at in expired:
        logger.warning(
            "Dropping stale active-turn registration for task %s after %.0fs "
            "— its turn never released it, and its environment is eligible "
            "for cleanup again",
            task_id[:8],
            clock - started_at,
        )
    return live


def reset() -> None:
    """Forget every registration. For tests and process-reuse harnesses."""
    with _lock:
        _active_turns.clear()
