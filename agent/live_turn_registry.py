"""Process-global registry of the turns that are in flight right now.

Several Discord sessions, cron jobs and maintenance forks share one local
backend, and two separate subsystems need the same fact — "which turns are
running?" — so both read this one registry.

**The background-review idle gate (H-034).** Background review used to be gated
and cancelled purely through the agent that owned it: it started even while a
*different* session was mid-turn, and only the owning agent's next turn stopped
it. Observed effect: a skill-library review enqueued right after one session
finished kept the single backend slot busy, and the still-live session's own
calls came back 503 — housekeeping starving user-facing work. Two process-wide
facts fix that: every conversation turn registers for its duration, so a review
can ask "is anything else in flight anywhere?" before it starts and while it
runs; and every in-flight review registers a cancel hook, so a live turn in
*any* session can stop it.

**Sandbox retention (H-026).** ``tools.terminal_tool`` retires a task's
environment once it has gone ``lifetime_seconds`` without a *tool* call. That
measure reads a turn parked inside a single model API call as idle: local
backends routinely spend five to eight minutes prefilling a large context, so
the sweep tore the environment down mid-turn and the next tool call rebuilt it,
losing the session cwd and whatever shell state the turn had established. Each
turn record therefore carries its task id, and :func:`active_task_ids` tells
the sweep which environments to leave alone. Scope note: it protects the
*environment*. A background process whose tracked wrapper exited early still
loses its handle; that is a separate defect with a separate fix.

Sessions are identified in messages by ``agent.redact.session_key_fingerprint``
output only. Gateway session keys embed Discord channel/thread/participant IDs
and must never reach logs (H-014).

Everything here is best-effort coordination, never correctness-critical: a
process that never registers a turn behaves exactly as it did before.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# A turn a user is waiting on: CLI, gateway session, subagent, API server, cron.
TURN_KIND_LIVE = "live"
# Housekeeping that occupies the backend but nobody is waiting on: the
# background-review fork itself, the curator sweep.
TURN_KIND_MAINTENANCE = "maintenance"

# Idle-gate defaults, mirrored in hermes_cli/config_defaults.py under
# auxiliary.background_review. Duplicated as literals so a stub/partial config
# in tests still yields the shipped behavior.
_DEFAULT_IDLE_GATE_ENABLED = True
_DEFAULT_IDLE_GATE_POLL_SECONDS = 5.0
_DEFAULT_IDLE_GATE_MAX_WAIT_SECONDS = 300.0

# Safety valve for a turn that never reaches ``end_turn``. Well above the
# longest turns seen in practice (multi-hour agentic sessions): this is a leak
# stop, not a turn timeout, and expiring sooner would reintroduce the mid-turn
# sandbox teardown the task-id half of this registry exists to prevent.
MAX_TURN_AGE_SECONDS = 12 * 3600.0

_lock = threading.RLock()
_turns: Dict[int, Dict[str, Any]] = {}
_reviews: Dict[int, "BackgroundReviewHandle"] = {}
# Never reused, so excluding a stale token can never mask a live turn.
_token_seq = itertools.count(1)


# ---------------------------------------------------------------------------
# Identification helpers
# ---------------------------------------------------------------------------

def session_label(agent: Any) -> str:
    """Opaque, log-safe label for the agent's session."""
    try:
        from agent.redact import session_key_fingerprint

        return session_key_fingerprint(getattr(agent, "session_id", None))
    except Exception:
        return "unknown"


def is_maintenance_agent(agent: Any) -> bool:
    """True for the background-review fork and the curator sweep.

    Both tag themselves with the ``background_review`` write origin; the review
    fork also carries an explicit marker so the check does not depend on
    memory-provenance plumbing staying where it is.
    """
    if getattr(agent, "_is_background_review_fork", False):
        return True
    try:
        from tools.skill_provenance import BACKGROUND_REVIEW

        marker = BACKGROUND_REVIEW
    except Exception:
        marker = "background_review"
    return (getattr(agent, "_memory_write_origin", "") or "") == marker


# ---------------------------------------------------------------------------
# Turn registration
# ---------------------------------------------------------------------------

def begin_turn(
    agent: Any,
    kind: str = TURN_KIND_LIVE,
    task_id: Optional[str] = None,
) -> int:
    """Register a running turn and return its token.

    The token is also recorded on the agent (``_live_turn_tokens``) so its own
    post-turn review can exclude it — the review is spawned from inside
    ``finalize_turn``, i.e. while the owning turn is still registered.

    *task_id* names the sandbox this turn is working in, and is what
    :func:`active_task_ids` reports to the terminal-tool idle sweep. A turn
    that cannot name one still registers for the idle gate; it simply holds no
    environment open.
    """
    token = next(_token_seq)
    record = {
        "token": token,
        "kind": kind if kind in (TURN_KIND_LIVE, TURN_KIND_MAINTENANCE) else TURN_KIND_LIVE,
        "session": session_label(agent),
        "platform": str(getattr(agent, "platform", "") or "") or "unknown",
        "task_id": task_id if isinstance(task_id, str) and task_id else "",
        # Monotonic: the timestamp is only ever read as an age, and a wall-clock
        # step (NTP, sleep/wake) must not expire a live turn's registration.
        "started_at": time.monotonic(),
    }
    with _lock:
        _turns[token] = record
    try:
        tokens = getattr(agent, "_live_turn_tokens", None)
        if not isinstance(tokens, set):
            tokens = set()
            agent._live_turn_tokens = tokens
        tokens.add(token)
    except Exception:
        pass
    return token


def end_turn(agent: Any, token: Optional[int]) -> None:
    """Unregister a turn. Safe to call with an unknown or ``None`` token."""
    if token is None:
        return
    with _lock:
        _turns.pop(token, None)
    try:
        tokens = getattr(agent, "_live_turn_tokens", None)
        if isinstance(tokens, set):
            tokens.discard(token)
    except Exception:
        pass


def active_turns(exclude: Iterable[int] = ()) -> List[Dict[str, Any]]:
    """Snapshot of registered turns, newest last."""
    skip = set(exclude or ())
    with _lock:
        return [
            dict(record)
            for token, record in sorted(_turns.items())
            if token not in skip
        ]


def active_task_ids(
    *,
    now: Optional[float] = None,
    max_age_seconds: Optional[float] = None,
) -> Set[str]:
    """Return the task ids whose turn is still running.

    Registrations older than *max_age_seconds* are dropped and reported — a
    turn that outlives the cap has lost its ``end_turn``, and leaving it in
    place would keep its environment alive (and the idle gate closed) for the
    rest of the process. The sweep is done here rather than in
    :func:`active_turns` because the terminal-tool cleanup is the caller that
    runs on a timer; the gate sees the same freed entries because both read the
    one registry.

    *now* is a ``time.monotonic()`` reading; tests pass one to age entries out
    without waiting.
    """
    clock = time.monotonic() if now is None else float(now)
    cap = MAX_TURN_AGE_SECONDS if max_age_seconds is None else float(max_age_seconds)

    expired: List[Dict[str, Any]] = []
    with _lock:
        for token, record in list(_turns.items()):
            if clock - record["started_at"] > cap:
                _turns.pop(token, None)
                expired.append(record)
        live = {
            record["task_id"] for record in _turns.values() if record["task_id"]
        }

    # The session field is already a fingerprint (H-014); the task prefix is
    # enough to say which environment was affected.
    for record in expired:
        logger.warning(
            "Dropping stale active-turn registration for task %s (%s/%s) after "
            "%.0fs — its turn never released it, and its environment is "
            "eligible for cleanup again",
            (record["task_id"] or "unknown")[:8],
            record["kind"],
            record["session"],
            clock - record["started_at"],
        )
    return live


def backend_busy_reason(exclude: Iterable[int] = ()) -> Optional[str]:
    """Describe what is occupying the backend, or ``None`` when it is idle.

    The string is log-safe: session identities are fingerprints, never raw
    gateway session keys.
    """
    busy = active_turns(exclude=exclude)
    if not busy:
        return None
    shown = ", ".join(
        f"{record['kind']}/{record['platform']}/{record['session']}"
        for record in busy[:3]
    )
    if len(busy) > 3:
        shown = f"{shown}, +{len(busy) - 3} more"
    noun = "turn" if len(busy) == 1 else "turns"
    return f"{len(busy)} {noun} in flight ({shown})"


# ---------------------------------------------------------------------------
# Background-review registration + cross-session cancellation
# ---------------------------------------------------------------------------

class BackgroundReviewHandle:
    """Cancel surface for one in-flight background review.

    Holds the review's cancellation event (checked before the fork makes any
    API call) and, once the fork exists, the fork itself so ``interrupt()``
    can abort a call already on the wire.
    """

    __slots__ = ("token", "session", "cancellation_event", "_agent", "_cancelled", "_lock")

    def __init__(self, token: int, session: str, cancellation_event: Any) -> None:
        self.token = token
        self.session = session
        self.cancellation_event = cancellation_event
        self._agent: Any = None
        self._cancelled = False
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def attach_agent(self, review_agent: Any) -> bool:
        """Publish the fork for cancellation. False if it was already cancelled.

        The caller must abandon the review on False: a cancel that landed
        during fork construction would otherwise never reach the new agent.
        """
        with self._lock:
            if self._cancelled:
                return False
            self._agent = review_agent
            return True

    def cancel(self, reason: str) -> None:
        """Flag the review cancelled and request its abort off-thread.

        The cancelled flag and the cancellation event are set synchronously —
        they are what every gate in the review's startup path reads, and the
        registry contract is that they hold the moment ``cancel`` returns. The
        ``interrupt()`` itself is dispatched to a daemon thread: this runs on a
        LIVE TURN's thread, and a review whose abort hook is slow or wedged
        must never hold up the foreground (#84423, the same reason
        ``agent/background_review.py`` offloads its own interrupt in
        ``_interrupt_background_review``).
        """
        with self._lock:
            self._cancelled = True
            review_agent = self._agent
            event = self.cancellation_event
        if event is not None:
            try:
                event.set()
            except Exception:
                logger.debug("background-review cancel event set failed", exc_info=True)
        if review_agent is None:
            return

        session = self.session

        def _interrupt() -> None:
            try:
                review_agent.interrupt(reason)
            except Exception:
                logger.debug(
                    "background-review interrupt failed for %s", session, exc_info=True
                )

        try:
            threading.Thread(
                target=_interrupt,
                daemon=True,
                name="bg-review-cancel",
            ).start()
        except Exception:
            logger.debug(
                "background-review cancellation thread failed for %s; "
                "interrupting inline",
                session,
                exc_info=True,
            )
            _interrupt()


def register_background_review(
    *,
    cancellation_event: Any = None,
    session: str = "unknown",
) -> BackgroundReviewHandle:
    """Publish a starting review so any live turn can cancel it."""
    handle = BackgroundReviewHandle(
        token=next(_token_seq),
        session=session,
        cancellation_event=cancellation_event if cancellation_event is not None else threading.Event(),
    )
    with _lock:
        _reviews[handle.token] = handle
    return handle


def unregister_background_review(handle: Optional[BackgroundReviewHandle]) -> None:
    """Idempotent removal — safe from a ``finally`` that may run twice."""
    if handle is None:
        return
    with _lock:
        _reviews.pop(handle.token, None)


def active_background_reviews() -> List[BackgroundReviewHandle]:
    with _lock:
        return list(_reviews.values())


def cancel_background_reviews(reason: str) -> int:
    """Cancel every in-flight review in this process. Returns how many.

    Called at the start of every live turn, whatever session it belongs to —
    that cross-session reach is the whole point (CR-008).
    """
    handles = active_background_reviews()
    if not handles:
        return 0
    for handle in handles:
        handle.cancel(reason)
    logger.debug(
        "cancelled %d background review(s): %s", len(handles), reason
    )
    return len(handles)


# ---------------------------------------------------------------------------
# Idle-gate configuration
# ---------------------------------------------------------------------------

def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")) or parsed <= 0:
        return default
    return parsed


def idle_gate_settings() -> tuple:
    """Return ``(enabled, poll_seconds, max_wait_seconds)`` from config."""
    enabled = _DEFAULT_IDLE_GATE_ENABLED
    poll = _DEFAULT_IDLE_GATE_POLL_SECONDS
    max_wait = _DEFAULT_IDLE_GATE_MAX_WAIT_SECONDS
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        auxiliary = config.get("auxiliary", {})
        task = (
            auxiliary.get("background_review", {})
            if isinstance(auxiliary, dict)
            else {}
        )
        if not isinstance(task, dict):
            return (enabled, poll, max_wait)
        if "idle_gate" in task:
            enabled = bool(task.get("idle_gate"))
        poll = _coerce_positive_float(task.get("idle_gate_poll_seconds"), poll)
        # 0 / negative means "wait forever"; the caller reads <= 0 that way.
        raw_max = task.get("idle_gate_max_wait_seconds", max_wait)
        try:
            max_wait = float(raw_max)
        except (TypeError, ValueError):
            pass
        if max_wait != max_wait:  # NaN
            max_wait = _DEFAULT_IDLE_GATE_MAX_WAIT_SECONDS
    except Exception:
        logger.debug("background-review idle gate config read failed", exc_info=True)
    return (enabled, poll, max_wait)


def reset_for_tests() -> None:
    """Drop all registrations. Tests only."""
    with _lock:
        _turns.clear()
        _reviews.clear()


__all__ = [
    "MAX_TURN_AGE_SECONDS",
    "TURN_KIND_LIVE",
    "TURN_KIND_MAINTENANCE",
    "BackgroundReviewHandle",
    "active_background_reviews",
    "active_task_ids",
    "active_turns",
    "backend_busy_reason",
    "begin_turn",
    "cancel_background_reviews",
    "end_turn",
    "idle_gate_settings",
    "is_maintenance_agent",
    "register_background_review",
    "reset_for_tests",
    "session_label",
    "unregister_background_review",
]
