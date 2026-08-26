"""Admission control for the one shared backend (H-032 / H-033, CR-008).

Several Discord sessions, cron jobs and maintenance forks push inference at a
single local backend that serves **one request at a time**. Nothing on the
Hermes side decided who went next, so everyone submitted at once and the
backend arbitrated by rejecting: 12 × ``503 worker inference queue timed out``
in one observation window, a session-swap ping-pong as short as 21 seconds, a
later session waiting 147 seconds for its first token, and — once background
review joined in — housekeeping starving a user who was still waiting.

Retrying on a fixed backoff made that worse in both directions: a retry could
fire while the backend was still busy (rejected again, queue deeper) or sleep
through the moment it went idle.

This module moves the decision to the sender. One permit per in-flight backend
call; whoever cannot have one waits in an ordered queue and is woken by the
**release of the previous call**, not by a timer. Two ordering rules:

* live turns (any session, any surface, cron included) outrank maintenance
  (the background-review fork, the curator sweep) — a review can never take
  the slot ahead of a user who is already waiting for it;
* within one class the queue is first-come-first-served, so the wait a session
  sees only ever shrinks.

Scope is deliberately one *call*, not one *turn*. An agentic turn spends most
of its wall time running tools with the backend idle; holding the permit across
the whole turn would hand one session a multi-hour lease on everyone else's
backend. Re-queueing per call is what keeps the slot shared.

**Engagement.** ``mode: "auto"`` (the default) engages only when the active
endpoint is a local one — the single-slot case this exists for. A hosted
provider keeps its own concurrency and rate limits and is untouched, so the
default is behaviour-identical for anyone not running a local backend. A
multi-slot local server raises ``max_concurrent_requests``.

**Only the conversation loop's own model call goes through here.** Auxiliary
work that shares the backend — compression, title generation — deliberately
does not, and adding it needs more than an ``acquire`` call: compression can
run on a fork/worker thread while the turn that started it is blocked waiting
for the result, and with one permit that is a deadlock, not a queue. The
re-entrancy guard below only covers nesting on a *single* thread. Anything
cross-thread has to be reasoned about first (H-031 / plan item 2-2).

**Everything here is best-effort.** Admission is a scheduling heuristic, never
a correctness boundary: a failed config read, an exception, or a wait that runs
past its deadline all fall through to "submit anyway", which is exactly the
behaviour that existed before this module. It must never be able to fail a
turn.

Sessions appear in logs as ``agent.redact.session_key_fingerprint`` output
only — gateway session keys embed Discord channel/thread/participant IDs and
must never be logged (H-014).
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lower sorts first. Live work — a user is waiting on it — always outranks
# housekeeping. The values are spaced so a future tier can slot between them.
PRIORITY_LIVE = 0
PRIORITY_MAINTENANCE = 10

# Mirrored in hermes_cli/config_defaults.py under agent.backend_scheduler.
# Duplicated as literals so a stub/partial config in tests still yields the
# shipped behavior.
_DEFAULT_MODE = "auto"
_DEFAULT_MAX_CONCURRENT = 1
_DEFAULT_QUEUE_WAIT_SECONDS = 0.0  # 0 = derive from measured call durations
_DEFAULT_QUEUE_WAIT_MIN_SECONDS = 120.0
_DEFAULT_QUEUE_WAIT_CAP_SECONDS = 1800.0
_DEFAULT_ASSUMED_CALL_SECONDS = 120.0
_DEFAULT_POLL_SECONDS = 1.0
_DEFAULT_NOTIFY_AFTER_SECONDS = 20.0
_DEFAULT_NOTIFY_INTERVAL_SECONDS = 60.0

# How many recent call durations feed the derived queue deadline. Large enough
# that one outlier cannot move the p90, small enough to track a context that
# has grown (prefill time on a local backend scales with it).
_DURATION_SAMPLES = 64

# A permit held longer than this is not a real call — it is a leaked release.
# Excluded from the statistics so one leak cannot inflate every later deadline.
_MAX_PLAUSIBLE_HOLD_SECONDS = 6 * 3600.0

# Retry floor used when the fixed backoff is short-circuited: the actual wait
# has moved into acquire(), which returns the moment the backend frees up.
_BUSY_RETRY_FLOOR_SECONDS = 1.0

_lock = threading.RLock()
_active: Dict[int, "Ticket"] = {}
_waiting: List["Ticket"] = []
_hold_durations: Deque[float] = deque(maxlen=_DURATION_SAMPLES)
_busy_signals = 0
# Never reused, so a stale ticket number can never collide with a live one.
_seq = itertools.count(1)
# Re-entrancy is per thread: a permit holder that somehow re-enters must not
# wait on itself. Cross-thread nesting is not possible today (subagents and
# auxiliary calls run between calls, never inside one), and this keeps it from
# becoming a deadlock if that ever changes.
_thread_local = threading.local()

# Provider wording for "the backend is busy, not broken". ``overloaded`` covers
# Anthropic's 529 overloaded_error; it only ever reaches a decision here when
# the scheduler is engaged, which auto mode limits to local endpoints.
_BUSY_TEXT_RE = re.compile(
    r"(inference\s+queue|queue\s+timed?\s*out|no\s+slot|slots?\s+(are\s+)?busy"
    r"|server\s+is\s+busy|overloaded|service\s+unavailable)",
    re.IGNORECASE,
)
_BUSY_STATUS_CODES = frozenset({503, 529})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerSettings:
    mode: str = _DEFAULT_MODE
    max_concurrent_requests: int = _DEFAULT_MAX_CONCURRENT
    queue_wait_seconds: float = _DEFAULT_QUEUE_WAIT_SECONDS
    queue_wait_min_seconds: float = _DEFAULT_QUEUE_WAIT_MIN_SECONDS
    queue_wait_cap_seconds: float = _DEFAULT_QUEUE_WAIT_CAP_SECONDS
    assumed_call_seconds: float = _DEFAULT_ASSUMED_CALL_SECONDS
    poll_seconds: float = _DEFAULT_POLL_SECONDS
    notify_after_seconds: float = _DEFAULT_NOTIFY_AFTER_SECONDS
    notify_interval_seconds: float = _DEFAULT_NOTIFY_INTERVAL_SECONDS


def _coerce_float(value: Any, default: float, *, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default
    if parsed < 0:
        return default
    if parsed == 0 and not allow_zero:
        return default
    return parsed


def _coerce_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value or "").strip().lower()
    if text in ("auto", "on", "off"):
        return text
    if text in ("true", "yes", "enabled", "always"):
        return "on"
    if text in ("false", "no", "disabled", "never"):
        return "off"
    return _DEFAULT_MODE


def settings() -> SchedulerSettings:
    """Read ``agent.backend_scheduler`` from config, defaults on any failure."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        agent_cfg = config.get("agent", {})
        raw = agent_cfg.get("backend_scheduler", {}) if isinstance(agent_cfg, dict) else {}
        if not isinstance(raw, dict):
            return SchedulerSettings()
    except Exception:
        logger.debug("backend scheduler config read failed", exc_info=True)
        return SchedulerSettings()

    try:
        max_concurrent = int(raw.get("max_concurrent_requests", _DEFAULT_MAX_CONCURRENT))
    except (TypeError, ValueError):
        max_concurrent = _DEFAULT_MAX_CONCURRENT
    if max_concurrent < 1:
        max_concurrent = _DEFAULT_MAX_CONCURRENT

    return SchedulerSettings(
        mode=_coerce_mode(raw.get("mode", _DEFAULT_MODE)),
        max_concurrent_requests=max_concurrent,
        # 0 is meaningful here: "derive the deadline from measurements".
        queue_wait_seconds=_coerce_float(
            raw.get("queue_wait_seconds", _DEFAULT_QUEUE_WAIT_SECONDS),
            _DEFAULT_QUEUE_WAIT_SECONDS,
            allow_zero=True,
        ),
        queue_wait_min_seconds=_coerce_float(
            raw.get("queue_wait_min_seconds", _DEFAULT_QUEUE_WAIT_MIN_SECONDS),
            _DEFAULT_QUEUE_WAIT_MIN_SECONDS,
        ),
        queue_wait_cap_seconds=_coerce_float(
            raw.get("queue_wait_cap_seconds", _DEFAULT_QUEUE_WAIT_CAP_SECONDS),
            _DEFAULT_QUEUE_WAIT_CAP_SECONDS,
        ),
        assumed_call_seconds=_coerce_float(
            raw.get("assumed_call_seconds", _DEFAULT_ASSUMED_CALL_SECONDS),
            _DEFAULT_ASSUMED_CALL_SECONDS,
        ),
        poll_seconds=_coerce_float(
            raw.get("poll_seconds", _DEFAULT_POLL_SECONDS), _DEFAULT_POLL_SECONDS
        ),
        # 0/negative disables the "you are still queued" notice.
        notify_after_seconds=_coerce_float(
            raw.get("notify_after_seconds", _DEFAULT_NOTIFY_AFTER_SECONDS),
            _DEFAULT_NOTIFY_AFTER_SECONDS,
            allow_zero=True,
        ),
        notify_interval_seconds=_coerce_float(
            raw.get("notify_interval_seconds", _DEFAULT_NOTIFY_INTERVAL_SECONDS),
            _DEFAULT_NOTIFY_INTERVAL_SECONDS,
        ),
    )


def _is_local_backend(base_url: str) -> bool:
    try:
        from agent.model_metadata import is_local_endpoint

        return bool(is_local_endpoint(base_url))
    except Exception:
        return False


def engaged_for(agent: Any, config: Optional[SchedulerSettings] = None) -> bool:
    """Whether this agent's backend calls go through the queue."""
    cfg = config if config is not None else settings()
    if cfg.mode == "off":
        return False
    if cfg.mode == "on":
        return True
    return _is_local_backend(str(getattr(agent, "base_url", "") or ""))


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Ticket:
    """One caller's claim on the backend.

    Identity comparison (``eq=False``) is deliberate: the queue removes a
    ticket by ``list.remove``, and field-wise equality could match a different
    caller's ticket that happens to carry the same values.

    ``reentrant`` tickets are bookkeeping only: the thread already holds the
    real permit, so the ticket carries no queue position and releasing it just
    unwinds the nesting depth.
    """

    seq: int = 0
    priority: int = PRIORITY_LIVE
    session: str = "unknown"
    enqueued_at: float = 0.0
    granted_at: Optional[float] = None
    waited_seconds: float = 0.0
    over_capacity: bool = False
    reentrant: bool = False
    event: threading.Event = field(default_factory=threading.Event)

    @property
    def sort_key(self) -> tuple:
        return (self.priority, self.seq)


class _ThreadState:
    __slots__ = ("depth", "ticket")

    def __init__(self) -> None:
        self.depth = 0
        self.ticket: Optional[Ticket] = None


def _thread_state() -> _ThreadState:
    state = getattr(_thread_local, "state", None)
    if state is None:
        state = _ThreadState()
        _thread_local.state = state
    return state


def _priority_for(agent: Any) -> int:
    try:
        from agent import live_turn_registry

        if live_turn_registry.is_maintenance_agent(agent):
            return PRIORITY_MAINTENANCE
    except Exception:
        logger.debug("backend scheduler priority classification failed", exc_info=True)
    return PRIORITY_LIVE


def _session_label(agent: Any) -> str:
    try:
        from agent import live_turn_registry

        return live_turn_registry.session_label(agent)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Queue mechanics (all callers hold ``_lock``)
# ---------------------------------------------------------------------------

def _grant_locked(ticket: Ticket, *, over_capacity: bool = False) -> None:
    ticket.granted_at = time.monotonic()
    ticket.waited_seconds = max(0.0, ticket.granted_at - ticket.enqueued_at)
    ticket.over_capacity = over_capacity
    _active[ticket.seq] = ticket


def _blockers_locked(priority: int, capacity: int) -> int:
    """How many claims must clear before a newcomer at *priority* runs."""
    ahead = len(_active)
    ahead += sum(1 for waiter in _waiting if waiter.priority <= priority)
    return max(0, ahead - (capacity - 1))


def _promote_locked(capacity: int) -> None:
    """Hand the freed permits to the best-ranked waiters, newest last."""
    while _waiting and len(_active) < capacity:
        _waiting.sort(key=lambda item: item.sort_key)
        nxt = _waiting.pop(0)
        _grant_locked(nxt)
        nxt.event.set()


def _percentile(samples: List[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _wait_deadline_seconds(
    cfg: SchedulerSettings, blockers: int, priority: int = PRIORITY_LIVE
) -> float:
    """Queue deadline, tied to what calls on this backend actually cost.

    A fixed timeout is wrong in both directions on a local backend: too short
    and a legitimate 6-minute prefill behind one other session is abandoned,
    too long and a wedged backend holds a user for the full value. Deriving it
    from the recent p90 hold time × the number of claims ahead tracks the real
    cost; the floor and cap keep a cold process (no samples yet) and a
    pathological outlier from producing an absurd deadline.

    Maintenance waits the full cap. Expiring is a fail-open — it submits
    anyway — and housekeeping that shoulders its way in beside a live turn is
    the H-034 inversion all over again. Nobody is waiting on a review, so the
    patient option is the right one for it.
    """
    if cfg.queue_wait_seconds > 0:
        return cfg.queue_wait_seconds
    if priority >= PRIORITY_MAINTENANCE:
        return cfg.queue_wait_cap_seconds
    with _lock:
        samples = list(_hold_durations)
    typical = _percentile(samples, 0.9)
    if typical <= 0:
        typical = cfg.assumed_call_seconds
    estimate = typical * max(1, blockers) * 2.0
    lower = min(cfg.queue_wait_min_seconds, cfg.queue_wait_cap_seconds)
    return max(lower, min(estimate, cfg.queue_wait_cap_seconds))


def _position(ticket: Ticket) -> int:
    """Claims that must clear before this ticket runs (0 = next)."""
    with _lock:
        ahead = len(_active)
        ahead += sum(
            1 for waiter in _waiting if waiter.sort_key < ticket.sort_key
        )
    return ahead


def _wait_for_grant(
    ticket: Ticket,
    deadline_seconds: float,
    cfg: SchedulerSettings,
    on_wait: Optional[Callable[[int, float], None]],
    should_abort: Optional[Callable[[], bool]],
) -> str:
    """Block until granted, aborted, or past the deadline.

    Returns ``"granted"``, ``"aborted"`` or ``"expired"``. The grant arrives on
    an event set by :func:`release`, so a waiter resumes the instant the
    backend frees up rather than at the end of a backoff.
    """
    end = ticket.enqueued_at + deadline_seconds
    next_notice = (
        ticket.enqueued_at + cfg.notify_after_seconds
        if cfg.notify_after_seconds > 0 and on_wait is not None
        else None
    )
    poll = max(0.05, cfg.poll_seconds)
    while True:
        now = time.monotonic()
        remaining = end - now
        if remaining <= 0:
            return "expired"
        if ticket.event.wait(min(poll, remaining)):
            return "granted"
        if should_abort is not None:
            try:
                if should_abort():
                    return "aborted"
            except Exception:
                logger.debug("backend scheduler abort check failed", exc_info=True)
        if next_notice is not None:
            now = time.monotonic()
            if now >= next_notice:
                try:
                    on_wait(_position(ticket), now - ticket.enqueued_at)
                except Exception:
                    logger.debug("backend scheduler wait notice failed", exc_info=True)
                next_notice = now + cfg.notify_interval_seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acquire(
    agent: Any = None,
    *,
    priority: Optional[int] = None,
    on_wait: Optional[Callable[[int, float], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    config: Optional[SchedulerSettings] = None,
) -> Optional[Ticket]:
    """Claim the backend, waiting in priority-then-arrival order.

    Returns the ticket to pass to :func:`release`, or ``None`` when the caller
    should just make the call: the scheduler is disengaged for this backend, or
    the wait was aborted, or something failed. ``None`` is the pre-scheduler
    behaviour, which is why every failure path returns it.

    *on_wait* is called with ``(claims_ahead, seconds_waited)`` while queued,
    first after ``notify_after_seconds`` and then every
    ``notify_interval_seconds``; *should_abort* is polled on the same cadence
    so an interrupted turn stops queueing.
    """
    try:
        cfg = config if config is not None else settings()
        if not engaged_for(agent, cfg):
            return None

        state = _thread_state()
        if state.depth > 0:
            with _lock:
                still_holding = (
                    state.ticket is not None and state.ticket.seq in _active
                )
            if still_holding:
                state.depth += 1
                return Ticket(reentrant=True)
            # Depth without a live permit means a release was skipped. Treat
            # the thread as fresh rather than handing out a ticket that would
            # let it bypass the queue forever.
            logger.debug("backend scheduler nesting depth reset (stale permit)")
            state.depth = 0
            state.ticket = None

        ticket = Ticket(
            seq=next(_seq),
            priority=priority if priority is not None else _priority_for(agent),
            session=_session_label(agent),
            enqueued_at=time.monotonic(),
        )
        capacity = cfg.max_concurrent_requests
        with _lock:
            blockers = _blockers_locked(ticket.priority, capacity)
            if blockers == 0 and len(_active) < capacity:
                _grant_locked(ticket)
                state.depth = 1
                state.ticket = ticket
                return ticket
            _waiting.append(ticket)
    except Exception:
        logger.debug("backend scheduler acquire failed", exc_info=True)
        return None

    deadline = _wait_deadline_seconds(cfg, blockers, ticket.priority)
    try:
        outcome = _wait_for_grant(ticket, deadline, cfg, on_wait, should_abort)
    except Exception:
        logger.debug("backend scheduler wait failed", exc_info=True)
        outcome = "expired"

    with _lock:
        if ticket.seq in _active:
            # A grant landed while we were deciding to give up; keep it.
            outcome = "granted"
        elif ticket in _waiting:
            _waiting.remove(ticket)
            if outcome != "aborted":
                # Fail open: submit anyway rather than fail the turn. Counting
                # it as active keeps the accounting honest, so the queue does
                # not also promote a fresh waiter on top of this one.
                _grant_locked(ticket, over_capacity=True)
                outcome = "granted"
        else:
            # Neither queued nor holding — the state was reset underneath us.
            outcome = "abandoned"

    state = _thread_state()
    if outcome != "granted":
        state.depth = 0
        state.ticket = None
        return None

    if ticket.over_capacity:
        logger.warning(
            "Backend queue wait exceeded %.0fs for %s/%s — submitting anyway "
            "(%d in flight, %d queued)",
            deadline,
            _priority_name(ticket.priority),
            ticket.session,
            len(_active),
            len(_waiting),
        )
    elif ticket.waited_seconds >= 1.0:
        logger.info(
            "Backend queue admitted %s/%s after %.1fs (%d in flight, %d queued)",
            _priority_name(ticket.priority),
            ticket.session,
            ticket.waited_seconds,
            len(_active),
            len(_waiting),
        )
    state.depth = 1
    state.ticket = ticket
    return ticket


def release(ticket: Optional[Ticket]) -> None:
    """Give the permit back and wake the best-ranked waiter immediately."""
    if ticket is None:
        return
    try:
        state = _thread_state()
        if ticket.reentrant:
            state.depth = max(0, state.depth - 1)
            return
        state.depth = 0
        state.ticket = None
        capacity = settings().max_concurrent_requests
        with _lock:
            _active.pop(ticket.seq, None)
            if ticket.granted_at is not None:
                held = time.monotonic() - ticket.granted_at
                if 0.0 < held < _MAX_PLAUSIBLE_HOLD_SECONDS:
                    _hold_durations.append(held)
            _promote_locked(capacity)
    except Exception:
        logger.debug("backend scheduler release failed", exc_info=True)


def _priority_name(priority: int) -> str:
    return "maintenance" if priority >= PRIORITY_MAINTENANCE else "live"


def looks_like_backend_busy(
    error: Any = None, status_code: Any = None
) -> bool:
    """True when a failure means "the backend is occupied", not "broken"."""
    try:
        code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        code = None
    if code is None and error is not None:
        for attr in ("status_code", "code"):
            raw = getattr(error, attr, None)
            try:
                code = int(raw)
                break
            except (TypeError, ValueError):
                continue
    if code in _BUSY_STATUS_CODES:
        return True
    if error is None:
        return False
    try:
        return bool(_BUSY_TEXT_RE.search(str(error)))
    except Exception:
        return False


def note_backend_busy() -> None:
    """Count a backend-busy rejection. Observability only."""
    global _busy_signals
    with _lock:
        _busy_signals += 1


def contended() -> bool:
    """True when someone else in this process holds or wants the backend."""
    with _lock:
        return bool(_active) or bool(_waiting)


def retry_wait_override(
    agent: Any,
    default_wait: float,
    *,
    error: Any = None,
    status_code: Any = None,
) -> Optional[float]:
    """Replacement backoff for a backend-busy retry, or ``None`` to keep it.

    A fixed backoff is the wrong instrument once the queue exists: the retry
    goes back to the top of the loop, where :func:`acquire` blocks until the
    backend is actually free. Sleeping first only delays that. So when the
    scheduler is engaged *and* another claim is outstanding in this process,
    the sleep collapses to a floor and the real wait becomes the queue.

    When nothing else in this process wants the backend, the rejection came
    from outside it — another Hermes, the server's own queue — and there is no
    release event to wait for, so the jittered backoff stands.
    """
    try:
        if not looks_like_backend_busy(error=error, status_code=status_code):
            return None
        note_backend_busy()
        if not engaged_for(agent):
            return None
        if not contended():
            return None
        return min(float(default_wait), _BUSY_RETRY_FLOOR_SECONDS)
    except Exception:
        logger.debug("backend scheduler retry override failed", exc_info=True)
        return None


def snapshot() -> Dict[str, Any]:
    """Log-safe view of the queue. Sessions are fingerprints (H-014)."""
    with _lock:
        return {
            "active": [
                {
                    "priority": _priority_name(t.priority),
                    "session": t.session,
                    "waited_seconds": round(t.waited_seconds, 3),
                    "over_capacity": t.over_capacity,
                }
                for t in sorted(_active.values(), key=lambda item: item.sort_key)
            ],
            "waiting": [
                {
                    "priority": _priority_name(t.priority),
                    "session": t.session,
                }
                for t in sorted(_waiting, key=lambda item: item.sort_key)
            ],
            "busy_signals": _busy_signals,
            "call_seconds_p90": round(_percentile(list(_hold_durations), 0.9), 3),
            "samples": len(_hold_durations),
        }


def record_call_duration(seconds: float) -> None:
    """Feed a measured call duration in. Tests and warm-start callers."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return
    if not (0.0 < value < _MAX_PLAUSIBLE_HOLD_SECONDS):
        return
    with _lock:
        _hold_durations.append(value)


def reset_for_tests() -> None:
    """Drop all queue state and statistics. Tests only."""
    global _busy_signals
    with _lock:
        for ticket in list(_waiting):
            ticket.event.set()
        _active.clear()
        _waiting.clear()
        _hold_durations.clear()
        _busy_signals = 0
    state = _thread_state()
    state.depth = 0
    state.ticket = None


__all__ = [
    "PRIORITY_LIVE",
    "PRIORITY_MAINTENANCE",
    "SchedulerSettings",
    "Ticket",
    "acquire",
    "contended",
    "engaged_for",
    "looks_like_backend_busy",
    "note_backend_busy",
    "record_call_duration",
    "release",
    "reset_for_tests",
    "retry_wait_override",
    "settings",
    "snapshot",
]
