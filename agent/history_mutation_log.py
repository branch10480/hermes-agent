"""One-line structured events for every mutation of the conversation history.

Why this exists
---------------
A provider prompt/KV cache reuses a prefix only while that prefix stays
byte-identical to the previous request.  Any component that rewrites a message
the provider has already seen invalidates the cache from that message forward,
and on a local inference server that is a cold re-prefill of everything after
it — measured at 5-7 minutes per break on a 100K-token session.  The existing
logs recorded *that* a prompt diverged at token N, never *who* wrote there, so
a divergence in the middle of a live turn could not be attributed to the
component responsible.

Two instruments
---------------
``log_history_mutation()``
    A component hands over the before/after transcript pair around its own
    rewrite.  This module finds the first divergent message, converts the index
    to an approximate token offset, and emits one line.

``PrefixStabilityTracker``
    Per-agent memory of the previous request's per-message fingerprints.  Every
    assembled request reports where it diverged from its predecessor, how much
    of the prompt was still reusable, and which mutation events were recorded
    in between.  It therefore also catches divergence from components that
    never call ``log_history_mutation`` — sanitizers, cache decoration,
    provider adapters — which show up with an empty ``since`` trail.  An empty
    trail on a real divergence is itself the finding.

Both instruments are estimates: message tokens come from the same rough
estimator the compressor uses, not from the provider's tokenizer.  They are
built to answer "which component moved the divergence point, and by how much",
which is a comparison between consecutive requests, not an accounting figure.

Privacy
-------
Neither instrument logs message content.  Events carry roles, indices, token
estimates and caller-supplied component/reason codes.  Sessions appear only as
the run-local fingerprint from ``agent.redact.session_key_fingerprint``, so a
persisted log can never be walked back to a Discord channel/thread ID.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger("agent.history_mutation")

# Trail bookkeeping is bounded so a long-lived gateway process cannot grow it:
# only the most recent mutations matter (they are drained by the next request),
# and only recently active sessions are retained.
_TRAIL_MAX_EVENTS = 16
_TRAIL_MAX_SESSIONS = 64

_trail_lock = threading.Lock()
_trails: "OrderedDict[str, deque[str]]" = OrderedDict()


def measurement_enabled() -> bool:
    """True when the mutation log is live.

    Every instrument checks this before doing any work: fingerprinting a
    100K-token request is cheap next to a model call but not free, and it runs
    on every API call of every session.
    """

    return logger.isEnabledFor(logging.INFO)


# ---------------------------------------------------------------------------
# Message fingerprints
# ---------------------------------------------------------------------------


def _wire_view(value: Any) -> Any:
    """Strip harness-private keys so bookkeeping never reads as a rewrite.

    Underscore-prefixed keys (``_row_id``, ``_thinking_prefill``, the
    DB-persisted marker the proactive prune stamps) are removed by the
    transport before the wire.  Fingerprinting them would report a cache-
    breaking mutation every time a purely local marker was stamped.
    """

    if isinstance(value, Mapping):
        return {
            k: _wire_view(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(value, (list, tuple)):
        return [_wire_view(v) for v in value]
    return value


def message_fingerprint(message: Any) -> str:
    """Stable short digest of one message's on-the-wire content."""

    try:
        payload = json.dumps(
            _wire_view(message),
            sort_keys=True,
            ensure_ascii=False,
            default=repr,
        )
    except Exception:
        # A message carrying an unserializable object still needs a
        # fingerprint; repr() is stable enough to detect "this changed".
        payload = repr(message)
    return hashlib.blake2b(
        payload.encode("utf-8", "surrogatepass"), digest_size=8
    ).hexdigest()


def message_fingerprints(messages: Sequence[Any] | None) -> list[str]:
    """Fingerprint each message in order."""

    return [message_fingerprint(m) for m in (messages or [])]


def first_divergent_index(
    before: Sequence[str], after: Sequence[str]
) -> int | None:
    """Index of the first message whose bytes differ, or None when identical.

    A pure append returns ``len(before)``: everything the provider already
    cached is still valid, which is the outcome every request-only injection
    is supposed to produce.
    """

    for idx, (lhs, rhs) in enumerate(zip(before, after)):
        if lhs != rhs:
            return idx
    if len(before) == len(after):
        return None
    return min(len(before), len(after))


def _role_at(messages: Sequence[Any] | None, index: int | None) -> str:
    if index is None or not messages or index >= len(messages):
        return "-"
    entry = messages[index]
    if isinstance(entry, Mapping):
        return str(entry.get("role") or "-")
    return "-"


def _estimate_tokens(messages: Sequence[Any] | None) -> int:
    if not messages:
        return 0
    try:
        from agent.model_metadata import estimate_messages_tokens_rough

        return int(estimate_messages_tokens_rough(list(messages)))
    except Exception:
        # Estimation is diagnostic only — never let it break a turn.
        return 0


def _fingerprint_session(session_id: Any) -> str:
    try:
        from agent.redact import session_key_fingerprint

        return session_key_fingerprint(session_id)
    except Exception:
        return "none"


def _fmt_ratio(value: float) -> str:
    return f"{value:.3f}"


def _fmt_fields(fields: Mapping[str, Any]) -> str:
    """Render one ``key=value`` line.

    Whitespace inside a value is collapsed to ``_``: the line format is
    space-separated, so a caller-supplied string with a space in it would
    silently split into bogus fields for anything parsing the log.
    """

    parts = []
    for key, value in fields.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "-"
        else:
            rendered = "_".join(str(value).split()) or "-"
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Mutation events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryMutationEvent:
    """One recorded rewrite of a message list.

    ``first_changed_index`` is None when the component turned out to be a
    no-op; callers may still log that (a no-op is evidence too) but the
    default is to stay silent so the log only carries real mutations.
    """

    component: str
    reason: str
    session: str
    turn_id: str
    first_changed_index: int | None
    first_changed_role: str
    stable_prefix_tokens: int
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    append_only: bool
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stable_prefix_ratio(self) -> float:
        """Share of the pre-mutation prompt the provider can still reuse.

        1.0 for an append (or a no-op): nothing the provider cached moved.
        """

        if self.tokens_before <= 0:
            return 1.0
        return max(0.0, min(1.0, self.stable_prefix_tokens / self.tokens_before))

    @property
    def invalidated_tokens(self) -> int:
        """Tokens the provider must re-prefill because of this mutation."""

        return max(0, self.tokens_before - self.stable_prefix_tokens)

    def as_log_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "component": self.component,
            "reason": self.reason,
            "session": self.session,
            "turn": self.turn_id or "-",
            "first_changed_index": self.first_changed_index,
            "first_changed_role": self.first_changed_role,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "invalidated_tokens": self.invalidated_tokens,
            "stable_prefix_ratio": _fmt_ratio(self.stable_prefix_ratio),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "append_only": self.append_only,
        }
        fields.update(self.extra)
        return fields

    def to_log_line(self) -> str:
        return "history_mutation " + _fmt_fields(self.as_log_fields())

    @property
    def trail_label(self) -> str:
        return f"{self.component}:{self.reason}"


def build_history_mutation_event(
    *,
    component: str,
    reason: str,
    before: Sequence[Any] | None,
    after: Sequence[Any] | None,
    session_id: Any = "",
    turn_id: Any = "",
    before_fingerprints: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> HistoryMutationEvent:
    """Diff two transcript snapshots into an event without logging it."""

    before_list = list(before or [])
    after_list = list(after or [])
    before_fps = (
        list(before_fingerprints)
        if before_fingerprints is not None
        else message_fingerprints(before_list)
    )
    after_fps = message_fingerprints(after_list)
    index = first_divergent_index(before_fps, after_fps)
    prefix = before_list if index is None else before_list[:index]
    return HistoryMutationEvent(
        component=component,
        reason=reason,
        session=_fingerprint_session(session_id),
        turn_id=str(turn_id or ""),
        first_changed_index=index,
        first_changed_role=_role_at(after_list, index),
        stable_prefix_tokens=_estimate_tokens(prefix),
        tokens_before=_estimate_tokens(before_list),
        tokens_after=_estimate_tokens(after_list),
        messages_before=len(before_list),
        messages_after=len(after_list),
        append_only=(index is None or index >= len(before_list)),
        extra=dict(extra or {}),
    )


def log_history_mutation(
    *,
    component: str,
    reason: str,
    before: Sequence[Any] | None,
    after: Sequence[Any] | None,
    session_id: Any = "",
    turn_id: Any = "",
    before_fingerprints: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    log_noop: bool = False,
) -> HistoryMutationEvent | None:
    """Emit one line describing a component's rewrite of the transcript.

    Returns the event (None when measurement is off, or when nothing changed
    and ``log_noop`` is False).  Never raises: a broken instrument must not be
    able to fail a turn.
    """

    if not measurement_enabled():
        return None
    try:
        event = build_history_mutation_event(
            component=component,
            reason=reason,
            before=before,
            after=after,
            session_id=session_id,
            turn_id=turn_id,
            before_fingerprints=before_fingerprints,
            extra=extra,
        )
    except Exception:
        logger.debug("history mutation event build failed", exc_info=True)
        return None
    if event.first_changed_index is None and not log_noop:
        return None
    logger.info("%s", event.to_log_line())
    _record_trail(event.session, event.trail_label)
    return event


def note_history_mutation_intent(
    *,
    component: str,
    reason: str,
    session_id: Any = "",
    turn_id: Any = "",
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Record a decision that did NOT rewrite history (a deferral, a skip).

    Deferrals are the other half of the attribution story: without them a
    reader cannot tell "the prune never became eligible" from "the prune was
    eligible and was held back to a boundary".
    """

    if not measurement_enabled():
        return
    fields: dict[str, Any] = {
        "component": component,
        "reason": reason,
        "session": _fingerprint_session(session_id),
        "turn": str(turn_id or "") or "-",
    }
    fields.update(dict(extra or {}))
    logger.info("history_mutation_intent %s", _fmt_fields(fields))


# ---------------------------------------------------------------------------
# Per-session mutation trail (drained by the next assembled request)
# ---------------------------------------------------------------------------


def _record_trail(session: str, label: str) -> None:
    with _trail_lock:
        trail = _trails.get(session)
        if trail is None:
            trail = deque(maxlen=_TRAIL_MAX_EVENTS)
            _trails[session] = trail
            while len(_trails) > _TRAIL_MAX_SESSIONS:
                _trails.popitem(last=False)
        _trails.move_to_end(session)
        trail.append(label)


def _drain_trail(session: str) -> list[str]:
    with _trail_lock:
        trail = _trails.get(session)
        if not trail:
            return []
        drained = list(trail)
        trail.clear()
        return drained


def reset_mutation_trails() -> None:
    """Drop every recorded trail (test isolation)."""

    with _trail_lock:
        _trails.clear()


# ---------------------------------------------------------------------------
# Request prefix stability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixObservation:
    """How much of one assembled request survived from the previous one."""

    baseline: bool
    first_changed_index: int | None
    first_changed_role: str
    reusable_prefix_tokens: int
    prompt_tokens: int
    previous_prompt_tokens: int
    message_count: int
    append_only: bool
    trail: tuple[str, ...]
    api_call: int
    session: str
    turn_id: str

    @property
    def prefix_reuse_ratio(self) -> float:
        """Upper bound on the provider's cache hit for this request.

        The provider can reuse at most the bytes it already has, so the share
        of this prompt that matched the previous request's prefix bounds the
        hit rate from above.  A cold first request reports 0.0 (nothing was
        cached yet), which is why callers judging prefix STABILITY skip
        baselines rather than averaging them in.
        """

        if self.baseline:
            return 0.0
        if self.prompt_tokens <= 0:
            return 1.0
        return max(0.0, min(1.0, self.reusable_prefix_tokens / self.prompt_tokens))

    @property
    def reprefill_tokens(self) -> int:
        """Tokens this request must prefill from cold."""

        return max(0, self.prompt_tokens - self.reusable_prefix_tokens)

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "turn": self.turn_id or "-",
            "api_call": self.api_call,
            "baseline": self.baseline,
            "first_changed_index": self.first_changed_index,
            "first_changed_role": self.first_changed_role,
            "prefix_reuse_ratio": _fmt_ratio(self.prefix_reuse_ratio),
            "reusable_prefix_tokens": self.reusable_prefix_tokens,
            "reprefill_tokens": self.reprefill_tokens,
            "prompt_tokens": self.prompt_tokens,
            "prev_prompt_tokens": self.previous_prompt_tokens,
            "messages": self.message_count,
            "append_only": self.append_only,
            "since": ",".join(self.trail) if self.trail else "-",
        }

    def to_log_line(self) -> str:
        return "prompt_prefix_stability " + _fmt_fields(self.as_log_fields())


class PrefixStabilityTracker:
    """Remembers the previous request so the next one can be compared to it.

    One tracker per agent.  ``observe`` is called with the fully assembled
    request — after every sanitizer, injection and cache decoration — because
    that is the byte sequence the provider actually caches.  Anything earlier
    would miss exactly the mutations this is meant to catch.
    """

    __slots__ = ("_fingerprints", "_prompt_tokens", "_lock")

    def __init__(self) -> None:
        self._fingerprints: list[str] | None = None
        self._prompt_tokens = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Forget the previous request (session switch, model switch)."""

        with self._lock:
            self._fingerprints = None
            self._prompt_tokens = 0

    def observe(
        self,
        messages: Sequence[Any] | None,
        *,
        session_id: Any = "",
        turn_id: Any = "",
        api_call: int = 0,
        log: bool = True,
    ) -> PrefixObservation | None:
        """Compare one assembled request against its predecessor.

        Returns None when measurement is disabled or the comparison failed —
        callers treat that as "no data", never as an error.
        """

        if not measurement_enabled():
            return None
        try:
            message_list = list(messages or [])
            fingerprints = message_fingerprints(message_list)
            prompt_tokens = _estimate_tokens(message_list)
            with self._lock:
                previous = self._fingerprints
                previous_tokens = self._prompt_tokens
                self._fingerprints = fingerprints
                self._prompt_tokens = prompt_tokens
            session = _fingerprint_session(session_id)
            if previous is None:
                observation = PrefixObservation(
                    baseline=True,
                    first_changed_index=None,
                    first_changed_role="-",
                    reusable_prefix_tokens=0,
                    prompt_tokens=prompt_tokens,
                    previous_prompt_tokens=0,
                    message_count=len(message_list),
                    append_only=False,
                    trail=tuple(_drain_trail(session)),
                    api_call=api_call,
                    session=session,
                    turn_id=str(turn_id or ""),
                )
            else:
                index = first_divergent_index(previous, fingerprints)
                prefix = (
                    message_list if index is None else message_list[:index]
                )
                observation = PrefixObservation(
                    baseline=False,
                    first_changed_index=index,
                    first_changed_role=_role_at(message_list, index),
                    reusable_prefix_tokens=_estimate_tokens(prefix),
                    prompt_tokens=prompt_tokens,
                    previous_prompt_tokens=previous_tokens,
                    message_count=len(message_list),
                    append_only=(index is None or index >= len(previous)),
                    trail=tuple(_drain_trail(session)),
                    api_call=api_call,
                    session=session,
                    turn_id=str(turn_id or ""),
                )
        except Exception:
            logger.debug("prefix stability observation failed", exc_info=True)
            return None
        if log:
            logger.info("%s", observation.to_log_line())
        return observation


def summarize_prefix_reuse(
    observations: Iterable[PrefixObservation | None],
) -> tuple[float, list[PrefixObservation]]:
    """Worst-case reuse ratio across non-baseline observations.

    Returns ``(worst_ratio, comparable_observations)``; the ratio is 1.0 when
    there is nothing to compare, so a session with a single request never
    reports a false regression.
    """

    comparable = [
        obs for obs in observations if obs is not None and not obs.baseline
    ]
    if not comparable:
        return 1.0, []
    return min(obs.prefix_reuse_ratio for obs in comparable), comparable


__all__ = [
    "HistoryMutationEvent",
    "PrefixObservation",
    "PrefixStabilityTracker",
    "build_history_mutation_event",
    "first_divergent_index",
    "log_history_mutation",
    "measurement_enabled",
    "message_fingerprint",
    "message_fingerprints",
    "note_history_mutation_intent",
    "reset_mutation_trails",
    "summarize_prefix_reuse",
]
