"""Core-owned direct-user authority state for one live conversation turn.

High-impact plugins may read the revision and atomically claim the cloud-egress
boundary, but model-authored tool arguments never control this state.  Missing,
stale, reused, or already-closed identities fail closed.
"""

from __future__ import annotations

import threading
from typing import Optional


_lock = threading.RLock()
_active_by_session: dict[str, tuple[str, str, str]] = {}
_states: dict[tuple[str, str, str], dict[str, object]] = {}
_seen_keys: set[tuple[str, str, str]] = set()


def _key(session_id: str, task_id: str, turn_id: str) -> Optional[tuple[str, str, str]]:
    if not all(
        isinstance(value, str) and value
        for value in (session_id, task_id, turn_id)
    ):
        return None
    return session_id, task_id, turn_id


def begin_turn(session_id: str, task_id: str, turn_id: str) -> bool:
    """Open a fresh revision-0 authority record for one server-created turn."""
    key = _key(session_id, task_id, turn_id)
    if key is None:
        return False
    with _lock:
        if key in _seen_keys:
            # Replaying even the exact same server identity invalidates the
            # original contender rather than sharing or reopening authority.
            if _active_by_session.get(session_id) == key:
                _active_by_session.pop(session_id, None)
                _states.pop(key, None)
            return False
        previous = _active_by_session.get(session_id)
        if previous is not None and previous != key:
            _states.pop(previous, None)
        _active_by_session[session_id] = key
        _states[key] = {
            "revision": 0,
            "phase": "open",
            "publication_claims": 0,
        }
        _seen_keys.add(key)
    return True


def revoke_for_correction(session_id: str, task_id: str, turn_id: str) -> Optional[int]:
    """Increment authority before best-effort correction hooks are invoked."""
    key = _key(session_id, task_id, turn_id)
    if key is None:
        return None
    with _lock:
        if _active_by_session.get(session_id) != key:
            return None
        state = _states.get(key)
        if state is None or state.get("phase") == "closed":
            return None
        revision = int(state.get("revision", 0)) + 1
        state["revision"] = revision
        return revision


def current_revision(
    session_id: str, task_id: str, turn_id: str
) -> Optional[int]:
    """Return the current revision only for the exact active open/claimed turn."""
    key = _key(session_id, task_id, turn_id)
    if key is None:
        return None
    with _lock:
        if _active_by_session.get(session_id) != key:
            return None
        state = _states.get(key)
        if state is None or state.get("phase") == "closed":
            return None
        revision = state.get("revision")
        return revision if type(revision) is int and revision >= 0 else None


def claim_cloud_egress(
    session_id: str,
    task_id: str,
    turn_id: str,
    expected_revision: int,
) -> bool:
    """Atomically cross the one-way cloud boundary for an exact live turn."""
    key = _key(session_id, task_id, turn_id)
    if key is None or type(expected_revision) is not int or expected_revision < 0:
        return False
    with _lock:
        if _active_by_session.get(session_id) != key:
            return False
        state = _states.get(key)
        if not (
            state
            and state.get("phase") == "open"
            and expected_revision == 0
            and state.get("revision") == 0
        ):
            return False
        state["phase"] = "egress_claimed"
        return True


def claim_publication(
    session_id: str,
    task_id: str,
    turn_id: str,
    expected_revision: int,
) -> bool:
    """Linearize one publication attempt against accepted corrections.

    A publication can be retried during the same untouched direct-user turn,
    so this claim is repeatable while revision zero remains active. The exact
    immutable publication token still provides one-writer serialization.
    """
    key = _key(session_id, task_id, turn_id)
    if (
        key is None
        or type(expected_revision) is not int
        or expected_revision != 0
    ):
        return False
    with _lock:
        if _active_by_session.get(session_id) != key:
            return False
        state = _states.get(key)
        if not state or state.get("revision") != 0:
            return False
        claims = state.get("publication_claims")
        if type(claims) is not int or claims < 0:
            return False
        state["publication_claims"] = claims + 1
        return True


def close_turn(session_id: str, task_id: str, turn_id: str) -> None:
    """Tombstone a completed turn so delayed deliveries cannot regain authority."""
    key = _key(session_id, task_id, turn_id)
    if key is None:
        return
    with _lock:
        if _active_by_session.get(session_id) != key:
            return
        _active_by_session.pop(session_id, None)
        _states.pop(key, None)


def forget_session(session_id: str) -> None:
    """Remove bounded authority state when a session is reset/finalized."""
    if not isinstance(session_id, str) or not session_id:
        return
    with _lock:
        active = _active_by_session.pop(session_id, None)
        if active is not None:
            _states.pop(active, None)
        for key in tuple(_states):
            if key[0] == session_id:
                _states.pop(key, None)
        for key in tuple(_seen_keys):
            if key[0] == session_id:
                _seen_keys.discard(key)
