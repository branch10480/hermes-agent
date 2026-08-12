"""Core-owned direct-user authority state for one live conversation turn.

High-impact plugins may read the revision and atomically claim the cloud-egress
boundary, but model-authored tool arguments never control this state.  Missing,
stale, reused, or already-closed identities fail closed.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from typing import Optional


_lock = threading.RLock()
_active_by_session: dict[str, tuple[str, str, str]] = {}
_states: dict[tuple[str, str, str], dict[str, object]] = {}
_seen_keys: set[tuple[str, str, str]] = set()
_bound_capabilities: dict[str, dict[str, object]] = {}


def _drop_capabilities_for_key(
    key: tuple[str, str, str], *, include_scheduled: bool = True,
) -> None:
    for digest, record in tuple(_bound_capabilities.items()):
        if (
            record.get("key") == key
            and (include_scheduled or record.get("authority_kind") != "scheduled")
        ):
            _bound_capabilities.pop(digest, None)


def _key(session_id: str, task_id: str, turn_id: str) -> Optional[tuple[str, str, str]]:
    if not all(
        isinstance(value, str) and value
        for value in (session_id, task_id, turn_id)
    ):
        return None
    return session_id, task_id, turn_id


def begin_turn(
    session_id: str,
    task_id: str,
    turn_id: str,
    *,
    provenance: str = "direct_user",
) -> bool:
    """Open authority only for core-proven direct-user or scheduled turns."""
    key = _key(session_id, task_id, turn_id)
    if key is None:
        return False
    with _lock:
        if key in _seen_keys:
            # Replaying even the exact same server identity invalidates the
            # original contender rather than sharing or reopening authority.
            if _active_by_session.get(session_id) == key:
                _drop_capabilities_for_key(key)
                _active_by_session.pop(session_id, None)
                _states.pop(key, None)
            return False
        previous = _active_by_session.get(session_id)
        if previous is not None and previous != key:
            _drop_capabilities_for_key(previous)
            _states.pop(previous, None)
        if provenance not in {"direct_user", "scheduled"}:
            _active_by_session.pop(session_id, None)
            _seen_keys.add(key)
            return False
        _active_by_session[session_id] = key
        _states[key] = {
            "revision": 0,
            "phase": "open",
            "publication_claims": 0,
            "bound_capability_issued": False,
            "provenance": provenance,
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
        _drop_capabilities_for_key(key)
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
            and state.get("provenance") == "direct_user"
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
    """Linearize a publication attempt against accepted corrections.

    Existing publication flows may issue a fresh immutable retry token after a
    transient pre-write failure, so this generic claim remains repeatable while
    the exact direct-user turn stays at revision zero.
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
        if state.get("provenance") != "direct_user":
            return False
        claims = state.get("publication_claims")
        if type(claims) is not int or claims < 0:
            return False
        state["publication_claims"] = claims + 1
        return True


def issue_bound_capability(
    session_id: str,
    task_id: str,
    turn_id: str,
    expected_revision: int,
    binding: str,
    *,
    authority_kind: str = "direct_user",
) -> Optional[str]:
    """Issue one opaque capability bound to exact plugin-validated bytes.

    The raw token is returned once and only its SHA-256 digest is retained.
    Binding semantics remain in the plugin; core only proves provenance,
    revision, exact byte equality, expiry, and one-shot consumption.
    """

    key = _key(session_id, task_id, turn_id)
    if (
        key is None
        or type(expected_revision) is not int
        or expected_revision != 0
        or authority_kind not in {"direct_user", "scheduled"}
        or not isinstance(binding, str)
        or not binding
        or len(binding.encode("utf-8")) > 4096
    ):
        return None
    with _lock:
        current_time = time.time()
        for digest, record in tuple(_bound_capabilities.items()):
            if float(record.get("expires_at", 0)) < current_time:
                _bound_capabilities.pop(digest, None)
        if _active_by_session.get(session_id) != key:
            return None
        state = _states.get(key)
        if not (
            state
            and state.get("phase") == "open"
            and state.get("revision") == 0
            and state.get("bound_capability_issued") is False
            and state.get("provenance") == authority_kind
        ):
            return None
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        _bound_capabilities[digest] = {
            "key": key,
            "binding": binding,
            "authority_kind": authority_kind,
            "expires_at": current_time + (86_400 if authority_kind == "scheduled" else 900),
        }
        state["bound_capability_issued"] = True
        return token


def consume_bound_capability(token: str, binding: str) -> bool:
    """Atomically consume one generic exact-binding capability."""

    if (
        not isinstance(token, str)
        or not token
        or len(token) > 256
        or not isinstance(binding, str)
        or not binding
        or len(binding.encode("utf-8")) > 4096
    ):
        return False
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _lock:
        record = _bound_capabilities.pop(digest, None)
        if record is None or float(record.get("expires_at", 0)) < time.time():
            return False
        stored_binding = str(record.get("binding", ""))
        if not secrets.compare_digest(stored_binding, binding):
            return False
        authority_kind = record.get("authority_kind")
        key = record.get("key")
        if not isinstance(key, tuple) or len(key) != 3:
            return False
        if authority_kind != "scheduled":
            state = _states.get(key)
            if not (
                state
                and _active_by_session.get(key[0]) == key
                and state.get("phase") == "open"
                and state.get("revision") == 0
            ):
                return False
        return True


def close_turn(session_id: str, task_id: str, turn_id: str) -> None:
    """Tombstone a completed turn so delayed deliveries cannot regain authority."""
    key = _key(session_id, task_id, turn_id)
    if key is None:
        return
    with _lock:
        if _active_by_session.get(session_id) != key:
            return
        _drop_capabilities_for_key(key, include_scheduled=False)
        _active_by_session.pop(session_id, None)
        _states.pop(key, None)


def forget_session(session_id: str) -> None:
    """Remove bounded authority state when a session is reset/finalized."""
    if not isinstance(session_id, str) or not session_id:
        return
    with _lock:
        active = _active_by_session.pop(session_id, None)
        if active is not None:
            _drop_capabilities_for_key(active, include_scheduled=False)
            _states.pop(active, None)
        for key in tuple(_states):
            if key[0] == session_id:
                _drop_capabilities_for_key(key, include_scheduled=False)
                _states.pop(key, None)
        for key in tuple(_seen_keys):
            if key[0] == session_id:
                _seen_keys.discard(key)
