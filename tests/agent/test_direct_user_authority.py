from __future__ import annotations

import threading

from agent.direct_user_authority import (
    begin_turn,
    claim_cloud_egress,
    claim_publication,
    close_turn,
    current_revision,
    forget_session,
    revoke_for_correction,
)


def test_revision_and_cloud_claim_are_exact_and_one_way():
    session_id = "authority-session"
    task_id = "authority-task"
    turn_id = "authority-turn"
    forget_session(session_id)

    assert begin_turn(session_id, task_id, turn_id)
    assert current_revision(session_id, task_id, turn_id) == 0
    assert revoke_for_correction(session_id, task_id, turn_id) == 1
    assert not claim_cloud_egress(session_id, task_id, turn_id, 0)
    assert not claim_cloud_egress(session_id, task_id, turn_id, 1)

    # Later corrections remain monotonic and cannot reopen high-impact
    # authority inside the same turn.
    assert revoke_for_correction(session_id, task_id, turn_id) == 2
    assert current_revision(session_id, task_id, turn_id) == 2
    assert not claim_cloud_egress(session_id, task_id, turn_id, 2)
    assert not claim_publication(session_id, task_id, turn_id, 2)

    close_turn(session_id, task_id, turn_id)
    assert current_revision(session_id, task_id, turn_id) is None
    assert revoke_for_correction(session_id, task_id, turn_id) is None
    forget_session(session_id)


def test_new_turn_invalidates_delayed_old_turn_delivery():
    session_id = "authority-rotation"
    forget_session(session_id)
    assert begin_turn(session_id, "task-old", "turn-old")
    assert begin_turn(session_id, "task-new", "turn-new")

    assert current_revision(session_id, "task-old", "turn-old") is None
    assert not claim_cloud_egress(session_id, "task-old", "turn-old", 0)
    assert current_revision(session_id, "task-new", "turn-new") == 0
    forget_session(session_id)


def test_reused_claimed_or_closed_turn_identity_never_reopens():
    for suffix, close_first in (("claimed", False), ("closed", True)):
        session_id = f"authority-reused-{suffix}"
        task_id = "authority-task"
        turn_id = "authority-turn"
        forget_session(session_id)
        assert begin_turn(session_id, task_id, turn_id)
        if close_first:
            close_turn(session_id, task_id, turn_id)
        else:
            assert claim_cloud_egress(session_id, task_id, turn_id, 0)

        assert not begin_turn(session_id, task_id, turn_id)
        assert current_revision(session_id, task_id, turn_id) is None
        assert not claim_cloud_egress(session_id, task_id, turn_id, 0)
        forget_session(session_id)


def test_duplicate_open_turn_identity_closes_original_contender():
    session_id = "authority-duplicate-open"
    task_id = "authority-task"
    turn_id = "authority-turn"
    forget_session(session_id)
    assert begin_turn(session_id, task_id, turn_id)
    assert not begin_turn(session_id, task_id, turn_id)
    assert current_revision(session_id, task_id, turn_id) is None
    assert not claim_cloud_egress(session_id, task_id, turn_id, 0)
    forget_session(session_id)


def test_cloud_claim_is_atomic_under_concurrent_attempts():
    session_id = "authority-concurrent"
    task_id = "authority-task"
    turn_id = "authority-turn"
    forget_session(session_id)
    assert begin_turn(session_id, task_id, turn_id)
    barrier = threading.Barrier(3)
    outcomes: list[bool] = []

    def claim() -> None:
        barrier.wait()
        outcomes.append(claim_cloud_egress(session_id, task_id, turn_id, 0))

    first = threading.Thread(target=claim)
    second = threading.Thread(target=claim)
    first.start()
    second.start()
    barrier.wait()
    first.join(timeout=1)
    second.join(timeout=1)

    assert sorted(outcomes) == [False, True]
    forget_session(session_id)


def test_publication_claim_linearizes_before_or_after_correction():
    session_id = "authority-publication"
    task_id = "authority-task"
    turn_id = "authority-turn"
    forget_session(session_id)
    assert begin_turn(session_id, task_id, turn_id)

    # The untouched direct turn can make more than one serialized attempt,
    # which supports a safe retry token after a transient publication error.
    assert claim_publication(session_id, task_id, turn_id, 0)
    assert claim_publication(session_id, task_id, turn_id, 0)
    assert not claim_publication(session_id, task_id, turn_id, False)
    assert revoke_for_correction(session_id, task_id, turn_id) == 1
    assert not claim_publication(session_id, task_id, turn_id, 0)
    assert not claim_publication(session_id, task_id, turn_id, 1)
    forget_session(session_id)

    # If correction wins the same registry lock first, no publication boundary
    # can be crossed afterward in that turn.
    assert begin_turn(session_id, task_id, "turn-corrected-first")
    assert revoke_for_correction(session_id, task_id, "turn-corrected-first") == 1
    assert not claim_publication(
        session_id, task_id, "turn-corrected-first", 0
    )
    forget_session(session_id)


def test_runtime_authority_keywords_require_explicit_keyword_parameters():
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    schema = {
        "name": "placeholder",
        "description": "authority signature contract",
        "parameters": {"type": "object", "properties": {}},
    }

    def explicit(_args, *, direct_user_authority_revision=0):
        return direct_user_authority_revision

    def variadic(_args, **_kwargs):
        return None

    def positional(_args, direct_user_authority_revision=0, /):
        return direct_user_authority_revision

    for name, handler in (
        ("authority_explicit", explicit),
        ("authority_variadic", variadic),
        ("authority_positional", positional),
    ):
        registry.register(
            name=name,
            toolset="test",
            schema={**schema, "name": name},
            handler=handler,
        )

    assert registry.handler_accepts_keyword(
        "authority_explicit", "direct_user_authority_revision"
    )
    assert not registry.handler_accepts_keyword(
        "authority_variadic", "direct_user_authority_revision"
    )
    assert not registry.handler_accepts_keyword(
        "authority_positional", "direct_user_authority_revision"
    )
