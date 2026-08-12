from __future__ import annotations

import threading

from agent.direct_user_authority import (
    begin_turn,
    claim_cloud_egress,
    claim_publication,
    close_turn,
    current_revision,
    consume_bound_capability,
    forget_session,
    issue_bound_capability,
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
    assert claim_publication(session_id, task_id, turn_id, 0)
    assert claim_publication(session_id, task_id, turn_id, 0)
    assert not claim_publication(session_id, task_id, turn_id, False)
    assert revoke_for_correction(session_id, task_id, turn_id) == 1
    assert not claim_publication(session_id, task_id, turn_id, 0)
    assert not claim_publication(session_id, task_id, turn_id, 1)
    forget_session(session_id)

    assert begin_turn(session_id, task_id, "turn-corrected-first")
    assert revoke_for_correction(session_id, task_id, "turn-corrected-first") == 1
    assert not claim_publication(
        session_id, task_id, "turn-corrected-first", 0
    )
    forget_session(session_id)


def test_bound_capability_is_opaque_exact_and_one_shot():
    session_id = "authority-bound-publication"
    task_id = "authority-task"
    turn_id = "authority-turn"
    binding = "plugin-validated:exact-binding-v1"
    forget_session(session_id)
    assert begin_turn(session_id, task_id, turn_id)
    token = issue_bound_capability(
        session_id, task_id, turn_id, 0, binding,
    )
    assert isinstance(token, str) and token not in binding
    assert not consume_bound_capability(token, binding + " ")
    assert not consume_bound_capability(token, binding)
    forget_session(session_id)

    assert begin_turn(session_id, task_id, "authority-turn-2")
    token = issue_bound_capability(
        session_id, task_id, "authority-turn-2", 0, binding,
    )
    assert token is not None
    assert consume_bound_capability(token, binding)
    assert not consume_bound_capability(token, binding)
    forget_session(session_id)


def test_untrusted_turn_never_opens_direct_authority():
    session_id = "authority-untrusted"
    forget_session(session_id)
    assert not begin_turn(
        session_id, "task", "turn", provenance="untrusted"
    )
    assert current_revision(session_id, "task", "turn") is None
    assert issue_bound_capability(
        session_id, "task", "turn", 0, "binding"
    ) is None
    forget_session(session_id)


def test_scheduled_bound_capability_survives_turn_close():
    session_id = "authority-scheduled"
    task_id = "authority-task"
    turn_id = "authority-turn"
    grant = "plugin-validated:scheduled-grant-v1"
    forget_session(session_id)
    assert begin_turn(
        session_id, task_id, turn_id, provenance="scheduled"
    )
    token = issue_bound_capability(
        session_id,
        task_id,
        turn_id,
        0,
        grant,
        authority_kind="scheduled",
    )
    assert token is not None
    close_turn(session_id, task_id, turn_id)
    assert consume_bound_capability(token, grant)
    assert not consume_bound_capability(token, grant)
    forget_session(session_id)


def test_replayed_scheduled_turn_revokes_its_original_capability():
    session_id = "authority-scheduled-replay"
    task_id = "authority-task"
    turn_id = "authority-turn"
    grant = "plugin-validated:scheduled-grant-v1"
    forget_session(session_id)
    assert begin_turn(session_id, task_id, turn_id, provenance="scheduled")
    token = issue_bound_capability(
        session_id, task_id, turn_id, 0, grant, authority_kind="scheduled",
    )
    assert token is not None
    assert not begin_turn(session_id, task_id, turn_id, provenance="scheduled")
    assert not consume_bound_capability(token, grant)
    forget_session(session_id)


def test_bound_capability_rejects_empty_or_oversized_bindings():
    session_id = "authority-invalid-binding"
    forget_session(session_id)
    assert begin_turn(session_id, "task", "turn")
    assert issue_bound_capability(
        session_id, "task", "turn", 0, ""
    ) is None
    assert issue_bound_capability(
        session_id, "task", "turn", 0, "x" * 4097
    ) is None
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
