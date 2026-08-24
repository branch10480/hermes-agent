from unittest.mock import patch

from agent.context_checkpoint_hooks import (
    context_pressure_context,
    notify_post_context_compression,
    notify_pre_context_compression,
)


def test_pressure_context_combines_supported_result_shapes():
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[
                {"context": "save current state"},
                "restore prior checkpoint",
                {"ignored": "value"},
                None,
            ],
        ) as invoke,
    ):
        result = context_pressure_context(
            session_id="s1", approx_tokens=80, threshold_tokens=100
        )

    assert result == "save current state\n\nrestore prior checkpoint"
    invoke.assert_called_once_with(
        "on_context_pressure",
        session_id="s1",
        approx_tokens=80,
        threshold_tokens=100,
    )


def test_pressure_context_is_bounded_and_fail_open():
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "x" * 20_000}],
        ),
    ):
        assert len(context_pressure_context(session_id="s1")) == 12_000

    with patch("hermes_cli.lifecycle.has_hook", side_effect=RuntimeError("boom")):
        assert context_pressure_context(session_id="s1") == ""


def test_compression_notifications_are_observer_only_and_fail_open():
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch("hermes_cli.lifecycle.invoke_hook") as invoke,
    ):
        notify_pre_context_compression(session_id="old", conversation_history=[])
        notify_post_context_compression(
            session_id="new", old_session_id="old", in_place=False
        )

    assert invoke.call_args_list[0].args == ("pre_context_compression",)
    assert invoke.call_args_list[0].kwargs["session_id"] == "old"
    assert invoke.call_args_list[1].args == ("post_context_compression",)
    assert invoke.call_args_list[1].kwargs["session_id"] == "new"

    with patch("hermes_cli.lifecycle.has_hook", side_effect=RuntimeError("boom")):
        notify_pre_context_compression(session_id="old")
        notify_post_context_compression(session_id="new")
