"""Unit tests for the structured history-mutation log.

The log is the measurement half of the prefix-stability work: before any
mutation was moved, the harness had to be able to say WHICH component
rewrote WHICH token position and why. These tests pin that contract —
one line per mutation, an accurate divergence point, and no message
content or raw session identifier anywhere in the output.
"""

from __future__ import annotations

import logging

import pytest

from agent.history_mutation_log import (
    HistoryMutationEvent,
    PrefixStabilityTracker,
    build_history_mutation_event,
    first_divergent_index,
    log_history_mutation,
    measurement_enabled,
    message_fingerprint,
    message_fingerprints,
    note_history_mutation_intent,
    reset_mutation_trails,
    summarize_prefix_reuse,
)

LOGGER_NAME = "agent.history_mutation"


@pytest.fixture(autouse=True)
def _clean_trails():
    reset_mutation_trails()
    yield
    reset_mutation_trails()


@pytest.fixture()
def enabled(caplog):
    """Turn the instrument on and capture its lines."""
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        yield caplog


def _text(role: str, body: str) -> dict:
    return {"role": role, "content": body}


def _history(n: int = 6, size: int = 400) -> list[dict]:
    msgs = [_text("system", "S" * size), _text("user", "U" * size)]
    for i in range(n):
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": f"result-{i}-" + ("R" * size),
            }
        )
    return msgs


def _lines(caplog, prefix: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER_NAME and r.getMessage().startswith(prefix)
    ]


def _fields(line: str) -> dict[str, str]:
    # "history_mutation k=v k=v ..." — values never contain spaces.
    return dict(part.split("=", 1) for part in line.split(" ")[1:] if "=" in part)


class TestFingerprints:
    def test_identical_messages_share_a_fingerprint(self):
        assert message_fingerprint(_text("user", "hi")) == message_fingerprint(
            _text("user", "hi")
        )

    def test_content_change_changes_the_fingerprint(self):
        assert message_fingerprint(_text("user", "hi")) != message_fingerprint(
            _text("user", "ho")
        )

    def test_harness_private_keys_are_not_a_mutation(self):
        """The transport strips underscore keys, so stamping one is not a
        cache break — the DB-persisted marker the prune stamps would
        otherwise report every message as rewritten."""
        plain = _text("tool", "payload")
        stamped = {**plain, "_db_persisted": True, "_row_id": 41}
        assert message_fingerprint(plain) == message_fingerprint(stamped)

    def test_unserializable_content_still_fingerprints(self):
        class Weird:
            pass

        assert message_fingerprint({"role": "user", "content": Weird()})


class TestDivergence:
    def test_identical_lists_report_none(self):
        fps = message_fingerprints(_history())
        assert first_divergent_index(fps, fps) is None

    def test_pure_append_diverges_past_the_end(self):
        before = _history()
        after = before + [_text("user", "trailing block")]
        idx = first_divergent_index(
            message_fingerprints(before), message_fingerprints(after)
        )
        assert idx == len(before)

    def test_mid_history_rewrite_reports_that_index(self):
        before = _history()
        after = [dict(m) for m in before]
        after[5]["content"] = "[pruned]"
        idx = first_divergent_index(
            message_fingerprints(before), message_fingerprints(after)
        )
        assert idx == 5

    def test_truncation_reports_the_first_missing_index(self):
        before = _history()
        after = before[:-4]
        idx = first_divergent_index(
            message_fingerprints(before), message_fingerprints(after)
        )
        assert idx == len(after)


class TestLogHistoryMutation:
    def test_append_preserves_the_whole_prefix(self, enabled):
        before = _history()
        after = before + [_text("user", "[checkpoint instruction]")]
        event = log_history_mutation(
            component="context_pressure_checkpoint",
            reason="request_only_context",
            before=before,
            after=after,
            session_id="chan:123:thread:456",
            turn_id="turn-7",
        )
        assert isinstance(event, HistoryMutationEvent)
        assert event.append_only is True
        assert event.first_changed_index == len(before)
        assert event.stable_prefix_ratio == 1.0
        assert event.invalidated_tokens == 0

        line = _lines(enabled, "history_mutation ")[0]
        fields = _fields(line)
        assert fields["component"] == "context_pressure_checkpoint"
        assert fields["reason"] == "request_only_context"
        assert fields["append_only"] == "true"
        assert fields["stable_prefix_ratio"] == "1.000"
        assert fields["turn"] == "turn-7"

    def test_in_place_rewrite_reports_position_and_cost(self, enabled):
        """The regression this whole change exists to prevent: appending an
        instruction to the active user turn instead of the tail."""
        before = _history()
        after = [dict(m) for m in before]
        after[1]["content"] = after[1]["content"] + "\n\n[checkpoint]"
        event = log_history_mutation(
            component="context_pressure_checkpoint",
            reason="active_user_turn_amend",
            before=before,
            after=after,
            session_id="s",
        )
        assert event is not None
        assert event.first_changed_index == 1
        assert event.first_changed_role == "user"
        assert event.append_only is False
        assert event.stable_prefix_ratio < 1.0
        assert event.invalidated_tokens > 0

        fields = _fields(_lines(enabled, "history_mutation ")[0])
        assert fields["first_changed_index"] == "1"
        assert fields["first_changed_role"] == "user"
        assert fields["append_only"] == "false"

    def test_prune_reports_the_reclaim_and_the_break_point(self, enabled):
        before = _history(n=8, size=2000)
        after = [dict(m) for m in before]
        for msg in after[4:10]:
            if msg.get("role") == "tool":
                msg["content"] = "[old tool output pruned]"
        event = log_history_mutation(
            component="proactive_prune",
            reason="turn_boundary_commit",
            before=before,
            after=after,
            session_id="s",
            extra={"pruned_results": 3},
        )
        assert event is not None
        assert event.first_changed_index == 5
        assert event.tokens_after < event.tokens_before
        assert _fields(_lines(enabled, "history_mutation ")[0])["pruned_results"] == "3"

    def test_noop_is_silent_by_default(self, enabled):
        before = _history()
        assert (
            log_history_mutation(
                component="c", reason="r", before=before, after=list(before)
            )
            is None
        )
        assert _lines(enabled, "history_mutation ") == []

    def test_noop_can_be_logged_explicitly(self, enabled):
        before = _history()
        event = log_history_mutation(
            component="c",
            reason="r",
            before=before,
            after=list(before),
            log_noop=True,
        )
        assert event is not None and event.first_changed_index is None
        assert _fields(_lines(enabled, "history_mutation ")[0])["first_changed_index"] == "-"

    def test_raw_session_key_never_reaches_the_log(self, enabled):
        raw = "discord:987654321098765432:thread:123456789012345678"
        log_history_mutation(
            component="c",
            reason="r",
            before=_history(),
            after=_history() + [_text("user", "x")],
            session_id=raw,
        )
        line = _lines(enabled, "history_mutation ")[0]
        assert raw not in line
        assert "987654321098765432" not in line

    def test_message_content_never_reaches_the_log(self, enabled):
        secret = "TOPSECRETPAYLOAD"
        before = _history()
        after = before + [_text("user", secret)]
        log_history_mutation(
            component="c", reason="r", before=before, after=after, session_id="s"
        )
        assert secret not in _lines(enabled, "history_mutation ")[0]

    def test_disabled_logger_does_no_work(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert measurement_enabled() is False
            assert (
                log_history_mutation(
                    component="c",
                    reason="r",
                    before=_history(),
                    after=_history()[:-1],
                    session_id="s",
                )
                is None
            )
            assert _lines(caplog, "history_mutation ") == []

    def test_builder_is_usable_without_logging(self):
        before = _history()
        event = build_history_mutation_event(
            component="c", reason="r", before=before, after=before + [_text("user", "x")]
        )
        assert event.append_only is True


class TestMutationIntent:
    def test_deferral_is_recorded_without_a_transcript(self, enabled):
        note_history_mutation_intent(
            component="proactive_prune",
            reason="deferred_to_turn_boundary",
            session_id="s",
            turn_id="turn-2",
            extra={"observed_tokens": 118_000},
        )
        fields = _fields(_lines(enabled, "history_mutation_intent ")[0])
        assert fields["component"] == "proactive_prune"
        assert fields["reason"] == "deferred_to_turn_boundary"
        assert fields["observed_tokens"] == "118000"

    def test_disabled_logger_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            note_history_mutation_intent(component="c", reason="r", session_id="s")
            assert _lines(caplog, "history_mutation_intent ") == []


class TestPrefixStabilityTracker:
    def test_first_request_is_a_baseline(self, enabled):
        tracker = PrefixStabilityTracker()
        obs = tracker.observe(_history(), session_id="s", api_call=1)
        assert obs is not None and obs.baseline is True
        assert obs.prefix_reuse_ratio == 0.0
        assert _fields(_lines(enabled, "prompt_prefix_stability ")[0])["baseline"] == "true"

    def test_appending_keeps_the_prefix_fully_reusable(self, enabled):
        tracker = PrefixStabilityTracker()
        history = _history()
        tracker.observe(history, session_id="s", api_call=1)
        obs = tracker.observe(
            history + [_text("user", "[tail block]")], session_id="s", api_call=2
        )
        assert obs is not None and obs.baseline is False
        assert obs.append_only is True
        assert obs.prefix_reuse_ratio > 0.99
        assert obs.reprefill_tokens < 100

    def test_mid_history_rewrite_shows_up_as_lost_reuse(self, enabled):
        tracker = PrefixStabilityTracker()
        history = _history(n=10, size=3000)
        tracker.observe(history, session_id="s", api_call=1)
        rewritten = [dict(m) for m in history]
        rewritten[3]["content"] = "[pruned]"
        obs = tracker.observe(rewritten, session_id="s", api_call=2)
        assert obs is not None
        assert obs.first_changed_index == 3
        assert obs.prefix_reuse_ratio < 0.5
        assert obs.reprefill_tokens > 0

    def test_trail_attributes_the_divergence_to_a_component(self, enabled):
        tracker = PrefixStabilityTracker()
        history = _history(n=6, size=1000)
        tracker.observe(history, session_id="s", api_call=1)

        rewritten = [dict(m) for m in history]
        rewritten[4]["content"] = "[pruned]"
        log_history_mutation(
            component="proactive_prune",
            reason="turn_boundary_commit",
            before=history,
            after=rewritten,
            session_id="s",
        )
        obs = tracker.observe(rewritten, session_id="s", api_call=2)
        assert obs is not None
        assert obs.trail == ("proactive_prune:turn_boundary_commit",)
        assert "proactive_prune:turn_boundary_commit" in _lines(
            enabled, "prompt_prefix_stability "
        )[-1]

    def test_unattributed_divergence_has_an_empty_trail(self, enabled):
        """A rewrite by a component that does not report itself still shows
        up — with `since=-`, which is the signal to go looking for it."""
        tracker = PrefixStabilityTracker()
        history = _history(n=6, size=1000)
        tracker.observe(history, session_id="s", api_call=1)
        rewritten = [dict(m) for m in history]
        rewritten[2]["content"] = "silently rewritten"
        obs = tracker.observe(rewritten, session_id="s", api_call=2)
        assert obs is not None and obs.trail == ()
        assert _fields(_lines(enabled, "prompt_prefix_stability ")[-1])["since"] == "-"

    def test_trails_are_scoped_per_session(self, enabled):
        tracker_a = PrefixStabilityTracker()
        tracker_b = PrefixStabilityTracker()
        history = _history()
        tracker_a.observe(history, session_id="session-a", api_call=1)
        tracker_b.observe(history, session_id="session-b", api_call=1)
        log_history_mutation(
            component="proactive_prune",
            reason="turn_boundary_commit",
            before=history,
            after=history + [_text("user", "x")],
            session_id="session-a",
        )
        obs_b = tracker_b.observe(history, session_id="session-b", api_call=2)
        assert obs_b is not None and obs_b.trail == ()
        obs_a = tracker_a.observe(history, session_id="session-a", api_call=2)
        assert obs_a is not None and obs_a.trail

    def test_reset_makes_the_next_request_a_baseline(self, enabled):
        tracker = PrefixStabilityTracker()
        history = _history()
        tracker.observe(history, session_id="s", api_call=1)
        tracker.reset()
        obs = tracker.observe(history, session_id="s", api_call=2)
        assert obs is not None and obs.baseline is True

    def test_disabled_logger_returns_no_observation(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert PrefixStabilityTracker().observe(_history(), session_id="s") is None


class TestSummarize:
    def test_worst_ratio_wins(self, enabled):
        tracker = PrefixStabilityTracker()
        history = _history(n=8, size=2000)
        observations = [tracker.observe(history, session_id="s", api_call=1)]
        grown = history + [_text("user", "tail")]
        observations.append(tracker.observe(grown, session_id="s", api_call=2))
        broken = [dict(m) for m in grown]
        broken[3]["content"] = "[pruned]"
        observations.append(tracker.observe(broken, session_id="s", api_call=3))

        worst, comparable = summarize_prefix_reuse(observations)
        assert len(comparable) == 2  # the baseline is excluded
        assert worst < 0.5

    def test_nothing_to_compare_is_not_a_regression(self):
        worst, comparable = summarize_prefix_reuse([None])
        assert worst == 1.0 and comparable == []


class TestRequestOnlyTailBlockShape:
    """The trailing block must be valid for every provider shape.

    Chat-completions accepts a user turn after tool results directly. The
    Anthropic wire does not have a ``tool`` role at all: results become
    ``tool_result`` blocks inside a user turn, and consecutive user turns
    are merged. These tests run the real converter so "provider-valid"
    is verified against the adapter rather than asserted in a comment.
    """

    @staticmethod
    def _tool_loop_tail() -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do the work"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "result"},
        ]

    def test_tail_block_lands_after_tool_results(self):
        from agent.conversation_loop import _append_request_only_tail_block

        messages = self._tool_loop_tail()
        assert (
            _append_request_only_tail_block(
                messages, "[checkpoint]", component="c", reason="r"
            )
            is True
        )
        assert messages[-1] == {"role": "user", "content": "[checkpoint]"}

    def test_anthropic_conversion_keeps_roles_alternating(self):
        from agent.anthropic_adapter import convert_messages_to_anthropic
        from agent.conversation_loop import _append_request_only_tail_block

        messages = self._tool_loop_tail()
        _append_request_only_tail_block(
            messages, "[checkpoint]", component="c", reason="r"
        )
        system, converted = convert_messages_to_anthropic(messages)
        assert system == "sys"
        roles = [m["role"] for m in converted]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles

        # The trailing text merged into the tool_result turn, and the
        # tool_result blocks still come first in it.
        tail = converted[-1]
        assert tail["role"] == "user"
        types = [b.get("type") for b in tail["content"]]
        assert types[0] == "tool_result"
        assert types[-1] == "text"
        assert tail["content"][-1]["text"].strip() == "[checkpoint]"

    def test_trailing_user_turn_takes_the_text_in_place(self):
        from agent.conversation_loop import _append_request_only_tail_block

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do the work"},
        ]
        assert (
            _append_request_only_tail_block(
                messages, "[checkpoint]", component="c", reason="r"
            )
            is True
        )
        assert len(messages) == 2
        assert messages[-1]["content"].endswith("[checkpoint]")

    def test_multimodal_user_tail_gets_a_trailing_text_part(self):
        from agent.conversation_loop import _append_request_only_tail_block

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "look at this"}],
            }
        ]
        _append_request_only_tail_block(
            messages, "[checkpoint]", component="c", reason="r"
        )
        assert messages[-1]["content"][-1]["text"].strip() == "[checkpoint]"

    def test_unanswered_tool_calls_fall_back_without_dropping_the_text(self):
        """Unreachable in the loop (the sanitizer stubs results in first),
        but the instruction must never be silently dropped if it happens."""
        from agent.conversation_loop import _append_request_only_tail_block

        messages = [
            {"role": "user", "content": "do the work"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
        ]
        assert (
            _append_request_only_tail_block(
                messages, "[checkpoint]", component="c", reason="r"
            )
            is False
        )
        assert len(messages) == 2
        assert "[checkpoint]" in messages[0]["content"]


class TestLogLineFormat:
    def test_whitespace_in_a_field_cannot_split_the_line(self, enabled):
        before = _history()
        log_history_mutation(
            component="c",
            reason="r",
            before=before,
            after=before + [_text("user", "x")],
            session_id="s",
            extra={"note": "two words here"},
        )
        fields = _fields(_lines(enabled, "history_mutation ")[0])
        assert fields["note"] == "two_words_here"
        assert fields["append_only"] == "true"
