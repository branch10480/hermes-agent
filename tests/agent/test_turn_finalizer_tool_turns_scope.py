"""Turn-scoped vs. session-cumulative tool-turn counts in the "Turn ended" log.

Before this change, the "Turn ended" diagnostic logged a single ``tool_turns``
field that actually scanned the *entire* in-memory transcript (session-
cumulative across every turn), while the sibling ``api_calls`` field on the
same log line is turn-scoped. That mismatch produced misleading lines such as
``api_calls=1/50 ... tool_turns=76`` on a long-running session. The log now
carries both: ``tool_turns`` (this turn only, anchored at
``agent._persist_user_message_idx`` — the same index ``turn_context``/
``conversation_loop`` maintain for the persist-flush boundary) and
``history_tool_turns`` (the old, session-cumulative scan).
"""

import logging

from agent.turn_finalizer import finalize_turn

LOGGER_NAME = "agent.conversation_loop"


class _StubBudget:
    used = 1
    max_total = 50
    remaining = 49


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, persist_user_message_idx=None):
        self.max_iterations = 50
        self.iteration_budget = _StubBudget()
        self.context_compressor = None
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._persist_user_message_idx = persist_user_message_idx
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _tool_call_msg(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "function": {"name": "read_file", "arguments": "{}"}}
        ],
    }


def _make_messages():
    """Two prior turns (3 tool-call rows) + the current turn (1 tool-call row).

    The current-turn user message sits at index 9; only the single tool-call
    row after it (index 10) belongs to "this turn".
    """
    return [
        {"role": "user", "content": "turn 1"},                  # 0
        _tool_call_msg("a"),                                     # 1
        {"role": "tool", "tool_call_id": "a", "content": "x"},   # 2
        _tool_call_msg("b"),                                     # 3
        {"role": "tool", "tool_call_id": "b", "content": "x"},   # 4
        {"role": "assistant", "content": "done with turn 1"},    # 5
        {"role": "user", "content": "turn 2"},                   # 6
        _tool_call_msg("c"),                                     # 7
        {"role": "tool", "tool_call_id": "c", "content": "x"},   # 8
        {"role": "user", "content": "turn 3 (current)"},         # 9
        _tool_call_msg("d"),                                     # 10
        {"role": "tool", "tool_call_id": "d", "content": "x"},   # 11
    ]


def _run(agent, messages, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = finalize_turn(
            agent,
            final_response="all done",
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=None,
            effective_task_id="task-1",
            turn_id="turn-1",
            user_message="turn 3 (current)",
            original_user_message="turn 3 (current)",
            _should_review_memory=False,
            _turn_exit_reason="text_response(finish_reason=stop)",
        )
    return result


def _turn_end_record(caplog):
    matches = [r for r in caplog.records if r.message.startswith("Turn ended:")]
    assert len(matches) == 1, caplog.text
    return matches[0]


class TestToolTurnsScope:
    def test_turn_scope_counts_only_current_turn_tool_calls(self, caplog):
        messages = _make_messages()
        # The anchor turn_context/conversation_loop maintain for "where this
        # turn's user message lives in `messages`".
        agent = _StubAgent(persist_user_message_idx=9)
        result = _run(agent, messages, caplog)

        record = _turn_end_record(caplog)
        assert "tool_turns=1 " in record.message
        assert "history_tool_turns=4 " in record.message
        assert result["final_response"] == "all done"

    def test_falls_back_to_last_user_message_when_anchor_missing(self, caplog):
        """No ``_persist_user_message_idx`` set: approximate via the last
        user-role row in ``messages`` rather than raising or silently
        reverting to a full-session scan."""
        messages = _make_messages()
        agent = _StubAgent(persist_user_message_idx=None)
        _run(agent, messages, caplog)

        record = _turn_end_record(caplog)
        # Same expectation as the exact-anchor case: the last user message in
        # this transcript is still index 9, so the fallback lands on the same
        # turn-scoped count.
        assert "tool_turns=1 " in record.message
        assert "history_tool_turns=4 " in record.message

    def test_falls_back_when_anchor_out_of_range(self, caplog):
        messages = _make_messages()
        agent = _StubAgent(persist_user_message_idx=999)
        _run(agent, messages, caplog)

        record = _turn_end_record(caplog)
        assert "tool_turns=1 " in record.message
        assert "history_tool_turns=4 " in record.message

    def test_no_current_turn_tool_calls_yields_zero_turn_scope(self, caplog):
        messages = _make_messages()[:10]  # up through the current-turn user msg
        agent = _StubAgent(persist_user_message_idx=9)
        _run(agent, messages, caplog)

        record = _turn_end_record(caplog)
        assert "tool_turns=0 " in record.message
        assert "history_tool_turns=3 " in record.message

    def test_log_line_field_order_and_format(self, caplog):
        """Pin the exact field layout so downstream parsers (castle's hstat
        ``SLOT_TURN_END_RE``) can be checked against it: ``tool_turns=%d`` is
        immediately followed by ``history_tool_turns=%d``, both ahead of
        ``last_msg_role=``, and ``session=`` remains the trailing field."""
        messages = _make_messages()
        agent = _StubAgent(persist_user_message_idx=9)
        _run(agent, messages, caplog)

        record = _turn_end_record(caplog)
        assert (
            "budget=1/50 tool_turns=1 history_tool_turns=4 "
            "last_msg_role=assistant response_len=8 session=sess-1"
        ) in record.message
