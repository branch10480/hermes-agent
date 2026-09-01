"""Prompt-prefix stability across a 100K-token tool loop.

Two mutations used to rewrite history the provider had already cached, in
the middle of a live turn:

* the context-pressure checkpoint appended its request-only instruction to
  the ACTIVE USER TURN, which in a tool loop sits tens of thousands of
  tokens behind the tail;
* the proactive tool-result prune committed its rewrite between two calls
  of the same turn.

Each one cost a cold re-prefill of everything after the rewrite — measured
at 5-7 minutes per break on a local model. These tests drive
``run_conversation()`` through a real tool loop over a ~120K-token session
with a real ``ContextCompressor``, capture the exact message list handed to
the provider on every call, and assert the prefix stays reusable.

The requests are captured from ``api_kwargs["messages"]``, i.e. after every
sanitizer, injection and normalization pass — the bytes the provider
actually caches. Anything measured earlier would miss the mutations.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.history_mutation_log import (
    PrefixStabilityTracker,
    reset_mutation_trails,
    summarize_prefix_reuse,
)
from run_agent import AIAgent

# A 100K-class session: 40 tool iterations whose results are far above the
# prune's 8,000-char summarize floor, so the deterministic prune has real
# reclaimable content outside the protected tail.
_TOOL_ITERATIONS = 40
_TOOL_RESULT_CHARS = 12_000
_CONTEXT_LENGTH = 393_216
_PRUNE_TOKENS = 114_688  # the configured local-LLM trigger; not tuned here


@pytest.fixture(autouse=True)
def _clean_trails():
    reset_mutation_trails()
    yield
    reset_mutation_trails()


def _tool_call(i: int):
    return SimpleNamespace(
        id=f"call_{i}",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )


def _tool_response(i: int):
    msg = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call(i)],
    )
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _stop_response():
    msg = SimpleNamespace(
        content="done", reasoning_content=None, reasoning=None, tool_calls=None
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _seeded_history() -> list[dict]:
    """A long finished tool loop, distinct results (no dedup shortcuts)."""
    history: list[dict] = [
        {"role": "user", "content": "start the long investigation"}
    ]
    for i in range(_TOOL_ITERATIONS):
        history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"seed_{i}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": f"q{i}"}),
                        },
                    }
                ],
            }
        )
        history.append(
            {
                "role": "tool",
                "name": "web_search",
                "tool_call_id": f"seed_{i}",
                "content": json.dumps(
                    {"i": i, "payload": f"{i:04d}" * (_TOOL_RESULT_CHARS // 4)}
                ),
            }
        )
    history.append({"role": "assistant", "content": "here is what I found so far"})
    return history


def _real_compressor() -> ContextCompressor:
    return ContextCompressor(
        model="local/dwarfstar-test",
        threshold_percent=0.50,
        protect_first_n=3,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=_CONTEXT_LENGTH,
        proactive_prune_tokens=_PRUNE_TOKENS,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=4_096,
    )


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:18080/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=12,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.tool_delay = 0
    a.save_trajectories = False
    a.compression_enabled = True
    a.session_id = "prefix-stability-session"
    a.context_compressor = _real_compressor()
    return a


def _run(agent, *, tool_iterations: int, checkpoint_on: set[int] | None = None):
    """Drive a real turn; return (result, captured request message lists)."""

    responses = [_tool_response(i) for i in range(tool_iterations)]
    responses.append(_stop_response())
    captured: list[list[dict]] = []
    call_index = {"n": 0}

    def _create(*_args, **kwargs):
        captured.append(kwargs.get("messages") or [])
        idx = call_index["n"]
        call_index["n"] += 1
        return responses[min(idx, len(responses) - 1)]

    agent.client.chat.completions.create.side_effect = _create

    fired = checkpoint_on or set()

    def _pressure(**_kwargs):
        # The plugin checkpoint hook: request-only text that appears on
        # specific calls, mirroring a context-pressure level crossing.
        return (
            "[Checkpoint: save durable working state now. Request-only.]"
            if call_index["n"] in fired
            else ""
        )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "agent.context_checkpoint_hooks.context_pressure_context",
            side_effect=_pressure,
        ),
        patch(
            "run_agent.handle_function_call",
            # Distinct payload per iteration, for the same reason
            # ``_seeded_history`` uses distinct results: the tool-loop
            # guardrail collapses a byte-identical repeat into a short
            # "refer to the earlier result" note. With a constant blob every
            # iteration after the first adds ~250 tokens instead of ~3,000,
            # and the tool loop this file exists to measure never grows.
            lambda name, args, task_id=None, **kwargs: json.dumps(
                {
                    "ok": True,
                    "call": call_index["n"],
                    "blob": f"{call_index['n']:04d}" * (_TOOL_RESULT_CHARS // 4),
                }
            ),
        ),
    ):
        result = agent.run_conversation(
            "continue the investigation", conversation_history=_seeded_history()
        )
    return result, captured


def _observe(captured: list[list[dict]]) -> list:
    tracker = PrefixStabilityTracker()
    return [
        tracker.observe(msgs, session_id="prefix-stability-session", api_call=i)
        for i, msgs in enumerate(captured)
    ]


@pytest.fixture()
def measuring(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="agent.history_mutation"):
        yield caplog


class TestPrefixStability:
    def test_session_is_large_enough_to_be_meaningful(self, agent, measuring):
        """Guard the premise: a small session would pass vacuously."""
        _result, captured = _run(agent, tool_iterations=2)
        observations = _observe(captured)
        assert observations[0].prompt_tokens > 100_000
        assert observations[0].message_count > 80

    def test_checkpoint_and_prune_keep_the_prefix_reusable(self, agent, measuring):
        """The acceptance criterion: >= 95% of every request stays reusable
        while the checkpoint fires and the prune becomes eligible."""
        _result, captured = _run(
            agent, tool_iterations=3, checkpoint_on={1, 3}
        )
        assert len(captured) == 4

        observations = _observe(captured)
        worst, comparable = summarize_prefix_reuse(observations)
        assert comparable, "expected at least one comparable request"
        assert worst >= 0.95, [
            (o.api_call, o.first_changed_index, round(o.prefix_reuse_ratio, 4))
            for o in comparable
        ]

    def test_no_request_rewrites_the_settled_history(self, agent, measuring):
        """Nothing may diverge inside the pre-existing transcript: that is
        the region a provider has already cached, and the region both the
        checkpoint amend and the mid-turn prune used to rewrite."""
        seeded = len(_seeded_history())
        _result, captured = _run(
            agent, tool_iterations=3, checkpoint_on={1, 2, 3}
        )
        for obs in _observe(captured):
            if obs.baseline or obs.first_changed_index is None:
                continue
            assert obs.first_changed_index >= seeded, (
                f"call {obs.api_call} rewrote settled history at index "
                f"{obs.first_changed_index} (seeded={seeded})"
            )

    def test_checkpoint_text_reaches_the_provider(self, agent, measuring):
        """Moving the instruction to the tail must not drop it."""
        _result, captured = _run(agent, tool_iterations=2, checkpoint_on={1})
        marker = "[Checkpoint: save durable working state now."

        def _carries_marker(messages: list[dict]) -> bool:
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str) and marker in content:
                    return True
                if isinstance(content, list) and any(
                    marker in str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                ):
                    return True
            return False

        assert _carries_marker(captured[1])
        # Request-only: it must not survive into the next request or the
        # durable transcript.
        assert not _carries_marker(captured[2])

    def test_checkpoint_lands_in_the_tail_message(self, agent, measuring):
        _result, captured = _run(agent, tool_iterations=2, checkpoint_on={1})
        tail = captured[1][-1]
        assert tail.get("role") == "user"
        assert "[Checkpoint:" in str(tail.get("content"))

    def test_legacy_in_place_amend_would_fail_this_bar(self, agent, measuring):
        """Mutation check: with the pre-change injection restored, the same
        session drops well below the 95% bar. Without this, the test above
        could pass for reasons unrelated to the fix."""
        from agent.conversation_loop import _append_answer_only_recovery_prompt

        def _legacy(api_messages, prompt, **_kwargs):
            _append_answer_only_recovery_prompt(api_messages, prompt)
            return False

        with patch(
            "agent.conversation_loop._append_request_only_tail_block",
            side_effect=_legacy,
        ):
            _result, captured = _run(
                agent, tool_iterations=3, checkpoint_on={1, 3}
            )
        worst, comparable = summarize_prefix_reuse(_observe(captured))
        assert comparable
        assert worst < 0.95


class TestDeferredPrune:
    def test_prune_is_not_committed_mid_turn(self, agent, measuring):
        """The post-tool gate records eligibility; it does not rewrite."""
        prune_calls: list[int] = []
        real_prune = agent.context_compressor.prune_tool_results_only

        def _spy(messages, current_tokens=None):
            prune_calls.append(len(messages))
            return real_prune(messages, current_tokens=current_tokens)

        agent.context_compressor.prune_tool_results_only = _spy
        _result, captured = _run(agent, tool_iterations=3)

        # Exactly one prune attempt for the whole turn, at the boundary.
        assert len(prune_calls) == 1
        # And it ran after the final API call: no request saw a pruned body.
        for messages in captured:
            assert not any(
                isinstance(m.get("content"), str)
                and m["content"].startswith("[Tool result summarized")
                for m in messages
            )

    def test_prune_is_committed_at_the_turn_boundary(self, agent, measuring):
        result, _captured = _run(agent, tool_iterations=3)
        assert result["completed"] is True
        rows = result["messages"]
        settled_tool_rows = [
            m
            for m in rows[: len(_seeded_history()) - 20]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        assert settled_tool_rows, "expected settled tool rows to inspect"
        shrunk = [
            m
            for m in settled_tool_rows
            if len(str(m.get("content", ""))) < _TOOL_RESULT_CHARS
        ]
        assert shrunk, "the deferred prune never committed at the boundary"

    def test_boundary_commit_is_logged_with_a_position(self, agent, measuring):
        _result, _captured = _run(agent, tool_iterations=3)
        lines = [
            r.getMessage()
            for r in measuring.records
            if r.name == "agent.history_mutation"
        ]
        deferrals = [
            ln
            for ln in lines
            if "component=proactive_prune" in ln
            and "reason=deferred_to_turn_boundary" in ln
        ]
        commits = [
            ln
            for ln in lines
            if "component=proactive_prune" in ln
            and "reason=turn_boundary_commit" in ln
        ]
        assert len(deferrals) == 1, lines
        assert len(commits) == 1, lines
        assert "first_changed_index=" in commits[0]
        assert "invalidated_tokens=" in commits[0]

    def test_raising_prune_does_not_fail_the_turn(self, agent, measuring):
        def _boom(messages, current_tokens=None):
            raise RuntimeError("boom")

        agent.context_compressor.prune_tool_results_only = _boom
        result, _captured = _run(agent, tool_iterations=2)
        assert result["completed"] is True


class TestBoundaryFlushHelper:
    """Direct coverage of the boundary flush's caller contract."""

    def test_no_deferral_is_a_no_op(self):
        from agent.conversation_loop import _flush_deferred_tool_result_prune

        messages = [{"role": "user", "content": "hi"}]
        agent = SimpleNamespace(session_id="s", context_compressor=object())
        assert (
            _flush_deferred_tool_result_prune(agent, messages, None) is messages
        )

    def test_engine_without_the_method_is_a_no_op(self):
        """Plugin context engines predating the hook lack the method."""
        from agent.conversation_loop import _flush_deferred_tool_result_prune

        messages = [{"role": "user", "content": "hi"}]
        agent = SimpleNamespace(
            session_id="s", context_compressor=SimpleNamespace()
        )
        assert (
            _flush_deferred_tool_result_prune(agent, messages, 120_000)
            is messages
        )

    def test_input_object_back_commits_nothing(self):
        """The engine may report a count while returning the input list —
        the identity gate must refuse that commit."""
        from agent.conversation_loop import _flush_deferred_tool_result_prune

        messages = [{"role": "user", "content": "hi"}]
        agent = SimpleNamespace(
            session_id="s",
            context_compressor=SimpleNamespace(
                last_prompt_tokens=0,
                prune_tool_results_only=lambda msgs, current_tokens=None: (
                    msgs,
                    5,
                ),
            ),
        )
        assert (
            _flush_deferred_tool_result_prune(agent, messages, 120_000)
            is messages
        )

    def test_fresher_reading_overrides_the_deferred_one(self):
        """A compression later in the turn is itself a boundary and has
        already pruned; its smaller reading must stand this one down."""
        from agent.conversation_loop import _flush_deferred_tool_result_prune

        seen: list[int] = []

        def _prune(msgs, current_tokens=None):
            seen.append(current_tokens)
            return msgs, 0

        agent = SimpleNamespace(
            session_id="s",
            context_compressor=SimpleNamespace(
                last_prompt_tokens=30_000, prune_tool_results_only=_prune
            ),
        )
        _flush_deferred_tool_result_prune(
            agent, [{"role": "user", "content": "hi"}], 120_000
        )
        assert seen == [30_000]


class TestPruneArmingGate:
    """A disabled or below-trigger prune must not be recorded as deferred."""

    def test_disabled_prune_is_not_deferred(self):
        from agent.conversation_loop import _proactive_prune_armed

        assert (
            _proactive_prune_armed(
                SimpleNamespace(proactive_prune_tokens=0), 500_000
            )
            is False
        )

    def test_below_trigger_is_not_armed(self):
        from agent.conversation_loop import _proactive_prune_armed

        assert (
            _proactive_prune_armed(
                SimpleNamespace(proactive_prune_tokens=_PRUNE_TOKENS), 100_000
            )
            is False
        )

    def test_at_or_above_trigger_is_armed(self):
        from agent.conversation_loop import _proactive_prune_armed

        assert _proactive_prune_armed(
            SimpleNamespace(proactive_prune_tokens=_PRUNE_TOKENS), _PRUNE_TOKENS
        )

    def test_engine_hiding_the_trigger_is_treated_as_armed(self):
        """Duck-typed engines are consulted at the boundary exactly as the
        pre-change loop consulted them every iteration."""
        from agent.conversation_loop import _proactive_prune_armed

        assert _proactive_prune_armed(SimpleNamespace(), 10)

    def test_disabled_prune_produces_no_deferral_line(self, agent, measuring):
        agent.context_compressor.proactive_prune_tokens = 0
        _result, _captured = _run(agent, tool_iterations=2)
        assert not [
            r.getMessage()
            for r in measuring.records
            if r.name == "agent.history_mutation"
            and "reason=deferred_to_turn_boundary" in r.getMessage()
        ]
