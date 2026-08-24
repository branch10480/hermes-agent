from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.final_response_sanitization import FINAL_ANSWER_END, FINAL_ANSWER_START


JAPANESE_ANSWER = (
    "## 結論\n\n調査結果を日本語でまとめます。内部の作業計画は回答本文へ含めず、"
    "コードやURL、製品名などの英語表記だけを必要に応じて残します。"
    "Discordで長文が分割されても、一つ目の投稿はこの日本語の結論から始まります。"
    "保存される会話履歴にも同じ完成回答だけを残すため、次のターンへ漏れたメモを再送しません。"
)


def _run_response(
    raw_response: str,
    *,
    user_message: str = "この件を日本語で詳しく調査して",
    conversation_history: list[dict] | None = None,
    model: str = "deepseek-v4-flash-0731-2-4bit-mixed",
):
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:18088/v1",
            model=model,
            platform="discord",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=raw_response, finish_reason="stop")
    ]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    persisted: list[list[dict]] = []

    def record_flush(messages, conversation_history=None):
        persisted.append([dict(m) for m in messages if isinstance(m, dict)])
        return True

    with (
        patch.object(agent, "_flush_messages_to_session_db", side_effect=record_flush),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            user_message,
            conversation_history=conversation_history,
        )
    return result, persisted


def _assert_sanitized(result, persisted):
    assert result["final_response"] == JAPANESE_ANSWER
    assert result["messages"][-1]["content"] == JAPANESE_ANSWER
    assert any(rows[-1]["content"] == JAPANESE_ANSWER for rows in persisted)
    assert all(FINAL_ANSWER_START not in rows[-1]["content"] for rows in persisted)
    assert all(FINAL_ANSWER_END not in rows[-1]["content"] for rows in persisted)


def test_explicit_boundary_answer_is_returned_and_persisted_without_markers():
    raw = (
        "Unpublished internal planning.\n"
        f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}\n"
        "Unpublished tail."
    )

    _assert_sanitized(*_run_response(raw))


def test_explicit_boundary_is_model_independent_before_persistence():
    raw = f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}"

    for model in ("qwen38-mtplx-optimized-speed", "future-local-parent-v9"):
        _assert_sanitized(*_run_response(raw, model=model))


def test_structural_fallback_answer_is_returned_and_persisted():
    leaked = (
        "I have enough verified evidence from the tool results. "
        "Let me compile the answer and avoid retrying the denied operation.\n\n"
        "Let me write the final answer in Japanese.\n\n"
        + JAPANESE_ANSWER
    )

    result, persisted = _run_response(leaked)
    _assert_sanitized(result, persisted)
    assert all("I have enough" not in rows[-1]["content"] for rows in persisted)


def test_async_completion_uses_the_prior_human_request_and_persists_only_final():
    synthetic = (
        "[ASYNC DELEGATION BATCH COMPLETE — batch-123]\n"
        "A background subagent has finished.\n--- RESULT ---\nCompleted."
    )
    leaked = (
        "I have enough verified evidence from the delegated tool. I should now "
        "compile the final response and tell the user the completed result.\n\n"
        + JAPANESE_ANSWER
    )

    result, persisted = _run_response(
        leaked,
        user_message=synthetic,
        conversation_history=[
            {"role": "user", "content": "この件を日本語で詳しく調査して"},
            {"role": "assistant", "content": "調査を開始します。"},
        ],
    )

    _assert_sanitized(result, persisted)
    assert "I have enough" not in result["messages"][-1]["content"]


def test_internal_tool_narration_is_kept_in_history_but_not_projected():
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:18088/v1",
            model="deepseek-v4-flash-0731-2-4bit-mixed",
            platform="discord",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    narration = (
        "The delegated tool returned enough verified evidence. I should not retry "
        "the denied operation, and I will now inspect the remaining result."
    )
    tool_call = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
    agent.valid_tool_names = {"web_search"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content=narration,
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        ),
        _mock_response(content=JAPANESE_ANSWER, finish_reason="stop"),
    ]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    projected = MagicMock()
    agent.interim_assistant_callback = projected

    with (
        patch("run_agent.handle_function_call", return_value="verified result"),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("この件を日本語で詳しく調査して")

    assert any(
        message.get("role") == "assistant" and message.get("content") == narration
        for message in result["messages"]
    )
    assert all(narration not in str(call.args[0]) for call in projected.call_args_list)
