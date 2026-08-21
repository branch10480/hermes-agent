from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.final_response_sanitization import FINAL_ANSWER_END, FINAL_ANSWER_START


JAPANESE_ANSWER = (
    "## 結論\n\n調査結果を日本語でまとめます。内部の作業計画は回答本文へ含めず、"
    "コードやURL、製品名などの英語表記だけを必要に応じて残します。"
    "Discordで長文が分割されても、一つ目の投稿はこの日本語の結論から始まります。"
    "保存される会話履歴にも同じ完成回答だけを残すため、次のターンへ漏れたメモを再送しません。"
)


def _run_response(raw_response: str):
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
            model="deepseek-v4-flash-0731-2-4bit-mixed",
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
        result = agent.run_conversation("この件を日本語で詳しく調査して")
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
