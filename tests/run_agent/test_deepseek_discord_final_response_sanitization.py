from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_sanitized_answer_is_returned_and_persisted(tmp_path):
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response

    japanese_answer = (
        "## 結論\n\n調査結果を日本語でまとめます。内部の作業計画は回答本文へ含めず、"
        "コードやURL、製品名などの英語表記だけを必要に応じて残します。"
        "Discordで長文が分割されても、一つ目の投稿はこの日本語の結論から始まります。"
        "保存される会話履歴にも同じ完成回答だけを残すため、次のターンへ漏れたメモを再送しません。"
    )
    leaked = (
        "Now I have enough. Let me compile the answer.\n\n"
        "Key facts:\n- enough evidence\n\n"
        "Let me write the final answer in Japanese.\n\n"
        + japanese_answer
    )

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
        _mock_response(content=leaked, finish_reason="stop")
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

    assert result["final_response"] == japanese_answer
    assert result["messages"][-1]["content"] == japanese_answer
    assert any(rows[-1]["content"] == japanese_answer for rows in persisted)
    assert all("Now I have enough" not in rows[-1]["content"] for rows in persisted)
