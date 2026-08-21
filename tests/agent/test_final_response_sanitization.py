from __future__ import annotations

import pytest

from agent.final_response_sanitization import (
    FINAL_ANSWER_END,
    FINAL_ANSWER_START,
    sanitize_deepseek_discord_final_response,
)


MODEL = "deepseek-v4-flash-0731-2-4bit-mixed"
JAPANESE_ANSWER = (
    "## 結論\n\n調査結果をまとめます。今回確認した設定では、回答本文の前に内部の作業計画が"
    "混ざっていました。修正後は完成した日本語の回答だけを投稿し、技術用語や固有名詞は"
    "元の表記を保ちます。これにより分割された最初の投稿も自然な日本語から始まります。"
)


def _sanitize(content: str, **overrides):
    kwargs = {
        "model": MODEL,
        "platform": "discord",
        "user_message": "この内容を日本語で詳しく調査して",
    }
    kwargs.update(overrides)
    return sanitize_deepseek_discord_final_response(content, **kwargs)


def test_extracts_one_explicit_final_answer_pair():
    marked = (
        "Internal notes stay outside.\n"
        f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}\n"
        "Trailing scratch stays outside."
    )

    assert _sanitize(marked) == JAPANESE_ANSWER


@pytest.mark.parametrize(
    "ambiguous",
    [
        f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}",
        f"{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}",
        (
            f"{FINAL_ANSWER_START}\n{FINAL_ANSWER_START}\n"
            f"{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}"
        ),
        (
            f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n"
            f"{FINAL_ANSWER_END}\n{FINAL_ANSWER_END}"
        ),
        f"{FINAL_ANSWER_END}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_START}",
        f"{FINAL_ANSWER_START}\n\n{FINAL_ANSWER_END}",
        (
            "```text\n"
            f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}\n"
            "```"
        ),
    ],
)
def test_invalid_or_ambiguous_marker_pairs_fail_open(ambiguous):
    assert _sanitize(ambiguous) == ambiguous


def test_structural_fallback_removes_planning_without_exact_opening_phrase():
    leaked = (
        "Now I have enough evidence from the available results. "
        "Let me compile the answer and tell the user only what was verified.\n\n"
        "Key facts:\n- The context is large.\n- The user is Japanese.\n\n"
        "Let me write the final answer in Japanese.\n\n"
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_structural_fallback_removes_denied_tool_narration():
    leaked = (
        "The terminal operation and execute_code were denied by smart approval. "
        "Interesting — I should not retry them because the output file already "
        "provides enough verified evidence to answer the user.\n\n"
        "Let me write final answer in Japanese, natural phrasing, concise but informative."
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_structural_fallback_handles_response_heading_and_auth_evaluation():
    leaked = (
        "response\n\nAuth status confirms the scheduled cron job was available. "
        "I have verified the tool results and should not repeat the delegation call. "
        "The evidence is enough to report the outcome.\n\n"
        "Final answer in Japanese, concise.\n\n"
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_structural_fallback_handles_long_third_person_tool_narration():
    leaked = (
        "Authentication status output lists the background research workflow, "
        "search operations, cron metadata, and tool routing details. The auth "
        "records cover each lookup and output channel in the operational trace. "
        "No user-visible conclusion appears in this long English work record.\n\n"
        "Final response in Japanese, concise.\n\n"
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_preserves_japanese_markdown_when_switch_has_no_newline():
    leaked = (
        "The terminal command was blocked. I have enough verified evidence from the "
        "tool result, and I should not retry. I can answer the user now.\n\n"
        "Let me provide the final answer in Japanese:"
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_preserves_english_technical_terms_urls_and_code_in_japanese_answer():
    answer = (
        "## 結論\n\nDeepSeek V4 Flash と oMLX の設定は正常です。"
        "API は https://example.com/v1 を使い、`max_tokens = 4096` のように指定します。"
        "英語の製品名やコードは翻訳せず、そのまま残すのが適切です。"
    )

    assert _sanitize(answer) == answer


@pytest.mark.parametrize(
    "legitimate",
    [
        "response\n\n" + JAPANESE_ANSWER,
        "Performance Analysis\n\nDeepSeek V4 Flash and oMLX\n\n" + JAPANESE_ANSWER,
        (
            "> I will write the final answer in Japanese after checking the tool result.\n\n"
            + JAPANESE_ANSWER
        ),
        (
            "```python\n# I will write the final answer in Japanese\nprint('tool result')\n```\n\n"
            + JAPANESE_ANSWER
        ),
        "The terminal command was denied by the approval policy.\n\n" + JAPANESE_ANSWER,
    ],
)
def test_structural_fallback_preserves_legitimate_content(legitimate):
    assert _sanitize(legitimate) == legitimate


def test_fails_open_for_non_discord_non_deepseek_or_english_request():
    marked = f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}"

    assert _sanitize(marked, platform="cli") == marked
    assert _sanitize(marked, model="gpt-5.6") == marked
    assert _sanitize(marked, user_message="Please answer in English") == marked
    assert _sanitize(marked, user_message="英語で回答して") == marked


def test_preserves_explicit_bilingual_request():
    marked = f"{FINAL_ANSWER_START}\n{JAPANESE_ANSWER}\n{FINAL_ANSWER_END}"

    assert _sanitize(marked, user_message="日本語と英語を両方併記して") == marked


def test_does_not_add_an_apology_after_cleanup():
    leaked = (
        "I have checked the available tool results and verified enough evidence. "
        "Let me compile the final response without retrying the denied operation.\n"
        "I will write the final response in Japanese.\n"
        + JAPANESE_ANSWER
    )

    cleaned = _sanitize(leaked)
    assert cleaned == JAPANESE_ANSWER
    assert "申し訳" not in cleaned
