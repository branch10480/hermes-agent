from __future__ import annotations

from agent.final_response_sanitization import (
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


def test_removes_confirmed_english_planning_prefix():
    leaked = (
        "Now I have enough. Let me compile the answer.\n\n"
        "Key facts:\n- The context is large.\n- The user is Japanese.\n\n"
        "Let me write the final answer in Japanese.\n\n"
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_removes_denied_tool_narration_before_japanese_answer():
    leaked = (
        "The terminal python3 date conversion and execute_code both got denied "
        "by smart approval. Interesting — these are read-only operations, but "
        "they were denied. I shouldn't retry them. I already have enough evidence "
        "from the cron list and output file to answer.\n\n"
        "Let me write final answer in Japanese, natural phrasing, concise but informative."
        + JAPANESE_ANSWER
    )

    assert _sanitize(leaked) == JAPANESE_ANSWER


def test_preserves_japanese_first_sentence_when_switch_has_no_newline():
    answer = (
        "2時に予定されていたジョブは実際には動いていました。"
        "見えなかったのは、配信が意図的に抑制されたためです。"
        "確認した記録では実行結果も正常に保存されていました。"
    )
    leaked = (
        "The terminal command was blocked by smart approval.\n\n"
        "Let me write final answer in Japanese, natural phrasing, concise but informative."
        + answer
    )

    assert _sanitize(leaked) == answer


def test_preserves_english_technical_terms_urls_and_code_in_japanese_answer():
    answer = (
        "## 結論\n\nDeepSeek V4 Flash と oMLX の設定は正常です。"
        "API は https://example.com/v1 を使い、`max_tokens = 4096` のように指定します。"
        "英語の製品名やコードは翻訳せず、そのまま残すのが適切です。"
    )

    assert _sanitize(answer) == answer


def test_fails_open_without_explicit_switch_marker():
    ambiguous = "Now I have enough. " + JAPANESE_ANSWER

    assert _sanitize(ambiguous) == ambiguous


def test_preserves_denied_tool_report_without_japanese_switch_marker():
    legitimate_report = (
        "The terminal command was denied by the approval policy.\n\n"
        + JAPANESE_ANSWER
    )

    assert _sanitize(legitimate_report) == legitimate_report


def test_fails_open_for_non_discord_or_non_deepseek_or_english_request():
    leaked = (
        "Now I have enough. Let me compile the answer.\n"
        "Let me write the final answer in Japanese.\n" + JAPANESE_ANSWER
    )

    assert _sanitize(leaked, platform="cli") == leaked
    assert _sanitize(leaked, model="gpt-5.6") == leaked
    assert _sanitize(leaked, user_message="Please answer in English") == leaked


def test_does_not_add_an_apology_after_cleanup():
    leaked = (
        "Key facts:\n- ready\n"
        "I will write the final response in Japanese.\n" + JAPANESE_ANSWER
    )

    cleaned = _sanitize(leaked)
    assert cleaned == JAPANESE_ANSWER
    assert "申し訳" not in cleaned
