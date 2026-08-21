"""Narrow, provider-specific cleanup for completed assistant answers.

This module intentionally operates only at the final-response boundary.  It
must never rewrite tool turns, quoted English, code, or ordinary multilingual
answers.
"""

from __future__ import annotations

import re
from typing import Any


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_DEEPSEEK_V4_FLASH_MODEL_RE = re.compile(r"deepseek[-_/ ]v4[-_/ ]flash", re.I)

# Observed DeepSeek V4 Flash leak signatures.  Requiring one at the beginning
# and a separate Japanese-answer switch marker keeps the sanitizer fail-open.
_META_PREFIX_RE = re.compile(
    r"\A\s*(?:"
    r"now\s+i\s+have\s+enough\b|"
    r"let\s+me\s+(?:compile|compose|craft|prepare|write)\s+(?:the\s+)?answer\b|"
    r"key\s+facts\s*:|"
    r"since\s+the\s+user\s+is\s+japanese\b|"
    # DeepSeek sometimes narrates a denied tool call as ordinary content
    # before deciding to answer from the evidence it already has.  Keep this
    # signature narrow: a known tool/terminal noun must begin the response and
    # the same first line must explicitly say the operation was denied/blocked.
    r"the\s+(?:terminal|execute_code|read_file|web_search|browser\w*|tool)\b"
    r"[^\n]{0,240}\b(?:got|was|were)\s+(?:denied|blocked)\b"
    r")",
    re.I,
)
_JAPANESE_ANSWER_SWITCH_RE = re.compile(
    r"(?im)^\s*(?:let\s+me|i\s+(?:will|should|need\s+to))"
    r"[^\n]{0,220}\b(?:final\s+answer|answer|response)\b"
    r"[^\n]{0,160}?\b(?:in\s+japanese|japanese)\b"
)


def _text_from_user_message(user_message: Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if not isinstance(user_message, list):
        return ""
    parts: list[str] = []
    for item in user_message:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def sanitize_deepseek_discord_final_response(
    content: str,
    *,
    model: str,
    platform: str | None,
    user_message: Any,
) -> str:
    """Remove a confirmed English planning preamble from a Japanese answer.

    The cleanup is deliberately conservative: it applies only to Discord,
    DeepSeek V4 Flash, a Japanese user turn, an observed meta-planning prefix,
    an explicit switch-to-Japanese marker, and a substantial Japanese answer
    after that marker.  Any ambiguity returns the original content unchanged.
    """

    if not isinstance(content, str) or not content:
        return content
    if (platform or "").lower() != "discord":
        return content
    if not _DEEPSEEK_V4_FLASH_MODEL_RE.search(model or ""):
        return content
    if not _JAPANESE_RE.search(_text_from_user_message(user_message)):
        return content
    if not _META_PREFIX_RE.search(content):
        return content

    switches = list(_JAPANESE_ANSWER_SWITCH_RE.finditer(content))
    if not switches:
        return content
    tail = content[switches[-1].end():]
    first_japanese = _JAPANESE_RE.search(tail)
    if first_japanese is None:
        return content

    # Some local-model responses concatenate the English switch sentence and
    # the Japanese answer without a newline.  Find the last sentence/newline
    # boundary before the first Japanese character so Markdown prefixes such
    # as ``## `` are preserved while the remaining English switch prose is
    # removed.
    before_japanese = tail[:first_japanese.start()]
    boundary = max(
        before_japanese.rfind("\n"),
        before_japanese.rfind("."),
        before_japanese.rfind("!"),
        before_japanese.rfind("?"),
    )
    candidate = tail[boundary + 1:].lstrip()

    # A short Japanese phrase after an English quotation can occur naturally;
    # only treat a sizeable completed answer as proof of the leak shape.
    if len(_JAPANESE_RE.findall(candidate)) < 40:
        return content
    return candidate


__all__ = ["sanitize_deepseek_discord_final_response"]
