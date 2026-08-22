"""Conservative cleanup for DeepSeek Discord final answers.

DeepSeek V4 Flash can occasionally place English scratch planning in ordinary
``content`` after a tool trajectory. The preferred contract is an explicit
pair of final-answer markers. A structural fallback handles unmarked leaks,
but only when several independent signals agree; ambiguity always preserves
the original response.
"""

from __future__ import annotations

import re
from typing import Any


FINAL_ANSWER_START = "<<<HERMES_FINAL_ANSWER_V1>>>"
FINAL_ANSWER_END = "<<<END_HERMES_FINAL_ANSWER_V1>>>"

_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DEEPSEEK_V4_FLASH_MODEL_RE = re.compile(r"deepseek[-_/ ]v4[-_/ ]flash", re.I)
_START_MARKER_RE = re.compile(
    rf"(?m)^[ \t]*{re.escape(FINAL_ANSWER_START)}[ \t]*\r?$"
)
_END_MARKER_RE = re.compile(
    rf"(?m)^[ \t]*{re.escape(FINAL_ANSWER_END)}[ \t]*\r?$"
)
_FENCE_RE = re.compile(r"(?m)^[ \t]*(```|~~~)")
_ASYNC_DELEGATION_COMPLETE_RE = re.compile(
    r"^\s*\[ASYNC DELEGATION(?: BATCH)? COMPLETE\s+[—-]", re.I
)

_EXPLICIT_NON_JAPANESE_REQUEST_RE = re.compile(
    r"(?:英語で(?:回答|返答|答え|書い|説明|出力|まとめ)|"
    r"(?:日本語|英語)[^。！？\n]{0,20}(?:両方|併記)|"
    r"日英(?:両方|併記)|"
    r"\b(?:answer|respond|reply|write|explain)\b[^.!?\n]{0,32}"
    r"\bin\s+english\b|"
    r"\b(?:bilingual|both\s+japanese\s+and\s+english|"
    r"both\s+english\s+and\s+japanese)\b)",
    re.I,
)

# The switch is intentionally semantic rather than a list of observed opening
# phrases. Prefix scoring below must independently prove that the preceding
# prose looks like internal deliberation.
_FINAL_JAPANESE_SWITCH_RE = re.compile(
    r"(?im)^[ \t]*"
    r"(?=[^\n]{0,360}\bjapanese\b)"
    r"(?=[^\n]{0,360}\b(?:final\s+(?:answer|response)|"
    r"(?:write|give|provide|compose|compile|craft|prepare)\s+"
    r"(?:the\s+)?(?:final\s+)?(?:answer|response))\b)"
    r"[^\n]{0,360}?\bjapanese\b"
)

_META_FEATURES = (
    re.compile(
        r"\b(?:let\s+me|i(?:'ll|'ve|'m|\s+(?:will|should|need|have|can|must))|"
        r"we\s+(?:need|should|can)|actually|interesting)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:terminal|execute_code|read_file|web_search|browser\w*|tool(?:\s+call)?|"
        r"cron(?:\s+job)?|delegat\w*|auth(?:entication)?\s+status|api\s+call|"
        r"output\s+file)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:evidence|key\s+facts?|confirm(?:s|ed)?|show(?:s|ed)?|result(?:s)?|"
        r"enough|denied|blocked|failed|available|verified)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:final\s+(?:answer|response)|answer\s+plan|tell\s+the\s+user|"
        r"answer\s+from|respond\s+with|shouldn(?:'|’)t\s+retry|do\s+not\s+retry|"
        r"concise|natural\s+phrasing|compile\s+the\s+answer)\b",
        re.I,
    ),
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


def _effective_user_text(
    user_message: Any,
    conversation_messages: list[dict[str, Any]] | None,
) -> str:
    """Resolve the human request behind an async-completion synthetic turn.

    Async delegation completion is injected as a model-facing user message in
    English.  Treating that transport envelope as the user's language disables
    Japanese-only cleanup on the follow-up response.  The protocol header is a
    stable Hermes-owned boundary, so only that exact synthetic turn may inherit
    the most recent genuine Japanese user request.
    """

    current = _text_from_user_message(user_message)
    if _JAPANESE_RE.search(current) or not _ASYNC_DELEGATION_COMPLETE_RE.match(current):
        return current
    for message in reversed(conversation_messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        candidate = _text_from_user_message(message.get("content"))
        if not candidate or candidate == current:
            continue
        if _ASYNC_DELEGATION_COMPLETE_RE.match(candidate):
            continue
        if _JAPANESE_RE.search(candidate):
            return candidate
    return current


def _inside_fenced_code(content: str, position: int) -> bool:
    active: str | None = None
    for match in _FENCE_RE.finditer(content, 0, position):
        token = match.group(1)
        if active is None:
            active = token
        elif active == token:
            active = None
    return active is not None


def _extract_explicit_final_answer(content: str) -> str | None:
    starts = list(_START_MARKER_RE.finditer(content))
    ends = list(_END_MARKER_RE.finditer(content))
    if len(starts) != 1 or len(ends) != 1:
        return None
    start, end = starts[0], ends[0]
    if start.end() > end.start():
        return None
    if _inside_fenced_code(content, start.start()) or _inside_fenced_code(
        content, end.start()
    ):
        return None
    candidate = content[start.end() : end.start()].strip()
    if not candidate or not _JAPANESE_RE.search(candidate):
        return None
    return candidate


def _looks_like_internal_english_prefix(prefix: str) -> bool:
    stripped = prefix.lstrip()
    if not stripped or stripped.startswith(("```", "~~~", ">")):
        return False
    latin_count = len(_LATIN_RE.findall(prefix))
    japanese_count = len(_JAPANESE_RE.findall(prefix))
    if latin_count < 60 or latin_count < 3 * max(1, japanese_count):
        return False
    feature_count = sum(bool(pattern.search(prefix)) for pattern in _META_FEATURES)
    # The explicit final-Japanese transition and substantial Japanese suffix
    # are scored separately by the caller. A long English operational preface
    # therefore needs one meta category; a shorter one needs two. This covers
    # third-person tool/research narration without depending on a first-person
    # phrase list, while block quotes and code fences remain fail-open above.
    return feature_count >= (1 if latin_count >= 180 else 2)


def _candidate_after_switch(content: str, switch: re.Match[str]) -> str | None:
    tail = content[switch.end() :]
    first_japanese = _JAPANESE_RE.search(tail)
    if first_japanese is None:
        return None

    # A model can concatenate the switch sentence and answer. Cut at the last
    # sentence separator before the first Japanese character while preserving
    # a following Markdown prefix such as ``## ``.
    before_japanese = tail[: first_japanese.start()]
    boundary = max(
        before_japanese.rfind("\n"),
        before_japanese.rfind("."),
        before_japanese.rfind("!"),
        before_japanese.rfind("?"),
        before_japanese.rfind(":"),
    )
    candidate = tail[boundary + 1 :].lstrip()
    if len(_JAPANESE_RE.findall(candidate)) < 40:
        return None
    return candidate


def _extract_structural_fallback(content: str) -> str | None:
    for switch in reversed(list(_FINAL_JAPANESE_SWITCH_RE.finditer(content))):
        prefix = content[: switch.start()]
        if not _looks_like_internal_english_prefix(prefix):
            continue
        candidate = _candidate_after_switch(content, switch)
        if candidate is not None:
            return candidate
    return None


def _paragraph_spans(content: str) -> list[tuple[int, int, str]]:
    """Return non-empty blank-line-delimited blocks with source offsets."""

    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)", content, re.S):
        spans.append((match.start(), match.end(), match.group(0)))
    return spans


def _looks_like_internal_meta_block(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped or stripped.startswith(("```", "~~~", ">")):
        return False
    latin_count = len(_LATIN_RE.findall(text))
    japanese_count = len(_JAPANESE_RE.findall(text))
    if latin_count < 60 or latin_count < 2 * max(1, japanese_count):
        return False
    matched = [bool(pattern.search(text)) for pattern in _META_FEATURES]
    # Require both an explicit planning/decision signal and an independent
    # operational/evidence signal.  A legitimate sentence such as “The
    # terminal command was denied” therefore remains visible.
    return sum(matched) >= 2 and (matched[0] or matched[3]) and (matched[1] or matched[2])


def _looks_like_japanese_answer_suffix(text: str) -> bool:
    japanese_count = len(_JAPANESE_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    return japanese_count >= 80 and japanese_count * 2 >= latin_count


def _is_answer_leading_markdown_block(text: str) -> bool:
    stripped = text.strip()
    return bool(
        len(stripped) <= 160
        and (
            re.fullmatch(r"#{1,6}[ \t]+\S[^\n]*", stripped)
            or re.fullmatch(r"\*\*\S[^\n]*\*\*", stripped)
        )
    )


def _extract_unmarked_paragraph_fallback(content: str) -> str | None:
    """Extract a Japanese suffix after the last high-confidence meta block.

    DeepSeek sometimes omits both the explicit boundary and the traditional
    “final answer in Japanese” switch.  It can even draft a Japanese answer,
    return to English self-evaluation, then write the real Japanese answer.
    Searching candidate boundaries from the end keeps the final draft while
    requiring two independent meta classes immediately before it.
    """

    spans = _paragraph_spans(content)
    if len(spans) < 2:
        return None
    for index in range(len(spans) - 1, 0, -1):
        candidate_start = index
        while (
            candidate_start > 0
            and _is_answer_leading_markdown_block(spans[candidate_start - 1][2])
        ):
            candidate_start -= 1
        candidate = content[spans[candidate_start][0] :].strip()
        if not _looks_like_japanese_answer_suffix(candidate):
            continue
        # One model thought may be split across adjacent paragraphs (e.g.
        # evidence assessment followed by an answer plan), so score a bounded
        # block ending immediately before the candidate.
        for width in range(1, min(3, candidate_start) + 1):
            block = "\n\n".join(
                item[2] for item in spans[candidate_start - width : candidate_start]
            )
            if _looks_like_internal_meta_block(block):
                return candidate
    return None


def should_suppress_deepseek_discord_interim_content(
    content: str,
    *,
    model: str,
    platform: str | None,
    user_message: Any,
    conversation_messages: list[dict[str, Any]] | None = None,
) -> bool:
    """Return True only for high-confidence English internal tool narration.

    The caller suppresses UI projection only; the assistant/tool-call row stays
    in history so provider tool-call pairing and subsequent reasoning remain
    intact.
    """

    if not isinstance(content, str) or not content:
        return False
    if (platform or "").lower() != "discord":
        return False
    if not _DEEPSEEK_V4_FLASH_MODEL_RE.search(model or ""):
        return False
    user_text = _effective_user_text(user_message, conversation_messages)
    if not _JAPANESE_RE.search(user_text):
        return False
    if _EXPLICIT_NON_JAPANESE_REQUEST_RE.search(user_text):
        return False
    return _looks_like_internal_meta_block(content)


def sanitize_deepseek_discord_final_response(
    content: str,
    *,
    model: str,
    platform: str | None,
    user_message: Any,
    conversation_messages: list[dict[str, Any]] | None = None,
) -> str:
    """Return only a confirmed Japanese final answer, otherwise fail open."""

    if not isinstance(content, str) or not content:
        return content
    if (platform or "").lower() != "discord":
        return content
    if not _DEEPSEEK_V4_FLASH_MODEL_RE.search(model or ""):
        return content
    user_text = _effective_user_text(user_message, conversation_messages)
    if not _JAPANESE_RE.search(user_text):
        return content
    if _EXPLICIT_NON_JAPANESE_REQUEST_RE.search(user_text):
        return content

    explicit = _extract_explicit_final_answer(content)
    if explicit is not None:
        return explicit
    structural = _extract_structural_fallback(content)
    if structural is not None:
        return structural
    paragraph_fallback = _extract_unmarked_paragraph_fallback(content)
    return paragraph_fallback if paragraph_fallback is not None else content


__all__ = [
    "FINAL_ANSWER_END",
    "FINAL_ANSWER_START",
    "sanitize_deepseek_discord_final_response",
    "should_suppress_deepseek_discord_interim_content",
]
