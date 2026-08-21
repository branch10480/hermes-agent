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


def sanitize_deepseek_discord_final_response(
    content: str,
    *,
    model: str,
    platform: str | None,
    user_message: Any,
) -> str:
    """Return only a confirmed Japanese final answer, otherwise fail open."""

    if not isinstance(content, str) or not content:
        return content
    if (platform or "").lower() != "discord":
        return content
    if not _DEEPSEEK_V4_FLASH_MODEL_RE.search(model or ""):
        return content
    user_text = _text_from_user_message(user_message)
    if not _JAPANESE_RE.search(user_text):
        return content
    if _EXPLICIT_NON_JAPANESE_REQUEST_RE.search(user_text):
        return content

    explicit = _extract_explicit_final_answer(content)
    if explicit is not None:
        return explicit
    structural = _extract_structural_fallback(content)
    return structural if structural is not None else content


__all__ = [
    "FINAL_ANSWER_END",
    "FINAL_ANSWER_START",
    "sanitize_deepseek_discord_final_response",
]
