"""Generic plugin lifecycle helpers for compaction-independent checkpoints.

Hermes owns context pressure and compression boundaries, while plugins own
the policy and persistence behind a checkpoint.  This module is the narrow
adapter between those layers: it exposes no checkpoint-specific storage and
never mutates the durable transcript.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable


logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 12_000


def _collect_context(results: Iterable[Any]) -> str:
    """Collect bounded API-only context from lifecycle hook results."""

    parts: list[str] = []
    remaining = _MAX_CONTEXT_CHARS
    for result in results:
        if isinstance(result, dict):
            value = result.get("context")
        elif isinstance(result, str):
            value = result
        else:
            continue
        if not isinstance(value, str) or not value.strip() or remaining <= 0:
            continue
        piece = value.strip()[:remaining]
        parts.append(piece)
        remaining -= len(piece)
    return "\n\n".join(parts)


def context_pressure_context(**kwargs: Any) -> str:
    """Return plugin context for one request under observed context pressure.

    The hook fires for every fully assembled request so a plugin can both
    detect threshold crossings and deliver one-shot post-compression recovery
    context.  Returned text is request-only and must never be persisted as a
    synthetic user turn.
    """

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook("on_context_pressure"):
            return ""
        return _collect_context(invoke_hook("on_context_pressure", **kwargs))
    except Exception:
        logger.warning("on_context_pressure hook failed", exc_info=True)
        return ""


def notify_pre_context_compression(**kwargs: Any) -> None:
    """Notify plugins before any transcript content is compacted away."""

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if has_hook("pre_context_compression"):
            invoke_hook("pre_context_compression", **kwargs)
    except Exception:
        logger.warning("pre_context_compression hook failed", exc_info=True)


def notify_post_context_compression(**kwargs: Any) -> None:
    """Notify plugins after a compression boundary commits successfully."""

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if has_hook("post_context_compression"):
            invoke_hook("post_context_compression", **kwargs)
    except Exception:
        logger.warning("post_context_compression hook failed", exc_info=True)


__all__ = [
    "context_pressure_context",
    "notify_post_context_compression",
    "notify_pre_context_compression",
]
