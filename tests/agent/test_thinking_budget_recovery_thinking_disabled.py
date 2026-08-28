"""Reasoning-length-stop recovery must disable thinking, not merely lower it.

The answer-only recovery fires when a turn burned its whole output budget on
reasoning and returned no answer. Its one job is to write the conclusion the
model already reached, so it needs no reasoning at all. Clamping effort to
"low" left a thinking budget in place: on a local Qwen route (custom provider
behind an OpenAI-compatible proxy) the 4,096-token recovery spent everything on
reasoning three times in one session, each time producing the empty-answer
"Thinking Budget Exhausted" failure the recovery exists to prevent.
"""

from types import SimpleNamespace

from agent.conversation_loop import (
    _THINKING_BUDGET_RECOVERY_MAX_TOKENS,
    _apply_thinking_budget_recovery_overrides,
)


def _agent(provider: str, api_mode: str = "chat_completions") -> SimpleNamespace:
    return SimpleNamespace(provider=provider, api_mode=api_mode)


def test_custom_openai_compatible_route_disables_thinking():
    """The verified failing route gets an explicit chat-template thinking off."""
    api_kwargs = {
        "model": "qwen3-flash",
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "web_search"}}],
        "tool_choice": "auto",
        "max_tokens": 65536,
    }

    controls = _apply_thinking_budget_recovery_overrides(_agent("custom"), api_kwargs)

    assert api_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs.enable_thinking=false" in controls
    # The answer-only boundary and the 4,096 output ceiling are unchanged.
    assert "tools" not in api_kwargs
    assert "tool_choice" not in api_kwargs
    assert api_kwargs["max_tokens"] == _THINKING_BUDGET_RECOVERY_MAX_TOKENS


def test_custom_route_keeps_unrelated_chat_template_kwargs():
    """Only the thinking switch is rewritten; other template controls survive."""
    api_kwargs = {
        "max_tokens": 8192,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True, "tool_style": "qwen"},
        },
    }

    _apply_thinking_budget_recovery_overrides(_agent("custom"), api_kwargs)

    assert api_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "tool_style": "qwen",
    }


def test_managed_route_keeps_effort_clamp_and_sends_no_template_switch():
    """Vendors that 400 on unknown body fields keep the graceful-degradation clamp."""
    api_kwargs = {
        "max_tokens": 32000,
        "reasoning_effort": "high",
        "extra_body": {"reasoning": {"enabled": True, "effort": "high"}},
    }

    controls = _apply_thinking_budget_recovery_overrides(
        _agent("openrouter"), api_kwargs
    )

    assert "chat_template_kwargs" not in api_kwargs["extra_body"]
    assert api_kwargs["reasoning_effort"] == "low"
    assert api_kwargs["extra_body"]["reasoning"]["effort"] == "low"
    assert controls == "reasoning_effort=low+reasoning.effort=low"


def test_thinking_type_switch_is_flipped_and_paired_effort_dropped():
    """Routes with a native thinking switch (Kimi shape) turn it off outright.

    The chat-completions transport emits ``extra_body.thinking.type`` together
    with a top-level ``reasoning_effort``, and drops the effort when thinking is
    disabled. The recovery override produces the same pairing.
    """
    api_kwargs = {
        "max_tokens": 32000,
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    controls = _apply_thinking_budget_recovery_overrides(_agent("kimi"), api_kwargs)

    assert api_kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in api_kwargs
    assert controls == "thinking.type=disabled"


def test_anthropic_messages_route_gets_no_chat_template_switch():
    """The switch is a chat-completions body field — never sent on other wires."""
    api_kwargs = {"max_tokens": 32000}

    _apply_thinking_budget_recovery_overrides(
        _agent("custom", api_mode="anthropic_messages"), api_kwargs
    )

    assert "extra_body" not in api_kwargs
