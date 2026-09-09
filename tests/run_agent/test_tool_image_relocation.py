"""Tests for tool-image relocation recovery.

Some OpenAI-compatible local servers (antirez/ds4 ``ds4-server``) accept
image parts only on ``role: "user"`` messages and reject an image-bearing
``role: "tool"`` message with a generic HTTP 400 ``invalid JSON request``.
Rather than dropping the pixels (``_try_strip_image_parts_from_tool_messages``),
``_try_relocate_tool_image_parts_to_user_message`` moves them into a user
message that follows the tool-result block, and records the (provider, model)
so later requests in the session relocate preemptively.
"""

from __future__ import annotations


def _make_agent(provider: str = "custom", model: str = "deepseek-v4-flash"):
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    return agent


def _img(n: int) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,img{n}"}}


def _tool_msg(call_id: str, parts: list, name: str = "vision_analyze") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": parts}


class TestRelocateHelper:
    def test_no_messages_returns_false(self):
        agent = _make_agent()
        assert agent._try_relocate_tool_image_parts_to_user_message([]) is False
        assert agent._try_relocate_tool_image_parts_to_user_message(None) is False
        assert not hasattr(agent, "_tool_image_relocation_models")

    def test_string_and_text_only_tool_messages_untouched(self):
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "a", "content": "plain"},
            _tool_msg("b", [{"type": "text", "text": "hello"}]),
        ]
        snapshot = [dict(m) for m in msgs]
        assert agent._try_relocate_tool_image_parts_to_user_message(msgs) is False
        assert msgs == snapshot

    def test_images_move_to_user_message_after_tool_block(self):
        agent = _make_agent()
        original_parts = [{"type": "text", "text": "Image loaded"}, _img(1)]
        msgs = [
            {"role": "user", "content": "look at these"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
            {"role": "tool", "tool_call_id": "a", "content": "terminal output"},
            _tool_msg("b", original_parts),
            _tool_msg("c", [_img(2)]),
            {"role": "assistant", "content": "answer"},
        ]
        assert agent._try_relocate_tool_image_parts_to_user_message(msgs) is True

        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "tool", "tool", "user", "assistant"]

        # Tool messages keep their ids and are text-only now.
        assert msgs[2]["content"] == "terminal output"
        assert msgs[3]["tool_call_id"] == "b"
        assert msgs[3]["content"] == "Image loaded"
        assert msgs[4]["content"] == "[image attached in the following user message]"
        for m in msgs[2:5]:
            assert not agent._content_has_image_parts(m["content"])

        # The relocated user message carries both images, in order, plus a note.
        relocated = msgs[5]["content"]
        assert relocated[0]["type"] == "text"
        assert "vision_analyze" in relocated[0]["text"]
        assert relocated[1:] == [_img(1), _img(2)]

        # The canonical parts list handed in was not mutated.
        assert original_parts == [{"type": "text", "text": "Image loaded"}, _img(1)]

    def test_each_tool_block_gets_its_own_user_message(self):
        agent = _make_agent()
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            _tool_msg("a", [_img(1)]),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
            _tool_msg("b", [_img(2)]),
        ]
        assert agent._try_relocate_tool_image_parts_to_user_message(msgs) is True
        assert [m["role"] for m in msgs] == ["assistant", "tool", "user", "assistant", "tool", "user"]
        assert msgs[2]["content"][1:] == [_img(1)]
        assert msgs[5]["content"][1:] == [_img(2)]

    def test_helper_does_not_learn_by_itself(self):
        """Learning is the caller's job after the relocated retry succeeds."""
        agent = _make_agent()
        msgs = [_tool_msg("a", [_img(1)])]
        assert agent._try_relocate_tool_image_parts_to_user_message(msgs) is True
        assert not hasattr(agent, "_tool_image_relocation_models")
        assert agent._tool_images_relocate_preemptively() is False

    def test_remember_marks_provider_model_for_preemption(self):
        agent = _make_agent()
        agent._remember_tool_image_relocation()
        assert ("custom", "deepseek-v4-flash") in agent._tool_image_relocation_models
        assert agent._tool_images_relocate_preemptively() is True
        other = _make_agent(model="other-model")
        assert other._tool_images_relocate_preemptively() is False

    def test_remember_without_model_is_noop(self):
        agent = _make_agent(model="")
        agent._remember_tool_image_relocation()
        assert not hasattr(agent, "_tool_image_relocation_models")

    def test_images_prepend_to_following_real_user_turn(self):
        """A user turn right after the tool block (user interjected after the
        previous turn completed) must absorb the images instead of getting a
        second adjacent user message — strict-alternation servers reject
        user/user, and the reactive path runs after the merge pass."""
        agent = _make_agent()
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            _tool_msg("a", [{"type": "text", "text": "loaded"}, _img(1)]),
            {"role": "user", "content": "and now?", "name": "alice"},
            {"role": "assistant", "content": "ok"},
        ]
        original_user = msgs[2]
        assert agent._try_relocate_tool_image_parts_to_user_message(msgs) is True
        assert [m["role"] for m in msgs] == ["assistant", "tool", "user", "assistant"]
        assert msgs[1]["content"] == "loaded"
        merged = msgs[2]
        assert merged["name"] == "alice"
        assert merged["content"][0]["type"] == "text"
        assert merged["content"][1] == _img(1)
        assert merged["content"][-1] == {"type": "text", "text": "and now?"}
        # The caller's user dict is untouched.
        assert original_user["content"] == "and now?"

    def test_images_prepend_to_following_multimodal_user_turn(self):
        agent = _make_agent()
        msgs = [
            _tool_msg("a", [_img(1)]),
            {"role": "user", "content": [{"type": "text", "text": "hi"}, _img(9)]},
        ]
        assert agent._try_relocate_tool_image_parts_to_user_message(msgs) is True
        assert [m["role"] for m in msgs] == ["tool", "user"]
        assert msgs[1]["content"][1] == _img(1)
        assert msgs[1]["content"][2:] == [{"type": "text", "text": "hi"}, _img(9)]

    def test_no_consecutive_user_messages_ever_produced(self):
        agent = _make_agent()
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            _tool_msg("a", [_img(1)]),
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
            _tool_msg("b", [_img(2)]),
        ]
        agent._try_relocate_tool_image_parts_to_user_message(msgs)
        roles = [m["role"] for m in msgs]
        assert roles == ["assistant", "tool", "user", "assistant", "tool", "user"]
        assert not any(a == b == "user" for a, b in zip(roles, roles[1:]))


class TestRetryStateFlag:
    def test_flags_default_false(self):
        from agent.turn_retry_state import TurnRetryState
        state = TurnRetryState()
        assert state.tool_image_relocation_retry_attempted is False
        assert state.tool_image_relocation_pending is False
