"""``filter_api_tools`` hook: hide tool schemas from the wire request only."""

from types import SimpleNamespace

import agent.conversation_loop as conversation_loop
import hermes_cli.lifecycle as lifecycle


def _tools():
    return [
        {"type": "function", "function": {"name": "terminal"}},
        {"type": "function", "function": {"name": "delegate_task"}},
        {"name": "flat_tool"},
    ]


def test_hook_hides_named_tools_and_leaves_input_untouched(monkeypatch):
    seen = {}

    def fake_invoke(hook_name, **kwargs):
        seen["hook"] = hook_name
        seen.update(kwargs)
        return [["terminal", "not-a-tool"], None, "junk", ("flat_tool",)]

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "filter_api_tools")
    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke)
    tools = _tools()
    agent = SimpleNamespace(session_id="s1", platform="discord", model="main-model")

    kept = conversation_loop._filter_api_tools_for_request(agent, tools, task_id="t1")

    assert [conversation_loop._api_tool_name(t) for t in kept] == ["delegate_task"]
    assert seen["hook"] == "filter_api_tools"
    assert seen["tool_names"] == ["terminal", "delegate_task", "flat_tool"]
    assert (seen["session_id"], seen["task_id"], seen["platform"], seen["model"]) == (
        "s1", "t1", "discord", "main-model",
    )
    assert len(tools) == 3  # the agent's own tool list is not mutated


def test_no_hook_or_empty_result_keeps_every_tool(monkeypatch):
    tools = _tools()
    agent = SimpleNamespace(session_id="s1", platform="cli", model="m")

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    assert conversation_loop._filter_api_tools_for_request(agent, tools) is tools

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda hook_name, **kwargs: [None, []])
    assert conversation_loop._filter_api_tools_for_request(agent, tools) is tools
    assert conversation_loop._filter_api_tools_for_request(agent, []) == []


def test_hook_error_fails_open(monkeypatch):
    tools = _tools()
    agent = SimpleNamespace(session_id="s1", platform="cli", model="m")
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)

    def boom(hook_name, **kwargs):
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr(lifecycle, "invoke_hook", boom)
    assert conversation_loop._filter_api_tools_for_request(agent, tools) is tools
