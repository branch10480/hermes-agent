"""The queue, exercised through a real turn (H-032 / H-033, CR-008).

``tests/agent/test_backend_scheduler.py`` pins the queue's semantics against
its own API. This file checks the part that unit tests cannot: that
``agent/conversation_loop.py`` actually routes its backend call through the
queue, in the right span — claim before submitting, release the moment the
response comes back.

Two agents run a turn at the same time against one mock backend that reports
the highest concurrency it ever saw. Engaged, the backend never sees two at
once; disengaged, it does — which is what makes the first assertion mean
something rather than being satisfied by threads that never overlapped.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import backend_scheduler, live_turn_registry


class _CountingHandler(BaseHTTPRequestHandler):
    """Mock provider that records how many requests overlapped."""

    lock = threading.Lock()
    in_flight = 0
    peak = 0
    arrivals: list = []
    hold_seconds = 0.35
    # When set, requests block on this instead of sleeping, so a test can hold
    # the backend open and inspect the queue that forms behind it.
    gate: "threading.Event | None" = None

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode() or "{}")
        cls = type(self)
        with cls.lock:
            cls.in_flight += 1
            cls.peak = max(cls.peak, cls.in_flight)
            cls.arrivals.append(_first_user_text(payload))
        try:
            # Occupy the backend for long enough that a second submission
            # would overlap if nothing were holding it back.
            if cls.gate is not None:
                cls.gate.wait(timeout=30)
            else:
                time.sleep(cls.hold_seconds)
        finally:
            with cls.lock:
                cls.in_flight -= 1

        # The agent loop prefers streaming for its stale-stream health checks,
        # so a non-SSE reply reads as an empty stream and burns the retry
        # budget instead of exercising the queue once per turn.
        if payload.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        body = json.dumps(
            {
                "id": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # silence stderr logging
        pass


def _first_user_text(payload: dict) -> str:
    for message in payload.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def _turn_arrivals(prompts) -> list:
    """Only the turns' own calls.

    Agent construction probes the endpoint for its context length before any
    turn runs; those requests carry no user message and are sequential, so
    they cannot contribute to the concurrency peak.
    """
    wanted = set(prompts)
    return [text for text in _CountingHandler.arrivals if text in wanted]


@pytest.fixture
def backend():
    _CountingHandler.in_flight = 0
    _CountingHandler.peak = 0
    _CountingHandler.arrivals = []
    _CountingHandler.gate = None
    # Threading, not the single-threaded HTTPServer: a server that
    # serialises requests itself would report peak concurrency 1 no matter
    # what the agent did, and the control test below would be vacuous.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _clean_scheduler():
    backend_scheduler.reset_for_tests()
    live_turn_registry.reset_for_tests()
    yield
    backend_scheduler.reset_for_tests()
    live_turn_registry.reset_for_tests()


def _configure(monkeypatch, **overrides):
    cfg = dataclasses.replace(
        backend_scheduler.SchedulerSettings(
            mode="on",
            max_concurrent_requests=1,
            queue_wait_seconds=30.0,
            poll_seconds=0.05,
            notify_after_seconds=0,
        ),
        **overrides,
    )
    monkeypatch.setattr(backend_scheduler, "settings", lambda: cfg)
    return cfg


def _make_agent(base_url: str, session_id: str):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="openai-compat",
        model="test-model",
        max_iterations=3,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        platform="cli",
    )
    agent.session_id = session_id
    return agent


def _run_turns(base_url: str, prompts):
    """Start one turn per prompt, all at once; return them in finish order."""
    agents = [_make_agent(base_url, f"session-{i}") for i, _ in enumerate(prompts)]
    finished: list = []
    finished_lock = threading.Lock()
    ready = threading.Barrier(len(prompts))

    def _turn(agent, prompt):
        ready.wait(timeout=10)
        agent.run_conversation(prompt, conversation_history=[], task_id=None)
        with finished_lock:
            finished.append(prompt)

    threads = [
        threading.Thread(target=_turn, args=(agent, prompt), daemon=True)
        for agent, prompt in zip(agents, prompts)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "a turn never finished"
    return finished


def test_engaged_scheduler_keeps_the_backend_to_one_call(backend, monkeypatch):
    """H-032/H-033: nothing is submitted while the backend is occupied."""
    _configure(monkeypatch)
    prompts = ["first turn", "second turn"]
    _run_turns(backend, prompts)

    assert sorted(_turn_arrivals(prompts)) == sorted(prompts)
    assert _CountingHandler.peak == 1, (
        "the conversation loop submitted while the backend was busy — "
        f"peak concurrency {_CountingHandler.peak}"
    )


def test_disengaged_scheduler_lets_both_turns_pile_on(backend, monkeypatch):
    """Control: without the queue the two turns really do collide.

    Without this, the assertion above would also pass on a machine where the
    threads simply never overlapped.
    """
    _configure(monkeypatch, mode="off")
    prompts = ["first turn", "second turn"]
    _run_turns(backend, prompts)

    assert sorted(_turn_arrivals(prompts)) == sorted(prompts)
    assert _CountingHandler.peak == 2


def test_permit_is_released_before_the_next_call_of_the_same_turn(
    backend, monkeypatch
):
    """The claim covers one call, not the whole turn.

    A turn that held its permit across tool execution would lease the backend
    to one session for as long as the turn ran. After the turn, nothing is
    left holding anything.
    """
    _configure(monkeypatch)
    agent = _make_agent(backend, "solo")
    agent.run_conversation("hello", conversation_history=[], task_id=None)

    snapshot = backend_scheduler.snapshot()
    assert snapshot["active"] == []
    assert snapshot["waiting"] == []
    # The completed call was measured, which is what the derived queue
    # deadline reads (item 3).
    assert snapshot["samples"] >= 1
    assert snapshot["call_seconds_p90"] >= _CountingHandler.hold_seconds / 2


def test_a_live_turn_takes_the_slot_ahead_of_queued_maintenance(
    backend, monkeypatch
):
    """The H-034 priority inversion, through the real loop this time.

    Background review holds the backend; a second review and a user's turn
    both arrive while it is busy, the review first. When the slot frees up the
    user goes next — housekeeping that queued earlier waits.
    """
    _configure(monkeypatch)
    holder = _make_agent(backend, "review-holder")
    holder._is_background_review_fork = True
    later_review = _make_agent(backend, "review-later")
    later_review._is_background_review_fork = True
    user = _make_agent(backend, "user")

    # Only now: construction probes the endpoint for its context length, and
    # those requests would otherwise sit on the gate too.
    gate = threading.Event()
    _CountingHandler.gate = gate

    def _turn(agent, prompt):
        agent.run_conversation(prompt, conversation_history=[], task_id=None)

    threads = {
        name: threading.Thread(target=_turn, args=(agent, name), daemon=True)
        for name, agent in (
            ("holding review", holder),
            ("later review", later_review),
            ("user turn", user),
        )
    }

    threads["holding review"].start()
    assert _wait_for(lambda: "holding review" in _CountingHandler.arrivals), (
        "the first review never reached the backend"
    )

    threads["later review"].start()
    assert _wait_for(lambda: _waiting_priorities() == ["maintenance"])
    threads["user turn"].start()
    assert _wait_for(lambda: len(_waiting_priorities()) == 2)

    # Ordered before anything is released: the queue itself is what decides.
    assert _waiting_priorities() == ["live", "maintenance"]

    gate.set()
    for thread in threads.values():
        thread.join(timeout=60)
        assert not thread.is_alive(), "a turn never finished"

    ordered = [t for t in _CountingHandler.arrivals if t in threads]
    assert ordered == ["holding review", "user turn", "later review"]


def _waiting_priorities() -> list:
    return [entry["priority"] for entry in backend_scheduler.snapshot()["waiting"]]


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
