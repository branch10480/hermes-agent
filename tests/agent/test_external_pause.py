"""External video jobs must park the same review instead of losing or retrying work."""
import threading
import time
from types import SimpleNamespace

import pytest

from agent import backend_scheduler, external_pause
from agent.conversation_compression import CompressionCommitFence, run_compress_context_with_progress_timeout

URL = 'http://127.0.0.1:18088/v1'


@pytest.fixture
def reservation(tmp_path, monkeypatch):
    path = tmp_path / 'reservation.json'
    cfg = {'agent': {'backend_scheduler': {'external_pause_file': str(path),
           'external_pause_base_url': URL}},
           'auxiliary': {'background_review': {'idle_delay_seconds': 0, 'idle_gate_max_wait_seconds': 5}}}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: cfg)
    backend_scheduler.reset_for_tests()
    yield path
    path.unlink(missing_ok=True)
    backend_scheduler.reset_for_tests()


def join(thread):
    thread.join(3)
    assert not thread.is_alive()


def test_parked_request_uses_no_backend_permit_and_resumes_once(reservation, monkeypatch):
    parked = threading.Event()
    monkeypatch.setattr(external_pause.logger, "info", lambda *a: parked.set())
    reservation.write_text('occupied')
    called = []
    owner = SimpleNamespace(base_url=URL, session_id='test')
    def run():
        ticket = backend_scheduler.acquire(owner)
        try:
            called.append('provider')
        finally:
            backend_scheduler.release(ticket)
    thread = threading.Thread(target=run)
    thread.start()
    assert parked.wait(3)
    assert called == []
    assert backend_scheduler.snapshot()['active'] == []
    reservation.unlink()
    join(thread)
    assert called == ['provider']


def test_cancel_during_pause_never_dispatches(reservation):
    reservation.write_text('occupied')
    cancelled = threading.Event()
    outcome = []
    def run():
        try:
            backend_scheduler.acquire(SimpleNamespace(base_url=URL), should_abort=cancelled.is_set)
            outcome.append('bad dispatch')
        except InterruptedError:
            outcome.append('cancelled')
    thread = threading.Thread(target=run)
    thread.start()
    cancelled.set()
    join(thread)
    assert outcome == ['cancelled']


@pytest.mark.parametrize('body', [external_pause.REJECTION,
    {'error': external_pause.REJECTION}, {'error': {'message': external_pause.REJECTION}},
    {'message': external_pause.REJECTION}])
def test_explicit_rejection_reuses_request_without_replaying_other_work(reservation, body):
    class Rejected(Exception):
        status_code = 503
    Rejected.body = body
    refused = threading.Event()
    request = {'messages': [{'role': 'user', 'content': 'unchanged'}]}
    received, done = [], []
    def provider():
        received.append(request)
        if len(received) == 1:
            reservation.write_text('occupied')
            refused.set()
            raise Rejected()
        return 'ok'
    thread = threading.Thread(target=lambda: done.append(external_pause.call(provider, URL)))
    thread.start()
    assert refused.wait(3)
    assert len(received) == 1
    reservation.unlink()
    join(thread)
    assert done == ['ok']
    assert received == [request, request]
    assert received[0] is received[1]
    with pytest.raises(RuntimeError):
        external_pause.call(lambda: (_ for _ in ()).throw(RuntimeError('ambiguous failure')), URL)


def test_other_endpoint_not_affected(reservation):
    reservation.write_text('occupied')
    assert external_pause.call(lambda: 'ok', 'https://example.invalid/v1') == 'ok'


def test_auxiliary_pause_releases_and_restores_outer_permit(reservation, monkeypatch):
    from agent import auxiliary_client
    from agent.auxiliary_client import _relay_sync_completion
    monkeypatch.setattr(auxiliary_client, '_relay_auxiliary_metadata', lambda **kw: None)
    started, paused, completed = threading.Event(), threading.Event(), []
    released = threading.Event()
    original_release = backend_scheduler.release
    def release(ticket):
        original_release(ticket)
        released.set()
    monkeypatch.setattr(backend_scheduler, "release", release)
    owner = SimpleNamespace(base_url=URL, session_id='review')
    def run():
        ticket = backend_scheduler.acquire(owner)
        reservation.write_text('occupied')
        started.set()
        try:
            _relay_sync_completion(owner, {'messages': []}, create=lambda kw: completed.append(kw))
        finally:
            backend_scheduler.release(ticket)
            paused.set()
    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(3)
    assert released.wait(3)
    assert backend_scheduler.snapshot()['active'] == []
    reservation.unlink()
    join(thread)
    assert len(completed) == 1
    assert backend_scheduler.snapshot()['active'] == []


def test_review_timer_survives_hours_and_starts_only_once(reservation, monkeypatch):
    from tests.agent.test_background_review_idle_gate import (
        _review_owner, _use_recording_timer, _RecordingTimer, _ImmediateThread,
    )
    import run_agent
    owner = _review_owner()
    owner.base_url = URL
    _use_recording_timer(monkeypatch)
    now, starts = [10.0], []
    monkeypatch.setattr(external_pause.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(run_agent.threading, 'Thread',
                        lambda **kw: starts.append(kw) or _ImmediateThread(target=lambda: None))
    reservation.write_text('occupied')
    owner._spawn_background_review(messages_snapshot=[{'role': 'user', 'content': 'hi'}], review_skills=True)
    now[0] += 7200
    _RecordingTimer.instances[-1].fire()
    assert starts == []
    reservation.unlink()
    now[0] += 5
    timer = _RecordingTimer.instances[-1]
    timer.fire()
    timer.fire()
    assert len(starts) == 1
    owner._cancel_background_review_timer()


def test_compression_outer_idle_and_total_budgets_exclude_pause(reservation, monkeypatch):
    monkeypatch.setattr(backend_scheduler, 'auxiliary_endpoint', lambda *a, **kw: URL)
    reservation.write_text('occupied')
    now, parked = [time.monotonic()], threading.Event()
    real_clock = external_pause.WaitClock
    class TestClock(real_clock):
        def __init__(self):
            super().__init__(clock=lambda: now[0])
        def park(self):
            super().park()
            parked.set()
    monkeypatch.setattr(external_pause, "WaitClock", TestClock)
    results, timeouts = [], []
    messages = [{'role': 'user', 'content': 'same'}]
    fence = CompressionCommitFence()
    def worker(worker_fence):
        external_pause.wait(URL, should_abort=lambda: worker_fence.deadline_exceeded, poll=.01)
        return messages, 'same prompt'
    def run():
        results.append(run_compress_context_with_progress_timeout(
            worker=worker, messages=messages, system_prompt_fallback='fallback',
            idle_timeout_seconds=.05, total_ceiling_seconds=.1, fence=fence,
            telemetry_agent=SimpleNamespace(base_url=URL), stall_fallback=False,
            on_timeout=lambda *a: timeouts.append(a)))
    thread = threading.Thread(target=run)
    thread.start()
    assert parked.wait(3)
    now[0] += 7200
    assert thread.is_alive()
    assert not fence.deadline_exceeded
    reservation.unlink()
    join(thread)
    assert results == [(messages, 'same prompt')]
    assert timeouts == []


def controlled_attempt_clock(monkeypatch):
    now = [time.monotonic()]
    real_clock = external_pause.WaitClock
    class TestClock(real_clock):
        def __init__(self):
            super().__init__(clock=lambda: now[0])
    monkeypatch.setattr(external_pause, "WaitClock", TestClock)
    return now


def test_reservation_does_not_hide_stalled_dispatched_provider(reservation, monkeypatch):
    now = controlled_attempt_clock(monkeypatch)
    monkeypatch.setattr(backend_scheduler, 'auxiliary_endpoint', lambda *a, **kw: URL)
    entered, finish = threading.Event(), threading.Event()
    timeouts = []
    def worker(fence):
        entered.set()
        finish.wait(5)
        return [], 'provider result'
    result = []
    thread = threading.Thread(target=lambda: result.append(run_compress_context_with_progress_timeout(
        worker=worker, messages=[], system_prompt_fallback='fallback',
        idle_timeout_seconds=.1, total_ceiling_seconds=2,
        telemetry_agent=SimpleNamespace(base_url=URL), stall_fallback=False,
        on_timeout=lambda *a: timeouts.append(a))))
    thread.start()
    assert entered.wait(3)
    reservation.write_text('occupied')
    now[0] += 10
    try:
        join(thread)
        assert timeouts
        assert result == [([], 'fallback')]
    finally:
        finish.set()


def test_reservation_does_not_hide_commit_overrun(reservation, monkeypatch):
    now = controlled_attempt_clock(monkeypatch)
    monkeypatch.setattr(backend_scheduler, 'auxiliary_endpoint', lambda *a, **kw: URL)
    entered, finish, overrun = threading.Event(), threading.Event(), threading.Event()
    def worker(fence):
        assert fence.begin_commit()
        entered.set()
        try:
            finish.wait(5)
            return [], 'committed'
        finally:
            fence.finish_commit()
    result = []
    thread = threading.Thread(target=lambda: result.append(run_compress_context_with_progress_timeout(
        worker=worker, messages=[], system_prompt_fallback='fallback',
        idle_timeout_seconds=.05, total_ceiling_seconds=.1,
        telemetry_agent=SimpleNamespace(base_url=URL), stall_fallback=False,
        on_commit_overrun=lambda *a: overrun.set())))
    thread.start()
    assert entered.wait(3)
    reservation.write_text('occupied')
    now[0] += 10
    try:
        assert overrun.wait(3)
        assert thread.is_alive()
    finally:
        finish.set()
        join(thread)
    assert result == [([], 'committed')]
