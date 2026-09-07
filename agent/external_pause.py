"""Pause local requests at provider boundaries without changing their payloads."""
from __future__ import annotations

import logging
from contextvars import ContextVar
import os
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)
REJECTION = "exclusive local inference job is running"


def configured_path(base_url):
    from hermes_cli.config import load_config_readonly
    if not base_url:
        return None
    try:
        raw = load_config_readonly().get("agent", {}).get("backend_scheduler", {})
    except Exception:
        # Optional scheduling only; the proxy remains the admission boundary.
        return None
    if not isinstance(raw, dict):
        return None
    endpoint = raw.get("external_pause_base_url")
    if not endpoint or str(base_url or "").rstrip("/") != str(endpoint).rstrip("/"):
        return None
    path = Path(raw["external_pause_file"]).expanduser()
    if not path.is_absolute():
        raise ValueError("external_pause_file must be absolute")
    return path


def active_path(path):
    if path is None:
        return False
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


def active(base_url):
    return active_path(configured_path(base_url))


def wait(base_url, *, should_abort=None, poll=1.0):
    path = configured_path(base_url)
    if not active_path(path):
        if should_abort and should_abort():
            raise InterruptedError("external inference wait cancelled")
        return
    # The observation that starts this wait also yields the whole permit stack.
    # A separate earlier probe would leave a check-to-wait race.
    from agent import backend_scheduler
    with backend_scheduler.yield_for_external_pause(base_url, should_abort=should_abort):
        started = time.monotonic()
        budget = wait_clock.get()
        if budget is not None:
            budget.park()
        try:
            logger.info("Local inference paused for external workload")
            while active_path(path):
                if should_abort and should_abort():
                    raise InterruptedError("external inference wait cancelled")
                time.sleep(poll)
            if should_abort and should_abort():
                raise InterruptedError("external inference wait cancelled")
        finally:
            if budget is not None:
                budget.unpark()
        logger.info("Local inference resumed after %.1fs", time.monotonic() - started)


def rejected(error):
    if getattr(error, "status_code", None) != 503:
        return False
    body = getattr(error, "body", None)
    if body == REJECTION:
        return True
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    return detail == REJECTION or isinstance(detail, dict) and detail.get("message") == REJECTION


def call(callback, base_url, *, should_abort=None):
    """Retry only the proxy's explicit pre-admission refusal, never an ambiguous call."""
    while True:
        wait(base_url, should_abort=should_abort)
        try:
            return callback()
        except Exception as exc:
            if configured_path(base_url) is None or not rejected(exc):
                raise
            # The holder may release between the rejection and this check.
            # Re-submit the same arguments; no ordinary retry/fallback budget.


class PauseClock:
    """Monotonic budget clock; external reservation time is not a timeout."""
    def __init__(self, base_url, *, clock=None):
        self.path = configured_path(base_url)
        self.clock = clock or time.monotonic
        self.lock = threading.Lock()
        self.real = self.clock()
        self.value = self.real
        self.paused = active_path(self.path)

    def __call__(self):
        with self.lock:
            now = self.clock()
            paused = active_path(self.path)
            if not (self.paused or paused):
                self.value += now - self.real
            self.real, self.paused = now, paused
            return self.value


wait_clock = ContextVar("external_inference_wait_clock", default=None)


class WaitClock:
    """One attempt's clock; only its actual parked intervals are excluded."""
    def __init__(self, *, clock=None):
        self.clock = clock or time.monotonic
        self.lock = threading.Lock()
        self.total = 0.0
        self.since = None
        self.waiters = 0

    @property
    def paused(self):
        with self.lock:
            return self.waiters > 0

    def park(self):
        with self.lock:
            if self.waiters == 0:
                self.since = self.clock()
            self.waiters += 1

    def unpark(self):
        with self.lock:
            self.waiters -= 1
            if self.waiters == 0:
                self.total += self.clock() - self.since
                self.since = None

    def __call__(self):
        with self.lock:
            now = self.clock()
            paused = now - self.since if self.since is not None else 0.0
            return now - self.total - paused
