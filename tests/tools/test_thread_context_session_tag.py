"""H-030: the ``hermes_logging`` session tag must survive worker-thread hops.

``agent.tool_executor``'s parallel-tool dispatch runs each tool on a
``ThreadPoolExecutor`` worker via
``tools.thread_context.propagate_context_to_thread``. That helper already
carries ContextVars and the CLI approval/sudo callbacks across the thread
hop, but ``hermes_logging``'s per-conversation session tag lives in
``threading.local`` — which, unlike ``contextvars.Context``, is never
inherited by a new thread — so log lines from a parallel-tool worker lost
their ``[session_id]`` prefix. ``get_session_context()`` plus explicit
capture/install/clear in ``propagate_context_to_thread`` closes that gap.
"""

import concurrent.futures
import logging
import threading

import hermes_logging
from tools.thread_context import propagate_context_to_thread


class TestSessionTagPropagation:
    def test_session_id_visible_inside_propagated_worker_thread(self):
        hermes_logging.set_session_context("main-session-abc")
        try:
            captured = {}

            def _worker():
                captured["session_id"] = hermes_logging.get_session_context()

            wrapped = propagate_context_to_thread(_worker)
            t = threading.Thread(target=wrapped)
            t.start()
            t.join(timeout=5)
        finally:
            hermes_logging.clear_session_context()

        assert captured["session_id"] == "main-session-abc"

    def test_worker_thread_log_record_carries_session_tag(self, caplog):
        logger = logging.getLogger("agent.tool_executor")

        hermes_logging.set_session_context("sess-xyz")
        try:
            def _worker():
                logger.info("worker did a thing")

            wrapped = propagate_context_to_thread(_worker)
            with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
                t = threading.Thread(target=wrapped)
                t.start()
                t.join(timeout=5)
        finally:
            hermes_logging.clear_session_context()

        records = [r for r in caplog.records if r.message == "worker did a thing"]
        assert len(records) == 1
        # session_tag is stamped by hermes_logging's global LogRecord factory
        # (installed at module import) from the *emitting thread's*
        # threading.local — this is the exact field the H-030 bug dropped.
        assert getattr(records[0], "session_tag", None) == " [sess-xyz]"

    def test_no_active_session_leaves_worker_thread_untagged(self):
        """No session set on the parent thread (e.g. a maintenance/background
        call with no live turn) must not raise, and the worker sees no tag."""
        assert hermes_logging.get_session_context() is None
        captured = {}

        def _worker():
            captured["session_id"] = hermes_logging.get_session_context()

        wrapped = propagate_context_to_thread(_worker)
        t = threading.Thread(target=wrapped)
        t.start()
        t.join(timeout=5)

        assert captured["session_id"] is None

    def test_session_tag_does_not_leak_across_reused_pool_thread(self):
        """A ThreadPoolExecutor reuses idle worker threads across tasks. The
        tag installed for one tool call's dispatch must be cleared before the
        thread is handed the next, unrelated task."""
        results = []

        def _worker():
            results.append(hermes_logging.get_session_context())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            hermes_logging.set_session_context("sess-1")
            try:
                pool.submit(propagate_context_to_thread(_worker)).result(timeout=5)
            finally:
                hermes_logging.clear_session_context()

            # Second dispatch on the same pool (single worker => same OS
            # thread), with no session active on the parent this time.
            pool.submit(propagate_context_to_thread(_worker)).result(timeout=5)

        assert results == ["sess-1", None]

    def test_parent_thread_session_unaffected_by_worker(self):
        """Propagation is a one-way copy into the worker; it must not let the
        worker's clear-on-exit reach back and clobber the parent thread's own
        (unrelated) session tag."""
        hermes_logging.set_session_context("parent-session")
        try:
            def _worker():
                hermes_logging.set_session_context("worker-session")

            wrapped = propagate_context_to_thread(_worker)
            t = threading.Thread(target=wrapped)
            t.start()
            t.join(timeout=5)

            assert hermes_logging.get_session_context() == "parent-session"
        finally:
            hermes_logging.clear_session_context()
