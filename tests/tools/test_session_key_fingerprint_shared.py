"""redact.py and process_registry.py must fingerprint a session key the same
way (F-6).

Both modules log an opaque, run-local fingerprint of the session key instead of
the raw value (which embeds Discord channel / thread / participant IDs). They
used to hold *separate* per-process salts, so one session logged as two
different fingerprints across the two logs, defeating correlation.
process_registry now delegates to redact.py's single source.
"""

from __future__ import annotations

from agent.redact import session_key_fingerprint
from tools.process_registry import _session_key_fingerprint

SESSION_KEY = "discord:112233445566778899:998877665544332211:123456789012345678"


def test_both_modules_agree_on_the_fingerprint():
    assert _session_key_fingerprint(SESSION_KEY) == session_key_fingerprint(SESSION_KEY)


def test_empty_key_is_none_in_both():
    assert _session_key_fingerprint("") == session_key_fingerprint("") == "none"


def test_fingerprint_does_not_leak_the_raw_key():
    fp = _session_key_fingerprint(SESSION_KEY)
    assert "112233445566778899" not in fp
    assert "998877665544332211" not in fp
    assert "123456789012345678" not in fp
    # blake2s digest_size=6 → 12 hex chars.
    assert len(fp) == 12


def test_distinct_keys_get_distinct_fingerprints():
    other = "discord:000000000000000000:111111111111111111:222222222222222222"
    assert _session_key_fingerprint(SESSION_KEY) != _session_key_fingerprint(other)
