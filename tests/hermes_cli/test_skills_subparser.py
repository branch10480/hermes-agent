"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse

from tests.hermes_cli.conftest import fresh_hermes_module_imports


def test_no_duplicate_skills_subparser():
    """Ensure 'skills' subparser is only registered once to avoid Python 3.11+ crash.

    Python 3.11 changed argparse to raise an exception on duplicate subparser
    names instead of silently overwriting (see CPython #94331).

    This test will fail with:
        argparse.ArgumentError: argument command: conflicting subparser: skills

    if the duplicate 'skills' registration is reintroduced.
    """
    # Force fresh import of the module where parser is constructed.
    # If there are duplicate 'skills' subparsers, this import will raise
    # argparse.ArgumentError at module load time.
    #
    # The eviction must be undone afterwards: leaving a fresh hermes_cli.main
    # in sys.modules orphans the module object every other test file bound at
    # import time, which silently defeats their patches (incident H-035).
    with fresh_hermes_module_imports():
        try:
            import hermes_cli.main  # noqa: F401
        except argparse.ArgumentError as e:
            if "conflicting subparser" in str(e):
                raise AssertionError(
                    f"Duplicate subparser detected: {e}. "
                    "See issue #898 for details."
                ) from e
            raise
