"""Tests for the real-checkout git-mutation guard in ``tests/conftest.py``.

The guard is what stands between a mis-mocked ``hermes update`` test and the
developer's own working tree (incident H-035). If it ever stops classifying
correctly it fails open and silently, so its contract is pinned here:

* a work-destroying git command aimed at THIS checkout is reported,
* the same command aimed at any other repository is not,
* read-only git is never reported.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from tests.conftest import REAL_REPO_ROOT, describe_real_repo_git_mutation

# Captured at collection time, before the function-scoped guard fixture has
# wrapped anything — the only moment this file can see the unguarded original.
_UNGUARDED_SUBPROCESS_RUN = subprocess.run


@pytest.mark.parametrize(
    "cmd",
    [
        ["git", "stash", "push", "--include-untracked", "-m", "hermes-update-autostash-x"],
        ["git", "checkout", "main"],
        ["git", "reset", "--hard", "origin/main"],
        ["git", "pull", "--ff-only"],
        ["git", "clean", "-fdx"],
        ["git", "commit", "-m", "nope"],
        # The wrapper form must not launder it.
        ["bash", "-c", f"cd {REAL_REPO_ROOT} && git stash push"],
        # ...nor the -C form from a directory that is itself innocent.
        ["git", "-C", str(REAL_REPO_ROOT), "stash", "push"],
    ],
)
def test_mutating_git_against_the_real_checkout_is_reported(cmd):
    assert describe_real_repo_git_mutation(cmd, str(REAL_REPO_ROOT)) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        ["git", "status", "--porcelain"],
        ["git", "rev-parse", "--verify", "refs/stash"],
        ["git", "ls-files", "--unmerged"],
        ["git", "log", "-1", "--format=%H"],
        ["git", "config", "--get", "remote.origin.url"],
        ["git", "branch", "--show-current"],
        ["git", "--version"],
        # A command that merely mentions git in an argument is not a git call.
        ["echo", "git reset --hard"],
    ],
)
def test_read_only_git_against_the_real_checkout_is_allowed(cmd):
    assert describe_real_repo_git_mutation(cmd, str(REAL_REPO_ROOT)) is None


def test_mutating_git_against_another_repo_is_allowed(tmp_path):
    """A ``tmp_path`` fixture repo is exactly what correctly-mocked tests use."""
    cmd = ["git", "stash", "push", "--include-untracked"]
    assert describe_real_repo_git_mutation(cmd, str(tmp_path)) is None

    # And it really runs — the guard must not block the legitimate case.
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "seed.txt").write_text("seed", encoding="utf-8")
    subprocess.run(git + ["add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        git + ["commit", "-q", "-m", "seed"], cwd=tmp_path, check=True, capture_output=True
    )

    (tmp_path / "seed.txt").write_text("dirty", encoding="utf-8")
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _make_repo_with_worktree(tmp_path):
    """A temp repository plus one linked worktree, both ready for commits."""
    main = tmp_path / "main"
    main.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    (main / "seed.txt").write_text("seed", encoding="utf-8")
    subprocess.run(git + ["add", "-A"], cwd=main, check=True, capture_output=True)
    subprocess.run(
        git + ["commit", "-q", "-m", "seed"], cwd=main, check=True, capture_output=True
    )
    linked = tmp_path / "linked"
    subprocess.run(
        git + ["worktree", "add", "-q", str(linked)],
        cwd=main,
        check=True,
        capture_output=True,
    )
    return main, linked


def test_a_suite_running_inside_a_worktree_protects_that_worktree(tmp_path, monkeypatch):
    """The guard must not fail open when the checkout IS a linked worktree.

    A worktree's ``.git`` is a pointer FILE, not a directory. Comparing it to a
    ``<root>/.git`` path never matches, so a guard that does not follow the
    pointer waves through every mutation aimed at the tree it protects.
    """
    from tests import conftest as guard

    main, linked = _make_repo_with_worktree(tmp_path)
    monkeypatch.setattr(guard, "REAL_REPO_ROOT", linked)
    monkeypatch.setattr(
        guard, "_REAL_REPO_GIT_DIRS", guard._git_dirs_identifying_checkout(linked)
    )

    stash = ["git", "stash", "push", "--include-untracked"]
    # The worktree the suite lives in...
    assert describe_real_repo_git_mutation(stash, str(linked)) is not None
    # ...and the repo it is linked to, which owns the shared refs/stash.
    assert describe_real_repo_git_mutation(stash, str(main)) is not None
    # Read-only git and unrelated repositories stay untouched.
    assert describe_real_repo_git_mutation(["git", "status"], str(linked)) is None
    other = tmp_path / "other"
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    assert describe_real_repo_git_mutation(stash, str(other)) is None


def test_a_worktree_of_this_checkout_is_protected(tmp_path):
    """The mirror case: the suite runs in the main repo, git runs in a worktree.

    Both directions matter because the two share ``refs/stash``; the worktree's
    identity must therefore include the common git dir.
    """
    from tests import conftest as guard

    main, linked = _make_repo_with_worktree(tmp_path)
    main_dirs = guard._git_dirs_identifying_checkout(main)
    linked_dirs = guard._git_dirs_identifying_checkout(linked)

    assert main_dirs, "a plain checkout must resolve its .git directory"
    assert linked_dirs & main_dirs, "a worktree must share identity with its repo"
    assert linked_dirs - main_dirs, "and still carry its own worktree git dir"


def test_guard_is_actually_installed_around_subprocess():
    """Catches the guard being disabled wholesale (renamed fixture, bad merge).

    A correct classifier is worthless if nothing calls it, so check that the
    fixture really has wrapped ``subprocess.run`` for this test.
    """
    assert subprocess.run is not _UNGUARDED_SUBPROCESS_RUN


def test_cwd_defaults_to_the_process_working_directory():
    """``cwd=None`` must be resolved, not waved through.

    ``hermes update`` reaches the real repo precisely because pytest's own
    working directory is the checkout and several git calls pass no ``cwd``.
    """
    if REAL_REPO_ROOT.resolve() != pathlib.Path.cwd().resolve():
        pytest.skip("pytest was not invoked from the repository root")
    assert describe_real_repo_git_mutation(["git", "stash", "push"], None) is not None
