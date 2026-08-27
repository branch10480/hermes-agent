"""Tirith pre-exec security scanning wrapper.

Runs the tirith binary as a subprocess to scan commands for content-level
threats (homograph URLs, pipe-to-interpreter, terminal injection, etc.).

Exit code is the verdict source of truth:
  0 = allow, 1 = block, 2 = warn

JSON stdout enriches findings/summary and never escalates the verdict.  The
single downgrade is the .app lookalike_tld false-positive suppression in
check_command_security (see the comment there for its guard conditions).
Operational failures (spawn error, timeout, unknown exit code) respect
the fail_open config setting. Programming errors propagate.

Fail-open is per surface. A local CLI/TUI turn happens in front of the
person who typed it, so an unavailable scanner degrades to the historical
allow (``security.tirith_fail_open``, default true). A gateway or cron turn
has no such witness — the conversation arrives over chat or a timer — so an
unavailable scanner blocks instead (``security.tirith_fail_open_gateway``,
default false). Those surfaces also never reach for the network: an
unpinned auto-download is a supply-chain decision no unattended turn should
be making for itself.

Auto-install: if tirith is not found on PATH or at the configured path,
it is automatically downloaded from GitHub releases to $HERMES_HOME/bin/tirith.
The download always verifies SHA-256 checksums.  When cosign is available on
PATH, provenance verification (GitHub Actions workflow signature) is also
performed.  If cosign is not installed, the download proceeds with SHA-256
verification only — still secure via HTTPS + checksum, just without supply
chain provenance proof.  Installation runs in a background thread so startup
never blocks.  Gateway and cron turns opt out of it entirely (see above).
"""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_REPO = "sheeki03/tirith"

# Cosign provenance verification — pinned to the specific release workflow
_COSIGN_IDENTITY_REGEXP = f"^https://github.com/{_REPO}/\\.github/workflows/release\\.yml@refs/tags/v"
_COSIGN_ISSUER = "https://token.actions.githubusercontent.com"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _load_security_config() -> dict:
    """Load security settings from config.yaml, with env var overrides."""
    defaults = {
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
        "tirith_fail_open_gateway": False,
    }
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly().get("security", {}) or {}
    except Exception:
        cfg = {}

    return {
        "tirith_enabled": _env_bool("TIRITH_ENABLED", cfg.get("tirith_enabled", defaults["tirith_enabled"])),
        "tirith_path": os.getenv("TIRITH_BIN", cfg.get("tirith_path", defaults["tirith_path"])),
        "tirith_timeout": _env_int("TIRITH_TIMEOUT", cfg.get("tirith_timeout", defaults["tirith_timeout"])),
        "tirith_fail_open": _env_bool("TIRITH_FAIL_OPEN", cfg.get("tirith_fail_open", defaults["tirith_fail_open"])),
        "tirith_fail_open_gateway": _env_bool(
            "TIRITH_FAIL_OPEN_GATEWAY",
            cfg.get("tirith_fail_open_gateway", defaults["tirith_fail_open_gateway"]),
        ),
    }


# ---------------------------------------------------------------------------
# Surface-aware fail policy
# ---------------------------------------------------------------------------

# Surfaces with nobody watching the terminal the agent is driving. Kept in
# sync with tools.approval.REMOTE_TOOL_SURFACES via _current_surface().
_REMOTE_SURFACES = frozenset({"gateway", "cron"})

# What the owner has to do to clear a fail-closed verdict. Deliberately free
# of any host path: this text reaches the model, and on a gateway it is
# rendered straight into a chat message.
_PIN_REMEDIATION = (
    "The Hermes owner must install the pinned Tirith build on the host and "
    "point security.tirith_path at it, then restart the gateway."
)


def _current_surface() -> str:
    """Return the surface driving this scan: 'gateway', 'cron', or 'local'.

    ``tools.approval`` owns surface detection and is the only caller of
    ``check_command_security``, so the import below is a warm module lookup
    rather than a real import. If it ever fails, the security layer itself is
    broken — the moment to be strict, not permissive — so we report the
    stricter remote surface.
    """
    try:
        from tools.approval import get_current_tool_surface
        return get_current_tool_surface()
    except Exception:
        _warn_once(
            "tirith_surface_unavailable",
            "tirith could not determine the calling surface; "
            "treating it as a gateway surface (fail closed)",
        )
        return "gateway"


def _effective_fail_open(cfg: dict, surface: str) -> bool:
    """Return the fail-open policy that applies to *surface*.

    Remote surfaces read ``tirith_fail_open_gateway`` (default false) and can
    never end up more permissive than the global ``tirith_fail_open``, so an
    owner who turns fail-open off everywhere cannot be re-opened by a stale
    gateway key.
    """
    if surface not in _REMOTE_SURFACES:
        return bool(cfg.get("tirith_fail_open", True))
    return bool(cfg.get("tirith_fail_open_gateway", False)) and bool(
        cfg.get("tirith_fail_open", True)
    )


def _unavailable_verdict(reason_code: str, command: str, *,
                         fail_open: bool, remote: bool,
                         summary: str, closed_summary: str | None = None) -> dict:
    """Build the verdict for a scan that could not produce one.

    *summary* and *closed_summary* are the historical local-surface wordings,
    kept verbatim so CLI behaviour and its tests are untouched. Remote
    surfaces get a structured finding instead: a reason code the caller can
    branch on and remediation text, with no exception strings or host paths
    that would leak the install layout into a chat transcript.
    """
    if fail_open:
        return {"action": "allow", "findings": [], "summary": summary}
    if not remote:
        return {
            "action": "block",
            "findings": [],
            "summary": closed_summary or f"{summary} (fail-closed)",
        }

    # Digest the command into the rule_id. Approval persistence keys off
    # ``tirith:<rule_id>``, so a single shared id would let one session
    # approval re-open the gate for every later command in that session.
    digest = hashlib.sha256(command.encode("utf-8", "replace")).hexdigest()[:12]
    return {
        "action": "block",
        "fail_closed": True,
        "reason_code": reason_code,
        "findings": [
            {
                "rule_id": f"{reason_code}:{digest}",
                "severity": "HIGH",
                "title": "Tirith security scanner unavailable",
                "description": (
                    f"This command could not be security-scanned "
                    f"(reason_code={reason_code}). Commands from chat and "
                    f"scheduled sessions are not allowed to run unscanned. "
                    f"{_PIN_REMEDIATION}"
                ),
                "reason_code": reason_code,
            }
        ],
        "summary": f"tirith unavailable ({reason_code}) — fail closed",
    }


# ---------------------------------------------------------------------------
# Auto-install
# ---------------------------------------------------------------------------

# Cached path after first resolution (avoids repeated shutil.which per command).
# _INSTALL_FAILED means "we tried and failed" — prevents retry on every command.
_resolved_path: str | None | bool = None
_INSTALL_FAILED = False  # sentinel: distinct from "not yet tried"
_install_failure_reason: str = ""  # reason tag when _resolved_path is _INSTALL_FAILED

# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures,
# disable tirith for the rest of the process to prevent agent hangs (#41400).
# Reset on successful execution (see _record_tirith_crash / check_command_security).
#
# Thread safety: _crash_count and _circuit_open are module-level globals
# mutated without a lock. check_command_security can be called from
# concurrent agent threads (gateway multi-session). The race is benign —
# at worst two threads both increment past _CRASH_LIMIT and both set
# _circuit_open = True, opening the breaker one call early. No data
# corruption or security bypass is possible. This intentionally matches
# the lock-free style of error counters in mcp_tool.py rather than the
# locked _warn_once pattern, because the worst case is harmless.
_CRASH_LIMIT = 3
_crash_count: int = 0
_circuit_open: bool = False


def _record_tirith_crash() -> None:
    """Increment the crash counter and open the circuit breaker if needed."""
    global _crash_count, _circuit_open
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open = True
        logger.warning(
            "tirith circuit breaker opened after %d consecutive failures; "
            "disabling for the rest of the process",
            _crash_count,
        )

# Background install thread coordination
_install_lock = threading.Lock()
_install_thread: threading.Thread | None = None

# Warning de-duplication. The spawn/path warnings live in the hot path —
# without this dedupe set, a Windows install where ``tirith`` isn't on PATH
# (e.g. background install thread still running, or install marked failed)
# spams ``tirith spawn failed: [WinError 2]...`` once per terminal command,
# easily filling errors.log with hundreds of identical lines.
_warned_messages: set[str] = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, message: str, *args) -> None:
    """``logger.warning`` but at-most-once per ``key`` for the process
    lifetime. Used to avoid drowning the log when a fail-open tirith
    misconfiguration fires on every command."""
    with _warned_lock:
        if key in _warned_messages:
            return
        _warned_messages.add(key)
    logger.warning(message, *args)


def _reset_spawn_warning_state() -> None:
    """Clear the warn-once dedupe set. Called when tirith is freshly
    (re)installed so a subsequent failure surfaces again — e.g. user
    deletes the binary mid-session.
    """
    with _warned_lock:
        _warned_messages.clear()

# Disk-persistent failure marker — avoids retry across process restarts
_MARKER_TTL = 86400  # 24 hours


def _get_hermes_home() -> str:
    """Return the Hermes home directory, respecting HERMES_HOME env var."""
    return str(get_hermes_home())


def _failure_marker_path() -> str:
    """Return the path to the install-failure marker file."""
    return os.path.join(_get_hermes_home(), ".tirith-install-failed")


def _read_failure_reason() -> str | None:
    """Read the failure reason from the disk marker.

    Returns the reason string, or None if the marker doesn't exist or is
    older than _MARKER_TTL.
    """
    try:
        p = _failure_marker_path()
        mtime = os.path.getmtime(p)
        if (time.time() - mtime) >= _MARKER_TTL:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _is_install_failed_on_disk() -> bool:
    """Check if a recent install failure was persisted to disk.

    Returns False (allowing retry) when:
    - No marker exists
    - Marker is older than _MARKER_TTL (24h)
    - Marker reason is 'cosign_missing' and cosign is now on PATH
    """
    reason = _read_failure_reason()
    if reason is None:
        return False
    if reason == "cosign_missing" and shutil.which("cosign"):
        _clear_install_failed()
        return False
    return True


def _mark_install_failed(reason: str = ""):
    """Persist install failure to disk to avoid retry on next process.

    Args:
        reason: Short tag identifying the failure cause. Use "cosign_missing"
                when cosign is not on PATH so the marker can be auto-cleared
                once cosign becomes available.
    """
    try:
        p = _failure_marker_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(reason)
    except OSError:
        pass


def _clear_install_failed():
    """Remove the failure marker after successful install."""
    # Reset the warn-once dedupe set so a subsequent failure (e.g. user
    # deletes the binary) surfaces in the log again instead of being
    # silently suppressed by a stale dedupe key from before the fix.
    _reset_spawn_warning_state()
    try:
        os.unlink(_failure_marker_path())
    except OSError:
        pass


def _hermes_bin_dir() -> str:
    """Return $HERMES_HOME/bin, creating it if needed."""
    d = os.path.join(_get_hermes_home(), "bin")
    os.makedirs(d, exist_ok=True)
    return d


def _detect_target() -> str | None:
    """Return the Rust target triple for the current platform, or None.

    Windows is intentionally unsupported — tirith does not ship a Windows
    build. Callers should treat `None` as "this platform will never have
    tirith" and silently fall back to pattern-matching guards.
    """
    system = platform.system()
    machine = platform.machine().lower()

    # Android (Termux) is ABI-compatible with Linux — reuse Linux binaries.
    if system == "Darwin":
        plat = "apple-darwin"
    elif system in {"Linux", "Android"}:
        plat = "unknown-linux-gnu"
    else:
        return None

    if machine in {"x86_64", "amd64"}:
        arch = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        arch = "aarch64"
    else:
        return None

    return f"{arch}-{plat}"


def is_platform_supported() -> bool:
    """True when tirith ships a prebuilt binary for this OS+arch.

    Used by callers (CLI banner, etc.) to distinguish "tirith failed to
    install" from "tirith was never going to install here" — the latter
    is silent because there is nothing the user can do about it.
    """
    return _detect_target() is not None


def _download_file(url: str, dest: str, timeout: int = 10):
    """Download a URL to a local file."""
    req = urllib.request.Request(url)
    from agent.secret_scope import get_secret
    token = get_secret("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _verify_cosign(checksums_path: str, sig_path: str, cert_path: str) -> bool | None:
    """Verify cosign provenance signature on checksums.txt.

    Returns:
        True  — cosign verified successfully
        False — cosign found but verification failed
        None  — cosign not available (not on PATH, or execution failed)

    The caller treats both False and None as "abort auto-install" — only
    True allows the install to proceed.
    """
    cosign = shutil.which("cosign")
    if not cosign:
        logger.info("cosign not found on PATH")
        return None

    try:
        result = subprocess.run(
            [cosign, "verify-blob",
             "--certificate", cert_path,
             "--signature", sig_path,
             "--certificate-identity-regexp", _COSIGN_IDENTITY_REGEXP,
             "--certificate-oidc-issuer", _COSIGN_ISSUER,
             checksums_path],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            logger.info("cosign provenance verification passed")
            return True
        else:
            logger.warning("cosign verification failed (exit %d): %s",
                          result.returncode, result.stderr.strip())
            return False
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("cosign execution failed: %s", exc)
        return None


def _verify_checksum(archive_path: str, checksums_path: str, archive_name: str) -> bool:
    """Verify SHA-256 of the archive against checksums.txt."""
    expected = None
    with open(checksums_path, encoding="utf-8") as f:
        for line in f:
            # Format: "<hash>  <filename>"
            parts = line.strip().split("  ", 1)
            if len(parts) == 2 and parts[1] == archive_name:
                expected = parts[0]
                break
    if not expected:
        logger.warning("No checksum entry for %s", archive_name)
        return False

    sha = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected:
        logger.warning("Checksum mismatch: expected %s, got %s", expected, actual)
        return False
    return True


def _extract_tirith_binary(tar: tarfile.TarFile, dest_dir: str, log) -> tuple[str | None, str]:
    """Extract the tirith binary from a release archive into dest_dir."""
    for member in tar.getmembers():
        if member.name == "tirith" or member.name.endswith("/tirith"):
            if ".." in member.name:
                continue
            if not member.isfile():
                log("tirith archive member is not a regular file: %s", member.name)
                return None, "binary_not_regular_file"
            src_file = tar.extractfile(member)
            if src_file is None:
                log("tirith binary could not be read from archive")
                return None, "binary_extract_failed"

            dest_path = os.path.join(dest_dir, "tirith")
            try:
                with open(dest_path, "wb") as out:
                    shutil.copyfileobj(src_file, out)
            finally:
                src_file.close()
            return dest_path, ""

    log("tirith binary not found in archive")
    return None, "binary_not_in_archive"


def _install_tirith(*, log_failures: bool = True) -> tuple[str | None, str]:
    """Download and install tirith to $HERMES_HOME/bin/tirith.

    Verifies provenance via cosign and SHA-256 checksum.
    Returns (installed_path, failure_reason).  On success failure_reason is "".
    failure_reason is a short tag used by the disk marker to decide if the
    failure is retryable (e.g. "cosign_missing" clears when cosign appears).
    """
    log = logger.warning if log_failures else logger.debug

    target = _detect_target()
    if not target:
        logger.info("tirith auto-install: unsupported platform %s/%s",
                     platform.system(), platform.machine())
        return None, "unsupported_platform"

    archive_name = f"tirith-{target}.tar.gz"
    base_url = f"https://github.com/{_REPO}/releases/latest/download"

    try:
        tmpdir = tempfile.mkdtemp(prefix="tirith-install-")
    except OSError as exc:
        log("tirith install failed: cannot create temp dir: %s", exc)
        return None, "no_space"
    try:
        archive_path = os.path.join(tmpdir, archive_name)
        checksums_path = os.path.join(tmpdir, "checksums.txt")
        sig_path = os.path.join(tmpdir, "checksums.txt.sig")
        cert_path = os.path.join(tmpdir, "checksums.txt.pem")

        logger.info("tirith not found — downloading latest release for %s...", target)

        try:
            _download_file(f"{base_url}/{archive_name}", archive_path)
            _download_file(f"{base_url}/checksums.txt", checksums_path)
        except Exception as exc:
            log("tirith download failed: %s", exc)
            return None, "download_failed"

        # Cosign provenance verification — preferred but not mandatory.
        # When cosign is available, we verify that the release was produced
        # by the expected GitHub Actions workflow (full supply chain proof).
        # Without cosign, SHA-256 checksum + HTTPS still provides integrity
        # and transport-level authenticity.
        cosign_verified = False
        if shutil.which("cosign"):
            try:
                _download_file(f"{base_url}/checksums.txt.sig", sig_path)
                _download_file(f"{base_url}/checksums.txt.pem", cert_path)
            except Exception as exc:
                logger.info("cosign artifacts unavailable (%s), proceeding with SHA-256 only", exc)
            else:
                cosign_result = _verify_cosign(checksums_path, sig_path, cert_path)
                if cosign_result is True:
                    cosign_verified = True
                elif cosign_result is False:
                    # Verification explicitly rejected — abort, the release
                    # may have been tampered with.
                    log("tirith install aborted: cosign provenance verification failed")
                    return None, "cosign_verification_failed"
                else:
                    # None = execution failure (timeout/OSError) — proceed
                    # with SHA-256 only since cosign itself is broken.
                    logger.info("cosign execution failed, proceeding with SHA-256 only")
        else:
            logger.info("cosign not on PATH — installing tirith with SHA-256 verification only "
                        "(install cosign for full supply chain verification)")

        if not _verify_checksum(archive_path, checksums_path, archive_name):
            return None, "checksum_failed"

        with tarfile.open(archive_path, "r:gz") as tar:
            src, reason = _extract_tirith_binary(tar, tmpdir, log)
            if src is None:
                return None, reason

        dest = os.path.join(_hermes_bin_dir(), "tirith")
        try:
            shutil.move(src, dest)
        except OSError:
            # Cross-device move (common in Docker, NFS): shutil.move() falls
            # back to copy2 + unlink, but copy2's metadata step can raise
            # PermissionError.  Use plain copy + manual chmod instead.
            try:
                shutil.copy(src, dest)
            except OSError:
                # Clean up partial dest to prevent a non-executable retry loop
                try:
                    os.unlink(dest)
                except OSError:
                    pass
                return None, "cross_device_copy_failed"
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        verification = "cosign + SHA-256" if cosign_verified else "SHA-256 only"
        logger.info("tirith installed to %s (%s)", dest, verification)
        return dest, ""

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _is_explicit_path(configured_path: str) -> bool:
    """Return True if the user explicitly configured a non-default tirith path."""
    return configured_path != "tirith"


def _resolve_tirith_path(configured_path: str, *, allow_download: bool = True) -> str:
    """Resolve the tirith binary path, auto-installing if necessary.

    If the user explicitly set a path (anything other than the bare "tirith"
    default), that path is authoritative — we never fall through to
    auto-download a different binary.

    For the default "tirith":
    1. PATH lookup via shutil.which
    2. $HERMES_HOME/bin/tirith (previously auto-installed)
    3. Auto-install from GitHub releases → $HERMES_HOME/bin/tirith

    ``allow_download=False`` stops at step 2. Gateway and cron turns pass it
    so an unattended session never pulls an unpinned binary off the network
    mid-command; they fail closed instead. It deliberately leaves the cached
    sentinels alone, so declining the network here cannot suppress a later
    local install attempt in the same process.

    Failed installs are cached for the process lifetime (and persisted to
    disk for 24h) to avoid repeated network attempts.
    """
    global _resolved_path, _install_failure_reason

    # Fast path: successfully resolved on a previous call.
    if _resolved_path is not None and _resolved_path is not _INSTALL_FAILED:
        return _resolved_path

    expanded = os.path.expanduser(configured_path)
    explicit = _is_explicit_path(configured_path)
    install_failed = _resolved_path is _INSTALL_FAILED

    # Platform has no tirith build (Windows etc.). Cache the verdict and
    # return the unexpanded configured path — the spawn loop will fail-open
    # via the dedupe'd OSError handler, but only after the first call; on
    # subsequent calls the fast-path above short-circuits before spawning.
    if not explicit and not is_platform_supported():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "unsupported_platform"
        return expanded

    # Explicit path: check it and stop. Never auto-download a replacement.
    if explicit:
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            _resolved_path = expanded
            return expanded
        # Also try shutil.which in case it's a bare name on PATH
        found = shutil.which(expanded)
        if found:
            _resolved_path = found
            return found
        logger.warning("Configured tirith path %r not found; scanning disabled", configured_path)
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "explicit_path_missing"
        return expanded

    # Default "tirith" — always re-run cheap local checks so a manual
    # install is picked up even after a previous network failure (P2 fix:
    # long-lived gateway/CLI recovers without restart).
    found = shutil.which("tirith")
    if found:
        _resolved_path = found
        _install_failure_reason = ""
        _clear_install_failed()
        return found

    hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
    if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        _clear_install_failed()
        return hermes_bin

    # Caller declined the network (gateway/cron). Return the configured name
    # so the spawn fails and the surface's fail policy decides, without
    # caching a sentinel that would suppress a later local install.
    if not allow_download:
        _warn_once(
            "tirith_download_declined",
            "tirith is not installed and this surface does not auto-download; "
            "scans will fail closed until the owner installs a pinned build",
        )
        return expanded

    # Local checks failed.  If a previous install attempt already failed,
    # skip the network retry — UNLESS the failure was "cosign_missing" and
    # cosign is now available (retryable cause resolved in-process).
    if install_failed:
        if _install_failure_reason == "cosign_missing" and shutil.which("cosign"):
            # Retryable cause resolved — clear sentinel and fall through to retry
            _resolved_path = None
            _install_failure_reason = ""
            _clear_install_failed()
            install_failed = False
        else:
            return expanded

    # If a background install thread is running, don't start a parallel one —
    # return the configured path; the OSError handler in check_command_security
    # will apply fail_open until the thread finishes.
    if _install_thread is not None and _install_thread.is_alive():
        return expanded

    # Check disk failure marker before attempting network download.
    # Preserve the marker's real reason so in-memory retry logic can
    # detect retryable causes (e.g. cosign_missing) without restart.
    disk_reason = _read_failure_reason()
    if disk_reason is not None and _is_install_failed_on_disk():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = disk_reason
        return expanded

    installed, reason = _install_tirith()
    if installed:
        _resolved_path = installed
        _install_failure_reason = ""
        _clear_install_failed()
        return installed

    # Install failed — cache the miss and persist reason to disk
    _resolved_path = _INSTALL_FAILED
    _install_failure_reason = reason
    _mark_install_failed(reason)
    return expanded


def _background_install(*, log_failures: bool = True):
    """Background thread target: download and install tirith."""
    global _resolved_path, _install_failure_reason
    with _install_lock:
        # Double-check after acquiring lock (another thread may have resolved)
        if _resolved_path is not None:
            return

        # Re-check local paths (may have been installed by another process)
        found = shutil.which("tirith")
        if found:
            _resolved_path = found
            _install_failure_reason = ""
            return

        hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
        if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
            _resolved_path = hermes_bin
            _install_failure_reason = ""
            return

        installed, reason = _install_tirith(log_failures=log_failures)
        if installed:
            _resolved_path = installed
            _install_failure_reason = ""
            _clear_install_failed()
        else:
            _resolved_path = _INSTALL_FAILED
            _install_failure_reason = reason
            _mark_install_failed(reason)


def ensure_installed(*, log_failures: bool = True, allow_download: bool = True):
    """Ensure tirith is available, downloading in background if needed.

    Quick PATH/local checks are synchronous; network download runs in a
    daemon thread so startup never blocks. Safe to call multiple times.
    Returns the resolved path immediately if available, or None.

    ``allow_download=False`` reduces this to the local checks. The gateway
    passes it: every turn that process serves is a remote surface, and those
    fail closed rather than run against a binary fetched from
    ``releases/latest`` at startup.
    """
    global _resolved_path, _install_thread, _install_failure_reason

    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return None

    # Already resolved from a previous call
    if _resolved_path is not None and _resolved_path is not _INSTALL_FAILED:
        path = _resolved_path
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

    # Platform has no tirith build (e.g. Windows) — don't probe PATH,
    # don't start a download thread, don't write a disk failure marker.
    # Pattern-matching guards still run; this path stays silent.
    if not is_platform_supported():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "unsupported_platform"
        return None

    configured_path = cfg["tirith_path"]
    explicit = _is_explicit_path(configured_path)
    expanded = os.path.expanduser(configured_path)

    # Explicit path: synchronous check only, no download
    if explicit:
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            _resolved_path = expanded
            return expanded
        found = shutil.which(expanded)
        if found:
            _resolved_path = found
            return found
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "explicit_path_missing"
        return None

    # Default "tirith" — quick local checks first (no network)
    found = shutil.which("tirith")
    if found:
        _resolved_path = found
        _install_failure_reason = ""
        _clear_install_failed()
        return found

    hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
    if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        _clear_install_failed()
        return hermes_bin

    # Caller declined the network — local checks are all there is. Leave the
    # sentinels untouched (see _resolve_tirith_path) so this is a refusal to
    # download, not a recorded install failure.
    if not allow_download:
        return None

    # If previously failed in-memory, check if the cause is now resolved
    if _resolved_path is _INSTALL_FAILED:
        if _install_failure_reason == "cosign_missing" and shutil.which("cosign"):
            _resolved_path = None
            _install_failure_reason = ""
            _clear_install_failed()
        else:
            return None

    # Check disk failure marker (skip network attempt for 24h, unless
    # the cosign_missing reason was resolved — handled by _is_install_failed_on_disk).
    # Preserve the marker's real reason for in-memory retry logic.
    disk_reason = _read_failure_reason()
    if disk_reason is not None and _is_install_failed_on_disk():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = disk_reason
        return None

    # Need to download — launch background thread so startup doesn't block
    if _install_thread is None or not _install_thread.is_alive():
        _install_thread = threading.Thread(
            target=_background_install,
            kwargs={"log_failures": log_failures},
            daemon=True,
        )
        _install_thread.start()

    return None  # Not available yet; commands will fail-open until ready


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

_MAX_FINDINGS = 50
_MAX_SUMMARY_LEN = 500


def check_command_security(command: str) -> dict:
    """Run tirith security scan on a command.

    Exit code determines action (0=allow, 1=block, 2=warn). JSON enriches
    findings/summary; the only verdict override is the .app lookalike_tld
    false-positive suppression documented inline below. Spawn failures and
    timeouts respect the fail-open
    policy of the calling surface (see the module docstring). Programming
    errors propagate.

    Returns:
        {"action": "allow"|"warn"|"block", "findings": [...], "summary": str}

    A fail-closed verdict additionally carries ``fail_closed: True`` and a
    ``reason_code`` so callers can tell "the scanner found something" from
    "the scanner never ran".
    """
    global _crash_count, _circuit_open

    cfg = _load_security_config()

    if not cfg["tirith_enabled"]:
        return {"action": "allow", "findings": [], "summary": ""}

    surface = _current_surface()
    remote = surface in _REMOTE_SURFACES
    fail_open = _effective_fail_open(cfg, surface)

    # Circuit breaker: if tirith has crashed _CRASH_LIMIT times in a row,
    # stop trying for the rest of the process.  Without this, a corrupted
    # or missing binary causes every tool call to hit the same spawn failure
    # → fail-open → agent retry loop, hanging the user for 20+ minutes
    # (issue #41400).  An open breaker still means "unscanned", so on a
    # fail-closed surface it denies instead of waving the command through.
    if _circuit_open:
        return _unavailable_verdict(
            "tirith_unavailable_circuit_open", command,
            fail_open=fail_open, remote=remote,
            summary="tirith disabled (circuit breaker)",
        )

    # Unsupported platform (Windows etc.) — tirith has no binary here and
    # never will. Skip the resolver entirely so we don't even try to spawn.
    # Pattern-matching guards still run via the rest of approval.py. This
    # stays allow on every surface: fail-closed here would brick a Windows
    # gateway permanently, with no install the owner could perform to fix it.
    if not is_platform_supported():
        return {"action": "allow", "findings": [], "summary": ""}

    tirith_path = _resolve_tirith_path(cfg["tirith_path"], allow_download=not remote)
    timeout = cfg["tirith_timeout"]

    if tirith_path is None:
        _warn_once(
            "tirith_path_none",
            "tirith path resolved to None; scanning disabled",
        )
        return _unavailable_verdict(
            "tirith_unavailable_path", command,
            fail_open=fail_open, remote=remote,
            summary="tirith path unavailable",
        )

    try:
        result = subprocess.run(
            [tirith_path, "check", "--json", "--non-interactive",
             "--shell", "posix", "--", command],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        # Covers FileNotFoundError, PermissionError, exec format error.
        # Dedupe by ``(errno, exc class)`` so a transient failure mode
        # surfaces once but doesn't drown the log on every command —
        # commonly seen on Windows when the configured path "tirith"
        # isn't on PATH yet (background install still running, or
        # install marked failed for the day).
        spawn_key = f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}"
        _warn_once(spawn_key, "tirith spawn failed: %s", exc)
        _record_tirith_crash()
        return _unavailable_verdict(
            "tirith_unavailable_spawn_failed", command,
            fail_open=fail_open, remote=remote,
            summary=f"tirith unavailable: {exc}",
            closed_summary=f"tirith spawn failed (fail-closed): {exc}",
        )
    except subprocess.TimeoutExpired:
        _warn_once(
            f"tirith_timeout:{timeout}",
            "tirith timed out after %ds",
            timeout,
        )
        _record_tirith_crash()
        return _unavailable_verdict(
            "tirith_unavailable_timeout", command,
            fail_open=fail_open, remote=remote,
            summary=f"tirith timed out ({timeout}s)",
            closed_summary="tirith timed out (fail-closed)",
        )

    # Map exit code to action
    exit_code = result.returncode
    if exit_code == 0:
        action = "allow"
        # Successful execution — reset circuit breaker
        _crash_count = 0
    elif exit_code == 1:
        action = "block"
    elif exit_code == 2:
        action = "warn"
    else:
        # Unknown exit code (includes signal-killed processes like -11/SIGSEGV)
        # — respect fail_open
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        _record_tirith_crash()
        return _unavailable_verdict(
            "tirith_unavailable_exit_code", command,
            fail_open=fail_open, remote=remote,
            summary=f"tirith exit code {exit_code} (fail-open)",
            closed_summary=f"tirith exit code {exit_code} (fail-closed)",
        )

    # Parse JSON for enrichment. Findings never escalate the exit code
    # verdict; the sole downgrade is the .app lookalike_tld suppression below.
    findings = []
    raw_findings = []
    summary = ""
    try:
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            raw_findings = []
        findings = raw_findings[:_MAX_FINDINGS]
        summary = (data.get("summary", "") or "")[:_MAX_SUMMARY_LEN]
    except (json.JSONDecodeError, AttributeError):
        # JSON parse failure degrades findings/summary, not the verdict
        logger.debug("tirith JSON parse failed, using exit code only")
        if action == "block":
            summary = "security issue detected (details unavailable)"
        elif action == "warn":
            summary = "security warning detected (details unavailable)"

    # Suppress warn/block verdicts that consist solely of lookalike_tld
    # findings for the .app TLD.  .app is a legitimate gTLD used by many
    # production services (api.bsky.app, etc.) and the "can be confused with
    # file extensions" heuristic generates false positives for normal API
    # calls.  Any other finding (including other lookalike_tld entries for
    # non-.app TLDs) preserves the original action.  The check runs over the
    # full untruncated finding list so a non-.app finding past the
    # _MAX_FINDINGS cap still preserves the verdict; a JSON parse failure
    # leaves raw_findings empty, so detail-less blocks are never downgraded.
    # A block is additionally downgraded only when every finding carries a
    # LOW/MEDIUM severity: tirith can escalate a Warn to Block via policy
    # override or correlation, and that escalation may not be represented as
    # a finding, so a HIGH/CRITICAL or severity-less block is never touched.
    if action in ("warn", "block") and raw_findings:
        non_suppressible = [f for f in raw_findings if not _is_app_tld_finding(f)]
        if not non_suppressible and (
            action == "warn" or _all_findings_low_or_medium(raw_findings)
        ):
            action = "allow"
            findings = []
            summary = ""

    return {"action": action, "findings": findings, "summary": summary}


_APP_TLD_QUOTED_RE = re.compile(r"'\.app'", re.IGNORECASE)
_SUPPRESSIBLE_SEVERITIES = frozenset({"low", "medium"})


def _all_findings_low_or_medium(findings: list) -> bool:
    """True when every finding carries an explicit LOW or MEDIUM severity."""
    for f in findings:
        sev = str(f.get("severity", "")).lower() if isinstance(f, dict) else ""
        if sev not in _SUPPRESSIBLE_SEVERITIES:
            return False
    return True


def _is_app_tld_finding(finding: dict) -> bool:
    """Return True if this finding is a lookalike_tld warning for the .app TLD only.

    Matches conservatively so a suppression miss fails safe: ``value``/``tld``
    must equal ``.app`` exactly, and free-text fields must contain the quoted
    ``'.app'`` form tirith's real schema uses ("Domain uses '.app' TLD ...").
    A bare ``.app`` substring (e.g. an upstream wording change listing several
    TLDs, or ``.application``) does NOT match.
    """
    if not isinstance(finding, dict):
        return False
    if finding.get("rule_id") != "lookalike_tld":
        return False
    for field in ("value", "tld"):
        val = finding.get(field)
        if val is not None and str(val).strip().lower() == ".app":
            return True
    for field in ("detail", "description", "message"):
        val = finding.get(field)
        if val is not None and _APP_TLD_QUOTED_RE.search(str(val)):
            return True
    return False
