"""Refuse agent-driven changes to the harness's own safety settings (H-018).

The approval mode, the denial breaker, the Tirith scanner, the toolset surface
and the sandbox/egress boundaries all live in profile configuration. An agent
turn that can rewrite them can permanently widen every gate it will meet on the
*next* turn, so on the surfaces where nobody is watching the tool call happen
(Discord/Slack/API and cron) those writes are refused outright rather than sent
through approval.

Two independent boundaries implement that refusal:

* ``find_safety_config_change_in_command()`` inspects a ``terminal`` command
  before it runs, so ``hermes config set security.tirith_enabled false`` never
  reaches a shell. Wrapper forms (``env FOO=1 hermes ...``, ``command hermes``,
  ``bash -lc '...'``, ``$(...)``) are resolved first.
* ``assert_config_key_writable()`` / ``assert_config_save_allowed()`` guard the
  in-process config writers in ``hermes_cli.config`` for anything that reaches
  them without going through a shell.

The command check is the primary boundary: it runs before a shell exists and
resolves wrapper/substitution/interpreter forms, so it catches the change even
when the in-process guard cannot see it. The in-process guard is a real
backstop rather than a no-op, though: when the terminal tool shells out on a
gateway/cron turn, ``tools/environments/local.py:_inject_session_context_env``
bridges the session's ``HERMES_SESSION_PLATFORM`` (and ``HERMES_CRON_SESSION``)
onto the child's environment from ``gateway.session_context._VAR_MAP``. A
nested ``hermes`` CLI in that child therefore re-derives ``surface ==
"gateway"`` (or ``"cron"``) through ``get_current_tool_surface``'s
``os.environ`` fallback and ``assert_config_key_writable`` fires there too. The
in-process guard also covers tool code that reaches the config writers inside
the agent process without a shell.

Nothing here exposes a runtime toggle: an "allow safety config changes" setting
would itself be the first thing an agent flipped. The owner changes these
values from outside the harness (a local shell, or whatever declaratively
manages the profile), and ``PROTECTED_SAFETY_CONFIG_PATTERNS`` covers the
``security.*`` subtree, so any future switch added there is protected too.
"""

from __future__ import annotations

import os
import re
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Iterator, Optional, Sequence

__all__ = [
    "PROTECTED_SAFETY_CONFIG_PATTERNS",
    "SAFETY_CONFIG_DENIED_REASON_CODE",
    "SAFETY_CONFIG_EFFECT_CLASS",
    "SAFETY_CONFIG_SAFE_ALTERNATIVE",
    "SafetyConfigChangeDenied",
    "assert_config_key_writable",
    "assert_config_save_allowed",
    "changed_protected_config_keys",
    "current_guarded_surface",
    "find_safety_config_change_in_command",
    "matched_protected_config_key",
    "owner_initiated_safety_config_change",
    "safety_config_denial_payload",
]


# ---------------------------------------------------------------------------
# What counts as a safety setting
# ---------------------------------------------------------------------------

# Dotted config paths, matched segment-wise. A pattern matches a key when the
# two lie on the same path — the key is the pattern, sits under it (``approvals``
# vs ``approvals.mode``), or *contains* it (``agent`` vs
# ``agent.disabled_toolsets``, which a whole-section overwrite would clobber).
# ``*`` matches exactly one segment.
PROTECTED_SAFETY_CONFIG_PATTERNS: frozenset = frozenset({
    # Approval gate: mode, cron mode, the Smart Approval policy, the denial
    # circuit breaker threshold, and the standing deny list.
    "approvals",
    # The permanent approval allowlist — an entry here skips the gate forever.
    "command_allowlist",
    # Legacy whole-session approval bypass.
    "yolo",

    # Scanner / redaction / instruction-file protection, including every
    # security.tirith_* key (enabled, path, timeout, fail_open,
    # fail_open_gateway) and any future switch added to this subtree.
    "security",
    "privacy.redact_pii",

    # Which tools the agent is handed, globally and per platform. This is the
    # "active-profile tool surface" of H-018.
    "toolsets",
    "platform_toolsets",
    "agent.disabled_toolsets",
    "code_execution.mode",

    # Code the harness runs on the agent's behalf.
    "hooks",
    "hooks_auto_accept",
    "skills.inline_shell",
    "skills.guard_agent_created",
    "skills.write_approval",
    "skills.external_dirs",
    "memory.write_approval",
    "delegation.subagent_auto_approve",

    # Where commands actually execute, and what they can reach from there.
    "terminal.backend",
    "terminal.home_mode",
    "terminal.env_passthrough",
    "terminal.shell_init_files",
    "terminal.auto_source_bashrc",
    "terminal.docker_forward_env",
    "terminal.docker_volumes",
    "terminal.docker_mount_cwd_to_workspace",
    "terminal.docker_network",
    "terminal.docker_extra_args",
    "terminal.docker_run_as_host_user",

    # Network egress control.
    "proxy",
    "browser.allow_private_urls",
    "browser.allow_unsafe_evaluate",
    "browser.restrict_evaluate",
    "browser.dialog_policy",

    # Where credentials are fetched from — ``secrets.*.binary_path`` alone is an
    # arbitrary-executable setting.
    "secrets",

    # Trust boundaries around inbound content and delivery.
    "gateway.strict",
    "gateway.trust_recent_files",
    "gateway.trust_recent_files_seconds",
    "gateway.media_delivery_allow_dirs",
    "gateway.multiplex_profile_allowlist",
    "cron.preflight",

    # Who is allowed to command the agent, per platform.
    "platforms.*.allow_admin_from",
    "platforms.*.user_allowed_commands",
    "platforms.*.group_allow_admin_from",
    "platforms.*.group_user_allowed_commands",
    "discord.allowed_channels",
    "discord.free_response_channels",
    "discord.require_mention",
    "discord.thread_require_mention",
    "discord.bots_require_inline_mention",
    "discord.server_actions",
    "discord.dm_role_auth_guild",
    "slack.allowed_channels",
    "slack.free_response_channels",
    "slack.require_mention",
    "slack.require_mention_channels",
    "slack.thread_require_mention",
    "slack.ignore_other_user_mentions",
    "telegram.allowed_chats",
    "mattermost.allowed_channels",
    "mattermost.free_response_channels",
    "mattermost.require_mention",
    "matrix.allowed_rooms",
    "matrix.free_response_rooms",
    "matrix.require_mention",

    # Dashboard authentication material.
    "dashboard.basic_auth",
    "dashboard.drain_auth",
    "dashboard.oauth",

    # A different protection class from everything above: the keys above decide
    # whether the *next* command is checked, whereas ``updates.*`` governs the
    # owner's ability to recover and audit the harness — the self-update channel
    # and its provenance/pinning. An agent turn that could disable updates or
    # repoint them at an attacker-controlled source would blunt the owner's
    # recovery path and their forensic view of what the harness is running. No
    # code path writes ``updates.*`` (it is dashboard-driven only), so guarding
    # it here cannot break an existing owner flow.
    "updates",
})

_PROTECTED_SEGMENTS: tuple = tuple(
    (pattern, tuple(pattern.split("."))) for pattern in sorted(PROTECTED_SAFETY_CONFIG_PATTERNS)
)


SAFETY_CONFIG_DENIED_REASON_CODE = "safety_config_change_denied"
SAFETY_CONFIG_EFFECT_CLASS = "safety_control_modification"
SAFETY_CONFIG_SAFE_ALTERNATIVE = (
    "Describe the change you want and why, and let the owner apply it from "
    "outside the harness — a local shell on the host, or whatever declaratively "
    "manages this profile. Continue the task with the current settings, or stop "
    "and report that it cannot proceed under them."
)


class SafetyConfigChangeDenied(PermissionError):
    """Raised when a remote-surface caller tries to move a safety setting."""

    def __init__(self, keys: Sequence[str], surface: str, action: str = "change"):
        self.keys = tuple(keys)
        self.surface = surface
        self.action = action
        self.reason_code = SAFETY_CONFIG_DENIED_REASON_CODE
        self.effect_class = SAFETY_CONFIG_EFFECT_CLASS
        self.safe_alternative = SAFETY_CONFIG_SAFE_ALTERNATIVE
        super().__init__(
            f"Blocked: cannot {action} the safety setting(s) "
            f"{', '.join(self.keys)} from the {surface} surface. "
            f"{SAFETY_CONFIG_SAFE_ALTERNATIVE}"
        )


def matched_protected_config_key(key: Any) -> Optional[str]:
    """Return the protected pattern *key* collides with, or ``None``.

    Matching is segment-wise and symmetric: setting ``approvals.mode`` matches
    the ``approvals`` pattern, and overwriting the whole ``agent`` section
    matches ``agent.disabled_toolsets`` because that write would take the
    protected leaf with it.
    """
    if not isinstance(key, str):
        return None
    normalized = key.strip().strip(".").lower()
    if not normalized:
        return None
    key_segments = tuple(seg for seg in normalized.split(".") if seg)
    if not key_segments:
        return None
    for pattern, pattern_segments in _PROTECTED_SEGMENTS:
        span = min(len(key_segments), len(pattern_segments))
        if all(
            pattern_segments[i] == "*" or pattern_segments[i] == key_segments[i]
            for i in range(span)
        ):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Surface binding
# ---------------------------------------------------------------------------

# Set only around a config write a human explicitly asked for through an
# authenticated, non-agent path (the ``/approvals`` slash command, which already
# re-checks gateway admin at its side-effect boundary). An agent cannot reach
# this: it emits tool calls and shell commands, never platform slash events, and
# a shell command runs in a separate process where this contextvar does not
# exist.
_OWNER_INITIATED: ContextVar = ContextVar("hermes_owner_initiated_safety_config", default=False)


@contextmanager
def owner_initiated_safety_config_change() -> Iterator[None]:
    """Mark the enclosed config write as explicitly requested by the owner."""
    token = _OWNER_INITIATED.set(True)
    try:
        yield
    finally:
        _OWNER_INITIATED.reset(token)


def _remote_surface() -> Optional[str]:
    """Return the remote surface driving this call, or ``None``.

    ``None`` means either a local surface (CLI/TUI/desktop, where a human owns
    the terminal) or an owner-initiated write, and the guard stands down.
    """
    if _OWNER_INITIATED.get():
        return None
    try:
        from tools.approval import REMOTE_TOOL_SURFACES, get_current_tool_surface

        surface = get_current_tool_surface()
    except Exception:
        # Never let a guard failure turn into an unguarded write; but an
        # import failure here means the approval layer itself is unavailable,
        # in which case there is no session context to protect either.
        return None
    return surface if surface in REMOTE_TOOL_SURFACES else None


def current_guarded_surface() -> Optional[str]:
    """Public name for :func:`_remote_surface` — lets callers skip setup work."""
    return _remote_surface()


# ---------------------------------------------------------------------------
# In-process config writers
# ---------------------------------------------------------------------------

def assert_config_key_writable(key: Any, *, action: str = "change") -> None:
    """Raise :class:`SafetyConfigChangeDenied` for a protected key on a remote surface."""
    matched = matched_protected_config_key(key)
    if matched is None:
        return
    surface = _remote_surface()
    if surface is None:
        return
    raise SafetyConfigChangeDenied([matched], surface, action)


_MISSING = object()


def _lookup(config: Any, path: Sequence[str]) -> Any:
    node = config
    for segment in path:
        if not isinstance(node, dict) or segment not in node:
            return _MISSING
        node = node[segment]
    return node


def _expand_pattern(
    pattern_segments: Sequence[str], *configs: Any
) -> Iterable[tuple]:
    """Yield concrete paths for *pattern_segments*, expanding ``*`` over *configs*."""
    paths: list = [()]
    nodes: list = [tuple(configs)]
    for segment in pattern_segments:
        next_paths: list = []
        next_nodes: list = []
        for path, node_group in zip(paths, nodes):
            if segment == "*":
                names: set = set()
                for node in node_group:
                    if isinstance(node, dict):
                        names.update(str(name) for name in node)
                for name in sorted(names):
                    next_paths.append(path + (name,))
                    next_nodes.append(
                        tuple(
                            node.get(name) if isinstance(node, dict) else None
                            for node in node_group
                        )
                    )
            else:
                next_paths.append(path + (segment,))
                next_nodes.append(
                    tuple(
                        node.get(segment) if isinstance(node, dict) else None
                        for node in node_group
                    )
                )
        paths, nodes = next_paths, next_nodes
    return paths


def changed_protected_config_keys(new_config: Any, old_config: Any) -> list:
    """Return the protected keys whose value differs between the two configs.

    Both sides must already be *effective* configs (user values merged over the
    schema defaults). Comparing a merged config against a raw one would report
    every defaulted safety key as a change on the first bulk save.
    """
    changed: list = []
    for pattern, pattern_segments in _PROTECTED_SEGMENTS:
        for path in _expand_pattern(pattern_segments, new_config, old_config):
            if _lookup(new_config, path) != _lookup(old_config, path):
                changed.append(".".join(path))
    return sorted(set(changed))


def assert_config_save_allowed(new_config: Any, old_config: Any) -> None:
    """Raise when a bulk config save would move a safety setting on a remote surface.

    Only an actual change is refused: the many internal callers that load the
    config, edit one unrelated key and save the whole document keep working.
    """
    surface = _remote_surface()
    if surface is None:
        return
    changed = changed_protected_config_keys(new_config, old_config)
    if changed:
        raise SafetyConfigChangeDenied(changed, surface, "save changes to")


# ---------------------------------------------------------------------------
# Shell command analysis
# ---------------------------------------------------------------------------

# Wrappers that hand their remaining argv to another program, so `env FOO=1
# command hermes config set ...` still reads as a `hermes` invocation.
_PASSTHROUGH_WRAPPERS = frozenset({
    "command", "doas", "env", "exec", "nice", "nohup", "setsid", "stdbuf",
    "sudo", "time", "timeout",
})
# ``env -u NAME`` / ``env -C dir`` / ``nice -n 5`` consume the token after them.
_WRAPPER_FLAGS_WITH_VALUE = frozenset({"-u", "-C", "-n", "--unset", "--chdir", "--adjustment"})
_SHELL_PROGRAMS = frozenset({"ash", "bash", "dash", "fish", "ksh", "sh", "zsh"})
_SHELL_COMMAND_LONG_FLAGS = frozenset({"--command"})

_HERMES_PROGRAMS = frozenset({"hermes"})
_PYTHON_PROGRAMS_RE = re.compile(r"^python(?:\d(?:\.\d+)?)?$")

_OPERATOR_CHARS = ";|&\n"


def _read_balanced(command: str, start: int, closer: str) -> tuple:
    """Read up to the matching *closer*, honoring quotes and nesting."""
    depth = 1
    quote = None
    i = start
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "\\" and i + 1 < n:
            i += 2
            continue
        elif closer == ")" and ch == "(":
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return command[start:i], i + 1
        i += 1
    return command[start:], n


def _scan_command(command: str) -> tuple:
    """Split *command* into top-level segments plus nested substitution bodies."""
    segments: list = []
    nested: list = []
    buf: list = []
    quote = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue
        if ch == "$" and command.startswith("$(", i):
            body, i = _read_balanced(command, i + 2, ")")
            nested.append(body)
            buf.append(" ")
            continue
        if ch == "`":
            body, i = _read_balanced(command, i + 1, "`")
            nested.append(body)
            buf.append(" ")
            continue
        if ch in _OPERATOR_CHARS or ch in "()":
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return [seg for seg in segments if seg.strip()], nested


def _tokenize(segment: str) -> list:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _is_env_assignment(token: str) -> bool:
    if "=" not in token or token.startswith("="):
        return False
    name = token.split("=", 1)[0]
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _strip_wrappers(tokens: list) -> list:
    """Drop env assignments and argv-passthrough wrappers from the front."""
    tokens = list(tokens)
    guard = 0
    while tokens and guard < 32:
        guard += 1
        head = tokens[0]
        if _is_env_assignment(head):
            tokens.pop(0)
            continue
        base = os.path.basename(head).lower()
        if base not in _PASSTHROUGH_WRAPPERS:
            break
        tokens.pop(0)
        # Consume the wrapper's own flags (and their values) plus, for
        # `timeout`, the duration that follows them.
        while tokens:
            token = tokens[0]
            if token == "--":
                tokens.pop(0)
                break
            if token.startswith("-") and len(token) > 1:
                tokens.pop(0)
                if token in _WRAPPER_FLAGS_WITH_VALUE and tokens:
                    tokens.pop(0)
                continue
            break
        if base == "timeout" and tokens and re.match(r"^\d+(?:\.\d+)?[smhd]?$", tokens[0]):
            tokens.pop(0)
    return tokens


def _shell_script_argument(tokens: list) -> Optional[str]:
    """Return the script a `sh -c`-style invocation would run, if any."""
    for index in range(1, len(tokens)):
        arg = tokens[index]
        if arg.startswith("--"):
            if arg in _SHELL_COMMAND_LONG_FLAGS:
                return tokens[index + 1] if index + 1 < len(tokens) else None
            if arg.startswith("--command="):
                return arg.split("=", 1)[1]
        elif arg.startswith("-") and len(arg) > 1 and "c" in arg[1:]:
            return tokens[index + 1] if index + 1 < len(tokens) else None
    return None


def _hermes_argv(tokens: list) -> Optional[list]:
    """Return the `hermes` argv this segment invokes, or ``None``."""
    if not tokens:
        return None
    base = os.path.basename(tokens[0]).lower()
    if base in _HERMES_PROGRAMS:
        return tokens[1:]
    # `python -m hermes_cli.main config set ...`
    if _PYTHON_PROGRAMS_RE.match(base):
        for index in range(1, len(tokens)):
            if tokens[index] == "-m" and index + 1 < len(tokens):
                module = tokens[index + 1]
                if module == "hermes" or module.startswith("hermes_cli") or module.startswith("hermes."):
                    return tokens[index + 2:]
                return None
            if tokens[index].startswith("-"):
                continue
            return None
    return None


def _first_non_flag(tokens: Sequence[str], start: int = 0) -> Optional[str]:
    for token in tokens[start:]:
        if token.startswith("-"):
            continue
        return token
    return None


# Subcommands whose argv this guard inspects for a protected write.
_GUARDED_HERMES_SUBCOMMANDS = frozenset({"config", "approvals", "tools"})


def _protected_key_in_hermes_argv(argv: list) -> Optional[str]:
    """Return the protected setting a `hermes ...` argv would move, if any.

    A value-taking global flag sits between ``hermes`` and its subcommand and
    its *value* is an arbitrary token: ``hermes -m foo config set
    security.tirith_enabled false`` puts ``foo`` where the subcommand would be,
    and ``hermes -m config config set ...`` even repeats the ``config`` word as
    the model value. The previous version treated the first non-flag token as
    *the* subcommand and returned once it was not one we guard, so any such
    value token made the whole check fall through — the guard was bypassed by
    prefixing ``-m``/``--resume``/``-z``/… before ``config``.

    Rather than re-implement (and have to track) the CLI's whole global-flag
    table, scan every position for a guarded subcommand token and evaluate each
    occurrence, returning the first that names a protected write. A false
    positive here only costs the agent a little convenience on the gateway/cron
    surfaces where this runs, so over-detection is the safe direction; a value
    that merely *equals* ``config`` but heads no ``set``/``unset``/``edit`` is
    still not matched, so ordinary invocations are left alone.
    """
    if not argv:
        return None
    for index, token in enumerate(argv):
        if not isinstance(token, str):
            continue
        subcommand = token.lower()
        if subcommand not in _GUARDED_HERMES_SUBCOMMANDS:
            continue
        rest = argv[index + 1:]

        if subcommand == "config":
            action = _first_non_flag(rest)
            if action in {"set", "unset"}:
                action_index = rest.index(action)
                key = _first_non_flag(rest, action_index + 1)
                matched = matched_protected_config_key(key)
                if matched is not None:
                    return matched
            elif action == "edit":
                # Opens $EDITOR on config.yaml; with EDITOR pointing at a
                # script that is an arbitrary rewrite of every setting below.
                return "config-file-editor"
            # Not a guarded write at this position: a later token may still be
            # the real subcommand (e.g. a `config` that was a flag value), so
            # keep scanning instead of returning.
            continue

        if subcommand == "approvals":
            # `hermes approvals suggest --apply N` merges patterns into
            # command_allowlist.
            if any(arg == "--apply" or arg.startswith("--apply=") for arg in rest):
                return "command_allowlist"
            continue

        if subcommand == "tools":
            action = _first_non_flag(rest)
            if action in {"enable", "disable"}:
                return "platform_toolsets"
            continue
    return None


# Programs that take program text as data — a `hermes config set` inside one of
# these is invisible to the tokenizer above, so their presence turns on the
# textual fallback below. Shells count too: a heredoc or a piped script reaches
# them without ever appearing as an argv token.
_INTERPRETER_PROGRAMS = frozenset({
    "awk", "eval", "gawk", "lua", "node", "osascript", "perl", "php", "ruby",
    "source", "tclsh",
})

# Belt-and-braces for forms the tokenizer cannot follow — a `hermes config set`
# buried inside `python -c`, `perl -e`, an `eval` string, or a heredoc. Only
# consulted when the command actually invokes an interpreter, so an ordinary
# `git commit -m "...hermes config set..."` is not caught by it. The ordering
# requirement (hermes … config … set/unset … protected key) narrows it further.
_EMBEDDED_CONFIG_WRITE_RE = re.compile(
    r"\bhermes\b[^\n]{0,80}?\bconfig\b\s+(?:set|unset)\b\s+(?:--\S+\s+)*([A-Za-z_][\w.*-]*)",
    re.IGNORECASE,
)


def _embedded_protected_key(command: str) -> Optional[str]:
    for match in _EMBEDDED_CONFIG_WRITE_RE.finditer(command):
        matched = matched_protected_config_key(match.group(1))
        if matched is not None:
            return matched
    return None


def find_safety_config_change_in_command(command: Any, *, _depth: int = 0) -> Optional[str]:
    """Return the protected setting *command* would change, or ``None``.

    The returned name always comes from the static protected table (or a fixed
    label such as ``config-file-editor``), never from the caller's text, so it
    is safe to echo back into a tool result or a log line.
    """
    if not isinstance(command, str) or not command.strip() or _depth > 4:
        return None

    segments, nested = _scan_command(command)
    for body in nested:
        matched = find_safety_config_change_in_command(body, _depth=_depth + 1)
        if matched is not None:
            return matched

    saw_interpreter = False
    for segment in segments:
        tokens = _strip_wrappers(_tokenize(segment))
        if not tokens:
            continue
        base = os.path.basename(tokens[0]).lower()
        if base in _SHELL_PROGRAMS:
            saw_interpreter = True
            script = _shell_script_argument(tokens)
            if script:
                matched = find_safety_config_change_in_command(script, _depth=_depth + 1)
                if matched is not None:
                    return matched
            continue
        if base in _INTERPRETER_PROGRAMS or _PYTHON_PROGRAMS_RE.match(base):
            saw_interpreter = True
        argv = _hermes_argv(tokens)
        if argv is None:
            continue
        matched = _protected_key_in_hermes_argv(argv)
        if matched is not None:
            return matched

    if saw_interpreter:
        return _embedded_protected_key(command)
    return None


def safety_config_denial_payload(matched_key: str, surface: str) -> dict:
    """Build the machine-readable deny body for a refused terminal command."""
    return {
        "output": "",
        "exit_code": -1,
        "error": (
            f"Blocked: changing the harness's own safety configuration "
            f"({matched_key}) is not permitted from the {surface} surface. "
            "These settings decide which commands get checked at all, so an "
            "agent turn cannot move them."
        ),
        "status": "blocked",
        "reason_code": SAFETY_CONFIG_DENIED_REASON_CODE,
        "effect_class": SAFETY_CONFIG_EFFECT_CLASS,
        "safe_alternative": SAFETY_CONFIG_SAFE_ALTERNATIVE,
        "protected_setting": matched_key,
        "surface": surface,
    }
