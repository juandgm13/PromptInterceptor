"""
Bash command wrapper for Windows compatibility.

Bash wrapping is applied automatically when running on Windows. Every Bash
tool call command is wrapped as ``bash -c "<command>"`` before forwarding to
the client, so AI clients can execute Unix commands on Windows as long as
bash is available (Git Bash, WSL, Cygwin, etc.).

PowerShell cmdlets (Verb-Noun pattern, e.g. Get-ChildItem, get-childitem) are
detected and left unchanged so they continue to work natively.
"""

import json
import re
import sys
from typing import Tuple

# PowerShell approved verbs (case-insensitive). Commands whose first token is
# <verb>-<anything> with a verb in this set are treated as PS cmdlets and not wrapped.
_PS_VERBS = frozenset({
    'add', 'clear', 'close', 'compress', 'complete', 'confirm', 'connect',
    'convert', 'convertfrom', 'convertto', 'copy', 'debug', 'deny',
    'disable', 'dismount', 'edit', 'enable', 'enter', 'exit', 'expand',
    'export', 'find', 'format', 'get', 'group', 'hide', 'import',
    'initialize', 'install', 'invoke', 'join', 'limit', 'lock', 'measure',
    'mount', 'move', 'new', 'open', 'optimize', 'out', 'pop', 'protect',
    'push', 'read', 'receive', 'redo', 'register', 'remove', 'rename',
    'repair', 'request', 'reset', 'resize', 'resolve', 'restart', 'restore',
    'save', 'search', 'select', 'send', 'set', 'show', 'skip', 'sort',
    'split', 'start', 'stop', 'submit', 'suspend', 'switch', 'sync',
    'test', 'trace', 'undo', 'unlock', 'uninstall', 'unprotect',
    'unregister', 'update', 'use', 'wait', 'watch', 'write',
})


# Compiled regexes for Windows-specific bash fixes
_WIN_PATH_DQ_RE = re.compile(r'"([A-Za-z]:\\{1,2}[^"]*)"')
_WIN_PATH_SQ_RE = re.compile(r"'([A-Za-z]:\\{1,2}[^']*)'")
_WIN_PATH_RE    = re.compile(r'[A-Za-z]:\\{1,2}[^\s"\'&|;<>]*')
_NULL_REDIR_RE  = re.compile(r'([\d&]?>>?)\s*/?nul\b', re.IGNORECASE)


def _norm_win_path(s: str) -> str:
    return s.replace('\\\\', '/').replace('\\', '/')


def _fix_windows_command(command: str) -> str:
    """Fix Windows-specific patterns so they work in bash (Git Bash / WSL).

    - Backslash paths:  C:\\foo\\bar  →  C:/foo/bar  (quoted and unquoted)
    - Windows null device:  2>/nul  →  2>/dev/null
    """
    # Double-quoted paths first so spaces inside quotes are preserved
    command = _WIN_PATH_DQ_RE.sub(lambda m: '"' + _norm_win_path(m.group(1)) + '"', command)
    command = _WIN_PATH_SQ_RE.sub(lambda m: "'" + _norm_win_path(m.group(1)) + "'", command)
    # Unquoted paths (stop at whitespace / shell metacharacters)
    command = _WIN_PATH_RE.sub(lambda m: _norm_win_path(m.group(0)), command)
    # Null device redirect
    command = _NULL_REDIR_RE.sub(r'\1/dev/null', command)
    return command


def _is_powershell_command(command: str) -> bool:
    """Return True if the command looks like a PowerShell cmdlet (Verb-Noun)."""
    first_token = command.strip().split()[0] if command.strip() else ''
    if '-' not in first_token:
        return False
    verb = first_token.split('-')[0].lower()
    return verb in _PS_VERBS


def is_windows_bash_mode() -> bool:
    """Return True when running on Windows (sys.platform == 'win32')."""
    return sys.platform == 'win32'


def wrap_bash_command(command: str) -> str:
    """Wrap a Unix command so it runs via ``bash -c "..."``.

    Returns the command unchanged if it is already wrapped, empty, or looks
    like a PowerShell cmdlet (Verb-Noun pattern).

    Windows-specific fixes (path backslashes, /nul redirects) are applied
    before wrapping, and also to already-wrapped ``bash -c "..."`` commands
    so that inner paths are corrected regardless.
    """
    if not command:
        return command
    if _is_powershell_command(command):
        return command
    command = _fix_windows_command(command)
    if command.startswith('bash '):
        return command
    # Escape backslashes first, then double-quotes, so the shell sees them correctly
    escaped = command.replace('\\', '\\\\').replace('"', '\\"')
    return f'bash -c "{escaped}"'


def patch_anthropic_body(body: dict) -> Tuple[dict, bool]:
    """
    Wrap Bash tool_use command inputs in an Anthropic /v1/messages body.
    Returns (patched_body, was_changed).
    """
    content = body.get('content')
    if not isinstance(content, list):
        return body, False

    new_content = []
    changed = False
    for block in content:
        if block.get('type') == 'tool_use' and block.get('name', '').lower() == 'bash':
            inp = dict(block.get('input') or {})
            cmd = inp.get('command', '')
            wrapped = wrap_bash_command(cmd)
            if wrapped != cmd:
                inp['command'] = wrapped
                block = {**block, 'input': inp}
                changed = True
        new_content.append(block)

    if not changed:
        return body, False
    return {**body, 'content': new_content}, True


def patch_openai_body(body: dict) -> Tuple[dict, bool]:
    """
    Wrap Bash tool_call command inputs in an OpenAI /v1/chat/completions body.
    Returns (patched_body, was_changed).
    """
    choices = body.get('choices')
    if not isinstance(choices, list) or not choices:
        return body, False
    msg = choices[0].get('message', {})
    tool_calls = msg.get('tool_calls')
    if not isinstance(tool_calls, list):
        return body, False

    new_tcs = []
    changed = False
    for tc in tool_calls:
        fn = tc.get('function', {})
        if fn.get('name', '').lower() == 'bash':
            args_str = fn.get('arguments', '{}')
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                cmd = args.get('command', '')
                wrapped = wrap_bash_command(cmd)
                if wrapped != cmd:
                    args['command'] = wrapped
                    tc = {**tc, 'function': {**fn, 'arguments': json.dumps(args)}}
                    changed = True
            except (json.JSONDecodeError, ValueError):
                pass
        new_tcs.append(tc)

    if not changed:
        return body, False
    new_msg = {**msg, 'tool_calls': new_tcs}
    return {**body, 'choices': [{**choices[0], 'message': new_msg}]}, True
