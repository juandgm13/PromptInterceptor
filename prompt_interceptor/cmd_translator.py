"""
Bash → Windows CMD command translator for PromptInterceptor.

Translates Unix bash commands in LLM tool calls to their Windows CMD
equivalents before forwarding responses to the client. Activated by setting
``windows_cmd_mode: true`` in config.json.
"""

import json
import re
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Core translation helpers
# ---------------------------------------------------------------------------

def _translate_segment(cmd: str) -> str:
    """Translate one simple (non-compound) bash command to Windows CMD."""
    cmd = cmd.strip()
    if not cmd:
        return cmd

    # Strip leading 'sudo'
    cmd = re.sub(r'^sudo\s+', '', cmd)

    # Replace $VAR / ${VAR} with %VAR%
    cmd = re.sub(r'\$\{(\w+)\}', r'%\1%', cmd)
    cmd = re.sub(r'\$(\w+)', r'%\1%', cmd)

    # ls
    if re.match(r'^ls\b', cmd):
        rest = cmd[2:].strip()
        flags = re.findall(r'-(\S+)', rest)
        paths = re.sub(r'-\S+\s*', '', rest).strip()
        merged = ''.join(flags)
        show_hidden = 'a' in merged
        flag_str = ' /a' if show_hidden else ''
        return f'dir{flag_str}{" " + paths if paths else ""}'

    # pwd → cd (prints current directory in CMD)
    if re.match(r'^pwd\b', cmd):
        return 'cd'

    # clear → cls
    if re.match(r'^clear\b', cmd):
        return 'cls'

    # which → where
    if re.match(r'^which\b', cmd):
        return re.sub(r'^which\b', 'where', cmd)

    # cat → type
    if re.match(r'^cat\b', cmd):
        return re.sub(r'^cat\b', 'type', cmd)

    # touch file [file ...] → type nul > file [& type nul > file ...]
    m = re.match(r'^touch\s+(.+)$', cmd)
    if m:
        targets = m.group(1).split()
        return ' & '.join(f'type nul > {t}' for t in targets)

    # rm
    if re.match(r'^rm\b', cmd):
        rest = cmd[2:].strip()
        flags = re.findall(r'-(\S+)', rest)
        paths = re.sub(r'-\S+\s*', '', rest).strip()
        recursive = any('r' in f.lower() for f in flags)
        if recursive:
            return f'rd /s /q {paths}' if paths else 'rd /s /q'
        return f'del {paths}' if paths else 'del'

    # cp → copy (strip flags like -r, -f, -p)
    if re.match(r'^cp\b', cmd):
        rest = re.sub(r'^cp\b\s*', '', cmd)
        rest = re.sub(r'-\S+\s*', '', rest).strip()
        return f'copy {rest}'

    # mv → move
    if re.match(r'^mv\b', cmd):
        rest = re.sub(r'^mv\b\s*', '', cmd)
        rest = re.sub(r'-\S+\s*', '', rest).strip()
        return f'move {rest}'

    # mkdir -p → mkdir (CMD creates intermediate dirs with /s on older Windows; just use mkdir)
    if re.match(r'^mkdir\b', cmd):
        return re.sub(r'^mkdir\s+-p\s+', 'mkdir ', cmd)

    # grep → findstr
    if re.match(r'^grep\b', cmd):
        rest = cmd[4:].strip()
        flags = re.findall(r'-(\S+)', rest)
        args = re.sub(r'-\S+\s*', '', rest).strip()
        merged = ''.join(flags)
        findstr_flags = []
        if 'r' in merged or 'R' in merged:
            findstr_flags.append('/s')
        if 'i' in merged:
            findstr_flags.append('/i')
        if 'n' in merged:
            findstr_flags.append('/n')
        flag_str = ' '.join(findstr_flags)
        return f'findstr {flag_str} {args}'.strip() if flag_str else f'findstr {args}'

    # find . -name pattern → dir /s /b pattern
    m = re.match(r'^find\s+\S+\s+-name\s+(.+)$', cmd)
    if m:
        pattern = m.group(1).strip('"\'')
        return f'dir /s /b {pattern}'

    # export VAR=val → set VAR=val
    if re.match(r'^export\b', cmd):
        return re.sub(r'^export\s+', 'set ', cmd)

    # python3 → python, pip3 → pip
    if re.match(r'^python3\b', cmd):
        return re.sub(r'^python3\b', 'python', cmd)
    if re.match(r'^pip3\b', cmd):
        return re.sub(r'^pip3\b', 'pip', cmd)

    # chmod / chown have no CMD equivalent — comment them out so the shell doesn't error
    if re.match(r'^ch(mod|own)\b', cmd):
        return f'rem {cmd}'

    return cmd


def _split_compound(cmd: str):
    """
    Yield (operator, segment) pairs, preserving &&, ||, | and converting ; to &&.
    Respects single- and double-quoted strings so operators inside quotes are ignored.
    """
    current: list = []
    i = 0
    in_sq = in_dq = False

    def flush(op):
        seg = ''.join(current).strip()
        current.clear()
        return op, seg

    while i < len(cmd):
        c = cmd[i]
        if c == "'" and not in_dq:
            in_sq = not in_sq
            current.append(c)
        elif c == '"' and not in_sq:
            in_dq = not in_dq
            current.append(c)
        elif not in_sq and not in_dq:
            two = cmd[i:i+2]
            if two in ('&&', '||'):
                yield flush(two)
                i += 2
                continue
            elif c == '|':
                yield flush('|')
            elif c == ';':
                yield flush('&&')  # ; in bash ≈ && in CMD
            else:
                current.append(c)
        else:
            current.append(c)
        i += 1

    seg = ''.join(current).strip()
    if seg:
        yield '', seg


def translate_bash_command(command: str) -> str:
    """
    Translate a bash command string (simple or compound) to Windows CMD.

    Handles &&, ||, |, ; compound operators and multi-line commands.
    Returns the command unchanged if it does not appear to be a Unix command.
    """
    if not command or not command.strip():
        return command

    # Multi-line: join lines with &&
    lines = [ln.strip() for ln in command.splitlines() if ln.strip()]
    if len(lines) > 1:
        command = ' && '.join(lines)

    parts = list(_split_compound(command))
    if not parts:
        return command

    result = []
    for op, seg in parts:
        translated = _translate_segment(seg)
        if op:
            result.append(f' {op} ')
        result.append(translated)

    return ''.join(result).strip()


# ---------------------------------------------------------------------------
# Format-specific patchers (return mutated body + changed flag)
# ---------------------------------------------------------------------------

def _patch_bash_input(inp: dict) -> Tuple[dict, bool]:
    """Translate the command field inside a Bash tool input dict."""
    cmd = inp.get('command', '')
    if not cmd:
        return inp, False
    translated = translate_bash_command(cmd)
    if translated == cmd:
        return inp, False
    return {**inp, 'command': translated}, True


def patch_anthropic_body(body: dict) -> Tuple[dict, bool]:
    """
    Translate Bash tool_use command inputs in an Anthropic /v1/messages body.
    Returns (patched_body, was_changed).
    """
    content = body.get('content')
    if not isinstance(content, list):
        return body, False

    new_content = []
    changed = False
    for block in content:
        if block.get('type') == 'tool_use' and block.get('name', '').lower() == 'bash':
            new_inp, patched = _patch_bash_input(dict(block.get('input') or {}))
            if patched:
                block = {**block, 'input': new_inp}
                changed = True
        new_content.append(block)

    if not changed:
        return body, False
    return {**body, 'content': new_content}, True


def patch_openai_body(body: dict) -> Tuple[dict, bool]:
    """
    Translate Bash tool_call command inputs in an OpenAI /v1/chat/completions body.
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
                new_args, patched = _patch_bash_input(args)
                if patched:
                    tc = {**tc, 'function': {**fn, 'arguments': json.dumps(new_args)}}
                    changed = True
            except (json.JSONDecodeError, ValueError):
                pass
        new_tcs.append(tc)

    if not changed:
        return body, False
    new_msg = {**msg, 'tool_calls': new_tcs}
    return {**body, 'choices': [{**choices[0], 'message': new_msg}]}, True


def patch_ollama_body(body: dict) -> Tuple[dict, bool]:
    """
    Translate Bash tool_call command inputs in an Ollama /api/chat body.
    Returns (patched_body, was_changed).
    """
    msg = body.get('message', {})
    tool_calls = msg.get('tool_calls')
    if not isinstance(tool_calls, list):
        return body, False

    new_tcs = []
    changed = False
    for tc in tool_calls:
        fn = tc.get('function', {})
        if fn.get('name', '').lower() == 'bash':
            args = dict(fn.get('arguments') or {})
            new_args, patched = _patch_bash_input(args)
            if patched:
                tc = {**tc, 'function': {**fn, 'arguments': new_args}}
                changed = True
        new_tcs.append(tc)

    if not changed:
        return body, False
    return {**body, 'message': {**msg, 'tool_calls': new_tcs}}, True
