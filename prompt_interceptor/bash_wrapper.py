"""
Bash command wrapper for Windows compatibility.

When windows_bash_mode is enabled in config.json, the proxy wraps every Bash
tool call command as ``bash -c "<command>"`` before forwarding to the client.
This lets Open Code (or any AI client) execute Unix commands on Windows as long
as bash is available (Git Bash, WSL, Cygwin, etc.).
"""

import json
from typing import Dict, Tuple


def wrap_bash_command(command: str) -> str:
    """
    Wrap a bash command so it runs via ``bash -c "..."``.

    Already-wrapped commands (starting with 'bash ') are returned unchanged
    to avoid double-wrapping.
    """
    if not command or command.startswith('bash '):
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
