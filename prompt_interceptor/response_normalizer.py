"""
Response normalizer for PromptInterceptor.

Detects and fixes structural issues in LLM responses before forwarding to the client:
- <think>...</think> tags embedded in response content  →  dedicated thinking field/block
- <tool_call>...</tool_call> XML embedded in thinking   →  proper structured tool_calls
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


_THINK_RE = re.compile(r'<think>([\s\S]*?)</think>', re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r'<tool_call>([\s\S]*?)</tool_call>', re.IGNORECASE)
_FUNCTION_XML_RE = re.compile(r'<function=([^\s>]+)>([\s\S]*?)</function>', re.IGNORECASE)
_PARAM_RE = re.compile(r'<parameter=([^\s>]+)>([\s\S]*?)</parameter>', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_tool_call_inner(raw: str) -> Optional[Dict]:
    """Parse the inner content of a <tool_call> block into a structured dict."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    fn_match = _FUNCTION_XML_RE.search(raw)
    if fn_match:
        name = fn_match.group(1)
        args: Dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(fn_match.group(2)):
            val = pm.group(2).strip()
            try:
                args[pm.group(1)] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                args[pm.group(1)] = val
        return {'name': name, 'arguments': args}
    return None


def _extract_tool_calls(text: str) -> Tuple[str, List[Dict]]:
    """Remove <tool_call> blocks from text; return (cleaned_text, parsed_calls)."""
    calls = [tc for m in _TOOL_CALL_RE.finditer(text)
              if (tc := _parse_tool_call_inner(m.group(1))) is not None]
    return _TOOL_CALL_RE.sub('', text).strip(), calls


def _extract_think_tags(content: str) -> Tuple[str, str]:
    """Extract <think>...</think> blocks; return (cleaned_content, thinking_text)."""
    parts = [m.group(1).strip() for m in _THINK_RE.finditer(content)]
    return _THINK_RE.sub('', content).strip(), '\n\n'.join(parts)


def _tc_name(tc: Dict) -> str:
    return tc.get('name') or tc.get('function', {}).get('name', 'tool')


def _tc_args(tc: Dict) -> Any:
    return tc.get('arguments') or tc.get('input') or tc.get('parameters') or {}


def _args_to_str(args: Any) -> str:
    if isinstance(args, dict):
        return json.dumps(args)
    if isinstance(args, str):
        return args
    return json.dumps(args)


def _args_to_dict(args: Any) -> Dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _to_ollama_tc(tc: Dict) -> Dict:
    return {'function': {'name': _tc_name(tc), 'arguments': _args_to_dict(_tc_args(tc))}}


def _to_openai_tc(tc: Dict, idx: int) -> Dict:
    return {
        'id': f'call_{idx:03d}',
        'type': 'function',
        'function': {'name': _tc_name(tc), 'arguments': _args_to_str(_tc_args(tc))},
    }


def _to_anthropic_tu(tc: Dict, idx: int) -> Dict:
    return {
        'type': 'tool_use',
        'id': f'toolu_{idx:03d}',
        'name': _tc_name(tc),
        'input': _args_to_dict(_tc_args(tc)),
    }


# ---------------------------------------------------------------------------
# Public normalizers — one per response format
# ---------------------------------------------------------------------------

def normalize_ollama_chat(body: Dict) -> Tuple[Dict, bool, str]:
    """
    Normalize an assembled Ollama /api/chat (or /api/generate) response body.
    Returns (corrected_body, was_corrected, fix_description).
    """
    if not isinstance(body, dict) or 'message' not in body:
        return body, False, ''

    msg = dict(body['message'])
    content: str = msg.get('content', '') or ''
    thinking: str = msg.get('thinking', '') or ''
    tool_calls: list = list(msg.get('tool_calls') or [])
    fixes: List[str] = []

    # Fix 1: <think> tags embedded in content → thinking field
    if _THINK_RE.search(content):
        clean, extra = _extract_think_tags(content)
        if extra:
            content = clean
            thinking = f'{thinking}\n\n{extra}'.strip() if thinking else extra
            fixes.append('think→thinking')

    # Fix 2: <tool_call> XML embedded in thinking → proper tool_calls
    if not tool_calls and _TOOL_CALL_RE.search(thinking):
        clean, calls = _extract_tool_calls(thinking)
        if calls:
            thinking = clean
            tool_calls = [_to_ollama_tc(tc) for tc in calls]
            fixes.append('thinking→tool_calls')

    # Fix 3: <tool_call> XML in response content → proper tool_calls
    if not tool_calls and _TOOL_CALL_RE.search(content):
        clean, calls = _extract_tool_calls(content)
        if calls:
            content = clean
            tool_calls = [_to_ollama_tc(tc) for tc in calls]
            fixes.append('content→tool_calls')

    # Fix 4: bash tool_calls missing description in arguments
    fixed_tcs = []
    for tc in tool_calls:
        fn = tc.get('function', {})
        if fn.get('name', '').lower() == 'bash':
            args = dict(fn.get('arguments') or {})
            if 'description' not in args:
                args['description'] = ''
                tc = {**tc, 'function': {**fn, 'arguments': args}}
                fixes.append('bash:description')
        fixed_tcs.append(tc)
    tool_calls = fixed_tcs

    if not fixes:
        return body, False, ''

    new_msg = {**msg, 'content': content}
    if thinking:
        new_msg['thinking'] = thinking
    elif 'thinking' in new_msg:
        del new_msg['thinking']
    if tool_calls:
        new_msg['tool_calls'] = tool_calls
    elif 'tool_calls' in new_msg:
        del new_msg['tool_calls']

    return {**body, 'message': new_msg}, True, ', '.join(fixes)


def normalize_openai_chat(body: Dict) -> Tuple[Dict, bool, str]:
    """
    Normalize an assembled OpenAI /v1/chat/completions response body.
    Returns (corrected_body, was_corrected, fix_description).
    """
    if not isinstance(body, dict) or not body.get('choices'):
        return body, False, ''
    choices = body['choices']
    if not choices or 'message' not in choices[0]:
        return body, False, ''

    msg = dict(choices[0]['message'])
    content: str = msg.get('content', '') or ''
    thinking: str = (msg.get('thinking') or msg.get('reasoning') or msg.get('reasoning_content') or '')
    tool_calls: list = list(msg.get('tool_calls') or [])
    fixes: List[str] = []

    if _THINK_RE.search(content):
        clean, extra = _extract_think_tags(content)
        if extra:
            content = clean
            thinking = f'{thinking}\n\n{extra}'.strip() if thinking else extra
            fixes.append('think→thinking')

    if not tool_calls and _TOOL_CALL_RE.search(thinking):
        clean, calls = _extract_tool_calls(thinking)
        if calls:
            thinking = clean
            tool_calls = [_to_openai_tc(tc, i) for i, tc in enumerate(calls)]
            fixes.append('thinking→tool_calls')

    if not tool_calls and _TOOL_CALL_RE.search(content):
        clean, calls = _extract_tool_calls(content)
        if calls:
            content = clean
            tool_calls = [_to_openai_tc(tc, i) for i, tc in enumerate(calls)]
            fixes.append('content→tool_calls')

    fixed_tcs = []
    for tc in tool_calls:
        fn = tc.get('function', {})
        if fn.get('name', '').lower() == 'bash':
            args_str = fn.get('arguments', '{}')
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                if isinstance(args, dict) and 'description' not in args:
                    args['description'] = ''
                    tc = {**tc, 'function': {**fn, 'arguments': json.dumps(args)}}
                    fixes.append('bash:description')
            except (json.JSONDecodeError, ValueError):
                pass
        fixed_tcs.append(tc)
    tool_calls = fixed_tcs

    if not fixes:
        return body, False, ''

    new_msg = {**msg, 'content': content}
    if thinking:
        new_msg['thinking'] = thinking
    if tool_calls:
        new_msg['tool_calls'] = tool_calls

    return {**body, 'choices': [{**choices[0], 'message': new_msg}]}, True, ', '.join(fixes)


def normalize_anthropic_messages(body: Dict) -> Tuple[Dict, bool, str]:
    """
    Normalize an assembled Anthropic /v1/messages response body.
    Returns (corrected_body, was_corrected, fix_description).
    """
    if not isinstance(body, dict) or not isinstance(body.get('content'), list):
        return body, False, ''

    blocks = body['content']
    new_blocks: List[Dict] = []
    tool_blocks: List[Dict] = []
    fixes: List[str] = []
    tu_idx = 0

    for block in blocks:
        btype = block.get('type', '')
        if btype == 'thinking':
            thinking_text: str = block.get('thinking', '') or ''
            if _TOOL_CALL_RE.search(thinking_text):
                clean, calls = _extract_tool_calls(thinking_text)
                if calls:
                    new_blocks.append({**block, 'thinking': clean})
                    for tc in calls:
                        tool_blocks.append(_to_anthropic_tu(tc, tu_idx))
                        tu_idx += 1
                    fixes.append('thinking→tool_use')
                    continue
            new_blocks.append(block)
        elif btype == 'text':
            text: str = block.get('text', '') or ''
            modified = False
            if _THINK_RE.search(text):
                clean, thinking_content = _extract_think_tags(text)
                if thinking_content:
                    new_blocks.append({'type': 'thinking', 'thinking': thinking_content})
                    text = clean
                    fixes.append('think→thinking')
                    modified = True
            if _TOOL_CALL_RE.search(text):
                clean, calls = _extract_tool_calls(text)
                if calls:
                    text = clean
                    for tc in calls:
                        tool_blocks.append(_to_anthropic_tu(tc, tu_idx))
                        tu_idx += 1
                    fixes.append('text→tool_use')
                    modified = True
            new_blocks.append({**block, 'text': text} if modified else block)
        elif btype == 'tool_use':
            inp = dict(block.get('input') or {})
            if block.get('name', '').lower() == 'bash' and 'description' not in inp:
                inp['description'] = ''
                block = {**block, 'input': inp}
                fixes.append('bash:description')
            new_blocks.append(block)
        else:
            new_blocks.append(block)

    if not fixes:
        return body, False, ''

    return {**body, 'content': new_blocks + tool_blocks}, True, ', '.join(fixes)


# ---------------------------------------------------------------------------
# SSE/NDJSON re-synthesizers (used when correction was applied to a stream)
# ---------------------------------------------------------------------------

def _sse_event(name: str, data: Dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def emit_anthropic_sse(body: Dict) -> bytes:
    """
    Re-synthesize a complete Anthropic SSE stream from a corrected response body.
    Used when normalization changes the content structure of a /v1/messages response.
    """
    parts: List[str] = []
    msg_id = body.get('id', 'msg_corrected')
    model = body.get('model', '')
    usage = body.get('usage', {})
    stop_reason = body.get('stop_reason', 'end_turn')

    parts.append(_sse_event('message_start', {
        'type': 'message_start',
        'message': {
            'id': msg_id, 'type': 'message', 'role': 'assistant',
            'content': [], 'model': model, 'stop_reason': None,
            'stop_sequence': None,
            'usage': {'input_tokens': usage.get('input_tokens', 0), 'output_tokens': 0},
        },
    }))
    parts.append(_sse_event('ping', {'type': 'ping'}))

    for idx, block in enumerate(body.get('content', [])):
        btype = block.get('type', 'text')
        if btype == 'thinking':
            parts.append(_sse_event('content_block_start', {
                'type': 'content_block_start', 'index': idx,
                'content_block': {'type': 'thinking', 'thinking': ''},
            }))
            parts.append(_sse_event('content_block_delta', {
                'type': 'content_block_delta', 'index': idx,
                'delta': {'type': 'thinking_delta', 'thinking': block.get('thinking', '')},
            }))
        elif btype == 'text':
            parts.append(_sse_event('content_block_start', {
                'type': 'content_block_start', 'index': idx,
                'content_block': {'type': 'text', 'text': ''},
            }))
            parts.append(_sse_event('content_block_delta', {
                'type': 'content_block_delta', 'index': idx,
                'delta': {'type': 'text_delta', 'text': block.get('text', '')},
            }))
        elif btype == 'tool_use':
            parts.append(_sse_event('content_block_start', {
                'type': 'content_block_start', 'index': idx,
                'content_block': {
                    'type': 'tool_use',
                    'id': block.get('id', f'toolu_{idx:03d}'),
                    'name': block.get('name', 'tool'),
                    'input': {},
                },
            }))
            parts.append(_sse_event('content_block_delta', {
                'type': 'content_block_delta', 'index': idx,
                'delta': {
                    'type': 'input_json_delta',
                    'partial_json': json.dumps(block.get('input', {})),
                },
            }))
        parts.append(_sse_event('content_block_stop', {'type': 'content_block_stop', 'index': idx}))

    parts.append(_sse_event('message_delta', {
        'type': 'message_delta',
        'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
        'usage': {'output_tokens': usage.get('output_tokens', 0)},
    }))
    parts.append(_sse_event('message_stop', {'type': 'message_stop'}))

    return ''.join(parts).encode('utf-8')


def emit_openai_sse(body: Dict) -> bytes:
    """
    Re-synthesize a complete OpenAI SSE stream from a corrected response body.
    Used when normalization changes the content structure of a /v1/chat/completions response.
    """
    parts: List[str] = []
    choices = body.get('choices', [])
    msg = choices[0].get('message', {}) if choices else {}
    base = {k: v for k, v in body.items() if k != 'choices'}
    content: str = msg.get('content', '') or ''
    thinking: str = msg.get('thinking', '') or ''
    tool_calls: list = msg.get('tool_calls') or []

    if thinking:
        parts.append(f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'thinking': thinking}, 'finish_reason': None}]})}\n\n")

    if content:
        parts.append(f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n")

    for i, tc in enumerate(tool_calls):
        parts.append(f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [{**tc, 'index': i}]}, 'finish_reason': None}]})}\n\n")

    finish_reason = 'tool_calls' if tool_calls else (choices[0].get('finish_reason', 'stop') if choices else 'stop')
    parts.append(f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n")
    parts.append("data: [DONE]\n\n")

    return ''.join(parts).encode('utf-8')
