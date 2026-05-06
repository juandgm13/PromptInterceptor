"""Tests for response_normalizer.py — structural auto-correction of LLM responses."""

import json
import pytest

from prompt_interceptor.response_normalizer import (
    normalize_ollama_chat,
    normalize_openai_chat,
    normalize_anthropic_messages,
    emit_anthropic_sse,
    emit_openai_sse,
)


# ---------------------------------------------------------------------------
# normalize_ollama_chat
# ---------------------------------------------------------------------------

def test_ollama_no_correction_passthrough():
    body = {"message": {"role": "assistant", "content": "Hello"}, "done": True}
    corrected, was_corrected, desc = normalize_ollama_chat(body)
    assert not was_corrected
    assert corrected is body


def test_ollama_no_message_field_passthrough():
    body = {"response": "Hello", "done": True}
    corrected, was_corrected, _ = normalize_ollama_chat(body)
    assert not was_corrected
    assert corrected is body


def test_ollama_think_tags_in_content():
    body = {
        "message": {"role": "assistant", "content": "<think>My reasoning</think>The answer"},
        "done": True,
    }
    corrected, was_corrected, desc = normalize_ollama_chat(body)
    assert was_corrected
    assert "think→thinking" in desc
    msg = corrected["message"]
    assert msg["content"] == "The answer"
    assert msg["thinking"] == "My reasoning"


def test_ollama_tool_call_json_in_thinking():
    tool_json = json.dumps({"name": "bash", "arguments": {"command": "ls -la"}})
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": f"I need to list files.\n<tool_call>{tool_json}</tool_call>",
        },
        "done": True,
    }
    corrected, was_corrected, desc = normalize_ollama_chat(body)
    assert was_corrected
    assert "thinking→tool_calls" in desc
    msg = corrected["message"]
    assert "<tool_call>" not in msg["thinking"]
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert msg["tool_calls"][0]["function"]["arguments"]["command"] == "ls -la"


def test_ollama_tool_call_xml_in_thinking():
    xml = "<function=bash><parameter=command>ls</parameter></function>"
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": f"<tool_call>{xml}</tool_call>",
        },
        "done": True,
    }
    corrected, was_corrected, desc = normalize_ollama_chat(body)
    assert was_corrected
    msg = corrected["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert msg["tool_calls"][0]["function"]["arguments"]["command"] == "ls"


def test_ollama_tool_call_in_content():
    tool_json = json.dumps({"name": "read_file", "arguments": {"path": "/etc/hosts"}})
    body = {
        "message": {"role": "assistant", "content": f"<tool_call>{tool_json}</tool_call>"},
        "done": True,
    }
    corrected, was_corrected, desc = normalize_ollama_chat(body)
    assert was_corrected
    assert "content→tool_calls" in desc
    msg = corrected["message"]
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"


def test_ollama_existing_tool_calls_not_overwritten():
    existing = [{"function": {"name": "real_tool", "arguments": {}}}]
    tool_json = json.dumps({"name": "fake_tool", "arguments": {}})
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": f"<tool_call>{tool_json}</tool_call>",
            "tool_calls": existing,
        },
        "done": True,
    }
    corrected, was_corrected, _ = normalize_ollama_chat(body)
    assert not was_corrected
    assert corrected["message"]["tool_calls"] == existing


def test_ollama_thinking_field_removed_when_empty_after_correction():
    tool_json = json.dumps({"name": "fn", "arguments": {}})
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": f"<tool_call>{tool_json}</tool_call>",
        },
        "done": True,
    }
    corrected, was_corrected, _ = normalize_ollama_chat(body)
    assert was_corrected
    assert corrected["message"].get("thinking", "") == ""


def test_ollama_non_dict_body():
    corrected, was_corrected, _ = normalize_ollama_chat("not a dict")  # type: ignore[arg-type]
    assert not was_corrected


# ---------------------------------------------------------------------------
# normalize_openai_chat
# ---------------------------------------------------------------------------

def test_openai_no_correction_passthrough():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}]
    }
    corrected, was_corrected, _ = normalize_openai_chat(body)
    assert not was_corrected
    assert corrected is body


def test_openai_no_choices_passthrough():
    body = {"choices": []}
    corrected, was_corrected, _ = normalize_openai_chat(body)
    assert not was_corrected


def test_openai_think_tags_in_content():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "<think>reason</think>answer"}, "finish_reason": "stop"}]
    }
    corrected, was_corrected, desc = normalize_openai_chat(body)
    assert was_corrected
    assert "think→thinking" in desc
    msg = corrected["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert msg["thinking"] == "reason"


def test_openai_tool_call_in_thinking():
    tool_json = json.dumps({"name": "search", "arguments": {"q": "hello"}})
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": f"<tool_call>{tool_json}</tool_call>",
            },
            "finish_reason": "stop",
        }]
    }
    corrected, was_corrected, desc = normalize_openai_chat(body)
    assert was_corrected
    assert "thinking→tool_calls" in desc
    msg = corrected["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "search"
    assert msg["tool_calls"][0]["id"] == "call_000"


def test_openai_tool_call_in_content():
    tool_json = json.dumps({"name": "write", "arguments": {"content": "hello"}})
    body = {
        "choices": [{
            "message": {"role": "assistant", "content": f"<tool_call>{tool_json}</tool_call>"},
            "finish_reason": "stop",
        }]
    }
    corrected, was_corrected, desc = normalize_openai_chat(body)
    assert was_corrected
    assert "content→tool_calls" in desc


def test_openai_reasoning_field_respected():
    """Existing 'reasoning' field treated as thinking source."""
    tool_json = json.dumps({"name": "fn", "arguments": {}})
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning": f"<tool_call>{tool_json}</tool_call>",
            },
            "finish_reason": "stop",
        }]
    }
    corrected, was_corrected, _ = normalize_openai_chat(body)
    assert was_corrected
    assert corrected["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "fn"


# ---------------------------------------------------------------------------
# normalize_anthropic_messages
# ---------------------------------------------------------------------------

def test_anthropic_no_correction_passthrough():
    body = {
        "content": [
            {"type": "thinking", "thinking": "I think..."},
            {"type": "text", "text": "The answer"},
        ]
    }
    corrected, was_corrected, _ = normalize_anthropic_messages(body)
    assert not was_corrected
    assert corrected is body


def test_anthropic_no_content_list_passthrough():
    body = {"content": "not a list"}
    corrected, was_corrected, _ = normalize_anthropic_messages(body)
    assert not was_corrected


def test_anthropic_tool_call_in_thinking():
    tool_json = json.dumps({"name": "bash", "arguments": {"command": "pwd"}})
    body = {
        "content": [
            {"type": "thinking", "thinking": f"Run command.\n<tool_call>{tool_json}</tool_call>"},
            {"type": "text", "text": "Done"},
        ]
    }
    corrected, was_corrected, desc = normalize_anthropic_messages(body)
    assert was_corrected
    assert "thinking→tool_use" in desc
    content = corrected["content"]
    thinking_blocks = [b for b in content if b["type"] == "thinking"]
    tool_blocks = [b for b in content if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "bash"
    assert tool_blocks[0]["input"]["command"] == "pwd"
    assert "<tool_call>" not in thinking_blocks[0]["thinking"]


def test_anthropic_think_tags_in_text_block():
    body = {
        "content": [
            {"type": "text", "text": "<think>reasoning</think>answer text"},
        ]
    }
    corrected, was_corrected, desc = normalize_anthropic_messages(body)
    assert was_corrected
    assert "think→thinking" in desc
    content = corrected["content"]
    thinking_blocks = [b for b in content if b["type"] == "thinking"]
    text_blocks = [b for b in content if b["type"] == "text"]
    assert thinking_blocks[0]["thinking"] == "reasoning"
    assert text_blocks[0]["text"] == "answer text"


def test_anthropic_tool_call_in_text_block():
    tool_json = json.dumps({"name": "list_files", "arguments": {"path": "."}})
    body = {
        "content": [
            {"type": "text", "text": f"<tool_call>{tool_json}</tool_call>"},
        ]
    }
    corrected, was_corrected, desc = normalize_anthropic_messages(body)
    assert was_corrected
    assert "text→tool_use" in desc
    tool_blocks = [b for b in corrected["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "list_files"


def test_anthropic_multiple_tool_calls():
    t1 = json.dumps({"name": "fn_a", "arguments": {"x": 1}})
    t2 = json.dumps({"name": "fn_b", "arguments": {"y": 2}})
    body = {
        "content": [
            {"type": "thinking", "thinking": f"<tool_call>{t1}</tool_call>\n<tool_call>{t2}</tool_call>"},
        ]
    }
    corrected, was_corrected, _ = normalize_anthropic_messages(body)
    assert was_corrected
    tool_blocks = [b for b in corrected["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 2
    assert tool_blocks[0]["name"] == "fn_a"
    assert tool_blocks[1]["name"] == "fn_b"
    assert tool_blocks[0]["id"] == "toolu_000"
    assert tool_blocks[1]["id"] == "toolu_001"


# ---------------------------------------------------------------------------
# emit_anthropic_sse
# ---------------------------------------------------------------------------

def test_emit_anthropic_sse_text_block():
    body = {
        "id": "msg_01",
        "model": "test-model",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello world"}],
    }
    result = emit_anthropic_sse(body)
    text = result.decode("utf-8")
    assert "message_start" in text
    assert "text_delta" in text
    assert "Hello world" in text
    assert "message_stop" in text
    assert "message_delta" in text


def test_emit_anthropic_sse_thinking_and_tool_use():
    body = {
        "id": "msg_02",
        "model": "qwen3",
        "stop_reason": "tool_use",
        "usage": {},
        "content": [
            {"type": "thinking", "thinking": "I need to call bash"},
            {"type": "tool_use", "id": "toolu_001", "name": "bash", "input": {"command": "ls"}},
        ],
    }
    result = emit_anthropic_sse(body)
    text = result.decode("utf-8")
    assert "thinking_delta" in text
    assert "I need to call bash" in text
    assert "input_json_delta" in text
    assert "bash" in text
    assert "content_block_stop" in text


def test_emit_anthropic_sse_valid_sse_format():
    """Each SSE event must have 'event:' and 'data:' lines."""
    body = {
        "id": "x",
        "model": "m",
        "stop_reason": "end_turn",
        "usage": {},
        "content": [{"type": "text", "text": "hi"}],
    }
    text = emit_anthropic_sse(body).decode("utf-8")
    event_lines = [l for l in text.splitlines() if l.startswith("event:")]
    data_lines = [l for l in text.splitlines() if l.startswith("data:")]
    assert len(event_lines) > 0
    assert len(data_lines) > 0
    for dl in data_lines:
        json.loads(dl[5:].strip())  # each data: line must be valid JSON


# ---------------------------------------------------------------------------
# emit_openai_sse
# ---------------------------------------------------------------------------

def test_emit_openai_sse_content_only():
    body = {
        "id": "cmpl-1",
        "model": "qwen3",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
    }
    result = emit_openai_sse(body)
    text = result.decode("utf-8")
    assert "Hello" in text
    assert "[DONE]" in text


def test_emit_openai_sse_with_tool_calls():
    body = {
        "id": "cmpl-2",
        "model": "qwen3",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_000",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    result = emit_openai_sse(body)
    text = result.decode("utf-8")
    assert "bash" in text
    assert "tool_calls" in text
    assert "[DONE]" in text


def test_emit_openai_sse_valid_format():
    body = {
        "id": "x",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
    text = emit_openai_sse(body).decode("utf-8")
    data_lines = [l for l in text.splitlines() if l.startswith("data:") and l.strip() != "data: [DONE]"]
    for dl in data_lines:
        json.loads(dl[5:].strip())  # each data: line must be valid JSON


def test_emit_openai_sse_with_thinking():
    """thinking in message generates a thinking delta chunk."""
    body = {
        "id": "cmpl-3",
        "model": "qwen3",
        "choices": [{
            "message": {"role": "assistant", "content": "answer", "thinking": "deep thoughts"},
            "finish_reason": "stop",
        }],
    }
    text = emit_openai_sse(body).decode("utf-8")
    assert "deep thoughts" in text
    assert "thinking" in text
    assert "answer" in text
    assert "[DONE]" in text


# ---------------------------------------------------------------------------
# Edge cases for internal helpers
# ---------------------------------------------------------------------------

def test_parse_tool_call_inner_unrecognized_returns_none():
    """Content that is neither JSON nor XML function format returns None."""
    from prompt_interceptor.response_normalizer import _parse_tool_call_inner
    result = _parse_tool_call_inner("just some random text")
    assert result is None


def test_args_to_str_with_string():
    """_args_to_str passes a string through unchanged."""
    from prompt_interceptor.response_normalizer import _args_to_str
    assert _args_to_str('{"key": "val"}') == '{"key": "val"}'


def test_args_to_str_with_other_type():
    """_args_to_str with a list falls back to json.dumps."""
    from prompt_interceptor.response_normalizer import _args_to_str
    result = _args_to_str([1, 2, 3])
    assert result == "[1, 2, 3]"


def test_args_to_dict_with_valid_json_string():
    """_args_to_dict parses a valid JSON string into a dict."""
    from prompt_interceptor.response_normalizer import _args_to_dict
    result = _args_to_dict('{"command": "ls"}')
    assert result == {"command": "ls"}


def test_args_to_dict_with_invalid_json_string():
    """_args_to_dict returns {} for invalid JSON strings."""
    from prompt_interceptor.response_normalizer import _args_to_dict
    result = _args_to_dict("not valid json")
    assert result == {}


def test_args_to_dict_with_other_type():
    """_args_to_dict returns {} for non-dict/non-str types."""
    from prompt_interceptor.response_normalizer import _args_to_dict
    result = _args_to_dict([1, 2, 3])
    assert result == {}


def test_ollama_empty_tool_calls_key_removed():
    """When message has tool_calls=[] and a think fix is applied, tool_calls key is deleted."""
    body = {
        "message": {
            "role": "assistant",
            "content": "<think>reasoning</think>answer",
            "tool_calls": [],
        },
        "done": True,
    }
    corrected, was_corrected, _ = normalize_ollama_chat(body)
    assert was_corrected
    assert "tool_calls" not in corrected["message"]


def test_openai_choices_missing_message_passthrough():
    """choices[0] without 'message' key returns body unchanged."""
    body = {"choices": [{"finish_reason": "stop"}]}
    corrected, was_corrected, _ = normalize_openai_chat(body)
    assert not was_corrected
    assert corrected is body


def test_anthropic_unknown_block_type_preserved():
    """Block types other than thinking/text are passed through unchanged."""
    image_block = {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
    body = {"content": [image_block, {"type": "text", "text": "caption"}]}
    corrected, was_corrected, _ = normalize_anthropic_messages(body)
    assert not was_corrected
    assert corrected is body


def test_malformed_tool_call_skipped():
    """<tool_call> block that can't be parsed is silently dropped from extraction."""
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "<tool_call>not json and no xml function tag</tool_call>",
        },
        "done": True,
    }
    corrected, was_corrected, _ = normalize_ollama_chat(body)
    assert not was_corrected
