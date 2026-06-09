"""Tests for cmd_translator.py — bash → Windows CMD translation."""

import json
import pytest

from prompt_interceptor.cmd_translator import (
    translate_bash_command,
    patch_anthropic_body,
    patch_openai_body,
    patch_ollama_body,
)


# ---------------------------------------------------------------------------
# translate_bash_command — single commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bash, expected", [
    ("ls", "dir"),
    ("ls -l", "dir"),
    ("ls -la", "dir /a"),
    ("ls -al", "dir /a"),
    ("ls -a src/", "dir /a src/"),
    ("pwd", "cd"),
    ("clear", "cls"),
    ("which python", "where python"),
    ("cat README.md", "type README.md"),
    ("touch foo.txt", "type nul > foo.txt"),
    ("touch a.txt b.txt", "type nul > a.txt & type nul > b.txt"),
    ("rm file.txt", "del file.txt"),
    ("rm -rf build/", "rd /s /q build/"),
    ("rm -r dist/", "rd /s /q dist/"),
    ("cp src.py dst.py", "copy src.py dst.py"),
    ("mv old.py new.py", "move old.py new.py"),
    ("mkdir -p a/b/c", "mkdir a/b/c"),
    ("grep pattern file.txt", "findstr pattern file.txt"),
    ("grep -r pattern src/", "findstr /s pattern src/"),
    ("grep -i pattern file.txt", "findstr /i pattern file.txt"),
    ("find . -name '*.py'", "dir /s /b *.py"),
    ("export FOO=bar", "set FOO=bar"),
    ("python3 script.py", "python script.py"),
    ("pip3 install requests", "pip install requests"),
    ("sudo rm -rf /tmp/x", "rd /s /q /tmp/x"),
])
def test_single_command_translations(bash, expected):
    assert translate_bash_command(bash) == expected


def test_env_var_dollar_replaced():
    assert translate_bash_command("echo $HOME") == "echo %HOME%"


def test_env_var_braces_replaced():
    assert translate_bash_command("echo ${PATH}") == "echo %PATH%"


def test_chmod_commented_out():
    result = translate_bash_command("chmod +x script.sh")
    assert result.startswith("rem ")


def test_chown_commented_out():
    result = translate_bash_command("chown user file.txt")
    assert result.startswith("rem ")


def test_unknown_command_returned_unchanged():
    assert translate_bash_command("git status") == "git status"


def test_empty_command_returned_unchanged():
    assert translate_bash_command("") == ""
    assert translate_bash_command("   ") == "   "


# ---------------------------------------------------------------------------
# translate_bash_command — compound commands
# ---------------------------------------------------------------------------

def test_compound_and_operator():
    result = translate_bash_command("ls && pwd")
    assert "dir" in result
    assert "cd" in result
    assert "&&" in result


def test_compound_semicolon_becomes_and():
    result = translate_bash_command("ls; pwd")
    assert "dir" in result
    assert "cd" in result
    assert "&&" in result


def test_compound_pipe_preserved():
    result = translate_bash_command("ls | grep py")
    assert "dir" in result
    assert "|" in result
    assert "findstr" in result


def test_compound_or_operator():
    result = translate_bash_command("mkdir tmp || echo exists")
    assert "mkdir" in result
    assert "||" in result


def test_multiline_command():
    cmd = "ls\npwd\nmkdir tmp"
    result = translate_bash_command(cmd)
    assert "dir" in result
    assert "cd" in result
    assert "mkdir" in result
    assert "&&" in result


# ---------------------------------------------------------------------------
# patch_anthropic_body
# ---------------------------------------------------------------------------

def test_patch_anthropic_translates_bash_tool_use():
    body = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls -la", "description": ""}},
        ]
    }
    patched, changed = patch_anthropic_body(body)
    assert changed is True
    assert patched["content"][0]["input"]["command"] == "dir /a"


def test_patch_anthropic_case_insensitive_name():
    body = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "pwd"}},
        ]
    }
    patched, changed = patch_anthropic_body(body)
    assert changed is True
    assert patched["content"][0]["input"]["command"] == "cd"


def test_patch_anthropic_non_bash_tool_unchanged():
    body = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/tmp/x"}},
        ]
    }
    patched, changed = patch_anthropic_body(body)
    assert changed is False
    assert patched is body


def test_patch_anthropic_no_content_unchanged():
    body = {"stop_reason": "end_turn"}
    patched, changed = patch_anthropic_body(body)
    assert changed is False
    assert patched is body


def test_patch_anthropic_already_cmd_unchanged():
    body = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "dir /a", "description": ""}},
        ]
    }
    patched, changed = patch_anthropic_body(body)
    assert changed is False


# ---------------------------------------------------------------------------
# patch_openai_body
# ---------------------------------------------------------------------------

def test_patch_openai_translates_bash_tool_call():
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "cat file.txt", "description": ""}'},
                }],
            },
            "finish_reason": "tool_calls",
        }]
    }
    patched, changed = patch_openai_body(body)
    assert changed is True
    args = json.loads(patched["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "type file.txt"


def test_patch_openai_non_bash_unchanged():
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "Read", "arguments": '{}'}}],
            },
            "finish_reason": "tool_calls",
        }]
    }
    patched, changed = patch_openai_body(body)
    assert changed is False
    assert patched is body


def test_patch_openai_no_tool_calls_unchanged():
    body = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
    patched, changed = patch_openai_body(body)
    assert changed is False


# ---------------------------------------------------------------------------
# patch_ollama_body
# ---------------------------------------------------------------------------

def test_patch_ollama_translates_bash_tool_call():
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "Bash", "arguments": {"command": "rm -rf dist/"}}}],
        },
        "done": True,
    }
    patched, changed = patch_ollama_body(body)
    assert changed is True
    assert patched["message"]["tool_calls"][0]["function"]["arguments"]["command"] == "rd /s /q dist/"


def test_patch_ollama_non_bash_unchanged():
    body = {
        "message": {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "Edit", "arguments": {"file_path": "x.py"}}}],
        },
        "done": True,
    }
    patched, changed = patch_ollama_body(body)
    assert changed is False
    assert patched is body
