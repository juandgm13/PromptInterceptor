"""Tests for bash_wrapper module."""
import json
import sys
from unittest.mock import patch
import pytest
from prompt_interceptor.bash_wrapper import wrap_bash_command, patch_anthropic_body, patch_openai_body, is_windows_bash_mode


class TestWrapBashCommand:
    def test_simple_command(self):
        assert wrap_bash_command("ls -la") == 'bash -c "ls -la"'

    def test_empty_command(self):
        assert wrap_bash_command("") == ""

    def test_none_command(self):
        assert wrap_bash_command(None) is None

    def test_already_wrapped(self):
        cmd = 'bash -c "ls -la"'
        assert wrap_bash_command(cmd) == cmd

    def test_double_quotes_escaped(self):
        result = wrap_bash_command('echo "hello world"')
        assert result == 'bash -c "echo \\"hello world\\""'

    def test_backslash_escaped(self):
        result = wrap_bash_command("cat C:\\path\\file.txt")
        assert result == 'bash -c "cat C:\\\\path\\\\file.txt"'

    def test_complex_command(self):
        result = wrap_bash_command("grep -r 'pattern' ./src | head -10")
        assert result.startswith('bash -c "')
        assert result.endswith('"')


class TestPatchAnthropicBody:
    def _make_body(self, command: str, name: str = "Bash"):
        return {
            "content": [
                {
                    "type": "tool_use",
                    "name": name,
                    "input": {"command": command, "description": ""},
                }
            ]
        }

    def test_wraps_bash_command(self):
        body = self._make_body("ls -la")
        patched, changed = patch_anthropic_body(body)
        assert changed is True
        assert patched["content"][0]["input"]["command"] == 'bash -c "ls -la"'

    def test_no_change_if_already_wrapped(self):
        body = self._make_body('bash -c "ls -la"')
        patched, changed = patch_anthropic_body(body)
        assert changed is False
        assert patched is body

    def test_non_bash_tool_not_modified(self):
        body = {
            "content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/x"}}
            ]
        }
        patched, changed = patch_anthropic_body(body)
        assert changed is False

    def test_case_insensitive_name(self):
        body = self._make_body("pwd", name="BASH")
        _, changed = patch_anthropic_body(body)
        assert changed is True

    def test_non_list_content(self):
        body = {"content": "not a list"}
        patched, changed = patch_anthropic_body(body)
        assert changed is False
        assert patched is body

    def test_empty_content(self):
        body = {"content": []}
        patched, changed = patch_anthropic_body(body)
        assert changed is False

    def test_preserves_other_fields(self):
        body = self._make_body("echo hi")
        body["id"] = "msg_123"
        patched, _ = patch_anthropic_body(body)
        assert patched["id"] == "msg_123"


class TestPatchOpenAIBody:
    def _make_body(self, command: str, name: str = "Bash"):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps({"command": command}),
                                }
                            }
                        ]
                    }
                }
            ]
        }

    def test_wraps_bash_command(self):
        body = self._make_body("ls -la")
        patched, changed = patch_openai_body(body)
        assert changed is True
        args = json.loads(patched["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert args["command"] == 'bash -c "ls -la"'

    def test_no_change_if_already_wrapped(self):
        body = self._make_body('bash -c "ls -la"')
        patched, changed = patch_openai_body(body)
        assert changed is False

    def test_non_bash_tool_not_modified(self):
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "Read", "arguments": json.dumps({"path": "/x"})}}
                        ]
                    }
                }
            ]
        }
        _, changed = patch_openai_body(body)
        assert changed is False

    def test_no_choices(self):
        body = {"choices": []}
        patched, changed = patch_openai_body(body)
        assert changed is False

    def test_no_tool_calls(self):
        body = {"choices": [{"message": {"content": "hi"}}]}
        patched, changed = patch_openai_body(body)
        assert changed is False

    def test_invalid_json_arguments_skipped(self):
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "Bash", "arguments": "not-json"}}
                        ]
                    }
                }
            ]
        }
        patched, changed = patch_openai_body(body)
        assert changed is False

    def test_case_insensitive_name(self):
        body = self._make_body("pwd", name="bash")
        _, changed = patch_openai_body(body)
        assert changed is True


class TestIsWindowsBashMode:
    def test_true_on_windows_platform(self):
        with patch.object(sys, 'platform', 'win32'):
            assert is_windows_bash_mode() is True

    def test_false_on_linux(self):
        with patch.object(sys, 'platform', 'linux'):
            assert is_windows_bash_mode() is False

    def test_false_on_darwin(self):
        with patch.object(sys, 'platform', 'darwin'):
            assert is_windows_bash_mode() is False
