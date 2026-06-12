"""Tests for bash_wrapper module."""
import json
import sys
from unittest.mock import patch
import pytest
from prompt_interceptor.bash_wrapper import (
    wrap_bash_command, patch_anthropic_body, patch_openai_body,
    is_windows_bash_mode, _fix_windows_command,
)


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

    def test_backslash_path_converted(self):
        # Windows path backslashes are converted to forward slashes before wrapping
        result = wrap_bash_command("cat C:\\path\\file.txt")
        assert result == 'bash -c "cat C:/path/file.txt"'

    def test_complex_command(self):
        result = wrap_bash_command("grep -r 'pattern' ./src | head -10")
        assert result.startswith('bash -c "')
        assert result.endswith('"')

    # PowerShell cmdlets — must NOT be wrapped
    def test_powershell_capitalized_not_wrapped(self):
        assert wrap_bash_command("Get-ChildItem") == "Get-ChildItem"

    def test_powershell_lowercase_not_wrapped(self):
        assert wrap_bash_command("get-childitem") == "get-childitem"

    def test_powershell_with_args_not_wrapped(self):
        assert wrap_bash_command("Get-ChildItem -Path C:\\Users") == "Get-ChildItem -Path C:\\Users"

    def test_powershell_set_location_not_wrapped(self):
        assert wrap_bash_command("Set-Location C:\\Projects") == "Set-Location C:\\Projects"

    def test_powershell_new_item_not_wrapped(self):
        assert wrap_bash_command("New-Item -ItemType Directory -Path .\\output") == \
               "New-Item -ItemType Directory -Path .\\output"

    def test_powershell_remove_item_not_wrapped(self):
        assert wrap_bash_command("Remove-Item -Recurse -Force .\\dist") == \
               "Remove-Item -Recurse -Force .\\dist"

    def test_powershell_write_host_not_wrapped(self):
        assert wrap_bash_command("Write-Host 'hello'") == "Write-Host 'hello'"

    def test_powershell_invoke_expression_not_wrapped(self):
        assert wrap_bash_command("Invoke-Expression $cmd") == "Invoke-Expression $cmd"

    def test_powershell_select_object_not_wrapped(self):
        assert wrap_bash_command("Select-Object -Property Name, Size") == \
               "Select-Object -Property Name, Size"


class TestFixWindowsCommand:
    """Unit tests for _fix_windows_command independently of wrapping."""

    def test_unquoted_path_backslashes(self):
        assert _fix_windows_command("ls C:\\Users\\juand") == "ls C:/Users/juand"

    def test_unquoted_path_double_backslashes(self):
        # LLMs sometimes double-escape: C:\\\\Users → C:/Users in actual string
        assert _fix_windows_command("ls C:\\\\Users\\\\juand") == "ls C:/Users/juand"

    def test_quoted_path_with_spaces(self):
        assert _fix_windows_command('ls "C:\\Program Files\\app"') == 'ls "C:/Program Files/app"'

    def test_single_quoted_path(self):
        assert _fix_windows_command("ls 'C:\\Users\\juand'") == "ls 'C:/Users/juand'"

    def test_null_redirect_2(self):
        assert _fix_windows_command("ls 2>/nul") == "ls 2>/dev/null"

    def test_null_redirect_plain(self):
        assert _fix_windows_command("ls >/nul") == "ls >/dev/null"

    def test_null_redirect_no_slash(self):
        assert _fix_windows_command("ls >nul") == "ls >/dev/null"

    def test_null_redirect_append(self):
        assert _fix_windows_command("ls 2>>/nul") == "ls 2>>/dev/null"

    def test_combined_error_case_1(self):
        # Exact pattern from reported error #1
        cmd = "ls -la C:\\Users\\juand\\Documents\\Repositorios\\MyUtils/"
        result = _fix_windows_command(cmd)
        assert result == "ls -la C:/Users/juand/Documents/Repositorios/MyUtils/"

    def test_combined_error_case_2(self):
        # Exact pattern from reported error #2
        cmd = 'dir /b "C:\\Users\\juand\\Documents\\Repositorios\\MyUtils" 2>/nul'
        result = _fix_windows_command(cmd)
        # dir /b is intentionally NOT translated; only path and /nul are fixed
        assert result == 'dir /b "C:/Users/juand/Documents/Repositorios/MyUtils" 2>/dev/null'

    def test_no_change_for_unix_paths(self):
        cmd = "ls /home/user/docs"
        assert _fix_windows_command(cmd) == cmd

    def test_no_change_for_relative_paths(self):
        cmd = "ls ./src/main.py"
        assert _fix_windows_command(cmd) == cmd

    def test_wrap_integrates_fixes(self):
        result = wrap_bash_command("ls C:\\Users\\juand 2>/nul")
        assert result == 'bash -c "ls C:/Users/juand 2>/dev/null"'

    def test_already_wrapped_bash_c_gets_path_fixed(self):
        # Commands already wrapped as bash -c should also get inner paths fixed
        cmd = 'bash -c "ls C:\\Users\\juand"'
        result = wrap_bash_command(cmd)
        assert result == 'bash -c "ls C:/Users/juand"'


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
