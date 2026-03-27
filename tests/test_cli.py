"""Tests for cli.py commands."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import prompt_interceptor.cli as cli_mod
from prompt_interceptor.cli import (
    _load_config,
    _save_config,
    cmd_logs,
    cmd_mode,
    cmd_reload_rules,
    cmd_start,
    cmd_status,
    cmd_stop,
    main,
    show_help,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def config_file(tmp_path):
    """Write a temporary config.json and patch _CONFIG_PATH to point at it."""
    cfg = {
        "proxy_port": 8080,
        "proxy_host": "0.0.0.0",
        "target": "http://localhost:11434",
        "mode": "passthrough",
        "dashboard_port": 9090,
        "dashboard_enabled": True,
        "rules": [],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))

    with patch.object(cli_mod, "_CONFIG_PATH", path):
        yield path


# ---------------------------------------------------------------------------
# _load_config / _save_config
# ---------------------------------------------------------------------------

def test_load_config_reads_file(config_file):
    data = _load_config()
    assert data["proxy_port"] == 8080


def test_load_config_missing_file(tmp_path):
    with patch.object(cli_mod, "_CONFIG_PATH", tmp_path / "nonexistent.json"):
        data = _load_config()
    assert data == {}


def test_save_config_writes_file(config_file):
    _save_config({"proxy_port": 9999})
    raw = json.loads(config_file.read_text())
    assert raw["proxy_port"] == 9999


# ---------------------------------------------------------------------------
# show_help
# ---------------------------------------------------------------------------

def test_show_help_prints(capsys):
    # Rich renders to stderr when not a TTY; just assert it doesn't raise.
    show_help()


# ---------------------------------------------------------------------------
# cmd_start
# ---------------------------------------------------------------------------

def test_cmd_start_calls_uvicorn(config_file):
    with patch("prompt_interceptor.cli.subprocess.run") as mock_run:
        cmd_start([])
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "uvicorn" in args
    assert "prompt_interceptor.main:app" in args


def test_cmd_start_uses_provided_port(config_file):
    with patch("prompt_interceptor.cli.subprocess.run") as mock_run:
        cmd_start(["7777"])
    args = mock_run.call_args[0][0]
    assert "7777" in args


def test_cmd_start_keyboard_interrupt(config_file):
    with patch("prompt_interceptor.cli.subprocess.run", side_effect=KeyboardInterrupt):
        cmd_start([])  # Should not raise


def test_cmd_start_uvicorn_not_found(config_file):
    with patch("prompt_interceptor.cli.subprocess.run", side_effect=FileNotFoundError):
        cmd_start([])  # Should not raise


# ---------------------------------------------------------------------------
# cmd_stop
# ---------------------------------------------------------------------------

def test_cmd_stop_prints_message():
    cmd_stop()  # Just checks it doesn't raise


# ---------------------------------------------------------------------------
# cmd_mode
# ---------------------------------------------------------------------------

def test_cmd_mode_no_args_shows_current(config_file):
    cmd_mode([])  # Should not raise


def test_cmd_mode_sets_valid_mode(config_file):
    cmd_mode(["intercept"])
    data = json.loads(config_file.read_text())
    assert data["mode"] == "intercept"


def test_cmd_mode_rejects_invalid():
    cmd_mode(["flying_spaghetti_monster"])  # Should print error but not raise


def test_cmd_mode_passthrough(config_file):
    cmd_mode(["passthrough"])
    data = json.loads(config_file.read_text())
    assert data["mode"] == "passthrough"


# ---------------------------------------------------------------------------
# cmd_reload_rules
# ---------------------------------------------------------------------------

def test_cmd_reload_rules():
    cmd_reload_rules()  # Should not raise


# ---------------------------------------------------------------------------
# cmd_logs
# ---------------------------------------------------------------------------

def test_cmd_logs_no_dir(tmp_path):
    # Point log_dir to a path that doesn't exist
    with patch("glob.glob", return_value=[]):
        with patch.object(Path, "exists", return_value=False):
            cmd_logs()  # Should print "No logs directory" and not raise


def test_cmd_logs_empty_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    with patch.object(Path, "exists", return_value=True):
        with patch("glob.glob", return_value=[]):
            cmd_logs()  # Should print "No log files yet"


def test_cmd_logs_with_files(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "req_001.json"
    log_file.write_text("{}")

    # Patch __file__ so log_dir resolves to tmp_path / "logs"
    with patch.object(cli_mod, "__file__", str(tmp_path / "cli.py")):
        cmd_logs()  # Should not raise


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

def test_cmd_status_with_config(config_file):
    cmd_status()  # Should not raise


def test_cmd_status_no_config(tmp_path):
    with patch.object(cli_mod, "_CONFIG_PATH", tmp_path / "missing.json"):
        cmd_status()  # Should print "config.json not found"


# ---------------------------------------------------------------------------
# main() dispatcher
# ---------------------------------------------------------------------------

def test_main_no_args_shows_help():
    with patch("prompt_interceptor.cli.sys.argv", ["proxy"]):
        with patch("prompt_interceptor.cli.show_help") as mock_help:
            main()
    mock_help.assert_called_once()


def test_main_help_flag():
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "--help"]):
        with patch("prompt_interceptor.cli.show_help") as mock_help:
            main()
    mock_help.assert_called_once()


def test_main_start_command(config_file):
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "start"]):
        with patch("prompt_interceptor.cli.subprocess.run"):
            main()


def test_main_stop_command():
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "stop"]):
        main()  # Should not raise


def test_main_mode_command(config_file):
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "mode", "intercept"]):
        main()
    data = json.loads(config_file.read_text())
    assert data["mode"] == "intercept"


def test_main_reload_rules_command():
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "reload-rules"]):
        main()  # Should not raise


def test_main_logs_command(tmp_path):
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "logs"]):
        with patch.object(Path, "exists", return_value=False):
            main()


def test_main_status_command(config_file):
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "status"]):
        main()  # Should not raise


def test_main_unknown_command():
    with patch("prompt_interceptor.cli.sys.argv", ["proxy", "explode"]):
        with patch("prompt_interceptor.cli.show_help") as mock_help:
            main()
    mock_help.assert_called_once()
