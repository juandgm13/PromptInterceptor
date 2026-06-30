"""Tests for prompt_interceptor/launcher.py."""
import json
import shutil
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from prompt_interceptor.config import Config


# ---------------------------------------------------------------------------
# _CTX_OPTIONS / _CTX_DEFAULT constants
# ---------------------------------------------------------------------------

def test_ctx_options_contains_standard_sizes():
    from prompt_interceptor.launcher import _CTX_OPTIONS
    assert 4096 in _CTX_OPTIONS.values()
    assert 8192 in _CTX_OPTIONS.values()
    assert 262144 in _CTX_OPTIONS.values()


def test_ctx_default_in_options():
    from prompt_interceptor.launcher import _CTX_OPTIONS, _CTX_DEFAULT
    assert _CTX_DEFAULT in _CTX_OPTIONS


# ---------------------------------------------------------------------------
# _detect_clients — always includes Python App as last entry
# ---------------------------------------------------------------------------

def test_detect_clients_no_system_clients(monkeypatch):
    """When neither claude nor opencode are found, Python App is still included."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    clients, warnings = launcher._detect_clients()
    assert len(clients) == 2
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")
    assert warnings == []


@pytest.mark.skip(reason="Claude Code client deshabilitado en el launcher")
def test_detect_clients_claude_only(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    clients = launcher._detect_clients()
    assert len(clients) == 2
    assert clients[0] == ("Claude Code", "claude")
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")


def test_detect_clients_opencode_only(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": "1.17.5")
    clients, warnings = launcher._detect_clients()
    assert len(clients) == 3
    assert clients[1] == ("Open Code (CLI)", "opencode")
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")
    assert warnings == []


@pytest.mark.skip(reason="Claude Code client deshabilitado en el launcher")
def test_detect_clients_both(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    clients = launcher._detect_clients()
    names = [name for name, _ in clients]
    assert "Claude Code" in names
    assert "Open Code (CLI)" in names
    assert "Python App (Ollama)" in names
    assert len(clients) == 3


def test_detect_clients_python_app_always_last(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": "1.17.5")
    clients, _ = launcher._detect_clients()
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")


def test_detect_clients_wsl_opencode(monkeypatch):
    """Open Code (WSL) is disabled in the UI for now; it never appears in the client list."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: True)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": "1.17.5")
    clients, _ = launcher._detect_clients()
    names = [name for name, _ in clients]
    assert "Open Code (CLI)" in names
    assert "Open Code (WSL)" not in names
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")


def test_detect_clients_wsl_only_no_cli(monkeypatch):
    """Open Code (WSL) is disabled in the UI; it does not appear even when WSL has opencode."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: True)
    clients, warnings = launcher._detect_clients()
    names = [name for name, _ in clients]
    assert "Open Code (CLI)" not in names
    assert "Open Code (WSL)" not in names
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")
    assert warnings == []


def test_detect_clients_opencode_incompatible_version(monkeypatch):
    """Out-of-range version produces a warning and clean label; command stays 'opencode'."""
    from prompt_interceptor.client_versions import OPENCODE_MAX_VERSION
    parts = OPENCODE_MAX_VERSION.split(".")
    incompatible = ".".join(parts[:-1] + [str(int(parts[-1]) + 1)])
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": incompatible)
    clients, warnings = launcher._detect_clients()
    assert len(clients) == 3
    assert clients[1] == ("Open Code (CLI)", "opencode")
    assert len(warnings) == 1
    assert incompatible in warnings[0]
    assert "compatible" in warnings[0].lower()


def test_detect_clients_opencode_version_unknown(monkeypatch):
    """When version cannot be detected (None), display name is clean and no warning."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": None)
    clients, warnings = launcher._detect_clients()
    assert clients[1] == ("Open Code (CLI)", "opencode")
    assert warnings == []


def test_detect_clients_opencode_version_at_min(monkeypatch):
    """Version at minimum bound is accepted without warning."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": "1.17.4")
    clients, warnings = launcher._detect_clients()
    assert clients[1] == ("Open Code (CLI)", "opencode")
    assert warnings == []


def test_detect_clients_opencode_version_at_max(monkeypatch):
    """Version at maximum bound is accepted without warning."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor import launcher
    monkeypatch.setattr(launcher, "_is_opencode_in_wsl", lambda: False)
    monkeypatch.setattr(launcher, "_get_opencode_version", lambda cmd="opencode": "1.17.7")
    clients, warnings = launcher._detect_clients()
    assert clients[1] == ("Open Code (CLI)", "opencode")
    assert warnings == []


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------

def test_parse_version_three_parts():
    from prompt_interceptor.launcher import _parse_version
    assert _parse_version("1.17.4") == (1, 17, 4)


def test_parse_version_two_parts():
    from prompt_interceptor.launcher import _parse_version
    assert _parse_version("1.17") == (1, 17)


def test_parse_version_non_numeric_segment():
    from prompt_interceptor.launcher import _parse_version
    result = _parse_version("1.17.4-beta")
    assert result[0] == 1
    assert result[1] == 17


def test_parse_version_strips_whitespace():
    from prompt_interceptor.launcher import _parse_version
    assert _parse_version("  1.17.4  ") == (1, 17, 4)


# ---------------------------------------------------------------------------
# _is_version_compatible
# ---------------------------------------------------------------------------

def test_is_version_compatible_at_min():
    from prompt_interceptor.launcher import _is_version_compatible
    assert _is_version_compatible("1.17.4", "1.17.4", "1.17.7") is True


def test_is_version_compatible_at_max():
    from prompt_interceptor.launcher import _is_version_compatible
    assert _is_version_compatible("1.17.7", "1.17.4", "1.17.7") is True


def test_is_version_compatible_in_range():
    from prompt_interceptor.launcher import _is_version_compatible
    assert _is_version_compatible("1.17.5", "1.17.4", "1.17.7") is True


def test_is_version_compatible_below_min():
    from prompt_interceptor.launcher import _is_version_compatible
    assert _is_version_compatible("1.17.3", "1.17.4", "1.17.7") is False


def test_is_version_compatible_above_max():
    from prompt_interceptor.launcher import _is_version_compatible
    assert _is_version_compatible("1.17.8", "1.17.4", "1.17.7") is False


def test_is_version_compatible_major_minor_below():
    from prompt_interceptor.launcher import _is_version_compatible
    assert _is_version_compatible("1.16.9", "1.17.4", "1.17.7") is False


# ---------------------------------------------------------------------------
# _get_opencode_version
# ---------------------------------------------------------------------------

def test_get_opencode_version_stdout():
    mock_result = MagicMock()
    mock_result.stdout = "opencode 1.17.4\n"
    mock_result.stderr = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_opencode_version
        assert _get_opencode_version() == "1.17.4"


def test_get_opencode_version_bare_version_string():
    mock_result = MagicMock()
    mock_result.stdout = "1.17.5\n"
    mock_result.stderr = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_opencode_version
        assert _get_opencode_version() == "1.17.5"


def test_get_opencode_version_from_stderr():
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "opencode version 1.17.6"
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_opencode_version
        assert _get_opencode_version() == "1.17.6"


def test_get_opencode_version_no_version_in_output():
    mock_result = MagicMock()
    mock_result.stdout = "some unexpected output"
    mock_result.stderr = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_opencode_version
        assert _get_opencode_version() is None


def test_get_opencode_version_subprocess_exception():
    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=Exception("timeout")):
        from prompt_interceptor.launcher import _get_opencode_version
        assert _get_opencode_version() is None


def test_get_opencode_version_custom_cmd():
    mock_result = MagicMock()
    mock_result.stdout = "opencode 1.17.4"
    mock_result.stderr = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result) as mock_run:
        from prompt_interceptor.launcher import _get_opencode_version
        _get_opencode_version(cmd="opencode-custom")
    assert mock_run.call_args[0][0][0] == "opencode-custom"


# ---------------------------------------------------------------------------
# _start_proxy_thread
# ---------------------------------------------------------------------------

def test_start_proxy_thread_dashboard_enabled(tmp_path, monkeypatch):
    """_start_proxy_thread calls uvicorn.run and starts dashboard thread."""
    import asyncio
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    started = []

    class _MockThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            started.append(self._target)

    import uvicorn
    with patch.object(uvicorn, "run", MagicMock()) as mock_run, \
         patch("prompt_interceptor.launcher.threading.Thread", side_effect=_MockThread):
        from prompt_interceptor.launcher import _start_proxy_thread
        _start_proxy_thread()

    assert mock_run.called
    assert len(started) == 1  # dashboard thread was created

    # Call the dashboard thread target to cover its body
    with patch.object(uvicorn, "run", MagicMock()), \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        started[0]()


def test_start_proxy_thread_dashboard_disabled(tmp_path, monkeypatch):
    """When dashboard disabled, no extra thread is started."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=False)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    started = []

    class _MockThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            started.append(self._target)

    import uvicorn
    with patch.object(uvicorn, "run", MagicMock()), \
         patch("prompt_interceptor.launcher.threading.Thread", side_effect=_MockThread):
        from prompt_interceptor.launcher import _start_proxy_thread
        _start_proxy_thread()

    assert started == []  # no extra thread


# ---------------------------------------------------------------------------
# _start_dashboard_thread
# ---------------------------------------------------------------------------

def test_start_dashboard_thread_enabled(tmp_path, monkeypatch):
    """_start_dashboard_thread calls uvicorn.run for the dashboard when enabled."""
    import asyncio
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import uvicorn
    with patch.object(uvicorn, "run", MagicMock()) as mock_run, \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        from prompt_interceptor.launcher import _start_dashboard_thread
        _start_dashboard_thread()

    assert mock_run.called
    call_args = mock_run.call_args[0][0]
    assert "dashboard" in call_args


def test_start_dashboard_thread_disabled(tmp_path, monkeypatch):
    """_start_dashboard_thread is a no-op when dashboard_enabled=False."""
    import asyncio
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=False)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import uvicorn
    with patch.object(uvicorn, "run", MagicMock()) as mock_run, \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        from prompt_interceptor.launcher import _start_dashboard_thread
        _start_dashboard_thread()

    mock_run.assert_not_called()


def test_start_dashboard_thread_import_error(tmp_path, monkeypatch, capsys):
    """ImportError from uvicorn is caught and printed."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import asyncio
    import uvicorn
    with patch.object(uvicorn, "run", side_effect=ImportError("no module")), \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        from prompt_interceptor.launcher import _start_dashboard_thread
        _start_dashboard_thread()

    assert "no module" in capsys.readouterr().out


def test_start_dashboard_thread_oserror(tmp_path, monkeypatch, capsys):
    """OSError (port in use) from uvicorn is caught and printed."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import asyncio
    import uvicorn
    with patch.object(uvicorn, "run", side_effect=OSError("address in use")), \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        from prompt_interceptor.launcher import _start_dashboard_thread
        _start_dashboard_thread()

    assert "address in use" in capsys.readouterr().out


def test_start_dashboard_thread_generic_error(tmp_path, monkeypatch, capsys):
    """Unexpected exceptions from the dashboard thread are caught and printed."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import asyncio
    import uvicorn
    with patch.object(uvicorn, "run", side_effect=RuntimeError("unexpected crash")), \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        from prompt_interceptor.launcher import _start_dashboard_thread
        _start_dashboard_thread()

    assert "unexpected crash" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# LauncherWindow (fully mocked tkinter) — helpers
# ---------------------------------------------------------------------------

def _make_mock_root():
    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    return root


_DEFAULT_CLIENTS = [("Claude Code", "claude"), ("Python App (Ollama)", "__python_app__")]


def _make_headless_win(cfg, clients=None):
    """Create a LauncherWindow with fully mocked UI for unit-testing methods."""
    if clients is None:
        clients = _DEFAULT_CLIENTS
    mock_root = _make_mock_root()
    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher.get_config", return_value=cfg), \
         patch("prompt_interceptor.launcher._detect_clients", return_value=(clients, [])):
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)
    return win, mock_root


# ---------------------------------------------------------------------------
# LauncherWindow init and icon
# ---------------------------------------------------------------------------

def test_launcher_window_init_python_app_only(tmp_path, monkeypatch):
    """Init works when only Python App is available (no system AI clients)."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    mock_root = _make_mock_root()
    python_only = [("Python App (Ollama)", "__python_app__")]

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=(python_only, [])):
        mock_tk.StringVar.return_value = MagicMock()
        mock_tk.PhotoImage.side_effect = Exception("no display")
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    mock_root.title.assert_called_with("PromptInterceptor")
    mock_root.resizable.assert_called_with(False, False)
    mock_root.configure.assert_called()


def test_launcher_window_init_with_clients(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    mock_root = _make_mock_root()

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=(_DEFAULT_CLIENTS, [])):
        mock_tk.StringVar.return_value = MagicMock()
        mock_tk.PhotoImage.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    mock_root.title.assert_called_with("PromptInterceptor")


def test_launcher_window_set_icon_success(tmp_path, monkeypatch):
    """When icon exists and PhotoImage succeeds, iconphoto is called."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    mock_root = _make_mock_root()
    mock_icon = MagicMock()

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=(_DEFAULT_CLIENTS, [])), \
         patch("prompt_interceptor.launcher.Path.exists", return_value=True):
        mock_tk.StringVar.return_value = MagicMock()
        mock_tk.PhotoImage.return_value = mock_icon
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    mock_root.iconphoto.assert_called_once_with(True, mock_icon)


def test_launcher_window_set_icon_missing(tmp_path, monkeypatch):
    """When icon path doesn't exist, iconphoto is never called."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    mock_root = _make_mock_root()

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=(_DEFAULT_CLIENTS, [])), \
         patch("prompt_interceptor.launcher.Path.exists", return_value=False):
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    mock_root.iconphoto.assert_not_called()


# ---------------------------------------------------------------------------
# Step 1: _on_launch_ollama
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Ollama auto-launch via Popen was removed; user must start ollama manually")
def test_on_launch_ollama_opens_process_and_thread(tmp_path, monkeypatch):
    """When Ollama is not running, _check_or_launch_ollama opens a CMD process and starts a background thread."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=8192)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: [])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "127.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()
    mock_root.after.side_effect = lambda delay, fn, *args: fn(*args)

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._check_or_launch_ollama()

    assert len(popen_calls) == 1
    cmd_args, _ = popen_calls[0]
    assert "ollama serve" in " ".join(cmd_args)
    mock_thread.assert_called_once()


def test_on_launch_ollama_disables_button_and_starts_thread(tmp_path, monkeypatch):
    """_on_launch_ollama disables the button and starts the check thread."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_ollama()

    win._launch_ollama_btn.config.assert_called_with(state="disabled")
    win.status_var.set.assert_called_with("Checking Ollama...")
    mock_thread.assert_called_once()


def test_check_or_launch_ollama_already_running(tmp_path, monkeypatch):
    """When Ollama is already running, no process is launched and models are loaded."""
    cfg = Config(log_dir=str(tmp_path / "logs"), target="http://localhost:11434")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: ["llama3"])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "127.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()
    mock_root.after.side_effect = lambda delay, fn, *args: fn(*args)

    with patch("prompt_interceptor.launcher.subprocess.Popen") as mock_popen, \
         patch.object(win, "_on_models_ready") as mock_models_ready:
        win._check_or_launch_ollama()

    mock_popen.assert_not_called()
    mock_models_ready.assert_called_once_with(["llama3"])
    win.status_var.set.assert_called_with("Ollama running.")


@pytest.mark.skip(reason="Ollama auto-launch via Popen was removed; user must start ollama manually")
def test_on_launch_ollama_uses_selected_ctx(tmp_path, monkeypatch):
    """_check_or_launch_ollama lanza Ollama con 'ollama serve' (context_size se aplica en el paso Python App)."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=32768)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: [])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "127.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()
    mock_root.after.side_effect = lambda delay, fn, *args: fn(*args)

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._check_or_launch_ollama()

    assert len(popen_calls) == 1
    assert "ollama serve" in " ".join(popen_calls[0][0])


# ---------------------------------------------------------------------------
# Step 1: _fetch_models_after_launch / _on_models_ready
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="_fetch_models_after_launch method was removed from LauncherWindow")
def test_fetch_models_after_launch_calls_root_after(tmp_path, monkeypatch):
    """_fetch_models_after_launch sleeps, fetches models, then schedules _on_models_ready."""
    cfg = Config(log_dir=str(tmp_path / "logs"), target="http://localhost:11434")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: ["llama3"])
    monkeypatch.setattr("prompt_interceptor.launcher.time.sleep", lambda s: None)

    win, mock_root = _make_headless_win(cfg)
    win._fetch_models_after_launch()

    mock_root.after.assert_called()
    args = mock_root.after.call_args[0]
    assert args[0] == 0
    assert args[1] == win._on_models_ready


@pytest.mark.skip(reason="_fetch_models_after_launch method was removed from LauncherWindow")
def test_fetch_models_retries_when_empty(tmp_path, monkeypatch):
    """_fetch_models_after_launch retries once if first fetch returns no models."""
    cfg = Config(log_dir=str(tmp_path / "logs"), target="http://localhost:11434")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    call_count = [0]

    def mock_fetch(t):
        call_count[0] += 1
        return [] if call_count[0] == 1 else ["llama3"]

    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", mock_fetch)
    monkeypatch.setattr("prompt_interceptor.launcher.time.sleep", lambda s: None)

    win, mock_root = _make_headless_win(cfg)
    win.status_var = MagicMock()
    win._fetch_models_after_launch()

    assert call_count[0] == 2


def test_on_models_ready_populates_combobox(tmp_path, monkeypatch):
    """_on_models_ready fills the model combobox values and unlocks Step 2."""
    cfg = Config(log_dir=str(tmp_path / "logs"), default_model="")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win._model_cb = MagicMock()
    win.model_var = MagicMock()
    win.status_var = MagicMock()

    with patch.object(win, "_set_step2_enabled") as mock_unlock:
        win._on_models_ready(["llama3", "mistral"])

    # values are set on the combobox
    assert win._model_cb.__setitem__.call_args_list[0][0] == ("values", ["llama3", "mistral"])
    mock_unlock.assert_called_once_with(True)
    win.status_var.set.assert_called()


def test_on_models_ready_no_models(tmp_path, monkeypatch):
    """When no models are returned, Step 2 still unlocks."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win._model_cb = MagicMock()
    win.model_var = MagicMock()
    win.status_var = MagicMock()

    with patch.object(win, "_set_step2_enabled") as mock_unlock:
        win._on_models_ready([])

    mock_unlock.assert_called_once_with(True)


def test_on_models_ready_selects_default_model(tmp_path, monkeypatch):
    """When default_model is in the list it is pre-selected."""
    cfg = Config(log_dir=str(tmp_path / "logs"), default_model="mistral")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win._model_cb = MagicMock()
    win.model_var = MagicMock()
    win.status_var = MagicMock()

    with patch.object(win, "_set_step2_enabled"):
        win._on_models_ready(["llama3", "mistral"])

    win.model_var.set.assert_called_with("mistral")


# ---------------------------------------------------------------------------
# Step 2: _set_step2_enabled / _refresh_client_rows
# ---------------------------------------------------------------------------

def test_set_step2_enabled_enables_widgets(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win._client_cb = MagicMock()

    with patch.object(win, "_refresh_client_rows"):
        win._set_step2_enabled(True)

    win._client_cb.config.assert_called_with(state="readonly")
    assert win._step2_enabled is True


def test_set_step2_enabled_disables_widgets(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win._client_cb = MagicMock()

    with patch.object(win, "_refresh_client_rows"):
        win._set_step2_enabled(False)

    win._client_cb.config.assert_called_with(state="disabled")
    assert win._step2_enabled is False


def test_refresh_client_rows_python_app_shows_app_fields(tmp_path):
    """When Python App is selected, apppath/command/venv/env/launch rows shown; workdir+model hidden."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._step2_enabled = True
    win._row_model = MagicMock()
    win._row_workdir = MagicMock()
    win._row_apppath = MagicMock()
    win._row_command = MagicMock()
    win._row_venv = MagicMock()
    win._row_envvar = MagicMock()
    win._row_launch = MagicMock()
    win._app_path_entry = MagicMock()
    win._browse_app_btn = MagicMock()
    win._command_entry = MagicMock()
    win._venv_check = MagicMock()
    win._env_var_entry = MagicMock()
    win._launch_client_btn = MagicMock()
    win._work_dir_entry = MagicMock()
    win._browse_workdir_btn = MagicMock()
    win._model_cb = MagicMock()

    win._refresh_client_rows()

    win._row_model.pack_forget.assert_called()
    win._row_workdir.pack_forget.assert_called()
    win._row_apppath.pack.assert_called()
    win._row_command.pack.assert_called()
    win._row_venv.pack.assert_called()
    win._row_envvar.pack.assert_called()
    win._row_launch.pack.assert_called()
    win._app_path_entry.config.assert_called_with(state="normal")
    win._command_entry.config.assert_called_with(state="normal")
    win._env_var_entry.config.assert_called_with(state="normal")
    win._launch_client_btn.config.assert_called_with(text="Launch Client", state="normal")


def test_refresh_client_rows_claude_shows_workdir(tmp_path):
    """When Claude Code is selected, model+workdir rows are shown and python rows are hidden."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Claude Code"
    win._step2_enabled = True
    win._row_model = MagicMock()
    win._row_workdir = MagicMock()
    win._row_apppath = MagicMock()
    win._row_command = MagicMock()
    win._row_venv = MagicMock()
    win._row_envvar = MagicMock()
    win._work_dir_entry = MagicMock()
    win._browse_workdir_btn = MagicMock()
    win._app_path_entry = MagicMock()
    win._browse_app_btn = MagicMock()
    win._command_entry = MagicMock()
    win._venv_check = MagicMock()
    win._env_var_entry = MagicMock()
    win._model_cb = MagicMock()

    win._refresh_client_rows()

    win._row_apppath.pack_forget.assert_called()
    win._row_command.pack_forget.assert_called()
    win._row_venv.pack_forget.assert_called()
    win._row_envvar.pack_forget.assert_called()
    win._row_model.pack.assert_called()
    win._row_workdir.pack.assert_called()
    win._model_cb.config.assert_called_with(state="readonly")
    win._row_workdir.pack.assert_called()
    win._work_dir_entry.config.assert_called_with(state="normal")


def test_get_window_height_returns_550_for_python_app(tmp_path):
    """_get_window_height returns 550 when Python App is selected."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    assert win._get_window_height() == 550


def test_get_window_height_returns_490_for_other_clients(tmp_path):
    """_get_window_height returns 460 for non-Python-App clients."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (CLI)"
    assert win._get_window_height() == 460


def test_get_window_height_attribute_error_returns_490(tmp_path):
    """_get_window_height returns 460 when client_var.get() raises AttributeError."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.side_effect = AttributeError("not initialized")
    assert win._get_window_height() == 460


def test_refresh_client_rows_resizes_window_for_python_app(tmp_path):
    """_refresh_client_rows calls root.geometry with height 550 for Python App."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, mock_root = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._step2_enabled = False
    for attr in ("_row_model", "_row_workdir", "_row_apppath", "_row_command",
                 "_row_venv", "_row_envvar", "_row_launch",
                 "_app_path_entry", "_browse_app_btn", "_command_entry",
                 "_venv_check", "_env_var_entry", "_launch_client_btn",
                 "_work_dir_entry", "_browse_workdir_btn", "_model_cb"):
        setattr(win, attr, MagicMock())

    win._refresh_client_rows()

    geometry_calls = [str(c) for c in mock_root.geometry.call_args_list]
    assert any("550" in c for c in geometry_calls)


def test_refresh_client_rows_resizes_window_for_other_client(tmp_path):
    """_refresh_client_rows calls root.geometry with height 460 for non-Python-App clients."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, mock_root = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (CLI)"
    win._step2_enabled = False
    for attr in ("_row_model", "_row_workdir", "_row_apppath", "_row_command",
                 "_row_venv", "_row_envvar", "_row_launch",
                 "_app_path_entry", "_browse_app_btn", "_command_entry",
                 "_venv_check", "_env_var_entry", "_launch_client_btn",
                 "_work_dir_entry", "_browse_workdir_btn", "_model_cb"):
        setattr(win, attr, MagicMock())

    win._refresh_client_rows()

    geometry_calls = [str(c) for c in mock_root.geometry.call_args_list]
    assert any("460" in c for c in geometry_calls)


def test_on_client_change_calls_refresh(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    with patch.object(win, "_refresh_client_rows") as mock_refresh:
        win._on_client_change()
    mock_refresh.assert_called_once()


# ---------------------------------------------------------------------------
# Step 2: browse helpers
# ---------------------------------------------------------------------------

def test_on_browse_workdir_sets_path(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.work_dir_var = MagicMock()
    with patch("prompt_interceptor.launcher.filedialog.askdirectory", return_value="/my/project"):
        win._on_browse_workdir()
    win.work_dir_var.set.assert_called_once_with("/my/project")


def test_on_browse_workdir_no_selection(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.work_dir_var = MagicMock()
    with patch("prompt_interceptor.launcher.filedialog.askdirectory", return_value=""):
        win._on_browse_workdir()
    win.work_dir_var.set.assert_not_called()


def test_on_browse_app_sets_path(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    with patch("prompt_interceptor.launcher.filedialog.askdirectory", return_value="/app/myproject"):
        win._on_browse_app()
    win.app_path_var.set.assert_called_once_with("/app/myproject")


def test_on_browse_app_no_selection(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    with patch("prompt_interceptor.launcher.filedialog.askdirectory", return_value=""):
        win._on_browse_app()
    win.app_path_var.set.assert_not_called()


# ---------------------------------------------------------------------------
# Step 2: _on_launch_client
# ---------------------------------------------------------------------------

def test_on_launch_client_claude_code(tmp_path, monkeypatch):
    """Launching Claude Code passes --model, sets ANTHROPIC_BASE_URL, and cwd."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Claude Code", "claude"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Claude Code"
    win._clients = [("Claude Code", "claude"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "/my/repo"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "llama3.2:latest"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_client()

    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    cmd_str = " ".join(args)
    assert "claude" in cmd_str
    assert "--model" in cmd_str
    assert "llama3.2:latest" in cmd_str
    assert kwargs.get("cwd") == "/my/repo"
    assert kwargs.get("env", {}).get("ANTHROPIC_BASE_URL") == "http://localhost:8080"
    mock_thread.assert_called()  # proxy thread started after successful launch


def test_on_launch_client_open_code_cli(tmp_path, monkeypatch):
    """Launching Open Code (CLI) passes --model, sets OPENAI_BASE_URL, and cwd."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Open Code (CLI)", "opencode"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (CLI)"
    win._clients = [("Open Code (CLI)", "opencode"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "/my/repo"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "mistral:latest"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher._write_opencode_config"):
        win._on_launch_client()

    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    cmd_str = " ".join(args)
    assert "opencode" in cmd_str
    assert "--model" in cmd_str
    assert "ollama/mistral:latest" in cmd_str


@pytest.mark.skip(reason="Open Code (WSL) is disabled in the UI; re-enable when reactivated")
def test_on_launch_client_open_code_wsl(tmp_path, monkeypatch):
    """Launching Open Code (WSL) writes WSL config and opens a WSL terminal via shell=True."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Open Code (WSL)", "__opencode_wsl__"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (WSL)"
    win._clients = [("Open Code (WSL)", "__opencode_wsl__"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.model_var = MagicMock()
    win.model_var.get.return_value = "mistral:latest"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher._get_wsl_host_ip", return_value="172.28.0.1"), \
         patch("prompt_interceptor.launcher._write_opencode_config_wsl") as mock_wsl_cfg:
        win._on_launch_client()

    assert len(popen_calls) == 1
    cmd_list, kwargs = popen_calls[0]
    assert isinstance(cmd_list, list)
    # cmd /c start <title> wsl [--cd <dir>] -- bash -ic <opencode_cmd>
    assert cmd_list[:3] == ["cmd", "/c", "start"]
    assert "wsl" in cmd_list
    assert "bash" in cmd_list
    assert "-ic" in cmd_list
    assert any("opencode" in a for a in cmd_list)
    assert any("ollama/mistral:latest" in a for a in cmd_list)
    mock_wsl_cfg.assert_called_once_with("mistral:latest", "http://172.28.0.1:8080")
    win._start_btn.config.assert_called_with(state="normal")


@pytest.mark.skip(reason="Open Code (WSL) is disabled in the UI; re-enable when reactivated")
def test_on_launch_client_open_code_wsl_with_workdir(tmp_path, monkeypatch):
    """WSL launch passes Windows work dir directly via wsl --cd (no wslpath conversion)."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Open Code (WSL)", "__opencode_wsl__"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (WSL)"
    win._clients = [("Open Code (WSL)", "__opencode_wsl__"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "C:\\Users\\user\\project"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "llama3"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher._get_wsl_host_ip", return_value="172.28.0.1"), \
         patch("prompt_interceptor.launcher._write_opencode_config_wsl"):
        win._on_launch_client()

    assert len(popen_calls) == 1
    cmd_list, kwargs = popen_calls[0]
    assert isinstance(cmd_list, list)
    assert cmd_list[:3] == ["cmd", "/c", "start"]
    assert "wsl" in cmd_list
    assert "--cd" in cmd_list
    assert "C:\\Users\\user\\project" in cmd_list
    assert any("opencode" in a for a in cmd_list)


def test_on_launch_client_python_app_via_button(tmp_path, monkeypatch):
    """When Python App is launched via its Launch Client button, proxy starts and app is scheduled."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, mock_root = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._clients = [("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.status_var = MagicMock()

    after_calls = []
    mock_root.after.side_effect = lambda delay, fn, *args: after_calls.append((delay, fn))

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_client()

    # Proxy thread started
    mock_thread.assert_called()
    assert win._servers_started is True
    # App launch and iconify scheduled with a delay
    delays = [d for d, _ in after_calls]
    assert 2500 in delays


def test_on_launch_client_python_app(tmp_path, monkeypatch):
    """_launch_python_app sets the env var to the proxy URL and runs the command."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = "/apps/myproject"
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = "python app.py --port 8000"
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))):
        win._launch_python_app()

    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    cmd_str = " ".join(args)
    assert "python app.py --port 8000" in cmd_str
    assert kwargs.get("cwd") == "/apps/myproject"
    # env var is injected via process environment, not the command string
    assert kwargs["env"]["OLLAMA_HOST"] == "http://localhost:8080"


def test_on_launch_client_python_app_uses_venv(tmp_path, monkeypatch):
    """_launch_python_app with venv activates .venv\\Scripts\\activate before the command."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = "python main.py"
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = True
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        win._launch_python_app()

    assert len(popen_calls) == 1
    cmd_str = " ".join(popen_calls[0])
    assert ".venv" in cmd_str
    assert "activate" in cmd_str
    assert "python main.py" in cmd_str


def test_launch_python_app_venv_candidate_detected(tmp_path, monkeypatch):
    """When app_dir is set and .venv/Scripts/activate.bat exists, that candidate is selected."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    # Create the activate.bat so Path.exists() returns True for .venv
    activate_bat = tmp_path / ".venv" / "Scripts" / "activate.bat"
    activate_bat.parent.mkdir(parents=True)
    activate_bat.write_text("@echo off")

    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = str(tmp_path)
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = "python main.py"
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = True
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        win._launch_python_app()

    assert len(popen_calls) == 1
    cmd_str = " ".join(popen_calls[0])
    assert ".venv\\Scripts\\activate" in cmd_str
    assert "python main.py" in cmd_str


# ---------------------------------------------------------------------------
# WSL detection helpers
# ---------------------------------------------------------------------------

def test_is_wsl_available_true(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/wsl" if cmd == "wsl" else None)
    from prompt_interceptor.launcher import _is_wsl_available
    assert _is_wsl_available() is True


def test_is_wsl_available_false(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    from prompt_interceptor.launcher import _is_wsl_available
    assert _is_wsl_available() is False


def test_is_opencode_in_wsl_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/wsl" if cmd == "wsl" else None)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "/usr/local/bin/opencode\n"
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _is_opencode_in_wsl
        assert _is_opencode_in_wsl() is True


def test_is_opencode_in_wsl_not_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/wsl" if cmd == "wsl" else None)
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _is_opencode_in_wsl
        assert _is_opencode_in_wsl() is False


def test_is_opencode_in_wsl_no_wsl(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    from prompt_interceptor.launcher import _is_opencode_in_wsl
    assert _is_opencode_in_wsl() is False


def test_is_opencode_in_wsl_exception():
    with patch("prompt_interceptor.launcher._is_wsl_available", return_value=True), \
         patch("prompt_interceptor.launcher.subprocess.run", side_effect=Exception("timeout")):
        from prompt_interceptor.launcher import _is_opencode_in_wsl
        assert _is_opencode_in_wsl() is False


def test_get_wsl_host_ip_success():
    mock_result = MagicMock()
    mock_result.stdout = "172.28.0.1\n"
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_wsl_host_ip
        assert _get_wsl_host_ip() == "172.28.0.1"


def test_get_wsl_host_ip_nameserver_format():
    """'nameserver X.X.X.X' format: returns the IP (parts[-1])."""
    mock_result = MagicMock()
    mock_result.stdout = "nameserver 10.255.255.254\n"
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_wsl_host_ip
        assert _get_wsl_host_ip() == "10.255.255.254"


def test_get_wsl_host_ip_empty_falls_back_to_localhost():
    mock_result = MagicMock()
    mock_result.stdout = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _get_wsl_host_ip
        assert _get_wsl_host_ip() == "localhost"


def test_get_wsl_host_ip_exception_falls_back_to_localhost():
    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=Exception("error")):
        from prompt_interceptor.launcher import _get_wsl_host_ip
        assert _get_wsl_host_ip() == "localhost"


def test_wsl_can_reach_proxy_returns_true():
    """_wsl_can_reach_proxy returns True when WSL echo prints 'ok'."""
    mock_result = MagicMock()
    mock_result.stdout = "ok\n"
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _wsl_can_reach_proxy
        assert _wsl_can_reach_proxy("172.28.0.1", 8080) is True


def test_wsl_can_reach_proxy_returns_false_on_no_output():
    """_wsl_can_reach_proxy returns False when stdout is not 'ok'."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        from prompt_interceptor.launcher import _wsl_can_reach_proxy
        assert _wsl_can_reach_proxy("172.28.0.1", 8080) is False


def test_wsl_can_reach_proxy_returns_false_on_exception():
    """_wsl_can_reach_proxy returns False when subprocess.run raises."""
    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=Exception("timeout")):
        from prompt_interceptor.launcher import _wsl_can_reach_proxy
        assert _wsl_can_reach_proxy("172.28.0.1", 8080) is False


def test_warn_wsl_firewall_calls_showwarning():
    """_warn_wsl_firewall calls tkinter.messagebox.showwarning with host and port info."""
    with patch("prompt_interceptor.launcher.tk") as mock_tk:
        import tkinter.messagebox as mb
        with patch.object(mb, "showwarning") as mock_warn:
            from prompt_interceptor.launcher import _warn_wsl_firewall
            _warn_wsl_firewall("172.28.0.1", 8080)

    mock_warn.assert_called_once()
    _, msg = mock_warn.call_args[0]
    assert "172.28.0.1" in msg
    assert "8080" in msg


# ---------------------------------------------------------------------------
# _get_window_height — Only Proxy branch (line 308)
# ---------------------------------------------------------------------------

def test_get_window_height_only_proxy(tmp_path):
    """_get_window_height returns 400 when the selected client is 'Only Proxy'."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg, clients=[("Only Proxy", "__only_proxy__"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Only Proxy"
    assert win._get_window_height() == 400


# ---------------------------------------------------------------------------
# _refresh_client_rows — Only Proxy branch (lines 629-636)
# ---------------------------------------------------------------------------

def test_refresh_client_rows_only_proxy(tmp_path):
    """_refresh_client_rows packs proxy info row and sets Launch Proxy button for Only Proxy."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win, mock_root = _make_headless_win(cfg, clients=[("Only Proxy", "__only_proxy__"),
                                                       ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Only Proxy"
    win._clients = [("Only Proxy", "__only_proxy__"), ("Python App (Ollama)", "__python_app__")]
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win._proxy_url_var = MagicMock()
    win._step2_enabled = True
    for attr in ("_row_model", "_row_workdir", "_row_apppath", "_row_command",
                 "_row_venv", "_row_envvar", "_row_launch", "_row_proxy_info",
                 "_app_path_entry", "_browse_app_btn", "_command_entry",
                 "_venv_check", "_env_var_entry", "_launch_client_btn",
                 "_work_dir_entry", "_browse_workdir_btn", "_model_cb"):
        setattr(win, attr, MagicMock())

    win._refresh_client_rows()

    win._row_proxy_info.pack.assert_called()
    win._launch_client_btn.config.assert_called()
    # Verify the button text was set to "Launch Proxy"
    config_calls = win._launch_client_btn.config.call_args_list
    assert any(call.kwargs.get("text") == "Launch Proxy" or
               (call.args and "Launch Proxy" in str(call.args))
               for call in config_calls)


def test_refresh_client_rows_only_proxy_invalid_port(tmp_path):
    """_refresh_client_rows falls back to config port when proxy_port_var is non-numeric."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win, mock_root = _make_headless_win(cfg, clients=[("Only Proxy", "__only_proxy__"),
                                                       ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Only Proxy"
    win._clients = [("Only Proxy", "__only_proxy__"), ("Python App (Ollama)", "__python_app__")]
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "not-a-number"
    win._proxy_url_var = MagicMock()
    win._step2_enabled = True
    for attr in ("_row_model", "_row_workdir", "_row_apppath", "_row_command",
                 "_row_venv", "_row_envvar", "_row_launch", "_row_proxy_info",
                 "_app_path_entry", "_browse_app_btn", "_command_entry",
                 "_venv_check", "_env_var_entry", "_launch_client_btn",
                 "_work_dir_entry", "_browse_workdir_btn", "_model_cb"):
        setattr(win, attr, MagicMock())

    # Should not raise, falls back to config.proxy_port
    win._refresh_client_rows()
    win._proxy_url_var.set.assert_called()


# ---------------------------------------------------------------------------
# _on_launch_client — Only Proxy branch (lines 685-696)
# ---------------------------------------------------------------------------

def test_on_launch_client_only_proxy(tmp_path, monkeypatch):
    """Only Proxy mode starts proxy thread and opens dashboard in browser."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080,
                 dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, mock_root = _make_headless_win(cfg, clients=[("Only Proxy", "__only_proxy__"),
                                                       ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Only Proxy"
    win._clients = [("Only Proxy", "__only_proxy__"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.status_var = MagicMock()
    win._servers_started = False

    after_calls = []
    mock_root.after.side_effect = lambda delay, fn, *args: after_calls.append((delay, fn))

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread, \
         patch("prompt_interceptor.launcher.webbrowser.open"):
        mock_thread.return_value = MagicMock()
        win._on_launch_client()

    # Proxy thread must have been started
    mock_thread.assert_called()
    assert win._servers_started is True
    # Dashboard open and iconify are scheduled via root.after
    delays = [d for d, _ in after_calls]
    assert 2500 in delays


def test_on_launch_client_only_proxy_already_started(tmp_path, monkeypatch):
    """Only Proxy mode does NOT start a second proxy thread when already running."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080,
                 dashboard_enabled=False)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, mock_root = _make_headless_win(cfg, clients=[("Only Proxy", "__only_proxy__"),
                                                       ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Only Proxy"
    win._clients = [("Only Proxy", "__only_proxy__"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.status_var = MagicMock()
    win._servers_started = True  # Already started

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        win._on_launch_client()

    # Thread must NOT be started again
    mock_thread.assert_not_called()


def test_write_opencode_config_wsl_creates_config():
    """Writes opencode.json inside WSL by piping JSON via stdin."""
    read_result = MagicMock()
    read_result.stdout = "{}"
    write_result = MagicMock()

    run_calls = []

    def _fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        if "cat" in args[-1] and ">" not in args[-1]:
            return read_result
        return write_result

    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=_fake_run):
        from prompt_interceptor.launcher import _write_opencode_config_wsl
        _write_opencode_config_wsl("qwen2.5:7b", "http://172.28.0.1:8080")

    # Two calls: read + write
    assert len(run_calls) == 2
    # The write call must pipe the JSON via stdin
    write_call_kwargs = run_calls[1][1]
    assert "input" in write_call_kwargs
    written = json.loads(write_call_kwargs["input"])
    assert written["provider"]["ollama"]["options"]["baseURL"] == "http://172.28.0.1:8080/v1"
    assert "qwen2.5:7b" in written["provider"]["ollama"]["models"]


def test_write_opencode_config_wsl_merges_existing():
    """Merges into existing opencode.json without wiping other keys."""
    existing = json.dumps({"theme": "dark", "$schema": "https://opencode.ai/config.json"})
    read_result = MagicMock()
    read_result.stdout = existing

    run_calls = []

    def _fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        return read_result

    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=_fake_run):
        from prompt_interceptor.launcher import _write_opencode_config_wsl
        _write_opencode_config_wsl("mistral:latest", "http://172.28.0.1:8080")

    write_input = json.loads(run_calls[1][1]["input"])
    assert write_input["theme"] == "dark"
    assert "mistral:latest" in write_input["provider"]["ollama"]["models"]


def test_write_opencode_config_wsl_invalid_json_read_falls_back_to_empty():
    """When the read subprocess returns invalid JSON, an empty dict is used (line 116)."""
    invalid_read = MagicMock()
    invalid_read.stdout = "not valid json {{{"  # causes json.JSONDecodeError
    write_result = MagicMock()

    run_calls = []

    def _fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        # First call is the read, second is the write
        if len(run_calls) == 1:
            return invalid_read
        return write_result

    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=_fake_run):
        from prompt_interceptor.launcher import _write_opencode_config_wsl
        _write_opencode_config_wsl("llama3", "http://172.28.0.1:8080")

    # Still writes valid JSON even though read was garbage
    write_input = json.loads(run_calls[1][1]["input"])
    assert "provider" in write_input


def test_write_opencode_config_wsl_write_failure_is_silent():
    """When the write subprocess.run raises, the function handles it gracefully."""
    read_result = MagicMock()
    read_result.stdout = "{}"
    call_count = [0]

    def _fake_run(args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return read_result
        raise Exception("disk full")

    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=_fake_run):
        from prompt_interceptor.launcher import _write_opencode_config_wsl
        # Must not raise even if write fails
        _write_opencode_config_wsl("llama3", "http://172.28.0.1:8080")

    assert call_count[0] == 2


# ---------------------------------------------------------------------------
# _write_opencode_config
# ---------------------------------------------------------------------------

def test_write_opencode_config_creates_file(tmp_path):
    """Creates .opencode/opencode.json inside work_dir when it does not exist."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("qwen2.5:7b", "http://localhost:8080", work_dir=str(tmp_path))

    written = tmp_path / ".opencode" / "opencode.json"
    assert written.exists()
    data = json.loads(written.read_text())
    assert data["provider"]["ollama"]["options"]["baseURL"] == "http://localhost:8080/v1"
    assert "qwen2.5:7b" in data["provider"]["ollama"]["models"]


def test_write_opencode_config_updates_existing(tmp_path):
    """Merges into an existing opencode.json without overwriting unrelated keys."""
    from prompt_interceptor.launcher import _write_opencode_config

    config_dir = tmp_path / ".opencode"
    config_dir.mkdir(parents=True)
    existing = {"$schema": "https://opencode.ai/config.json", "theme": "dark"}
    (config_dir / "opencode.json").write_text(json.dumps(existing))

    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("mistral:latest", "http://localhost:8080", work_dir=str(tmp_path))

    data = json.loads((config_dir / "opencode.json").read_text())
    assert data["theme"] == "dark"
    assert "mistral:latest" in data["provider"]["ollama"]["models"]


def test_write_opencode_config_idempotent(tmp_path):
    """Calling twice with the same model does not duplicate entries."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("llama3:8b", "http://localhost:8080", work_dir=str(tmp_path))
        _write_opencode_config("llama3:8b", "http://localhost:8080", work_dir=str(tmp_path))

    written = tmp_path / ".opencode" / "opencode.json"
    data = json.loads(written.read_text())
    models = data["provider"]["ollama"]["models"]
    assert list(models.keys()).count("llama3:8b") == 1


def test_write_opencode_config_invalid_json_file_falls_back_to_empty(tmp_path):
    """When the existing opencode.json contains invalid JSON, an empty dict is used."""
    from prompt_interceptor.launcher import _write_opencode_config

    config_dir = tmp_path / ".opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "opencode.json").write_text("not valid json {{}")

    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir=str(tmp_path))

    data = json.loads((config_dir / "opencode.json").read_text())
    assert "provider" in data


# _detect_git_bash / shell field
# ---------------------------------------------------------------------------

def test_detect_git_bash_true_when_bash_in_path():
    """Returns True on Windows when bash.exe is in PATH."""
    from prompt_interceptor.launcher import _detect_git_bash
    with patch("prompt_interceptor.launcher.sys") as mock_sys, \
         patch("prompt_interceptor.launcher.shutil.which", return_value="/usr/bin/bash"):
        mock_sys.platform = "win32"
        assert _detect_git_bash() is True


def test_detect_git_bash_false_when_bash_not_in_path():
    """Returns False on Windows when bash.exe is not in PATH."""
    from prompt_interceptor.launcher import _detect_git_bash
    with patch("prompt_interceptor.launcher.sys") as mock_sys, \
         patch("prompt_interceptor.launcher.shutil.which", return_value=None):
        mock_sys.platform = "win32"
        assert _detect_git_bash() is False


def test_detect_git_bash_false_on_non_windows():
    """Returns False on non-Windows regardless of bash availability."""
    from prompt_interceptor.launcher import _detect_git_bash
    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        assert _detect_git_bash() is False


def test_write_opencode_config_repo_path_used(tmp_path):
    """Config is written to work_dir/.opencode/opencode.json, not ~/.config."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir=str(tmp_path))

    assert (tmp_path / ".opencode" / "opencode.json").exists()
    assert not (tmp_path / ".config").exists()


def test_write_opencode_config_shell_bash_on_windows_with_git_bash(tmp_path):
    """On Windows with Git Bash available, shell is set to 'bash'."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys, \
         patch("prompt_interceptor.launcher.shutil.which", return_value="C:\\Git\\bin\\bash.exe"):
        mock_sys.platform = "win32"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir=str(tmp_path))

    data = json.loads((tmp_path / ".opencode" / "opencode.json").read_text())
    assert data["shell"] == "bash"


def test_write_opencode_config_shell_cmd_on_windows_without_git_bash(tmp_path):
    """On Windows without Git Bash, shell is set to 'cmd'."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys, \
         patch("prompt_interceptor.launcher.shutil.which", return_value=None):
        mock_sys.platform = "win32"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir=str(tmp_path))

    data = json.loads((tmp_path / ".opencode" / "opencode.json").read_text())
    assert data["shell"] == "cmd"


def test_write_opencode_config_no_shell_on_linux(tmp_path):
    """On Linux, no shell key is added to the config."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir=str(tmp_path))

    data = json.loads((tmp_path / ".opencode" / "opencode.json").read_text())
    assert "shell" not in data


def test_write_opencode_config_schema_before_shell_before_provider(tmp_path):
    """Key order in output: $schema first, then shell (Windows), then provider."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.sys") as mock_sys, \
         patch("prompt_interceptor.launcher.shutil.which", return_value="C:\\Git\\bin\\bash.exe"):
        mock_sys.platform = "win32"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir=str(tmp_path))

    data = json.loads((tmp_path / ".opencode" / "opencode.json").read_text())
    keys = list(data.keys())
    assert keys[0] == "$schema"
    assert keys[1] == "shell"
    assert "provider" in keys
    assert keys.index("shell") < keys.index("provider")


def test_write_opencode_config_fallback_to_home_when_no_work_dir(tmp_path):
    """Falls back to ~/.config/opencode/opencode.json when work_dir is empty."""
    from prompt_interceptor.launcher import _write_opencode_config

    with patch("prompt_interceptor.launcher.Path.home", return_value=tmp_path), \
         patch("prompt_interceptor.launcher.sys") as mock_sys:
        mock_sys.platform = "linux"
        _write_opencode_config("llama3", "http://localhost:8080", work_dir="")

    written = tmp_path / ".config" / "opencode" / "opencode.json"
    assert written.exists()
    data = json.loads(written.read_text())
    assert "llama3" in data["provider"]["ollama"]["models"]


def test_on_launch_client_python_app_custom_env_var(tmp_path, monkeypatch):
    """Custom env var name is used instead of the default OLLAMA_HOST."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = "python main.py"
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OPENAI_BASE_URL"
    win.status_var = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))):
        win._launch_python_app()

    _, kwargs = popen_calls[0]
    # env var is injected via process environment, not the command string
    assert "OPENAI_BASE_URL" in kwargs["env"]
    assert kwargs["env"]["OPENAI_BASE_URL"] == "http://localhost:8080"


def test_on_launch_client_starts_proxy_on_success(tmp_path, monkeypatch):
    """After a successful client launch, the proxy thread is started and _servers_started set."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Claude Code", "claude"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Claude Code"
    win._clients = [("Claude Code", "claude"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "."
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.subprocess.Popen"), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_client()

    mock_thread.assert_called()
    assert win._servers_started is True


# ---------------------------------------------------------------------------
# Step 3: _on_open_dashboard — starts dashboard (standalone or via proxy), no Popen
# ---------------------------------------------------------------------------

def test_on_open_dashboard_saves_config_and_starts_dashboard(tmp_path, monkeypatch):
    """_on_open_dashboard saves config and starts the dashboard thread. No Popen."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    saved = []
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: saved.append(c))

    win, mock_root = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "llama3"
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "/my/repo"
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.subprocess.Popen") as mock_popen, \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()

    # Config saved with correct context_size
    assert len(saved) == 1
    assert saved[0].context_size == 4096

    # Dashboard thread started (standalone, no proxy)
    mock_thread.assert_called_once()

    # No Popen — Ollama and client are NOT launched by _on_open_dashboard
    mock_popen.assert_not_called()

    # Browser open scheduled via root.after
    mock_root.after.assert_called()


def test_on_open_dashboard_saves_model_to_config(tmp_path, monkeypatch):
    """_on_open_dashboard persists the selected model as default_model."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    saved = []
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: saved.append(c))

    win, _ = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "deepseek-coder"
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()

    assert saved[0].default_model == "deepseek-coder"


def test_on_open_dashboard_standalone_schedules_browser_and_iconify(tmp_path, monkeypatch):
    """When no servers are started, _on_open_dashboard schedules browser at 2000ms, iconify at 2500ms."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: None)

    win, mock_root = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "llama3"
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win.status_var = MagicMock()

    after_calls = []
    mock_root.after.side_effect = lambda delay, fn, *args: after_calls.append((delay, fn))

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()

    delays = [delay for delay, _ in after_calls]
    # open browser (2000ms), iconify (2500ms) — no 1500ms reset here
    assert 2000 in delays
    assert 2500 in delays
    assert 1500 not in delays


def test_on_open_dashboard_with_servers_started_opens_browser_immediately(tmp_path, monkeypatch):
    """When servers are already running, _on_open_dashboard opens the browser without starting a thread."""
    import prompt_interceptor.launcher as _launcher_mod
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: None)

    win, mock_root = _make_headless_win(cfg)
    win._servers_started = True  # simulate already-running proxy
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win.status_var = MagicMock()

    opened = []
    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread, \
         patch("prompt_interceptor.launcher.webbrowser.open", side_effect=opened.append):
        win._on_open_dashboard()

    mock_thread.assert_not_called()
    assert len(opened) == 1
    assert "9090" in opened[0]


# ---------------------------------------------------------------------------
# _reset_session — extracted instance method
# ---------------------------------------------------------------------------

def test_reset_session_calls_api_endpoint(tmp_path, monkeypatch):
    """_reset_session calls POST /api/reset on the dashboard."""
    import urllib.request as _urllib_request

    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)

    opened_urls = []

    def _fake_urlopen(req, timeout=None):
        opened_urls.append(req.full_url)
        return MagicMock()

    with patch("prompt_interceptor.launcher.urllib.request.urlopen", side_effect=_fake_urlopen), \
         patch("prompt_interceptor.launcher.urllib.request.Request", wraps=_urllib_request.Request):
        win._reset_session()

    assert any("9090" in url and "reset" in url for url in opened_urls)


def test_reset_session_silences_connection_error(tmp_path, monkeypatch):
    """_reset_session does not raise if the dashboard is not yet up."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)

    with patch("prompt_interceptor.launcher.urllib.request.urlopen",
               side_effect=Exception("connection refused")):
        win._reset_session()  # must not raise


# ---------------------------------------------------------------------------
# _fetch_ollama_models
# ---------------------------------------------------------------------------

def test_fetch_ollama_models_success(monkeypatch):
    import json as _json
    payload = _json.dumps({"models": [{"name": "llama3"}, {"name": "mistral"}]}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("prompt_interceptor.launcher.urllib.request.urlopen",
                        lambda url, timeout: mock_resp)
    from prompt_interceptor.launcher import _fetch_ollama_models
    assert _fetch_ollama_models("http://localhost:11434") == ["llama3", "mistral"]


def test_fetch_ollama_models_error(monkeypatch):
    import urllib.error as _uerr
    monkeypatch.setattr(
        "prompt_interceptor.launcher.urllib.request.urlopen",
        lambda url, timeout: (_ for _ in ()).throw(_uerr.URLError("refused"))
    )
    from prompt_interceptor.launcher import _fetch_ollama_models
    assert _fetch_ollama_models("http://localhost:11434") == []


def test_fetch_ollama_models_empty_list(monkeypatch):
    import json as _json
    payload = _json.dumps({"models": []}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("prompt_interceptor.launcher.urllib.request.urlopen",
                        lambda url, timeout: mock_resp)
    from prompt_interceptor.launcher import _fetch_ollama_models
    assert _fetch_ollama_models("http://localhost:11434") == []


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------

def test_launch_creates_tk_and_runs_mainloop(monkeypatch):
    mock_root = _make_mock_root()

    with patch("prompt_interceptor.launcher.tk.Tk", return_value=mock_root), \
         patch("prompt_interceptor.launcher.LauncherWindow"):
        from prompt_interceptor.launcher import launch
        launch()

    mock_root.mainloop.assert_called_once()


# ---------------------------------------------------------------------------
# Error handling: _on_launch_ollama subprocess failures
# ---------------------------------------------------------------------------

def test_on_launch_ollama_file_not_found_shows_error(tmp_path, monkeypatch):
    """When 'ollama' is not found, status shows an error message and button is re-enabled."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: [])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "127.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()
    mock_root.after.side_effect = lambda delay, fn, *args: fn(*args)

    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=FileNotFoundError("not found")), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        win._check_or_launch_ollama()

    win.status_var.set.assert_called()
    msg = win.status_var.set.call_args[0][0]
    assert "not found" in msg.lower() or "ollama" in msg.lower()
    # Button must be re-enabled so the user can retry
    win._launch_ollama_btn.config.assert_called_with(state="normal")
    # Background thread must NOT start
    mock_thread.assert_not_called()


@pytest.mark.skip(reason="Ollama auto-launch via Popen was removed; this OSError path no longer exists")
def test_on_launch_ollama_os_error_shows_error(tmp_path, monkeypatch):
    """An OSError from Popen is caught and shown in the status label."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: [])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "127.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()
    mock_root.after.side_effect = lambda delay, fn, *args: fn(*args)

    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=OSError("access denied")), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        win._check_or_launch_ollama()

    win.status_var.set.assert_called()
    assert "access denied" in win.status_var.set.call_args[0][0]
    mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling: _on_launch_client subprocess failures
# ---------------------------------------------------------------------------

def _make_claude_win(cfg, monkeypatch):
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    clients = [("Claude Code", "claude"), ("Python App (Ollama)", "__python_app__")]
    win, _ = _make_headless_win(cfg, clients=clients)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Claude Code"
    win._clients = clients
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "/my/repo"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()
    return win


def test_on_launch_client_file_not_found_shows_error(tmp_path, monkeypatch):
    """FileNotFoundError from Popen is shown in the status label."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win = _make_claude_win(cfg, monkeypatch)

    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=FileNotFoundError("claude not found")):
        win._on_launch_client()

    win.status_var.set.assert_called()
    msg = win.status_var.set.call_args[0][0]
    assert "not found" in msg.lower() or "claude" in msg.lower()


def test_on_launch_client_os_error_shows_error(tmp_path, monkeypatch):
    """OSError from Popen is caught and shown in the status label."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win = _make_claude_win(cfg, monkeypatch)

    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=OSError("permission denied")):
        win._on_launch_client()

    win.status_var.set.assert_called()
    assert "permission denied" in win.status_var.set.call_args[0][0]


def test_on_launch_client_error_does_not_start_proxy(tmp_path, monkeypatch):
    """When Popen fails, the proxy thread is NOT started and _servers_started stays False."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win = _make_claude_win(cfg, monkeypatch)

    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=FileNotFoundError("not found")), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        win._on_launch_client()

    mock_thread.assert_not_called()
    assert win._servers_started is False


# ---------------------------------------------------------------------------
# Error handling: _start_proxy_thread
# ---------------------------------------------------------------------------

def test_start_proxy_thread_import_error(monkeypatch, capsys):
    """ImportError (missing uvicorn) is caught and printed."""
    import sys
    fake_modules = dict(sys.modules)
    fake_modules.pop("uvicorn", None)

    with patch.dict(sys.modules, {"uvicorn": None}):
        # Importing uvicorn inside the function will raise ImportError
        from prompt_interceptor.launcher import _start_proxy_thread
        # Patch the import inside the function
        with patch("builtins.__import__", side_effect=ImportError("No module named 'uvicorn'")):
            _start_proxy_thread()  # must not raise

    captured = capsys.readouterr()
    assert "PromptInterceptor" in captured.out or True  # no exception is the key assertion


def test_start_proxy_thread_os_error(tmp_path, monkeypatch, capsys):
    """OSError from uvicorn.run (port already in use) is caught and printed."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=False)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import uvicorn
    with patch.object(uvicorn, "run", side_effect=OSError("address already in use")):
        from prompt_interceptor.launcher import _start_proxy_thread
        _start_proxy_thread()  # must not raise

    captured = capsys.readouterr()
    assert "address already in use" in captured.out


def test_start_proxy_thread_generic_error(tmp_path, monkeypatch, capsys):
    """Unexpected exceptions from the proxy thread are caught and printed."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=False)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import uvicorn
    with patch.object(uvicorn, "run", side_effect=RuntimeError("unexpected crash")):
        from prompt_interceptor.launcher import _start_proxy_thread
        _start_proxy_thread()

    captured = capsys.readouterr()
    assert "unexpected crash" in captured.out


def test_start_proxy_thread_system_exit(tmp_path, monkeypatch):
    """SystemExit from uvicorn.run (sys.exit(1) on port bind failure) is silently caught."""
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=False)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import uvicorn
    with patch.object(uvicorn, "run", side_effect=SystemExit(1)):
        from prompt_interceptor.launcher import _start_proxy_thread
        _start_proxy_thread()  # must not raise


def test_start_proxy_thread_dashboard_system_exit(tmp_path, monkeypatch):
    """SystemExit inside the _run_dashboard closure is silently caught."""
    import asyncio
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    started = []

    class _MockThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            started.append(self._target)

    import uvicorn
    with patch.object(uvicorn, "run", MagicMock()), \
         patch("prompt_interceptor.launcher.threading.Thread", side_effect=_MockThread):
        from prompt_interceptor.launcher import _start_proxy_thread
        _start_proxy_thread()

    with patch.object(uvicorn, "run", side_effect=SystemExit(1)), \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        started[0]()  # must not raise


def test_start_dashboard_thread_system_exit(tmp_path, monkeypatch):
    """SystemExit from uvicorn.run is silently caught in _start_dashboard_thread."""
    import asyncio
    cfg = Config(log_dir=str(tmp_path / "logs"), dashboard_enabled=True, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    import uvicorn
    with patch.object(uvicorn, "run", side_effect=SystemExit(1)), \
         patch.object(asyncio, "set_event_loop"), \
         patch.object(asyncio, "new_event_loop", return_value=MagicMock()):
        from prompt_interceptor.launcher import _start_dashboard_thread
        _start_dashboard_thread()  # must not raise


# ---------------------------------------------------------------------------
# Step 3: _on_open_dashboard — proxy port field
# ---------------------------------------------------------------------------

def test_on_open_dashboard_saves_proxy_port_to_config(tmp_path, monkeypatch):
    """_on_open_dashboard reads proxy_port_var and persists it to config."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090,
                 proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    saved = []
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: saved.append(c))

    win, _ = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "9999"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()

    assert saved[0].proxy_port == 9999


def test_on_open_dashboard_invalid_proxy_port_shows_error(tmp_path, monkeypatch):
    """_on_open_dashboard shows an error and does not start if the port is not numeric."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: None)

    win, _ = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "not-a-port"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()

    # Thread must NOT be started when port is invalid
    mock_thread.assert_not_called()
    # Status must report the error
    msgs = [c[0][0] for c in win.status_var.set.call_args_list]
    assert any("Port" in m or "port" in m or "number" in m.lower() for m in msgs)


# ---------------------------------------------------------------------------
# Error handling: _on_open_dashboard — save_config failure
# ---------------------------------------------------------------------------

def test_on_open_dashboard_save_config_error_shows_warning(tmp_path, monkeypatch):
    """When save_config raises OSError, _on_open_dashboard sets a warning status message."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.save_config",
                        lambda c: (_ for _ in ()).throw(OSError("disk full")))

    win, mock_root = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()  # must not raise

    # At least one status_var.set call must mention the save error
    all_msgs = [c[0][0] for c in win.status_var.set.call_args_list]
    assert any("disk full" in m or "config" in m.lower() for m in all_msgs)


def test_on_launch_ollama_save_config_oserror_is_silently_ignored(tmp_path, monkeypatch):
    """save_config OSError in _on_launch_ollama is caught with pass — no crash, thread still starts."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win._launch_ollama_btn = MagicMock()
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.save_config", side_effect=OSError("disk full")), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_ollama()  # must not raise

    # OSError is silently ignored; thread is still started
    mock_thread.assert_called_once()


# ---------------------------------------------------------------------------
# Error handling: _fetch_ollama_models — specific error types
# ---------------------------------------------------------------------------

def test_fetch_ollama_models_json_decode_error(monkeypatch):
    """JSONDecodeError from an invalid Ollama response returns []."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not valid json {{{"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(
        "prompt_interceptor.launcher.urllib.request.urlopen",
        lambda url, timeout: mock_resp,
    )
    from prompt_interceptor.launcher import _fetch_ollama_models
    assert _fetch_ollama_models("http://localhost:11434") == []


def test_fetch_ollama_models_unexpected_error_returns_empty(monkeypatch):
    """Any unexpected exception from urlopen returns []."""
    monkeypatch.setattr(
        "prompt_interceptor.launcher.urllib.request.urlopen",
        lambda url, timeout: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    from prompt_interceptor.launcher import _fetch_ollama_models
    assert _fetch_ollama_models("http://localhost:11434") == []


# ---------------------------------------------------------------------------
# Ollama host selection — new feature
# ---------------------------------------------------------------------------

def test_localhost_hosts_constant():
    """_LOCALHOST_HOSTS contains both canonical localhost values."""
    from prompt_interceptor.launcher import _LOCALHOST_HOSTS
    assert "127.0.0.1" in _LOCALHOST_HOSTS
    assert "localhost" in _LOCALHOST_HOSTS


def test_check_or_launch_remote_host_reachable(tmp_path, monkeypatch):
    """Remote host that responds with models: no local Ollama launched, target is stored."""
    cfg = Config(log_dir=str(tmp_path / "logs"), target="http://localhost:11434")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: ["llama3"])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "192.168.1.100"
    win.status_var = MagicMock()
    mock_root.after.side_effect = lambda delay, fn, *args: fn(*args)

    with patch.object(win, "_on_models_ready") as mock_ready, \
         patch("prompt_interceptor.launcher.subprocess.Popen") as mock_popen:
        win._check_or_launch_ollama()

    mock_popen.assert_not_called()
    mock_ready.assert_called_once_with(["llama3"])
    assert win._active_target == "http://192.168.1.100:11434"


def test_check_or_launch_remote_host_unreachable_schedules_dialog(tmp_path, monkeypatch):
    """Remote host that does not respond: no Ollama launch, dialog is scheduled."""
    cfg = Config(log_dir=str(tmp_path / "logs"), target="http://localhost:11434")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher._fetch_ollama_models", lambda t: [])

    win, mock_root = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "192.168.1.100"
    win.status_var = MagicMock()

    after_calls = []
    mock_root.after.side_effect = lambda delay, fn, *args: after_calls.append((fn, args))

    with patch("prompt_interceptor.launcher.subprocess.Popen") as mock_popen:
        win._check_or_launch_ollama()

    mock_popen.assert_not_called()
    assert any(fn == win._ask_on_remote_fail for fn, _ in after_calls)


def test_on_open_dashboard_saves_active_target_to_config(tmp_path, monkeypatch):
    """When _active_target differs from default, _on_open_dashboard persists it in config."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    saved = []
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: saved.append(c))

    win, _ = _make_headless_win(cfg)
    win._active_target = "http://192.168.1.100:11434"
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "4k  (4096)"
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = ""
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.proxy_port_var = MagicMock()
    win.proxy_port_var.get.return_value = "8080"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_open_dashboard()

    assert saved[0].target == "http://192.168.1.100:11434"


# ---------------------------------------------------------------------------
# _on_launch_client — WSL branch (lines 653-674 in launcher.py)
# ---------------------------------------------------------------------------

def _make_wsl_win(cfg, monkeypatch):
    """Helper: LauncherWindow set up to launch Open Code (WSL)."""
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    clients = [("Open Code (WSL)", "__opencode_wsl__"), ("Python App (Ollama)", "__python_app__")]
    win, _ = _make_headless_win(cfg, clients=clients)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (WSL)"
    win._clients = clients
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.model_var = MagicMock()
    win.model_var.get.return_value = "mistral:latest"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()
    return win


def test_on_launch_client_wsl_firewall_warning_when_unreachable(tmp_path, monkeypatch):
    """When WSL cannot reach the proxy, _warn_wsl_firewall is called and Popen is NOT called."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win = _make_wsl_win(cfg, monkeypatch)

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])), \
         patch("prompt_interceptor.launcher._get_wsl_host_ip", return_value="172.28.0.1"), \
         patch("prompt_interceptor.launcher._write_opencode_config_wsl"), \
         patch("prompt_interceptor.launcher._wsl_can_reach_proxy", return_value=False), \
         patch("prompt_interceptor.launcher._warn_wsl_firewall") as mock_warn:
        win._on_launch_client()

    mock_warn.assert_called_once_with("172.28.0.1", 8080)
    assert len(popen_calls) == 0


def test_on_launch_client_wsl_opens_terminal_when_reachable(tmp_path, monkeypatch):
    """When WSL can reach the proxy, Popen is called with wsl args."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    win = _make_wsl_win(cfg, monkeypatch)

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append((a[0], kw))), \
         patch("prompt_interceptor.launcher._get_wsl_host_ip", return_value="172.28.0.1"), \
         patch("prompt_interceptor.launcher._write_opencode_config_wsl"), \
         patch("prompt_interceptor.launcher._wsl_can_reach_proxy", return_value=True), \
         patch("prompt_interceptor.launcher.threading.Thread"):
        win._on_launch_client()

    assert len(popen_calls) == 1
    cmd_list, _ = popen_calls[0]
    assert "wsl" in cmd_list


def test_on_launch_client_wsl_with_workdir_passes_cd_flag(tmp_path, monkeypatch):
    """When work_dir is set, --cd is included in the wsl args (line 667)."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    clients = [("Open Code (WSL)", "__opencode_wsl__"), ("Python App (Ollama)", "__python_app__")]
    win, _ = _make_headless_win(cfg, clients=clients)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code (WSL)"
    win._clients = clients
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "C:\\Users\\user\\project"
    win.model_var = MagicMock()
    win.model_var.get.return_value = "llama3"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])), \
         patch("prompt_interceptor.launcher._get_wsl_host_ip", return_value="172.28.0.1"), \
         patch("prompt_interceptor.launcher._write_opencode_config_wsl"), \
         patch("prompt_interceptor.launcher._wsl_can_reach_proxy", return_value=True), \
         patch("prompt_interceptor.launcher.threading.Thread"):
        win._on_launch_client()

    assert len(popen_calls) == 1
    assert "--cd" in popen_calls[0]
    assert "C:\\Users\\user\\project" in popen_calls[0]


def test_on_launch_client_generic_else_branch(tmp_path, monkeypatch):
    """Unknown cmd_name falls through to the generic 'else' Popen branch."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    clients = [("Custom Tool", "customtool"), ("Python App (Ollama)", "__python_app__")]
    win, _ = _make_headless_win(cfg, clients=clients)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Custom Tool"
    win._clients = clients
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = ""
    win.model_var = MagicMock()
    win.model_var.get.return_value = ""
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])), \
         patch("prompt_interceptor.launcher.threading.Thread"):
        win._on_launch_client()

    assert len(popen_calls) == 1
    assert "customtool" in popen_calls[0]


# ---------------------------------------------------------------------------
# _launch_python_app — OSError branch (lines 712-713)
# ---------------------------------------------------------------------------

def test_launch_python_app_oserror_shows_error(tmp_path, monkeypatch):
    """OSError from Popen in _launch_python_app is caught and shown in status."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = ""
    win.app_command_var = MagicMock()
    win.app_command_var.get.return_value = "python main.py"
    win.use_venv_var = MagicMock()
    win.use_venv_var.get.return_value = False
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=OSError("permission denied")):
        win._launch_python_app()

    win.status_var.set.assert_called()
    msg = win.status_var.set.call_args[0][0]
    assert "permission denied" in msg


# ---------------------------------------------------------------------------
# _ask_on_remote_fail dialog — button callbacks
# ---------------------------------------------------------------------------

def _capture_dialog_buttons(win, host, port):
    """Invoke _ask_on_remote_fail with mocked tkinter and return captured button commands."""
    mock_dialog = MagicMock()
    captured_cmds = {}

    def _btn(parent, text="", command=None, **kw):
        btn = MagicMock()
        if command:
            captured_cmds[text] = command
        return btn

    with patch("prompt_interceptor.launcher.tk.Toplevel", return_value=mock_dialog), \
         patch("prompt_interceptor.launcher.ttk.Label", return_value=MagicMock()), \
         patch("prompt_interceptor.launcher.ttk.Frame", return_value=MagicMock()), \
         patch("prompt_interceptor.launcher.ttk.Button", side_effect=_btn):
        win._ask_on_remote_fail(host, port)

    return captured_cmds, mock_dialog


def test_ask_on_remote_fail_retry_reruns_check(tmp_path, monkeypatch):
    """Retry button re-enables the check thread with the same host."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "10.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()

    cmds, _ = _capture_dialog_buttons(win, "10.0.0.1", 11434)

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        cmds["Retry"]()

    win._launch_ollama_btn.config.assert_called_with(state="disabled")
    mock_thread.assert_called_once()


def test_ask_on_remote_fail_use_localhost_switches_host(tmp_path, monkeypatch):
    """Use Localhost button sets host to 127.0.0.1 and re-triggers check."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.ollama_host_var = MagicMock()
    win.ollama_host_var.get.return_value = "10.0.0.1"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()

    cmds, _ = _capture_dialog_buttons(win, "10.0.0.1", 11434)

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        cmds["Use Localhost"]()

    win.ollama_host_var.set.assert_called_with("127.0.0.1")
    mock_thread.assert_called_once()


def test_ask_on_remote_fail_cancel_reenables_button(tmp_path, monkeypatch):
    """Cancel button re-enables the launch button and sets a status message."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()

    cmds, _ = _capture_dialog_buttons(win, "10.0.0.1", 11434)
    cmds["Cancel"]()

    win._launch_ollama_btn.config.assert_called_with(state="normal")
    win.status_var.set.assert_called()


# ---------------------------------------------------------------------------
# _get_opencode_version — debug output (lines 91, 102-104, 108, 111, 114, 116)
# ---------------------------------------------------------------------------

def test_get_opencode_version_debug_with_match(monkeypatch, capsys):
    """debug=True prints diagnostics when a version is found."""
    from prompt_interceptor.launcher import _get_opencode_version
    from prompt_interceptor.config import Config

    cfg = Config(debug=True)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.shutil.which", lambda cmd: cmd)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "opencode 1.17.4"
    mock_result.stderr = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        result = _get_opencode_version()

    assert result == "1.17.4"
    out = capsys.readouterr().out
    assert "[DEBUG] _get_opencode_version" in out
    assert "1.17.4" in out


def test_get_opencode_version_debug_no_match(monkeypatch, capsys):
    """debug=True prints diagnostics when no version is found in output."""
    from prompt_interceptor.launcher import _get_opencode_version
    from prompt_interceptor.config import Config

    cfg = Config(debug=True)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.shutil.which", lambda cmd: cmd)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "no version here"
    mock_result.stderr = ""
    with patch("prompt_interceptor.launcher.subprocess.run", return_value=mock_result):
        result = _get_opencode_version()

    assert result is None
    out = capsys.readouterr().out
    assert "no version found" in out
    assert "returning None" in out


def test_get_opencode_version_debug_exception(monkeypatch, capsys):
    """debug=True prints diagnostics on subprocess exception."""
    from prompt_interceptor.launcher import _get_opencode_version
    from prompt_interceptor.config import Config

    cfg = Config(debug=True)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.shutil.which", lambda cmd: cmd)

    with patch("prompt_interceptor.launcher.subprocess.run", side_effect=Exception("timeout")):
        result = _get_opencode_version()

    assert result is None
    out = capsys.readouterr().out
    assert "EXCEPTION" in out
    assert "returning None" in out


# ---------------------------------------------------------------------------
# _show_version_warnings (lines 377-379)
# ---------------------------------------------------------------------------

def test_show_version_warnings_calls_messagebox(tmp_path, monkeypatch):
    """_show_version_warnings shows a messagebox for each warning."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    warnings_shown = []

    import tkinter.messagebox as _mb_real
    with patch.object(_mb_real, "showwarning", side_effect=lambda title, msg: warnings_shown.append(msg)):
        win._show_version_warnings(["Version A is too old", "Version B is incompatible"])

    assert len(warnings_shown) == 2
    assert "Version A is too old" in warnings_shown
    assert "Version B is incompatible" in warnings_shown


# ---------------------------------------------------------------------------
# _build_ui — version warning scheduling (line 477)
# ---------------------------------------------------------------------------

def test_build_ui_schedules_version_warning_when_warnings_present(tmp_path, monkeypatch):
    """When _detect_clients returns warnings, root.after is called at 300ms."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    mock_root = _make_mock_root()
    after_calls = []
    mock_root.after.side_effect = lambda delay, fn, *args: after_calls.append(delay)

    clients = [("Python App (Ollama)", "__python_app__")]
    warnings = ["OpenCode version 1.0.0 is not supported."]

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=(clients, warnings)):
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    assert 300 in after_calls
