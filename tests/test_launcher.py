"""Tests for prompt_interceptor/launcher.py."""
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

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
# _detect_clients
# ---------------------------------------------------------------------------

def test_detect_clients_none_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    from prompt_interceptor.launcher import _detect_clients
    assert _detect_clients() == []


def test_detect_clients_claude_only(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    assert len(clients) == 1
    assert clients[0] == ("Claude Code", "claude")


def test_detect_clients_opencode_only(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    assert len(clients) == 1
    assert clients[0] == ("Open Code", "opencode")


def test_detect_clients_both(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    names = [name for name, _ in clients]
    assert "Claude Code" in names
    assert "Open Code" in names


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

    # Call the dashboard thread target to cover its body (lines 48-49)
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
# LauncherWindow (fully mocked tkinter)
# ---------------------------------------------------------------------------

def _make_mock_root():
    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    return root


def test_launcher_window_init_no_clients(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    mock_root = _make_mock_root()

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=[]):
        mock_tk.StringVar.return_value = MagicMock()
        # PhotoImage might fail without a display; ensure it raises to exercise the except branch
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=[("Claude Code", "claude")]):
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=[]), \
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=[]), \
         patch("prompt_interceptor.launcher.Path.exists", return_value=False):
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    mock_root.iconphoto.assert_not_called()


# ---------------------------------------------------------------------------
# LauncherWindow._on_start
# ---------------------------------------------------------------------------

def test_on_start_saves_config_and_opens_processes(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    saved = []
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", lambda c: saved.append(c))

    mock_root = _make_mock_root()
    ctx_var = MagicMock()
    ctx_var.get.return_value = "4k  (4096)"
    client_var = MagicMock()
    client_var.get.return_value = "Claude Code"

    popen_calls = []

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=[("Claude Code", "claude")]), \
         patch("prompt_interceptor.launcher.subprocess.Popen", side_effect=lambda *a, **kw: popen_calls.append(a)) as mock_popen, \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)
        win.ctx_var = ctx_var
        win.client_var = client_var
        win._clients = [("Claude Code", "claude")]
        win._on_start()

    assert len(saved) == 1
    assert saved[0].context_size == 4096
    # Ollama terminal + client terminal both opened
    assert len(popen_calls) >= 2
    mock_thread.assert_called_once()


def test_on_start_no_client_selected(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=4096, dashboard_port=9090)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)
    monkeypatch.setattr("prompt_interceptor.launcher.save_config", MagicMock())

    mock_root = _make_mock_root()
    ctx_var = MagicMock()
    ctx_var.get.return_value = "4k  (4096)"
    client_var = MagicMock()
    client_var.get.return_value = ""  # no client selected

    popen_calls = []

    with patch("prompt_interceptor.launcher.ttk"), \
         patch("prompt_interceptor.launcher.tk") as mock_tk, \
         patch("prompt_interceptor.launcher._detect_clients", return_value=[]), \
         patch("prompt_interceptor.launcher.subprocess.Popen", side_effect=lambda *a, **kw: popen_calls.append(a)), \
         patch("prompt_interceptor.launcher.threading.Thread"):
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)
        win.ctx_var = ctx_var
        win.client_var = client_var
        win._clients = []
        win._on_start()

    # Only Ollama terminal opened (no client terminal)
    assert len(popen_calls) == 1


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
