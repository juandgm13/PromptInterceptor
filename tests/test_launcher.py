"""Tests for prompt_interceptor/launcher.py."""
import shutil
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
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    assert len(clients) == 1
    assert clients[0] == ("Python App (Ollama)", "__python_app__")


def test_detect_clients_claude_only(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    assert len(clients) == 2
    assert clients[0] == ("Claude Code", "claude")
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")


def test_detect_clients_opencode_only(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    assert len(clients) == 2
    assert clients[0] == ("Open Code", "opencode")
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")


def test_detect_clients_both(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    names = [name for name, _ in clients]
    assert "Claude Code" in names
    assert "Open Code" in names
    assert "Python App (Ollama)" in names
    assert len(clients) == 3


def test_detect_clients_python_app_always_last(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    from prompt_interceptor.launcher import _detect_clients
    clients = _detect_clients()
    assert clients[-1] == ("Python App (Ollama)", "__python_app__")


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
         patch("prompt_interceptor.launcher._detect_clients", return_value=clients):
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=python_only):
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=_DEFAULT_CLIENTS):
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=_DEFAULT_CLIENTS), \
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
         patch("prompt_interceptor.launcher._detect_clients", return_value=_DEFAULT_CLIENTS), \
         patch("prompt_interceptor.launcher.Path.exists", return_value=False):
        mock_tk.StringVar.return_value = MagicMock()
        from prompt_interceptor.launcher import LauncherWindow
        win = LauncherWindow(mock_root)

    mock_root.iconphoto.assert_not_called()


# ---------------------------------------------------------------------------
# Step 1: _on_launch_ollama
# ---------------------------------------------------------------------------

def test_on_launch_ollama_opens_process_and_thread(tmp_path, monkeypatch):
    """_on_launch_ollama opens an Ollama CMD process and starts a background thread."""
    cfg = Config(log_dir=str(tmp_path / "logs"), context_size=8192)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "8k  (8192)"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])) as mock_popen, \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_ollama()

    assert len(popen_calls) == 1
    cmd_str = " ".join(popen_calls[0])
    assert "8192" in cmd_str
    assert "ollama serve" in cmd_str
    mock_thread.assert_called_once()
    win._launch_ollama_btn.config.assert_called_with(state="disabled")


def test_on_launch_ollama_uses_selected_ctx(tmp_path, monkeypatch):
    """Context size from ctx_var is correctly passed to the Ollama command."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.ctx_var = MagicMock()
    win.ctx_var.get.return_value = "32k (32768)"
    win.status_var = MagicMock()
    win._launch_ollama_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])), \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_launch_ollama()

    assert "32768" in " ".join(popen_calls[0])


# ---------------------------------------------------------------------------
# Step 1: _fetch_models_after_launch / _on_models_ready
# ---------------------------------------------------------------------------

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
    """_on_models_ready fills the model combobox and unlocks Step 2."""
    cfg = Config(log_dir=str(tmp_path / "logs"), default_model="")
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win._model_cb = MagicMock()
    win.model_var = MagicMock()
    win.status_var = MagicMock()

    with patch.object(win, "_set_step2_enabled") as mock_unlock:
        win._on_models_ready(["llama3", "mistral"])

    win._model_cb.config.assert_called_with(state="readonly")
    mock_unlock.assert_called_once_with(True)
    win.status_var.set.assert_called()


def test_on_models_ready_no_models(tmp_path, monkeypatch):
    """When no models are returned, combobox is set to normal and Step 2 still unlocks."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win._model_cb = MagicMock()
    win.model_var = MagicMock()
    win.status_var = MagicMock()

    with patch.object(win, "_set_step2_enabled") as mock_unlock:
        win._on_models_ready([])

    win._model_cb.config.assert_called_with(state="normal")
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
    win._launch_client_btn = MagicMock()

    with patch.object(win, "_refresh_client_rows"):
        win._set_step2_enabled(True)

    win._client_cb.config.assert_called_with(state="readonly")
    win._launch_client_btn.config.assert_called_with(state="normal")
    assert win._step2_enabled is True


def test_set_step2_enabled_disables_widgets(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win._client_cb = MagicMock()
    win._launch_client_btn = MagicMock()

    with patch.object(win, "_refresh_client_rows"):
        win._set_step2_enabled(False)

    win._client_cb.config.assert_called_with(state="disabled")
    win._launch_client_btn.config.assert_called_with(state="disabled")
    assert win._step2_enabled is False


def test_refresh_client_rows_python_app_shows_app_fields(tmp_path):
    """When Python App is selected, app/env rows are shown and workdir is hidden."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._step2_enabled = True
    win._row_workdir = MagicMock()
    win._row_apppath = MagicMock()
    win._row_envvar = MagicMock()
    win._app_path_entry = MagicMock()
    win._browse_app_btn = MagicMock()
    win._env_var_entry = MagicMock()
    win._work_dir_entry = MagicMock()
    win._browse_workdir_btn = MagicMock()

    win._refresh_client_rows()

    win._row_workdir.pack_forget.assert_called()
    win._row_apppath.pack.assert_called()
    win._row_envvar.pack.assert_called()
    win._app_path_entry.config.assert_called_with(state="normal")
    win._env_var_entry.config.assert_called_with(state="normal")


def test_refresh_client_rows_claude_shows_workdir(tmp_path):
    """When Claude Code is selected, work-dir row is shown and app fields are hidden."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Claude Code"
    win._step2_enabled = True
    win._row_workdir = MagicMock()
    win._row_apppath = MagicMock()
    win._row_envvar = MagicMock()
    win._work_dir_entry = MagicMock()
    win._browse_workdir_btn = MagicMock()
    win._app_path_entry = MagicMock()
    win._browse_app_btn = MagicMock()
    win._env_var_entry = MagicMock()

    win._refresh_client_rows()

    win._row_apppath.pack_forget.assert_called()
    win._row_envvar.pack_forget.assert_called()
    win._row_workdir.pack.assert_called()
    win._work_dir_entry.config.assert_called_with(state="normal")


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
    with patch("prompt_interceptor.launcher.filedialog.askopenfilename", return_value="/app/main.py"):
        win._on_browse_app()
    win.app_path_var.set.assert_called_once_with("/app/main.py")


def test_on_browse_app_no_selection(tmp_path):
    cfg = Config(log_dir=str(tmp_path / "logs"))
    win, _ = _make_headless_win(cfg)
    win.app_path_var = MagicMock()
    with patch("prompt_interceptor.launcher.filedialog.askopenfilename", return_value=""):
        win._on_browse_app()
    win.app_path_var.set.assert_not_called()


# ---------------------------------------------------------------------------
# Step 2: _on_launch_client
# ---------------------------------------------------------------------------

def test_on_launch_client_claude_code(tmp_path, monkeypatch):
    """Launching Claude Code runs `cd /d <dir> && claude` in a new CMD window."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Claude Code", "claude"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Claude Code"
    win._clients = [("Claude Code", "claude"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "/my/repo"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        win._on_launch_client()

    assert len(popen_calls) == 1
    cmd_str = " ".join(popen_calls[0])
    assert "claude" in cmd_str
    assert "/my/repo" in cmd_str
    assert "cd /d" in cmd_str
    win._start_btn.config.assert_called_with(state="normal")


def test_on_launch_client_open_code(tmp_path, monkeypatch):
    """Launching Open Code runs `cd /d <dir> && opencode` in a new CMD window."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg, clients=[("Open Code", "opencode"),
                                               ("Python App (Ollama)", "__python_app__")])
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Open Code"
    win._clients = [("Open Code", "opencode"), ("Python App (Ollama)", "__python_app__")]
    win.work_dir_var = MagicMock()
    win.work_dir_var.get.return_value = "/my/repo"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        win._on_launch_client()

    assert len(popen_calls) == 1
    cmd_str = " ".join(popen_calls[0])
    assert "opencode" in cmd_str


def test_on_launch_client_python_app(tmp_path, monkeypatch):
    """Launching Python App sets the env var to the proxy URL and runs python."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._clients = _DEFAULT_CLIENTS
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = "/apps/myapp/main.py"
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        win._on_launch_client()

    assert len(popen_calls) == 1
    cmd_str = " ".join(popen_calls[0])
    assert "OLLAMA_HOST" in cmd_str
    assert "http://localhost:8080" in cmd_str
    assert "python" in cmd_str
    assert "main.py" in cmd_str
    win._start_btn.config.assert_called_with(state="normal")


def test_on_launch_client_python_app_no_path(tmp_path, monkeypatch):
    """Python App launch is blocked when App Path is empty."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._clients = _DEFAULT_CLIENTS
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = "  "  # blank
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    with patch("prompt_interceptor.launcher.subprocess.Popen") as mock_popen:
        win._on_launch_client()

    mock_popen.assert_not_called()
    win._start_btn.config.assert_not_called()
    win.status_var.set.assert_called()


def test_on_launch_client_python_app_custom_env_var(tmp_path, monkeypatch):
    """Custom env var name is used instead of the default OLLAMA_HOST."""
    cfg = Config(log_dir=str(tmp_path / "logs"), proxy_port=8080)
    monkeypatch.setattr("prompt_interceptor.launcher.get_config", lambda: cfg)

    win, _ = _make_headless_win(cfg)
    win.client_var = MagicMock()
    win.client_var.get.return_value = "Python App (Ollama)"
    win._clients = _DEFAULT_CLIENTS
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = "/app/main.py"
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OPENAI_BASE_URL"
    win.status_var = MagicMock()
    win._start_btn = MagicMock()

    popen_calls = []
    with patch("prompt_interceptor.launcher.subprocess.Popen",
               side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        win._on_launch_client()

    cmd_str = " ".join(popen_calls[0])
    assert "OPENAI_BASE_URL" in cmd_str


def test_on_launch_client_unlocks_start_btn(tmp_path, monkeypatch):
    """After a successful client launch, the Start button is enabled."""
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
    win._start_btn = MagicMock()

    with patch("prompt_interceptor.launcher.subprocess.Popen"):
        win._on_launch_client()

    win._start_btn.config.assert_called_with(state="normal")


# ---------------------------------------------------------------------------
# Step 3: _on_start — only starts proxy + dashboard, no Popen
# ---------------------------------------------------------------------------

def test_on_start_saves_config_and_starts_proxy(tmp_path, monkeypatch):
    """_on_start saves config, starts proxy thread, and opens dashboard. No Popen."""
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
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = ""
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = "OLLAMA_HOST"
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.subprocess.Popen") as mock_popen, \
         patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_start()

    # Config saved with correct context_size
    assert len(saved) == 1
    assert saved[0].context_size == 4096

    # Proxy thread started
    mock_thread.assert_called_once()

    # No Popen — Ollama and client are NOT launched by _on_start
    mock_popen.assert_not_called()

    # Dashboard scheduled via root.after
    mock_root.after.assert_called()


def test_on_start_saves_model_to_config(tmp_path, monkeypatch):
    """_on_start persists the selected model as default_model."""
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
    win.app_path_var = MagicMock()
    win.app_path_var.get.return_value = ""
    win.env_var_var = MagicMock()
    win.env_var_var.get.return_value = ""
    win.status_var = MagicMock()

    with patch("prompt_interceptor.launcher.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        win._on_start()

    assert saved[0].default_model == "deepseek-coder"


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
