"""
PromptInterceptor Launcher - Desktop configuration window.
Uses only tkinter (Python built-in), no extra dependencies.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

from .client_versions import OPENCODE_MIN_VERSION, OPENCODE_MAX_VERSION
from .config import get_config, save_config
from ._version import __version__

_LOCALHOST_HOSTS = {"127.0.0.1", "localhost"}

_DEFAULT_OLLAMA_PORT = 11434


def _parse_ollama_host(text: str, default_port: int = _DEFAULT_OLLAMA_PORT) -> tuple:
    """
    Normalise whatever the user typed in the "Ollama Host" field.

    Accepts "192.168.1.50", "192.168.1.50:11434", "http://192.168.1.50:11434"
    and "my-server.local". Returns (target_url, hostname), where target_url
    always carries a scheme and no trailing slash (the proxy concatenates
    target + path, so a stray slash would produce "//api/chat").
    """
    text = (text or "").strip().rstrip("/")
    if not text:
        return f"http://127.0.0.1:{default_port}", "127.0.0.1"

    # urlsplit only recognises host/port when a scheme is present.
    if "://" not in text:
        text = f"http://{text}"

    parts = urllib.parse.urlsplit(text)
    host = parts.hostname or "127.0.0.1"
    try:
        port = parts.port or default_port
    except ValueError:
        # Malformed port (e.g. "1.2.3.4:11434:11434") — fall back to the default.
        port = default_port
    scheme = parts.scheme or "http"
    return f"{scheme}://{host}:{port}", host


def _format_ollama_host(target: str, default_port: int = _DEFAULT_OLLAMA_PORT) -> str:
    """Render a target URL for the host field, keeping a non-standard port visible."""
    parts = urllib.parse.urlsplit(target or "")
    host = parts.hostname or "127.0.0.1"
    try:
        port = parts.port
    except ValueError:
        port = None
    return host if port in (None, default_port) else f"{host}:{port}"


_CTX_OPTIONS = {
    "4k  (4096)":    4096,
    "8k  (8192)":    8192,
    "16k (16384)":   16384,
    "32k (32768)":   32768,
    "64k (65536)":   65536,
    "128k (131072)": 131072,
    "256k (262144)": 262144,
}

_CTX_DEFAULT = "32k (32768)"


def _is_wsl_available() -> bool:
    """Return True if wsl.exe is in PATH."""
    return shutil.which("wsl") is not None


def _detect_git_bash() -> bool:
    """Return True if Git Bash (bash.exe) is available on Windows."""
    if sys.platform != "win32":
        return False
    return shutil.which("bash") is not None


def _is_opencode_in_wsl() -> bool:
    """Return True if opencode is installed inside WSL."""
    if not _is_wsl_available():
        return False
    try:
        result = subprocess.run(
            ["wsl", "--", "which", "opencode"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _parse_version(v: str) -> tuple:
    """Convert '1.17.4' to (1, 17, 4). Non-numeric segments become 0."""
    parts = []
    for segment in v.strip().split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _is_version_compatible(version: str, min_v: str, max_v: str) -> bool:
    """Return True if min_v <= version <= max_v (inclusive, tuple comparison)."""
    return _parse_version(min_v) <= _parse_version(version) <= _parse_version(max_v)


def _get_opencode_version(cmd: str = "opencode") -> str | None:
    """Return the opencode version string (e.g. '1.17.4'), or None if undetectable.

    Resolves the full path via shutil.which first so that Windows .cmd/.bat
    wrappers are found by subprocess (which does not apply PATHEXT by default).
    Tries --version, then the 'version' subcommand, then -v.
    """
    debug = get_config().debug
    resolved = shutil.which(cmd) or cmd
    if debug:
        print(f"[DEBUG] _get_opencode_version: cmd={cmd!r}  resolved={resolved!r}")
    for args in ([resolved, "--version"], [resolved, "version"], [resolved, "-v"]):
        try:
            result = subprocess.run(
                args,
                capture_output=True, text=True, timeout=5
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            output = (stdout or stderr or "").strip()
            if debug:
                print(f"[DEBUG]   args={args}  rc={result.returncode}")
                print(f"[DEBUG]   stdout={stdout!r}")
                print(f"[DEBUG]   stderr={stderr!r}")
            match = re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', output)
            if match:
                if debug:
                    print(f"[DEBUG]   -> version={match.group(1)!r}")
                return match.group(1)
            if debug:
                print(f"[DEBUG]   -> no version found in output")
        except Exception as exc:
            if debug:
                print(f"[DEBUG]   args={args}  EXCEPTION: {type(exc).__name__}: {exc}")
    if debug:
        print(f"[DEBUG] _get_opencode_version: returning None")
    return None


def _get_wsl_host_ip() -> str:
    """Get the Windows host IP as seen from inside WSL (for proxy URL)."""
    try:
        result = subprocess.run(
            ["wsl", "--", "bash", "-c",
             "grep -m1 nameserver /etc/resolv.conf 2>/dev/null"],
            capture_output=True, text=True, timeout=5
        )
        line = result.stdout.strip()
        # "nameserver 10.255.255.254" -> "10.255.255.254"
        parts = line.split()
        if len(parts) >= 2:
            return parts[-1]
        if parts:
            return parts[0]
    except Exception:
        pass
    return "localhost"


def _wsl_can_reach_proxy(host_ip: str, port: int) -> bool:
    """Return True if WSL can TCP-connect to host_ip:port."""
    try:
        result = subprocess.run(
            ["wsl", "--", "bash", "-c",
             f"timeout 2 bash -c 'echo >/dev/tcp/{host_ip}/{port}' 2>/dev/null && echo ok"],
            capture_output=True, text=True, timeout=6
        )
        return result.stdout.strip() == "ok"
    except Exception:
        return False


def _warn_wsl_firewall(host_ip: str, port: int) -> None:
    """Show a dialog explaining the firewall issue and the fix command."""
    import tkinter.messagebox as mb
    rule_cmd = (
        f'netsh advfirewall firewall add rule name="PromptInterceptor" '
        f'protocol=TCP dir=in localport={port} action=allow'
    )
    mb.showwarning(
        "WSL cannot reach the proxy",
        f"WSL cannot connect to {host_ip}:{port}.\n\n"
        f"Common cause: Windows Firewall blocks port {port} from WSL2.\n\n"
        f"Fix — run this in PowerShell as Administrator:\n\n"
        f"{rule_cmd}\n\n"
        f"Then launch Open Code (WSL) again."
    )


def _write_opencode_config_wsl(model: str, proxy_url: str) -> None:
    """Write/update opencode.json inside WSL via stdin pipe."""
    read_cmd = "cat ~/.config/opencode/opencode.json 2>/dev/null || echo '{}'"
    try:
        result = subprocess.run(
            ["wsl", "--", "bash", "-c", read_cmd],
            capture_output=True, text=True, timeout=5
        )
        try:
            existing = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            existing = {}
    except Exception:
        existing = {}

    providers = existing.setdefault("provider", {})
    ollama = providers.setdefault("ollama", {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ollama",
    })
    ollama.setdefault("options", {})["baseURL"] = f"{proxy_url}/v1"
    ollama.setdefault("models", {})[model] = {"name": model}
    existing.setdefault("$schema", "https://opencode.ai/config.json")

    config_json = json.dumps(existing, indent=2)
    write_cmd = "mkdir -p ~/.config/opencode && cat > ~/.config/opencode/opencode.json"
    try:
        subprocess.run(
            ["wsl", "--", "bash", "-c", write_cmd],
            input=config_json, text=True, timeout=5
        )
        print(f"[PromptInterceptor] opencode WSL config written")
    except Exception as exc:
        print(f"[PromptInterceptor] Failed to write opencode WSL config: {exc}")


def _detect_clients() -> tuple:
    """Detect installed AI clients.

    Returns (clients, version_warnings) where:
      clients: list of (display_name, command) — always clean names
      version_warnings: list of warning strings for incompatible client versions
    """
    clients = [("Only Proxy", "__only_proxy__")]
    version_warnings = []
    # Claude Code disabled for now — proxy translation not yet stable for this client.
    # if shutil.which("claude"):
    #     clients.append(("Claude Code", "claude"))
    if shutil.which("opencode"):
        ver = _get_opencode_version()
        if ver is not None and not _is_version_compatible(ver, OPENCODE_MIN_VERSION, OPENCODE_MAX_VERSION):
            version_warnings.append(
                f"OpenCode is installed (version {ver}), but this version may not be "
                f"compatible with PromptInterceptor.\n\n"
                f"Supported versions: {OPENCODE_MIN_VERSION} – {OPENCODE_MAX_VERSION}\n\n"
                f"You can still use it, but some features may not work correctly."
            )
        clients.append(("Open Code (CLI)", "opencode"))
    # Open Code (WSL) support is implemented but disabled in the UI for now.
    # clients.append(("Open Code (WSL)", "__opencode_wsl__"))
    clients.append(("Python App (Ollama)", "__python_app__"))
    return clients, version_warnings


def _fetch_ollama_models(target: str) -> list:
    """Sync query to Ollama /api/tags. Returns [] if not reachable."""
    url = target + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except urllib.error.URLError as exc:
        print(f"[PromptInterceptor] Cannot reach Ollama at {url}: {exc}")
        return []
    except json.JSONDecodeError as exc:
        print(f"[PromptInterceptor] Invalid JSON from Ollama /api/tags: {exc}")
        return []
    except Exception as exc:
        print(f"[PromptInterceptor] Unexpected error fetching Ollama models: {exc}")
        return []


def _start_proxy_thread():
    """Run the FastAPI proxy and dashboard servers in background threads."""
    try:
        import asyncio
        import uvicorn

        asyncio.set_event_loop(asyncio.new_event_loop())
        config = get_config()

        if config.dashboard_enabled:
            def _run_dashboard():
                try:
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    uvicorn.run(
                        "prompt_interceptor.dashboard:app",
                        host="0.0.0.0",
                        port=config.dashboard_port,
                        log_level="warning",
                    )
                except SystemExit:
                    pass
            threading.Thread(target=_run_dashboard, daemon=True).start()

        uvicorn.run(
            "prompt_interceptor.main:app",
            host=config.proxy_host,
            port=config.proxy_port,
            log_level="info",
        )
    except ImportError as exc:
        print(f"[PromptInterceptor] Missing dependency for proxy server: {exc}")
    except OSError as exc:
        print(f"[PromptInterceptor] Failed to start proxy server (port in use?): {exc}")
    except SystemExit:
        pass
    except Exception as exc:
        print(f"[PromptInterceptor] Proxy thread error: {exc}")


def _start_dashboard_thread():
    """Run the FastAPI dashboard server standalone (no proxy)."""
    try:
        import asyncio
        import uvicorn

        asyncio.set_event_loop(asyncio.new_event_loop())
        config = get_config()
        if not config.dashboard_enabled:
            return
        uvicorn.run(
            "prompt_interceptor.dashboard:app",
            host="0.0.0.0",
            port=config.dashboard_port,
            log_level="warning",
        )
    except ImportError as exc:
        print(f"[PromptInterceptor] Missing dependency for dashboard: {exc}")
    except OSError as exc:
        print(f"[PromptInterceptor] Failed to start dashboard (port in use?): {exc}")
    except SystemExit:
        pass
    except Exception as exc:
        print(f"[PromptInterceptor] Dashboard thread error: {exc}")


def _write_opencode_config(model: str, proxy_url: str, work_dir: str = "") -> None:
    """Write/update .opencode/opencode.json in work_dir (or ~/.config/opencode if no work_dir)."""
    if work_dir:
        config_path = Path(work_dir) / ".opencode" / "opencode.json"
    else:
        config_path = Path.home() / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}

    providers = existing.setdefault("provider", {})
    ollama = providers.setdefault("ollama", {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ollama",
    })
    ollama.setdefault("options", {})["baseURL"] = f"{proxy_url}/v1"
    ollama.setdefault("models", {})[model] = {"name": model}

    # Enforce key order: $schema first, then shell (Windows only), then the rest
    existing.pop("$schema", None)
    existing.pop("shell", None)
    output: dict = {"$schema": "https://opencode.ai/config.json"}
    if sys.platform == "win32":
        output["shell"] = "bash" if _detect_git_bash() else "cmd"
    output.update(existing)

    config_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[PromptInterceptor] opencode config written to {config_path}")


class LauncherWindow:
    """Main launcher window for PromptInterceptor."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PromptInterceptor")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        self._step2_enabled = False
        self._servers_started = False
        self._active_target: str = get_config().target

        self._set_icon()
        self._build_ui()
        self._center_window(540, max(self._get_window_height(), self._required_height()))

    def _set_icon(self) -> None:
        icon_path = Path(__file__).parent.parent / "res" / "PromptInterceptor_Icon.png"
        if icon_path.exists():
            try:
                icon = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, icon)
                self.root._icon = icon  # prevent garbage collection
            except Exception:
                pass

    def _show_version_warnings(self, warnings: list) -> None:
        import tkinter.messagebox as mb
        for msg in warnings:
            mb.showwarning("Compatibility Warning", msg)

    def _get_window_height(self) -> int:
        """Return the appropriate window height for the currently selected client."""
        try:
            name = self.client_var.get()
        except AttributeError:
            return 460
        if name == "Python App (Ollama)":
            return 550
        if name == "Only Proxy":
            return 400
        return 460

    def _required_height(self) -> int:
        """Height tkinter says the packed content needs, or 0 when there is no real UI."""
        try:
            self.root.update_idletasks()
            return int(self.root.winfo_reqheight())
        except (TypeError, ValueError, AttributeError):
            return 0

    def _apply_window_height(self) -> None:
        """
        Resize to fit, never clipping the content.

        _get_window_height is a floor, not the last word: a bare constant silently
        cut off the Step 3 buttons and the status line once the layout grew.
        """
        height = max(self._get_window_height(), self._required_height())
        self.root.geometry(f"540x{height}")

    def _center_window(self, width: int, height: int) -> None:
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1a1a2e", foreground="#4fc3f7", font=("Segoe UI", 14, "bold"))
        style.configure("Section.TLabel", background="#1a1a2e", foreground="#666688", font=("Segoe UI", 8))
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TCombobox", fieldbackground="#16213e", background="#16213e", foreground="#e0e0e0",
                        selectbackground="#16213e", selectforeground="#e0e0e0")
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#16213e"), ("disabled", "#0d1117")],
                  foreground=[("readonly", "#e0e0e0"), ("disabled", "#555555")],
                  selectbackground=[("readonly", "#16213e")],
                  selectforeground=[("readonly", "#e0e0e0")])
        self.root.option_add("*TCombobox*Listbox.background", "#16213e")
        self.root.option_add("*TCombobox*Listbox.foreground", "#e0e0e0")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#4fc3f7")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#000000")
        style.configure("Action.TButton", background="#1e3a1e", foreground="#7ec87e",
                        font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("Start.TButton", background="#4fc3f7", foreground="#000000",
                        font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Exit.TButton", background="#333355", foreground="#e0e0e0",
                        font=("Segoe UI", 10), padding=8)
        style.map("Action.TButton",
                  background=[("active", "#2a4a2a"), ("disabled", "#1a1a2e")],
                  foreground=[("disabled", "#444444")])
        style.map("Start.TButton",
                  background=[("active", "#29b6f6"), ("disabled", "#1a2a3a")],
                  foreground=[("disabled", "#446688")])
        style.map("Exit.TButton", background=[("active", "#444466")])

        config = get_config()
        pad = {"padx": 20, "pady": 4}

        # ── Header ──
        ttk.Label(self.root, text="PromptInterceptor", style="Header.TLabel").pack(pady=(20, 2))
        ttk.Label(self.root, text="Ollama Traffic Interceptor for AI Clients",
                  font=("Segoe UI", 9)).pack(pady=(0, 2))
        ttk.Label(self.root, text=f"v{__version__}", style="Section.TLabel").pack(pady=(0, 6))

        # Proxy Port (global proxy setting, not Ollama-specific)
        row_port = ttk.Frame(self.root)
        row_port.pack(fill="x", **pad)
        ttk.Label(row_port, text="Proxy Port:", width=14, anchor="w").pack(side="left")
        self.proxy_port_var = tk.StringVar(value=str(config.proxy_port))
        ttk.Entry(row_port, textvariable=self.proxy_port_var, width=8).pack(side="left")

        # ── Step 1: Check Ollama ──
        ttk.Label(self.root, text="── Step 1: Check Ollama ──", style="Section.TLabel").pack(pady=(6, 0))

        # Ollama Host row. Keeps a non-standard port visible so it survives a
        # repaint instead of silently reverting to 11434 on the next check.
        initial_host = _format_ollama_host(config.target)
        row_host = ttk.Frame(self.root)
        row_host.pack(fill="x", **pad)
        ttk.Label(row_host, text="Ollama Host:", width=14, anchor="w").pack(side="left")
        self.ollama_host_var = tk.StringVar(value=initial_host)
        self._host_entry = ttk.Entry(row_host, textvariable=self.ollama_host_var, width=22)
        self._host_entry.pack(side="left")

        row1c = ttk.Frame(self.root)
        row1c.pack(fill="x", padx=20, pady=(2, 10))
        ttk.Label(row1c, text="", width=14).pack(side="left")
        self._launch_ollama_btn = ttk.Button(row1c, text="Check Ollama",
                                              style="Action.TButton",
                                              command=self._on_launch_ollama, width=18)
        self._launch_ollama_btn.pack(side="left")

        # ── Step 2: AI Client ──
        ttk.Label(self.root, text="── Step 2: AI Client ──", style="Section.TLabel").pack()

        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="AI Client:", width=14, anchor="w").pack(side="left")
        self._clients, _version_warnings = _detect_clients()
        self.client_var = tk.StringVar(value=self._clients[0][0])
        if _version_warnings:
            self.root.after(300, lambda w=_version_warnings: self._show_version_warnings(w))
        self._client_cb = ttk.Combobox(row2, textvariable=self.client_var,
                                        values=[name for name, _ in self._clients],
                                        state="disabled", width=22)
        self._client_cb.pack(side="left")
        self._client_cb.bind("<<ComboboxSelected>>", self._on_client_change)

        # Container for conditional client rows
        self._client_details = ttk.Frame(self.root)
        self._client_details.pack(fill="x")

        # Model row (Claude Code / Open Code only)
        self._row_model = ttk.Frame(self._client_details)
        ttk.Label(self._row_model, text="Model:", width=14, anchor="w").pack(side="left")
        self.model_var = tk.StringVar(value="")
        self._model_cb = ttk.Combobox(self._row_model, textvariable=self.model_var,
                                       values=[], width=22, state="disabled")
        self._model_cb.pack(side="left")

        # Work Dir row (Claude Code / Open Code)
        self._row_workdir = ttk.Frame(self._client_details)
        ttk.Label(self._row_workdir, text="Work Dir:", width=14, anchor="w").pack(side="left")
        self.work_dir_var = tk.StringVar(value=config.client_work_dir)
        self._work_dir_entry = ttk.Entry(self._row_workdir, textvariable=self.work_dir_var,
                                          width=26, state="disabled")
        self._work_dir_entry.pack(side="left", padx=(0, 4))
        self._browse_workdir_btn = ttk.Button(self._row_workdir, text="Browse…",
                                               command=self._on_browse_workdir,
                                               width=9, state="disabled")
        self._browse_workdir_btn.pack(side="left")

        # App Dir row (Python App — working directory)
        self._row_apppath = ttk.Frame(self._client_details)
        ttk.Label(self._row_apppath, text="App Dir:", width=14, anchor="w").pack(side="left")
        self.app_path_var = tk.StringVar(value=config.python_app_path)
        self._app_path_entry = ttk.Entry(self._row_apppath, textvariable=self.app_path_var,
                                          width=26, state="disabled")
        self._app_path_entry.pack(side="left", padx=(0, 4))
        self._browse_app_btn = ttk.Button(self._row_apppath, text="Browse…",
                                           command=self._on_browse_app,
                                           width=9, state="disabled")
        self._browse_app_btn.pack(side="left")

        # Command row (Python App)
        self._row_command = ttk.Frame(self._client_details)
        ttk.Label(self._row_command, text="Command:", width=14, anchor="w").pack(side="left")
        self.app_command_var = tk.StringVar(value=config.python_app_command or "python main.py")
        self._command_entry = ttk.Entry(self._row_command, textvariable=self.app_command_var,
                                         width=36, state="disabled")
        self._command_entry.pack(side="left")

        # Venv row (Python App)
        self._row_venv = ttk.Frame(self._client_details)
        ttk.Label(self._row_venv, text="", width=14).pack(side="left")
        self.use_venv_var = tk.BooleanVar(value=config.python_app_use_venv)
        self._venv_check = ttk.Checkbutton(self._row_venv, text="Use venv (auto: .venv / venv)",
                                            variable=self.use_venv_var, state="disabled")
        self._venv_check.pack(side="left")

        # Env Var row (Python App)
        self._row_envvar = ttk.Frame(self._client_details)
        ttk.Label(self._row_envvar, text="Ollama Env Var:", width=14, anchor="w").pack(side="left")
        self.env_var_var = tk.StringVar(value=config.python_app_env_var or "OLLAMA_HOST")
        self._env_var_entry = ttk.Entry(self._row_envvar, textvariable=self.env_var_var,
                                         width=26, state="disabled")
        self._env_var_entry.pack(side="left")

        # Context Size row (Python App only — controls OLLAMA_NUM_CTX at Ollama launch)
        self._row_ctx = ttk.Frame(self._client_details)
        ttk.Label(self._row_ctx, text="Context Size:", width=14, anchor="w").pack(side="left")
        default_ctx = next(
            (k for k, v in _CTX_OPTIONS.items() if v == config.context_size),
            _CTX_DEFAULT
        )
        self.ctx_var = tk.StringVar(value=default_ctx)
        self._ctx_cb = ttk.Combobox(self._row_ctx, textvariable=self.ctx_var,
                                     values=list(_CTX_OPTIONS.keys()),
                                     state="readonly", width=22)
        self._ctx_cb.pack(side="left")

        # Proxy URL info row (Only Proxy mode)
        self._row_proxy_info = ttk.Frame(self._client_details)
        ttk.Label(self._row_proxy_info, text="Proxy URL:", width=14, anchor="w").pack(side="left")
        self._proxy_url_var = tk.StringVar(value=f"http://localhost:{config.proxy_port}")
        ttk.Label(self._row_proxy_info, textvariable=self._proxy_url_var,
                  foreground="#4fc3f7", font=("Segoe UI", 10, "bold")).pack(side="left")

        self._row_launch = ttk.Frame(self._client_details)
        ttk.Label(self._row_launch, text="", width=14).pack(side="left")
        self._launch_client_btn = ttk.Button(self._row_launch, text="Launch Client",
                                              style="Action.TButton",
                                              command=self._on_launch_client,
                                              width=24, state="disabled")
        self._launch_client_btn.pack(side="left")

        # ── Step 3: Dashboard ──
        # Grouped in one frame so _refresh_client_rows can hide the whole step:
        # in Only Proxy mode the dashboard already opens on its own.
        self._step3_block = ttk.Frame(self.root)
        self._step3_block.pack(fill="x")
        ttk.Label(self._step3_block, text="── Step 3: Dashboard ──",
                  style="Section.TLabel").pack()
        step3_btns = ttk.Frame(self._step3_block)
        step3_btns.pack(pady=(8, 16))
        self._start_btn = ttk.Button(step3_btns, text="Open Dashboard", style="Start.TButton",
                                      command=self._on_open_dashboard, state="normal")
        self._start_btn.pack()

        # Status
        self.status_var = tk.StringVar(value="")
        self._status_label = ttk.Label(self.root, textvariable=self.status_var,
                                       foreground="#4fc3f7", font=("Segoe UI", 9))
        self._status_label.pack()

        # Exit lives outside Step 3 so it stays reachable in every mode.
        self._exit_block = ttk.Frame(self.root)
        self._exit_block.pack(pady=(8, 16))
        ttk.Button(self._exit_block, text="Exit", style="Exit.TButton",
                   command=self.root.destroy).pack()

        # Show correct conditional rows for initial client selection
        self._refresh_client_rows()

    # ── Step 1: Ollama ──

    def _on_launch_ollama(self) -> None:
        self._launch_ollama_btn.config(state="disabled")
        self.status_var.set("Checking Ollama...")
        threading.Thread(target=self._check_or_launch_ollama, daemon=True).start()

    def _check_or_launch_ollama(self) -> None:
        # The field accepts "host" or "host:port"; the port is no longer
        # inherited from config.target, so a remote Ollama on a non-standard
        # port can be reached from the UI alone.
        target, host = _parse_ollama_host(self.ollama_host_var.get())
        port = urllib.parse.urlsplit(target).port or _DEFAULT_OLLAMA_PORT
        is_local = host in _LOCALHOST_HOSTS

        models = _fetch_ollama_models(target)
        if models:
            self._active_target = target
            self.root.after(0, lambda: self.status_var.set("Ollama running."))
            self.root.after(0, self._on_models_ready, models)
            return

        if not is_local:
            self.root.after(0, self._ask_on_remote_fail, host, port)
            return

        self.root.after(0, lambda: self._launch_ollama_btn.config(state="normal"))
        self.root.after(0, lambda: self.status_var.set(
            "Ollama not running. Start it (ollama serve) and click Check again."
        ))

    def _ask_on_remote_fail(self, host: str, port: int) -> None:
        """Show a dialog when a remote Ollama host is unreachable."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Connection failed")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text=f"Cannot reach Ollama at {host}:{port}",
            font=("Segoe UI", 10, "bold"),
            background="#1a1a2e", foreground="#f48771",
        ).pack(padx=24, pady=(20, 6))
        ttk.Label(
            dialog,
            text="The server did not respond. What would you like to do?",
            background="#1a1a2e", foreground="#e0e0e0",
            font=("Segoe UI", 9),
        ).pack(padx=24, pady=(0, 16))

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(padx=24, pady=(0, 20))

        def _retry():
            dialog.destroy()
            self._launch_ollama_btn.config(state="disabled")
            self.status_var.set(f"Retrying connection to {host}:{port}...")
            threading.Thread(target=self._check_or_launch_ollama, daemon=True).start()

        def _use_localhost():
            dialog.destroy()
            self.ollama_host_var.set("127.0.0.1")
            self._launch_ollama_btn.config(state="disabled")
            self.status_var.set("Switching to localhost...")
            threading.Thread(target=self._check_or_launch_ollama, daemon=True).start()

        def _cancel():
            dialog.destroy()
            self._launch_ollama_btn.config(state="normal")
            self.status_var.set("Connection cancelled.")

        ttk.Button(btn_frame, text="Retry", style="Action.TButton",
                   command=_retry, width=14).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Use Localhost", style="Action.TButton",
                   command=_use_localhost, width=14).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", style="Exit.TButton",
                   command=_cancel, width=10).pack(side="left", padx=4)

        # Center dialog over launcher
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _on_models_ready(self, models: list) -> None:
        if models:
            self._model_cb["values"] = models
            config = get_config()
            initial = config.default_model if config.default_model in models else models[0]
            self.model_var.set(initial)
            self.status_var.set(f"Ollama ready — {len(models)} model(s) available.")
        else:
            self.status_var.set("Ollama found (no models — pull one with 'ollama pull <model>').")
        self._set_step2_enabled(True)

    # ── Step 2: Client ──

    def _set_step2_enabled(self, enabled: bool) -> None:
        self._step2_enabled = enabled
        self._client_cb.config(state="readonly" if enabled else "disabled")
        self._refresh_client_rows()
        # _launch_client_btn and _start_btn states are managed by _refresh_client_rows

    def _refresh_client_rows(self) -> None:
        """Show/hide and enable/disable conditional rows based on selected client."""
        self._row_model.pack_forget()
        self._row_workdir.pack_forget()
        self._row_apppath.pack_forget()
        self._row_command.pack_forget()
        self._row_venv.pack_forget()
        self._row_envvar.pack_forget()
        self._row_ctx.pack_forget()
        self._row_proxy_info.pack_forget()
        self._row_launch.pack_forget()

        name = self.client_var.get()
        cmd_name = next((c for n, c in self._clients if n == name), "")
        is_python = cmd_name == "__python_app__"
        is_only_proxy = cmd_name == "__only_proxy__"
        field_state = "normal" if self._step2_enabled else "disabled"

        # Step 3 is noise in Only Proxy mode: launching already opens the dashboard.
        # Re-packing needs before=, or tkinter would drop it below the status line.
        if is_only_proxy:
            self._step3_block.pack_forget()
        else:
            self._step3_block.pack(fill="x", before=self._status_label)

        if is_only_proxy:
            try:
                port = int(self.proxy_port_var.get().strip())
            except ValueError:
                port = get_config().proxy_port
            self._proxy_url_var.set(f"http://localhost:{port}")
            self._row_proxy_info.pack(fill="x", padx=20, pady=2)
            self._row_launch.pack(fill="x", padx=20, pady=(2, 10))
            self._launch_client_btn.config(text="Launch Proxy", state=field_state)
        elif is_python:
            self._row_ctx.pack(fill="x", padx=20, pady=2)
            self._row_apppath.pack(fill="x", padx=20, pady=2)
            self._row_command.pack(fill="x", padx=20, pady=2)
            self._row_venv.pack(fill="x", padx=20, pady=2)
            self._row_envvar.pack(fill="x", padx=20, pady=2)
            self._row_launch.pack(fill="x", padx=20, pady=(2, 10))
            self._ctx_cb.config(state="readonly" if self._step2_enabled else "disabled")
            self._app_path_entry.config(state=field_state)
            self._browse_app_btn.config(state=field_state)
            self._command_entry.config(state=field_state)
            self._venv_check.config(state=field_state)
            self._env_var_entry.config(state=field_state)
            self._launch_client_btn.config(text="Launch Client", state=field_state)
        else:
            self._row_model.pack(fill="x", padx=20, pady=2)
            self._row_workdir.pack(fill="x", padx=20, pady=2)
            self._row_launch.pack(fill="x", padx=20, pady=(2, 10))
            self._model_cb.config(state="readonly" if self._step2_enabled else "disabled")
            self._work_dir_entry.config(state=field_state)
            self._browse_workdir_btn.config(state=field_state)
            self._launch_client_btn.config(text="Launch Client", state=field_state)

        self._apply_window_height()

    def _on_client_change(self, event=None) -> None:
        self._refresh_client_rows()

    def _on_browse_workdir(self) -> None:
        path = filedialog.askdirectory(title="Select working directory for AI client")
        if path:
            self.work_dir_var.set(path)

    def _on_browse_app(self) -> None:
        path = filedialog.askdirectory(title="Select Python app directory")
        if path:
            self.app_path_var.set(path)

    def _on_launch_client(self) -> None:
        # Persist before any branch: all three proxy start points below (Only
        # Proxy, Python App and the external clients) run off config.json, so the
        # validated target and the typed proxy port must be on disk first.
        config = self._persist_ui_config()
        if config is None:
            return
        name = self.client_var.get()
        cmd_name = next((c for n, c in self._clients if n == name), "")
        work_dir = self.work_dir_var.get().strip() or None
        model = self.model_var.get().strip()
        proxy_url = f"http://localhost:{config.proxy_port}"
        env = os.environ.copy()

        if cmd_name == "__only_proxy__":
            if not self._servers_started:
                self._servers_started = True
                threading.Thread(target=_start_proxy_thread, daemon=True).start()
                self.root.after(1500, self._reset_session)
            if config.dashboard_enabled:
                dashboard_port = config.dashboard_port
                self.root.after(2000, lambda: webbrowser.open(
                    f"http://localhost:{dashboard_port}"
                ))
            self.status_var.set(f"Proxy started. Point your client at {proxy_url}")
            self.root.after(2500, self.root.iconify)
            return

        try:
            if cmd_name == "claude":
                env["ANTHROPIC_BASE_URL"] = proxy_url
                shell_cmd = f"claude --model {model}" if model else "claude"
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", shell_cmd],
                    cwd=work_dir,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            elif cmd_name == "opencode":
                _write_opencode_config(model, proxy_url, work_dir=work_dir or "")
                shell_cmd = f"opencode --model ollama/{model}" if model else "opencode"
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", shell_cmd],
                    cwd=work_dir,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            elif cmd_name == "__opencode_wsl__":
                wsl_host_ip = _get_wsl_host_ip()
                wsl_proxy_url = f"http://{wsl_host_ip}:{config.proxy_port}"
                _write_opencode_config_wsl(model, wsl_proxy_url)
                # Verify WSL can reach the proxy before opening the terminal.
                if not _wsl_can_reach_proxy(wsl_host_ip, config.proxy_port):
                    _warn_wsl_firewall(wsl_host_ip, config.proxy_port)
                    return
                opencode_cmd = f"opencode --model ollama/{model}" if model else "opencode"
                # Use "start wsl" so the system default terminal (WT/conhost) opens a WSL window.
                # bash -ic loads .bashrc so user-installed tools (npm/cargo/etc.) are in PATH.
                # wsl --cd accepts Windows paths directly.
                wsl_args = ["wsl"]
                if work_dir:
                    wsl_args += ["--cd", work_dir]
                wsl_args += ["--", "bash", "-ic", opencode_cmd]
                subprocess.Popen(
                    ["cmd", "/c", "start", "OpenCode (WSL)"] + wsl_args,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            elif cmd_name == "__python_app__":
                if not self._servers_started:
                    self._servers_started = True
                    threading.Thread(target=_start_proxy_thread, daemon=True).start()
                    self.root.after(1500, self._reset_session)
                self.root.after(2500, self._launch_python_app)
                self.status_var.set("Proxy starting... Python app will launch shortly.")
                self.root.after(2500, self.root.iconify)
                return
            else:
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", cmd_name],
                    cwd=work_dir,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
        except FileNotFoundError:
            self.status_var.set(f"Error: '{cmd_name}' not found. Is the client installed?")
            return
        except OSError as exc:
            self.status_var.set(f"Error launching {name}: {exc}")
            return

        if not self._servers_started:
            self._servers_started = True
            threading.Thread(target=_start_proxy_thread, daemon=True).start()
            self.root.after(1500, self._reset_session)
        self.status_var.set(f"{name} launched.")

    def _launch_python_app(self) -> None:
        """Launch the configured Python app (called after proxy is ready)."""
        config = get_config()
        app_dir = self.app_path_var.get().strip() or None
        command = self.app_command_var.get().strip() or "python main.py"
        env_var = self.env_var_var.get().strip() or "OLLAMA_HOST"
        use_venv = self.use_venv_var.get()
        proxy_url = f"http://localhost:{config.proxy_port}"
        if use_venv:
            # Auto-detect venv name: try .venv first, then venv
            venv_activate = ".venv\\Scripts\\activate"
            if app_dir:
                for candidate in (".venv", "venv"):
                    if Path(app_dir, candidate, "Scripts", "activate.bat").exists():
                        venv_activate = f"{candidate}\\Scripts\\activate"
                        break
            full_cmd = f'call {venv_activate} && {command}'
        else:
            full_cmd = command
        # Inject the env var through the process environment to avoid cmd.exe
        # quoting issues that arise when embedding `set "VAR=val"` in the command.
        child_env = os.environ.copy()
        child_env[env_var] = proxy_url
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "cmd", "/k", full_cmd],
                cwd=app_dir,
                env=child_env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.status_var.set("Python app launched.")
        except OSError as exc:
            self.status_var.set(f"Error launching Python app: {exc}")

    # ── Step 3: Dashboard ──

    def _reset_session(self) -> None:
        config = get_config()
        reset_url = f"http://localhost:{config.dashboard_port}/api/reset"
        try:
            req = urllib.request.Request(reset_url, data=b"{}", method="POST")
            req.add_header("Content-Type", "application/json")
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass

    def _persist_ui_config(self):
        """
        Write the current UI selections to config.json.

        Must run before the proxy starts: the proxy re-reads config.json on every
        request, so a target validated by "Check Ollama" that is never persisted
        leaves it forwarding to the stale value (usually localhost).

        Returns the saved Config, or None if the form is invalid (the reason is
        left in status_var).
        """
        config = get_config()
        try:
            config.proxy_port = int(self.proxy_port_var.get().strip())
        except ValueError:
            self.status_var.set("Error: Proxy Port must be a number.")
            return None
        config.context_size = _CTX_OPTIONS.get(self.ctx_var.get(), 4096)
        model = self.model_var.get().strip()
        if model:
            config.default_model = model
        config.client_work_dir = self.work_dir_var.get().strip()
        config.python_app_path = self.app_path_var.get().strip()
        config.python_app_command = self.app_command_var.get().strip() or "python main.py"
        config.python_app_use_venv = self.use_venv_var.get()
        config.python_app_env_var = self.env_var_var.get().strip()
        if hasattr(self, "_active_target"):
            config.target = self._active_target
        try:
            save_config(config)
        except OSError as exc:
            self.status_var.set(f"Warning: could not save config — {exc}")
        return config

    def _on_open_dashboard(self) -> None:
        config = self._persist_ui_config()
        if config is None:
            return

        dashboard_port = config.dashboard_port

        if not self._servers_started:
            threading.Thread(target=_start_dashboard_thread, daemon=True).start()
            self.root.after(2000, lambda: webbrowser.open(
                f"http://localhost:{dashboard_port}"
            ))
            self.status_var.set("Dashboard starting...")
        else:
            webbrowser.open(f"http://localhost:{dashboard_port}")
            self.status_var.set("Dashboard opened.")

        self.root.after(2500, self.root.iconify)


def launch() -> None:
    """Launch the PromptInterceptor desktop window."""
    root = tk.Tk()
    LauncherWindow(root)
    root.mainloop()
