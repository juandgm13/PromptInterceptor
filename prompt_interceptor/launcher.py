"""
PromptInterceptor Launcher - Desktop configuration window.
Uses only tkinter (Python built-in), no extra dependencies.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

from .config import get_config, save_config

_LOCALHOST_HOSTS = {"127.0.0.1", "localhost"}

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
        "WSL no puede alcanzar el proxy",
        f"WSL no puede conectarse a {host_ip}:{port}.\n\n"
        f"Causa habitual: Windows Firewall bloquea el puerto {port} desde WSL2.\n\n"
        f"Solución — ejecuta esto en PowerShell como Administrador:\n\n"
        f"{rule_cmd}\n\n"
        f"Después vuelve a lanzar Open Code (WSL)."
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


def _detect_clients() -> list:
    """Detect installed AI clients. Returns list of (display_name, command).
    Always includes 'Python App (Ollama)' as a custom option.
    """
    clients = []
    if shutil.which("claude"):
        clients.append(("Claude Code", "claude"))
    if shutil.which("opencode"):
        clients.append(("Open Code (CLI)", "opencode"))
    # Open Code (WSL) support is implemented but disabled in the UI for now.
    # clients.append(("Open Code (WSL)", "__opencode_wsl__"))
    clients.append(("Python App (Ollama)", "__python_app__"))
    return clients


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
                asyncio.set_event_loop(asyncio.new_event_loop())
                uvicorn.run(
                    "prompt_interceptor.dashboard:app",
                    host="0.0.0.0",
                    port=config.dashboard_port,
                    log_level="warning",
                )
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
    except Exception as exc:
        print(f"[PromptInterceptor] Proxy thread error: {exc}")


def _write_opencode_config(model: str, proxy_url: str) -> None:
    """Write/update ~/.config/opencode/opencode.json to point at the proxy with the selected model."""
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
    existing.setdefault("$schema", "https://opencode.ai/config.json")

    config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[PromptInterceptor] opencode config written to {config_path}")


class LauncherWindow:
    """Main launcher window for PromptInterceptor."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PromptInterceptor")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        self._step2_enabled = False
        self._active_target: str = get_config().target

        self._set_icon()
        self._build_ui()
        self._center_window(540, 490)

    def _set_icon(self) -> None:
        icon_path = Path(__file__).parent.parent / "res" / "PromptInterceptor_Icon.png"
        if icon_path.exists():
            try:
                icon = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, icon)
                self.root._icon = icon  # prevent garbage collection
            except Exception:
                pass

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
                  font=("Segoe UI", 9)).pack(pady=(0, 6))

        # Proxy Port (global proxy setting, not Ollama-specific)
        row_port = ttk.Frame(self.root)
        row_port.pack(fill="x", **pad)
        ttk.Label(row_port, text="Proxy Port:", width=14, anchor="w").pack(side="left")
        self.proxy_port_var = tk.StringVar(value=str(config.proxy_port))
        ttk.Entry(row_port, textvariable=self.proxy_port_var, width=8).pack(side="left")

        # ── Step 1: Ollama Server ──
        ttk.Label(self.root, text="── Step 1: Ollama Server ──", style="Section.TLabel").pack(pady=(6, 0))

        # Ollama Host row
        parsed_target = urllib.parse.urlparse(config.target)
        initial_host = parsed_target.hostname or "127.0.0.1"
        row_host = ttk.Frame(self.root)
        row_host.pack(fill="x", **pad)
        ttk.Label(row_host, text="Ollama Host:", width=14, anchor="w").pack(side="left")
        self.ollama_host_var = tk.StringVar(value=initial_host)
        self._host_entry = ttk.Entry(row_host, textvariable=self.ollama_host_var, width=22)
        self._host_entry.pack(side="left")

        row1 = ttk.Frame(self.root)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Context Size:", width=14, anchor="w").pack(side="left")
        default_ctx = next(
            (k for k, v in _CTX_OPTIONS.items() if v == config.context_size),
            _CTX_DEFAULT
        )
        self.ctx_var = tk.StringVar(value=default_ctx)
        ttk.Combobox(row1, textvariable=self.ctx_var,
                     values=list(_CTX_OPTIONS.keys()),
                     state="readonly", width=22).pack(side="left")

        row1c = ttk.Frame(self.root)
        row1c.pack(fill="x", padx=20, pady=(2, 10))
        ttk.Label(row1c, text="", width=14).pack(side="left")
        self._launch_ollama_btn = ttk.Button(row1c, text="Launch Ollama Server",
                                              style="Action.TButton",
                                              command=self._on_launch_ollama, width=24)
        self._launch_ollama_btn.pack(side="left")

        # ── Step 2: AI Client ──
        ttk.Label(self.root, text="── Step 2: AI Client ──", style="Section.TLabel").pack()

        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="AI Client:", width=14, anchor="w").pack(side="left")
        self._clients = _detect_clients()
        self.client_var = tk.StringVar(value=self._clients[0][0])
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
        self._venv_check = ttk.Checkbutton(self._row_venv, text="Use venv (.venv)",
                                            variable=self.use_venv_var, state="disabled")
        self._venv_check.pack(side="left")

        # Env Var row (Python App)
        self._row_envvar = ttk.Frame(self._client_details)
        ttk.Label(self._row_envvar, text="Ollama Env Var:", width=14, anchor="w").pack(side="left")
        self.env_var_var = tk.StringVar(value=config.python_app_env_var or "OLLAMA_HOST")
        self._env_var_entry = ttk.Entry(self._row_envvar, textvariable=self.env_var_var,
                                         width=26, state="disabled")
        self._env_var_entry.pack(side="left")

        self._row_launch = ttk.Frame(self._client_details)
        ttk.Label(self._row_launch, text="", width=14).pack(side="left")
        self._launch_client_btn = ttk.Button(self._row_launch, text="Launch Client",
                                              style="Action.TButton",
                                              command=self._on_launch_client,
                                              width=24, state="disabled")
        self._launch_client_btn.pack(side="left")

        # ── Step 3: Proxy ──
        ttk.Label(self.root, text="── Step 3: Proxy ──", style="Section.TLabel").pack()

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=(8, 16))
        self._start_btn = ttk.Button(btn_frame, text="Start", style="Start.TButton",
                                      command=self._on_start, state="disabled")
        self._start_btn.pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Exit", style="Exit.TButton",
                   command=self.root.destroy).pack(side="left", padx=8)

        # Status
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="#4fc3f7", font=("Segoe UI", 9)).pack()

        # Show correct conditional rows for initial client selection
        self._refresh_client_rows()

    # ── Step 1: Ollama ──

    def _on_launch_ollama(self) -> None:
        self._launch_ollama_btn.config(state="disabled")
        self.status_var.set("Checking Ollama...")
        ctx_value = _CTX_OPTIONS.get(self.ctx_var.get(), 32768)
        config = get_config()
        config.context_size = ctx_value
        try:
            save_config(config)
        except OSError:
            pass
        threading.Thread(target=self._check_or_launch_ollama, daemon=True).start()

    def _check_or_launch_ollama(self) -> None:
        host = self.ollama_host_var.get().strip() or "127.0.0.1"
        config = get_config()
        port = urllib.parse.urlparse(config.target).port or 11434
        target = f"http://{host}:{port}"
        is_local = host in _LOCALHOST_HOSTS

        models = _fetch_ollama_models(target)
        if models:
            self._active_target = target
            self.root.after(0, lambda: self.status_var.set("Ollama already running."))
            self.root.after(0, self._on_models_ready, models)
            return

        if not is_local:
            self.root.after(0, self._ask_on_remote_fail, host, port)
            return

        ctx_value = config.context_size or 32768
        self.root.after(0, lambda: self.status_var.set("Launching Ollama server..."))
        self._active_target = target
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", "cmd", "/k",
                 f"set OLLAMA_NUM_CTX={ctx_value} && ollama serve"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        except FileNotFoundError:
            self.root.after(0, lambda: self._launch_ollama_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_var.set(
                "Error: 'ollama' command not found. Is Ollama installed?"))
            return
        except OSError as exc:
            msg = str(exc)
            self.root.after(0, lambda: self._launch_ollama_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_var.set(f"Error launching Ollama: {msg}"))
            return
        threading.Thread(target=self._fetch_models_after_launch, daemon=True).start()

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

    def _fetch_models_after_launch(self) -> None:
        """Wait for Ollama to start, fetch downloaded models, then unlock Step 2."""
        target = getattr(self, "_active_target", None) or get_config().target
        time.sleep(2)
        models = _fetch_ollama_models(target)
        if not models:
            self.root.after(0, lambda: self.status_var.set("Retrying model fetch..."))
            time.sleep(3)
            models = _fetch_ollama_models(target)
        self.root.after(0, self._on_models_ready, models)

    def _on_models_ready(self, models: list) -> None:
        if models:
            self._model_cb["values"] = models
            config = get_config()
            initial = config.default_model if config.default_model in models else models[0]
            self.model_var.set(initial)
            self.status_var.set(f"Ollama ready — {len(models)} model(s) available.")
        else:
            self.status_var.set("Ollama launched (models unavailable — check Ollama manually).")
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
        self._row_launch.pack_forget()

        name = self.client_var.get()
        is_python = name == "Python App (Ollama)"
        field_state = "normal" if self._step2_enabled else "disabled"

        if is_python:
            self._row_apppath.pack(fill="x", padx=20, pady=2)
            self._row_command.pack(fill="x", padx=20, pady=2)
            self._row_venv.pack(fill="x", padx=20, pady=2)
            self._row_envvar.pack(fill="x", padx=20, pady=2)
            self._app_path_entry.config(state=field_state)
            self._browse_app_btn.config(state=field_state)
            self._command_entry.config(state=field_state)
            self._venv_check.config(state=field_state)
            self._env_var_entry.config(state=field_state)
            # For Python App, Start is enabled directly once Ollama is ready
            if self._step2_enabled:
                self._start_btn.config(state="normal")
        else:
            self._row_model.pack(fill="x", padx=20, pady=2)
            self._row_workdir.pack(fill="x", padx=20, pady=2)
            self._row_launch.pack(fill="x", padx=20, pady=(2, 10))
            self._model_cb.config(state="readonly" if self._step2_enabled else "disabled")
            self._work_dir_entry.config(state=field_state)
            self._browse_workdir_btn.config(state=field_state)
            self._launch_client_btn.config(state=field_state)

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
        config = get_config()
        name = self.client_var.get()
        cmd_name = next((c for n, c in self._clients if n == name), "")
        work_dir = self.work_dir_var.get().strip() or None
        model = self.model_var.get().strip()
        proxy_url = f"http://localhost:{config.proxy_port}"
        env = os.environ.copy()

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
                _write_opencode_config(model, proxy_url)
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

        self.status_var.set(f"{name} launched.")
        self._start_btn.config(state="normal")

    def _launch_python_app(self) -> None:
        """Launch the configured Python app (called after proxy is ready)."""
        config = get_config()
        app_dir = self.app_path_var.get().strip() or None
        command = self.app_command_var.get().strip() or "python main.py"
        env_var = self.env_var_var.get().strip() or "OLLAMA_HOST"
        use_venv = self.use_venv_var.get()
        proxy_url = f"http://localhost:{config.proxy_port}"
        if use_venv:
            full_cmd = (
                f'set {env_var}={proxy_url} && '
                f'.venv\\Scripts\\activate && {command}'
            )
        else:
            full_cmd = f'set {env_var}={proxy_url} && {command}'
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "cmd", "/k", full_cmd],
                cwd=app_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.status_var.set("Python app launched.")
        except OSError as exc:
            self.status_var.set(f"Error launching Python app: {exc}")

    # ── Step 3: Proxy ──

    def _on_start(self) -> None:
        config = get_config()
        try:
            config.proxy_port = int(self.proxy_port_var.get().strip())
        except ValueError:
            self.status_var.set("Error: Proxy Port must be a number.")
            return
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

        threading.Thread(target=_start_proxy_thread, daemon=True).start()

        dashboard_port = config.dashboard_port

        def _reset_session():
            reset_url = f"http://localhost:{dashboard_port}/api/reset"
            try:
                req = urllib.request.Request(reset_url, data=b"{}", method="POST")
                req.add_header("Content-Type", "application/json")
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                pass

        self.root.after(1500, _reset_session)
        self.root.after(2000, lambda: webbrowser.open(
            f"http://localhost:{dashboard_port}"
        ))

        if self.client_var.get() == "Python App (Ollama)":
            self.root.after(2500, self._launch_python_app)

        self.status_var.set("Proxy starting... Dashboard will open shortly.")
        self.root.after(2500, self.root.iconify)


def launch() -> None:
    """Launch the PromptInterceptor desktop window."""
    root = tk.Tk()
    LauncherWindow(root)
    root.mainloop()
