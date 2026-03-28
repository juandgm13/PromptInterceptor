"""
PromptInterceptor Launcher - Desktop configuration window.
Uses only tkinter (Python built-in), no extra dependencies.
"""
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

from .config import get_config, save_config

_CTX_OPTIONS = {
    "4k  (4096)":    4096,
    "8k  (8192)":    8192,
    "16k (16384)":   16384,
    "32k (32768)":   32768,
    "64k (65536)":   65536,
    "128k (131072)": 131072,
    "256k (262144)": 262144,
}

_CTX_DEFAULT = "4k  (4096)"


def _detect_clients() -> list:
    """Detect installed AI clients. Returns list of (display_name, command)."""
    clients = []
    if shutil.which("claude"):
        clients.append(("Claude Code", "claude"))
    if shutil.which("opencode"):
        clients.append(("Open Code", "opencode"))
    return clients


def _fetch_ollama_models(target: str) -> list:
    """Sync query to Ollama /api/tags. Returns [] if not reachable."""
    url = target + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _start_proxy_thread():
    """Run the FastAPI proxy and dashboard servers in background threads."""
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


class LauncherWindow:
    """Main launcher window for PromptInterceptor."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PromptInterceptor")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")

        self._set_icon()
        self._build_ui()
        self._center_window(540, 500)

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
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TCombobox", fieldbackground="#16213e", background="#16213e", foreground="#e0e0e0")
        style.configure("Start.TButton", background="#4fc3f7", foreground="#000000", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Exit.TButton", background="#333355", foreground="#e0e0e0", font=("Segoe UI", 10), padding=8)
        style.map("Start.TButton", background=[("active", "#29b6f6")])
        style.map("Exit.TButton", background=[("active", "#444466")])

        pad = {"padx": 20, "pady": 6}

        # Header
        ttk.Label(self.root, text="PromptInterceptor", style="Header.TLabel").pack(pady=(24, 4))
        ttk.Label(self.root, text="Ollama Traffic Interceptor for AI Clients", style="TLabel",
                  font=("Segoe UI", 9)).pack(pady=(0, 16))

        # Server row
        row1 = ttk.Frame(self.root)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Server:", width=14, anchor="w").pack(side="left")
        self.server_var = tk.StringVar(value="Ollama")
        server_cb = ttk.Combobox(row1, textvariable=self.server_var, values=["Ollama"],
                                  state="readonly", width=22)
        server_cb.pack(side="left")

        # Context size row
        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Context Size:", width=14, anchor="w").pack(side="left")
        config = get_config()
        default_ctx = next(
            (k for k, v in _CTX_OPTIONS.items() if v == config.context_size),
            _CTX_DEFAULT
        )
        self.ctx_var = tk.StringVar(value=default_ctx)
        ctx_cb = ttk.Combobox(row2, textvariable=self.ctx_var,
                               values=list(_CTX_OPTIONS.keys()),
                               state="readonly", width=22)
        ctx_cb.pack(side="left")

        # AI Client row
        row3 = ttk.Frame(self.root)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="AI Client:", width=14, anchor="w").pack(side="left")
        self._clients = _detect_clients()
        if self._clients:
            self.client_var = tk.StringVar(value=self._clients[0][0])
            client_cb = ttk.Combobox(row3, textvariable=self.client_var,
                                      values=[name for name, _ in self._clients],
                                      state="readonly", width=22)
            client_cb.pack(side="left")
            client_cb.bind("<<ComboboxSelected>>", self._on_client_change)
        else:
            self.client_var = tk.StringVar(value="")
            ttk.Label(row3, text="No clients detected (claude, opencode)",
                      foreground="#888888").pack(side="left")

        # Client path row
        row3b = ttk.Frame(self.root)
        row3b.pack(fill="x", padx=20, pady=2)
        ttk.Label(row3b, text="", width=14).pack(side="left")
        self.client_path_var = tk.StringVar(value=self._get_default_client_path())
        ttk.Entry(row3b, textvariable=self.client_path_var, width=26).pack(side="left", padx=(0, 4))
        ttk.Button(row3b, text="Browse...", command=self._on_browse_client, width=9).pack(side="left")

        # Model row
        row4 = ttk.Frame(self.root)
        row4.pack(fill="x", **pad)
        ttk.Label(row4, text="Model:", width=14, anchor="w").pack(side="left")
        ollama_models = _fetch_ollama_models(config.target)
        model_list = ollama_models or config.model_names
        initial_model = config.default_model or (model_list[0] if model_list else "")
        self.model_var = tk.StringVar(value=initial_model)
        self._model_cb = ttk.Combobox(row4, textvariable=self.model_var, values=model_list, width=22)
        self._model_cb.pack(side="left", padx=(0, 4))
        ttk.Button(row4, text="↻", command=self._on_refresh_models, width=3).pack(side="left")
        self._model_cb.bind("<<ComboboxSelected>>", lambda e: self._check_pull_needed())
        self._model_cb.bind("<FocusOut>", lambda e: self._check_pull_needed())

        # Pull button row (hidden initially)
        row4b = ttk.Frame(self.root)
        row4b.pack(fill="x", padx=20, pady=2)
        ttk.Label(row4b, text="", width=14).pack(side="left")
        self._pull_btn = ttk.Button(row4b, text="Pull model", command=self._on_pull_model)
        # .pack() is called conditionally by _check_pull_needed()

        # Spacer
        ttk.Frame(self.root).pack(pady=8)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=(4, 20))
        ttk.Button(btn_frame, text="Start", style="Start.TButton",
                   command=self._on_start).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Exit", style="Exit.TButton",
                   command=self.root.destroy).pack(side="left", padx=8)

        # Status label
        self.status_var = tk.StringVar(value="")
        self._status_label = ttk.Label(self.root, textvariable=self.status_var,
                                        foreground="#4fc3f7", font=("Segoe UI", 9))
        self._status_label.pack()

    # --- Client path helpers ---

    def _get_default_client_path(self) -> str:
        """Return the resolved path for the currently selected client."""
        if not self._clients:
            return ""
        name = self.client_var.get()
        cmd = next((c for n, c in self._clients if n == name), "")
        return shutil.which(cmd) or cmd

    def _on_client_change(self, event=None) -> None:
        """Update the path entry when the client selection changes."""
        self.client_path_var.set(self._get_default_client_path())

    def _on_browse_client(self) -> None:
        """Open a file dialog to pick a custom executable path."""
        path = filedialog.askopenfilename(
            title="Select AI client executable",
            filetypes=[("Executables", "*.exe *.cmd *.bat"), ("All files", "*.*")],
        )
        if path:
            self.client_path_var.set(path)

    # --- Model helpers ---

    def _on_refresh_models(self) -> None:
        """Re-query Ollama /api/tags and repopulate the model combobox."""
        self.status_var.set("Refreshing model list...")
        config = get_config()
        models = _fetch_ollama_models(config.target)
        if models:
            self._model_cb["values"] = models
            self.status_var.set(f"Found {len(models)} model(s).")
        else:
            self.status_var.set("Ollama not reachable — model list unchanged.")
        self._check_pull_needed()

    def _check_pull_needed(self) -> None:
        """Show or hide the Pull button based on whether the typed model is known."""
        model = self.model_var.get().strip()
        known = list(self._model_cb["values"])
        if model and model not in known:
            self._pull_btn.pack(side="left")
        else:
            self._pull_btn.pack_forget()

    def _on_pull_model(self) -> None:
        """Start pulling a model in a background thread."""
        model = self.model_var.get().strip()
        if not model:
            return
        self._pull_btn.config(state="disabled")
        self.status_var.set(f"Pulling {model}...")
        threading.Thread(target=self._pull_model_thread, args=(model,), daemon=True).start()

    def _pull_model_thread(self, model: str) -> None:
        """Background thread: runs `ollama pull <model>` and streams progress to the status label."""
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in proc.stdout:
                line = line.strip()
                if line:
                    self.root.after(0, lambda l=line: self.status_var.set(l))
            proc.wait()
            if proc.returncode == 0:
                self.root.after(0, self._on_pull_complete, model)
            else:
                self.root.after(0, lambda: self.status_var.set(f"Pull failed for {model}."))
                self.root.after(0, lambda: self._pull_btn.config(state="normal"))
        except FileNotFoundError:
            self.root.after(0, lambda: self.status_var.set("ollama not found in PATH."))
            self.root.after(0, lambda: self._pull_btn.config(state="normal"))

    def _on_pull_complete(self, model: str) -> None:
        """Called on the main thread after a successful pull."""
        self.status_var.set(f"Model '{model}' downloaded successfully.")
        self._pull_btn.pack_forget()
        self._on_refresh_models()

    # --- Start ---

    def _on_start(self) -> None:
        # Save context_size and default_model to config
        config = get_config()
        config.context_size = _CTX_OPTIONS.get(self.ctx_var.get(), 4096)
        model = self.model_var.get().strip()
        if model:
            config.default_model = model
            if model not in config.model_names:
                config.model_names = [model] + config.model_names
        save_config(config)

        ctx_value = config.context_size

        # Open Ollama terminal with OLLAMA_NUM_CTX env var
        subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k",
             f"set OLLAMA_NUM_CTX={ctx_value} && ollama serve"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        # Open AI client terminal using the (possibly custom) path from the entry
        client_path = self.client_path_var.get().strip()
        if client_path:
            subprocess.Popen(
                ["cmd", "/c", "start", "cmd", "/k", client_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

        # Start proxy server in background thread
        threading.Thread(target=_start_proxy_thread, daemon=True).start()

        # Open dashboard in browser after 2s delay
        self.root.after(2000, lambda: webbrowser.open(
            f"http://localhost:{config.dashboard_port}"
        ))

        self.status_var.set("Starting... Dashboard will open shortly.")
        self.root.after(2500, self.root.iconify)


def launch() -> None:
    """Launch the PromptInterceptor desktop window."""
    root = tk.Tk()
    LauncherWindow(root)
    root.mainloop()
