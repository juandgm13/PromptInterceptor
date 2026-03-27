"""
PromptInterceptor Launcher - Desktop configuration window.
Uses only tkinter (Python built-in), no extra dependencies.
"""
import shutil
import subprocess
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk
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
        self._center_window(420, 340)

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
        else:
            self.client_var = tk.StringVar(value="")
            ttk.Label(row3, text="No clients detected (claude, opencode)",
                      foreground="#888888").pack(side="left")

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

    def _on_start(self) -> None:
        # Save context_size to config
        config = get_config()
        config.context_size = _CTX_OPTIONS.get(self.ctx_var.get(), 4096)
        save_config(config)

        ctx_value = config.context_size

        # Open Ollama terminal with OLLAMA_NUM_CTX env var
        subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k",
             f"set OLLAMA_NUM_CTX={ctx_value} && ollama serve"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        # Open AI client terminal if one is selected
        client_name = self.client_var.get()
        if client_name:
            client_cmd = next(
                (cmd for name, cmd in self._clients if name == client_name), None
            )
            if client_cmd:
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", client_cmd],
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
