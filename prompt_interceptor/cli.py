#!/usr/bin/env python3
"""
CLI interface for PyProxy.
"""

import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# config.json lives next to this file
_CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def show_help() -> None:
    table = Table(title="PyProxy CLI")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="magenta")
    table.add_column("Example", style="yellow")

    for cmd, desc, ex in [
        ["start [port]",        "Start the proxy server",                  "proxy start"],
        ["stop",                "Stop the proxy server",                   "proxy stop"],
        ["mode <mode>",         "Switch mode (passthrough|intercept)",      "proxy mode intercept"],
        ["reload-rules",        "Reload rules from config.json",           "proxy reload-rules"],
        ["logs",                "Display recent log files",                "proxy logs"],
        ["status",              "Show current configuration",              "proxy status"],
        ["help",                "Show this help message",                  "proxy help"],
    ]:
        table.add_row(cmd, desc, ex)

    console.print(Panel(table, title="PyProxy CLI"))


def cmd_start(args: list) -> None:
    config = _load_config()
    port = args[0] if args else str(config.get("proxy_port", 8080))
    host = config.get("proxy_host", "0.0.0.0")

    console.print(f"[bold green]Starting PyProxy on {host}:{port}[/]")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "ollama_proxy.main:app",
                "--host", host,
                "--port", str(port),
                "--log-level", "info",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        console.print("[yellow]Proxy stopped.[/]")
    except FileNotFoundError:
        console.print("[red]uvicorn not found. Run: pip install uvicorn[/]")


def cmd_stop() -> None:
    console.print("[yellow]Use Ctrl+C in the terminal running the proxy to stop it.[/]")


def cmd_mode(args: list) -> None:
    if not args:
        config = _load_config()
        console.print(f"Current mode: [bold]{config.get('mode', 'passthrough')}[/]")
        return

    new_mode = args[0]
    if new_mode not in ("passthrough", "intercept"):
        console.print("[red]Invalid mode. Use 'passthrough' or 'intercept'[/]")
        return

    config = _load_config()
    config["mode"] = new_mode
    _save_config(config)
    console.print(f"[green]Mode set to: {new_mode}[/]")


def cmd_reload_rules() -> None:
    console.print("[yellow]Rules are reloaded on each request from config.json.[/]")
    console.print("[green]Done — the running proxy will pick up changes automatically.[/]")


def cmd_logs() -> None:
    import glob

    log_dir = Path(__file__).parent / "logs"
    if not log_dir.exists():
        console.print("[dim]No logs directory found.[/]")
        return

    files = sorted(glob.glob(str(log_dir / "**" / "req_*.json"), recursive=True))
    if not files:
        console.print("[dim]No log files yet.[/]")
        return

    for filepath in files[-10:]:
        size = Path(filepath).stat().st_size
        rel = Path(filepath).relative_to(log_dir.parent)
        console.print(f"[dim]{rel}[/] ({size:,} bytes)")


def cmd_status() -> None:
    config = _load_config()
    if not config:
        console.print("[red]config.json not found[/]")
        return

    console.print(Panel(
        f"Mode:    [bold]{config.get('mode', 'passthrough')}[/]\n"
        f"Port:    {config.get('proxy_port', 8080)}\n"
        f"Target:  {config.get('target', 'http://localhost:11434')}\n"
        f"Rules:   {len(config.get('rules', []))} configured\n"
        f"Dashboard: port {config.get('dashboard_port', 9090)}, "
        f"{'enabled' if config.get('dashboard_enabled') else 'disabled'}",
        title="PyProxy Status",
    ))


def main() -> None:
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        show_help()
        return

    command = args[0]
    rest = args[1:]

    if command == "start":
        cmd_start(rest)
    elif command == "stop":
        cmd_stop()
    elif command == "mode":
        cmd_mode(rest)
    elif command == "reload-rules":
        cmd_reload_rules()
    elif command == "logs":
        cmd_logs()
    elif command == "status":
        cmd_status()
    else:
        console.print(f"[red]Unknown command: {command}[/]")
        show_help()


if __name__ == "__main__":
    main()
