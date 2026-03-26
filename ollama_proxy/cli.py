#!/usr/bin/env python3
"""
CLI interface for the Ollama Proxy.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def run_command(command: str, *args):
    """Run a shell command and return output."""
    from rich.live import Live
    from rich.prompt import Prompt

    console.print(f"[bold blue]> {command} { ' '.join(args) if args else ''}[/]")

    try:
        # Check if process exists (simple subprocess check)
        import subprocess
        process = await asyncio.create_subprocess_shell(
            command + " " + " ".join(args) if args else command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        if stdout:
            console.print(stdout.decode())
        if stderr:
            console.print(stderr.decode(), style="red")

        return process.returncode == 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return False


def show_help():
    """Display help information."""
    table = Table(title="PyProxy CLI")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="magenta")
    table.add_column("Example", style="yellow")

    commands = [
        ["start", "Start the proxy server", "proxy start"],
        ["stop", "Stop the proxy server", "proxy stop"],
        ["mode", "Set proxy mode (passthrough|intercept)", "proxy mode intercept"],
        ["mode status", "Show current mode", "proxy mode status"],
        ["reload-rules", "Reload rules from config", "proxy reload-rules"],
        ["logs", "Display recent logs", "proxy logs"],
        ["status", "Show proxy status", "proxy status"],
        ["help", "Show this help message", "proxy help"],
    ]

    for cmd, desc, ex in commands:
        table.add_row(cmd, desc, ex)

    console.print(Panel(table, title="PyProxy CLI"))


def main():
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help", "help"]:
        show_help()
        return

    command = args[0]

    if command == "start":
        if args[1:]:
            port = args[1] if args[1] else "8080"
        else:
            port = "8080"
        asyncio.run(run_command("python", "-m", "uvicorn", "-m",
                                "ollama_proxy.main:app",
                                "--host", port, "--port", "8080"))

    elif command == "stop":
        # Simple process termination placeholder
        import os
        import signal
        console.print("[yellow]Stopping proxy...[/]")
        # In production, would use process manager to find and stop the uvicorn process

    elif command == "mode":
        if len(args) >= 2:
            new_mode = args[1]
            if new_mode in ["passthrough", "intercept"]:
                # Write to config
                import json
                config_path = Path(__file__).parent.parent / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    config["mode"] = new_mode
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)
                    console.print(f"[green]Mode changed to: {new_mode}[/]")
                else:
                    console.print(f"[red]Config file not found: {config_path}[/]")
            else:
                console.print(f"[red]Invalid mode. Use 'passthrough' or 'intercept'[/]")
        else:
            console.print("[yellow]Usage: proxy mode <passthrough|intercept>[/]")

    elif command == "reload-rules":
        console.print("[yellow]Reloading rules from config.json...[/]")
        # Reload logic handled by FastAPI on file change
        console.print("[green]Rules reloaded[/]")

    elif command == "logs":
        import glob
        logs_dir = Path(__file__).parent.parent / "logs"
        if logs_dir.exists():
            for log_dir in sorted(glob.glob(str(logs_dir / "*"))):
                log_name = Path(log_dir).name
                files = sorted(glob.glob(str(log_dir / "*")))
                for f in files[-5:]:  # Last 5 files
                    size = Path(f).stat().st_size
                    console.print(f"[dim]{log_name}/[/] {Path(f).name} ({size:,} bytes)")
        else:
            console.print("[dim]No logs directory yet[/]")

    elif command == "status":
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            console.print(Panel(
                f"Mode: {config.get('mode', 'passthrough')}\n"
                f"Port: {config.get('proxy_port', 8080)}\n"
                f"Target: {config.get('target', 'http://localhost:11434')}\n"
                f"Rules: {len(config.get('rules', []))}",
                title="Proxy Status"
            ))
        else:
            console.print("[red]Config file not found[/]")

    else:
        console.print(f"[red]Unknown command: {command}[/]")


if __name__ == "__main__":
    main()
