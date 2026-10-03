"""
Configuration management for PromptInterceptor.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration model for the proxy."""

    # Project metadata
    project_name: str = "PromptInterceptor"
    project_description: str = "PromptInterceptor - Ollama Traffic Interceptor for AI Clients"
    debug: bool = False
    debug_intercept: bool = False  # Enable detailed tracing of the intercept pipeline to stdout

    # Server configuration
    proxy_port: int = 8080
    proxy_host: str = "0.0.0.0"

    # Target Ollama server
    target: str = "http://localhost:11434"
    target_port: int = 11434

    # Proxy mode: "passthrough" or "intercept"
    mode: str = "passthrough"

    # Timeout in seconds for forwarding requests to Ollama
    timeout: int = 120

    # Seconds to hold an intercepted request waiting for a dashboard decision before auto-forwarding
    intercept_timeout: float = 30.0

    # Timeout for health checks (shorter than proxy timeout)
    health_timeout: float = 5.0

    # Logging configuration
    log_dir: str = "logs"
    log_size_limit: int = 1024 * 1024  # 1MB per file
    max_log_files: int = 10

    # CORS settings
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])

    # Rule configuration
    rules: List[Dict[str, Any]] = Field(default_factory=list)

    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_port: int = 9090

    # Context usage alert
    context_alert_enabled: bool = True
    context_alert_threshold: int = 80  # 0-100 percentage

    # Model names to watch for rules
    model_names: List[str] = Field(default_factory=lambda: ["llama3", "mistral"])

    # Context size for the Ollama server (passed as OLLAMA_NUM_CTX env var on start)
    context_size: int = 32768

    # Default model selected in the launcher
    default_model: str = ""

    # Launcher: working directory for Claude Code / Open Code
    client_work_dir: str = ""

    # Launcher: Python app entry point path
    python_app_path: str = ""

    # Launcher: command to launch the Python app (supports arguments)
    python_app_command: str = "python main.py"

    # Launcher: whether to activate a .venv before running the Python app
    python_app_use_venv: bool = False

    # Launcher: environment variable name the Python app uses to configure the Ollama host
    python_app_env_var: str = "OLLAMA_HOST"


# Location of config.json. Read at call time so tests can point it elsewhere.
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from JSON file."""
    config_path = path or CONFIG_PATH

    if not config_path.exists():
        return Config()

    try:
        with open(config_path) as f:
            data = json.load(f)
        return Config(**data)
    except (json.JSONDecodeError, KeyError, OSError, ValueError):
        return Config()


def save_config(config: Config) -> None:
    """Save configuration to JSON file.

    Raises:
        OSError: if the file cannot be written (e.g. permission denied).
    """
    config_path = CONFIG_PATH
    try:
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to save configuration to {config_path}: {exc}") from exc


def get_config() -> Config:
    """Get current configuration."""
    return load_config()


# Global config instance
_config = None


def get_global_config() -> Config:
    """Get global config instance (cached)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config():
    """Reload configuration from disk."""
    global _config
    _config = load_config()


def config_to_json(config: Config) -> dict:
    """Convert config to JSON-serializable dict."""
    return config.model_dump()
