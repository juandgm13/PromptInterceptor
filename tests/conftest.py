"""
Shared fixtures for PyProxy tests.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the repo root is on sys.path so `prompt_interceptor` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub out tkinter for headless CI environments where it is not installed.
# The launcher imports tkinter at module level; without this stub every test
# that touches prompt_interceptor.launcher would fail with ModuleNotFoundError.
_tk_stub = MagicMock()
for _mod in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"):
    sys.modules.setdefault(_mod, _tk_stub)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config(tmp_path):
    """Config with one model-switching rule, using tmp_path for logs."""
    from prompt_interceptor.config import Config
    return Config(
        proxy_port=8080,
        target="http://localhost:11434",
        mode="intercept",
        timeout=30,
        health_timeout=0.5,
        log_dir=str(tmp_path / "logs"),
        rules=[
            {
                "match": {
                    "path": "/api/chat",
                    "jsonpath": "$.model",
                    "value": ["llama3"],
                },
                "replace": {"jsonpath": "$.model", "value": "deepseek-coder"},
            }
        ],
    )


@pytest.fixture
def passthrough_config(tmp_path):
    """Config with no rules in passthrough mode."""
    from prompt_interceptor.config import Config
    return Config(
        log_dir=str(tmp_path / "logs"),
        mode="passthrough",
        rules=[],
    )


# ---------------------------------------------------------------------------
# Logger / engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logger(test_config):
    from prompt_interceptor.logger import TrafficLogger
    return TrafficLogger(test_config)


@pytest.fixture
def rule_engine(test_config, logger, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: test_config)
    from prompt_interceptor.rules_engine import RuleEngine
    return RuleEngine(logger)


# ---------------------------------------------------------------------------
# Mock Request factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_request():
    """Return a factory that builds mock FastAPI Request objects."""
    def _factory(body: dict, path: str = "/api/chat"):
        req = MagicMock()
        req.json = AsyncMock(return_value=body)
        req.url.path = path
        req.headers = {"content-type": "application/json"}
        return req
    return _factory


# ---------------------------------------------------------------------------
# Ollama mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def no_ollama():
    """Patch httpx.AsyncClient in health.py to simulate Ollama being offline (503)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        yield mock_client


# ---------------------------------------------------------------------------
# Proxy / dashboard ASGI clients
# ---------------------------------------------------------------------------

@pytest.fixture
def proxy_app(test_config, monkeypatch):
    """FastAPI proxy app with all get_config calls returning test_config."""
    import prompt_interceptor.main as main_mod
    import prompt_interceptor.proxy as proxy_mod
    import prompt_interceptor.rules_engine as re_mod
    import prompt_interceptor.logger as logger_mod
    import prompt_interceptor.cors_middleware as cors_mod
    import prompt_interceptor.health as health_mod

    for mod in (main_mod, proxy_mod, re_mod, logger_mod, cors_mod, health_mod):
        monkeypatch.setattr(mod, "get_config", lambda: test_config)

    from prompt_interceptor.main import create_app
    return create_app()


@pytest.fixture
async def proxy_client(proxy_app):
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(
        transport=ASGITransport(app=proxy_app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def dashboard_client():
    from httpx import AsyncClient, ASGITransport
    from prompt_interceptor.dashboard import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
