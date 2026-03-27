"""Tests for main.py app creation and routing."""

import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


def _mock_no_ollama():
    """Return a context manager patch that makes Ollama return 503."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client)


@pytest.fixture
async def client(proxy_app):
    async with AsyncClient(
        transport=ASGITransport(app=proxy_app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

def test_create_app_returns_fastapi(proxy_app):
    assert isinstance(proxy_app, FastAPI)


def test_module_level_app_exists():
    from prompt_interceptor.main import app
    assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "mode" in data


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

async def test_status_endpoint(client):
    with _mock_no_ollama():
        resp = await client.get("/status")
    assert resp.status_code == 200
    assert "proxy" in resp.json()


# ---------------------------------------------------------------------------
# /dashboard redirect
# ---------------------------------------------------------------------------

async def test_dashboard_redirect(client):
    resp = await client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 307
    assert "location" in resp.headers


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

async def test_api_models(client):
    with _mock_no_ollama():
        resp = await client.get("/api/models")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Proxy endpoints (non-streaming)
# ---------------------------------------------------------------------------

async def test_chat_endpoint(client):
    response_bytes = json.dumps({"model": "llama3", "done": True}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        resp = await client.post(
            "/api/chat",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200


async def test_generate_endpoint(client):
    response_bytes = json.dumps({"response": "hi", "done": True}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        resp = await client.post(
            "/api/generate",
            json={"model": "llama3", "prompt": "Say hi"},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Streaming endpoints
# ---------------------------------------------------------------------------

async def _mock_stream(*args, **kwargs):
    yield b'{"done":false}\n'
    yield b'{"done":true}\n'


async def test_stream_chat_endpoint(client):
    # httpx AsyncClient collects the full streaming body before returning,
    # so the patch must be active during both request and body consumption.
    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        async with client.stream(
            "POST",
            "/api/chat/stream",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            content = await resp.aread()
    assert resp.status_code == 200
    assert b"done" in content


async def test_stream_generate_endpoint(client):
    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        async with client.stream(
            "POST",
            "/api/generate/stream",
            json={"model": "llama3", "prompt": "hi"},
        ) as resp:
            content = await resp.aread()
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Timing header middleware
# ---------------------------------------------------------------------------

async def test_timing_header_added(client):
    response_bytes = b'{"done":true}'
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        resp = await client.post(
            "/api/chat",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert "x-proxy-time" in resp.headers


# ---------------------------------------------------------------------------
# __main__.py — entry point  (lines 5-13)
# ---------------------------------------------------------------------------

def test_main_module_calls_launch():
    with patch("prompt_interceptor.__main__.launch") as mock_launch:
        from prompt_interceptor.__main__ import main
        main()
    mock_launch.assert_called_once()


def test_main_module_is_callable():
    import prompt_interceptor.__main__ as m
    assert callable(m.main)


# ---------------------------------------------------------------------------
# /api/models — exception path  (lines 75-76)
# ---------------------------------------------------------------------------

async def test_api_models_exception(client):
    """/api/models returns {error: ...} when check_target_health raises."""
    with patch(
        "prompt_interceptor.main.check_target_health",
        new=AsyncMock(side_effect=RuntimeError("ollama down")),
    ):
        resp = await client.get("/api/models")

    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert "ollama down" in data["error"]


# ---------------------------------------------------------------------------
# main() function  (lines 103-116)
# ---------------------------------------------------------------------------

def test_main_with_dashboard_enabled():
    """main() starts dashboard thread and calls uvicorn.run."""
    import prompt_interceptor.main as main_mod
    import uvicorn

    threads_started = []

    class _MockThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            threads_started.append(self._target)

    with patch.object(uvicorn, "run", MagicMock()) as mock_run, \
         patch.object(main_mod.threading, "Thread", side_effect=_MockThread):
        from prompt_interceptor.main import main
        main()

    assert mock_run.called
    assert len(threads_started) == 1

    # Invoke the dashboard thread target to cover its body (line 107)
    with patch.object(uvicorn, "run", MagicMock()):
        threads_started[0]()


def test_main_with_dashboard_disabled():
    """main() skips dashboard thread when dashboard_enabled=False."""
    import prompt_interceptor.main as main_mod
    import uvicorn

    cfg = MagicMock()
    cfg.dashboard_enabled = False
    cfg.proxy_host = "0.0.0.0"
    cfg.proxy_port = 8080
    cfg.debug = False

    threads_started = []

    class _MockThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            threads_started.append(self._target)

    with patch("prompt_interceptor.main.get_config", return_value=cfg), \
         patch.object(uvicorn, "run", MagicMock()), \
         patch.object(main_mod.threading, "Thread", side_effect=_MockThread):
        from prompt_interceptor.main import main
        main()

    assert threads_started == []
