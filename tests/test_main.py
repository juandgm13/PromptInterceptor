"""Tests for main.py app creation and routing."""

import json
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
    return patch("ollama_proxy.health.httpx.AsyncClient", return_value=mock_client)


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
    from ollama_proxy.main import app
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
        "ollama_proxy.proxy._fetch_from_ollama",
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
        "ollama_proxy.proxy._fetch_from_ollama",
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
    with patch("ollama_proxy.proxy._stream_from_ollama", new=_mock_stream):
        async with client.stream(
            "POST",
            "/api/chat/stream",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            content = await resp.aread()
    assert resp.status_code == 200
    assert b"done" in content


async def test_stream_generate_endpoint(client):
    with patch("ollama_proxy.proxy._stream_from_ollama", new=_mock_stream):
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
        "ollama_proxy.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        resp = await client.post(
            "/api/chat",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert "x-proxy-time" in resp.headers
