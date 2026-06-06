"""Tests for main.py app creation and routing."""

import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


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

async def test_status_endpoint(client, no_ollama):
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

async def test_api_models(client, no_ollama):
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

def _make_stream_mock(chunks):
    async def fake_aiter_bytes():
        for c in chunks:
            yield c
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.aiter_bytes = fake_aiter_bytes
    fake_resp.aclose = AsyncMock()
    fake_client = MagicMock()
    fake_client.aclose = AsyncMock()
    return AsyncMock(return_value=(fake_client, fake_resp))


async def test_stream_chat_endpoint(client):
    mock = _make_stream_mock([b'{"done":false}\n', b'{"done":true}\n'])
    with patch("prompt_interceptor.proxy._start_streaming_request", new=mock):
        async with client.stream(
            "POST",
            "/api/chat/stream",
            json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            content = await resp.aread()
    assert resp.status_code == 200
    assert b"done" in content


async def test_stream_generate_endpoint(client):
    mock = _make_stream_mock([b'{"done":false}\n', b'{"done":true}\n'])
    with patch("prompt_interceptor.proxy._start_streaming_request", new=mock):
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
    """/api/models returns {error: ...} when get_models raises."""
    with patch(
        "prompt_interceptor.main._get_models",
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


async def test_v1_messages_endpoint_is_registered(proxy_app):
    """/v1/messages has a dedicated handler (not passthrough)."""
    from httpx import AsyncClient, ASGITransport
    import prompt_interceptor.proxy as proxy_mod

    response_bytes = json.dumps({"id": "msg_01", "type": "message"}).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=proxy_app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/messages",
                json={"model": "qwen3:9b", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            )

    assert resp.status_code == 200
    assert resp.json()["id"] == "msg_01"


async def test_v1_chat_completions_endpoint_is_registered(proxy_app):
    """/v1/chat/completions has a dedicated handler (not passthrough)."""
    from httpx import AsyncClient, ASGITransport

    response_bytes = json.dumps({
        "id": "chatcmpl-1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=proxy_app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"model": "qwen3:9b", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert resp.status_code == 200
    assert resp.json()["id"] == "chatcmpl-1"


def test_create_app_resets_mode_to_passthrough(monkeypatch):
    """create_app() always resets mode to passthrough on startup."""
    import prompt_interceptor.main as main_mod
    from prompt_interceptor.config import Config

    cfg = Config(mode="intercept")
    saved = []

    monkeypatch.setattr(main_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "save_config", lambda c: saved.append(c.mode))
    monkeypatch.setattr(main_mod, "reload_config", lambda: None)

    main_mod.create_app()

    assert saved == ["passthrough"]


def test_create_app_does_not_save_when_already_passthrough(monkeypatch):
    """create_app() skips save_config when mode is already passthrough."""
    import prompt_interceptor.main as main_mod
    from prompt_interceptor.config import Config

    cfg = Config(mode="passthrough")
    saved = []

    monkeypatch.setattr(main_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "save_config", lambda c: saved.append(c.mode))
    monkeypatch.setattr(main_mod, "reload_config", lambda: None)

    main_mod.create_app()

    assert saved == []


async def test_passthrough_endpoint_is_registered(proxy_app):
    """The catch-all route forwards unknown paths through handle_passthrough."""
    from httpx import AsyncClient, ASGITransport
    import prompt_interceptor.proxy as proxy_mod

    fake_response = MagicMock()
    fake_response.content = b'{"forwarded": true}'
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_response)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        async with AsyncClient(
            transport=ASGITransport(app=proxy_app), base_url="http://test"
        ) as c:
            resp = await c.get("/v1/models")

    assert resp.status_code == 200


