"""Tests for proxy.py handler functions and helpers."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prompt_interceptor.config import Config
from prompt_interceptor.logger import TrafficLogger
from prompt_interceptor.proxy import (
    _forward_headers,
    _HOP_BY_HOP,
    _inject_num_ctx,
    _parse_stream_response,
    _parse_sse_response,
    _parse_openai_sse_response,
    handle_chat_request,
    handle_generate_request,
    handle_stream_chat,
    handle_stream_generate,
    handle_v1_messages,
    handle_v1_chat_completions,
    handle_passthrough,
)
from prompt_interceptor.rules_engine import RuleEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_req(body: dict, path: str = "/api/chat"):
    req = MagicMock()
    req.json = AsyncMock(return_value=body)
    req.url.path = path
    req.headers = {"content-type": "application/json"}
    return req


@pytest.fixture
def cfg(tmp_path):
    return Config(log_dir=str(tmp_path / "logs"), mode="passthrough", timeout=5)


@pytest.fixture
def engine_and_logger(cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return engine, logger


# ---------------------------------------------------------------------------
# _forward_headers
# ---------------------------------------------------------------------------

def test_forward_headers_strips_hop_by_hop():
    headers = {
        "content-type": "application/json",
        "transfer-encoding": "chunked",
        "content-encoding": "gzip",
        "x-custom": "keep",
    }
    result = _forward_headers(headers)
    assert "transfer-encoding" not in result
    assert "content-encoding" not in result
    assert result["content-type"] == "application/json"
    assert result["x-custom"] == "keep"


def test_forward_headers_case_insensitive():
    # All hop-by-hop names are lowercase in _HOP_BY_HOP
    assert "connection" in _HOP_BY_HOP
    result = _forward_headers({"Connection": "keep-alive", "Accept": "application/json"})
    assert "Connection" not in result
    assert result["Accept"] == "application/json"


def test_forward_headers_empty():
    assert _forward_headers({}) == {}


# ---------------------------------------------------------------------------
# _inject_num_ctx
# ---------------------------------------------------------------------------

def test_inject_num_ctx_adds_option(monkeypatch):
    cfg = Config(context_size=16384)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    body = {"model": "llama3", "messages": []}
    result = _inject_num_ctx(body)
    assert result["options"]["num_ctx"] == 16384
    # original dict not mutated
    assert "options" not in body


def test_inject_num_ctx_does_not_overwrite_existing(monkeypatch):
    cfg = Config(context_size=16384)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    body = {"model": "llama3", "options": {"num_ctx": 4096, "temperature": 0.7}}
    result = _inject_num_ctx(body)
    # existing num_ctx is preserved
    assert result["options"]["num_ctx"] == 4096


def test_inject_num_ctx_none_body(monkeypatch):
    cfg = Config(context_size=16384)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    assert _inject_num_ctx(None) is None


def test_inject_num_ctx_zero_context_size(monkeypatch):
    cfg = Config(context_size=0)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    body = {"model": "llama3"}
    result = _inject_num_ctx(body)
    assert "options" not in result


# ---------------------------------------------------------------------------
# _parse_stream_response
# ---------------------------------------------------------------------------

def test_parse_stream_response_chat():
    chunks = [
        b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n',
        b'{"message":{"role":"assistant","content":" world"},"done":true,"total_duration":100}\n',
    ]
    result = _parse_stream_response(chunks)
    assert result["message"]["content"] == "Hello world"
    assert result["total_duration"] == 100


def test_parse_stream_response_generate():
    chunks = [
        b'{"response":"foo","done":false}\n',
        b'{"response":"bar","done":true}\n',
    ]
    result = _parse_stream_response(chunks)
    assert result["response"] == "foobar"


def test_parse_stream_response_empty():
    assert _parse_stream_response([]) is None


def test_parse_stream_response_invalid_json():
    chunks = [b"not json\n", b'{"response":"ok","done":true}\n']
    result = _parse_stream_response(chunks)
    assert result["response"] == "ok"


# ---------------------------------------------------------------------------
# handle_chat_request — success
# ---------------------------------------------------------------------------

async def test_handle_chat_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({"model": "llama3", "done": True}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req({"model": "llama3", "messages": [{"role": "user", "content": "hi"}]})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200
    assert json.loads(resp.body)["model"] == "llama3"


async def test_handle_chat_non_json_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain text")),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# handle_chat_request — error paths
# ---------------------------------------------------------------------------

async def test_handle_chat_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 408
    assert "timeout" in json.loads(resp.body)["error"].lower()


async def test_handle_chat_generic_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# handle_chat_request — intercept mode (forward / drop)
# ---------------------------------------------------------------------------

async def test_handle_chat_intercept_forward(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    response_bytes = json.dumps({"model": "llama3"}).encode()

    async def _fake_intercept(rid, method, path, headers, body):
        return "forward", body

    monkeypatch.setattr("prompt_interceptor.proxy.interceptor.intercept", _fake_intercept)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200


async def test_handle_chat_intercept_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    async def _fake_intercept(rid, method, path, headers, body):
        return "drop", body

    monkeypatch.setattr("prompt_interceptor.proxy.interceptor.intercept", _fake_intercept)

    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_chat_request(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_generate_request
# ---------------------------------------------------------------------------

async def test_handle_generate_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({"response": "hello", "done": True}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req({"model": "llama3", "prompt": "Say hello"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200


async def test_handle_generate_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("t")),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 408


async def test_handle_generate_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_generate_request(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_stream_chat
# ---------------------------------------------------------------------------

async def _mock_stream(*args, **kwargs):
    yield b'{"model":"llama3","done":false}\n'
    yield b'{"model":"llama3","done":true}\n'


async def test_handle_stream_chat_yields_chunks(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    # Consume body INSIDE the patch context: StreamingResponse is lazy
    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_stream_chat(req, engine, logger)
        assert resp.media_type == "application/x-ndjson"
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"llama3" in body


async def test_handle_stream_chat_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_stream_generate
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_yields_chunks(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_stream_generate(req, engine, logger)
        assert resp.media_type == "application/x-ndjson"
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"llama3" in body


async def test_handle_stream_generate_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# Rule application in handlers
# ---------------------------------------------------------------------------

async def test_handle_chat_rule_switches_model(tmp_path, monkeypatch):
    """When a rule matches, the forwarded body uses the replaced model."""
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        rules=[{
            "match": {"path": "/api/chat", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek-coder"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    captured = {}

    async def _fake_fetch(target, method, path, headers, body, timeout):
        captured["body"] = json.loads(body)
        return 200, {"content-type": "application/json"}, b'{"done": true}'

    # Auto-forward in intercept mode
    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek-coder", "messages": []})),
    )

    with patch("prompt_interceptor.proxy._fetch_from_ollama", new=_fake_fetch):
        req = make_req({"model": "llama3", "messages": []})
        await handle_chat_request(req, engine, logger)

    assert captured["body"]["model"] == "deepseek-coder"


# ---------------------------------------------------------------------------
# _fetch_from_ollama direct tests  (lines 47-57)
# ---------------------------------------------------------------------------

async def test_fetch_from_ollama_success():
    """Exercise the real body of _fetch_from_ollama with a mocked httpx client."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(return_value=b'{"done": true}')

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def request(self, *args, **kwargs):
            return mock_response

    from prompt_interceptor.proxy import _fetch_from_ollama
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=_FakeClient()):
        status, headers, body = await _fetch_from_ollama(
            "http://localhost:11434", "POST", "/api/chat",
            {"content-type": "application/json"}, b'{}', 5,
        )

    assert status == 200
    assert body == b'{"done": true}'
    assert "content-type" in headers


async def test_fetch_from_ollama_non_200():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.headers = {}
    mock_response.aread = AsyncMock(return_value=b'Service Unavailable')

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def request(self, *args, **kwargs):
            return mock_response

    from prompt_interceptor.proxy import _fetch_from_ollama
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=_FakeClient()):
        status, _, body = await _fetch_from_ollama(
            "http://localhost:11434", "GET", "/health", {}, None, 5,
        )

    assert status == 503
    assert b"Unavailable" in body


# ---------------------------------------------------------------------------
# _stream_from_ollama direct tests  (lines 77-89)
# ---------------------------------------------------------------------------

async def test_stream_from_ollama_yields_non_empty_chunks():
    """Exercise the real body of _stream_from_ollama with mocked httpx."""
    chunks_sent = [b'{"model":"llama3"}\n', b'', b'{"done":true}\n']

    async def _fake_aiter_bytes():
        for c in chunks_sent:
            yield c

    class _FakeStreamCtx:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def aiter_bytes(self):
            return _fake_aiter_bytes()

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def stream(self, *args, **kwargs):
            return _FakeStreamCtx()

    from prompt_interceptor.proxy import _stream_from_ollama
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=_FakeClient()):
        result = [c async for c in _stream_from_ollama(
            "http://localhost:11434", "POST", "/api/chat", {}, b'{}', 5,
        )]

    assert b'' not in result
    assert b'{"model":"llama3"}\n' in result
    assert b'{"done":true}\n' in result


# ---------------------------------------------------------------------------
# handle_chat_request — HTTPStatusError  (line 167)
# ---------------------------------------------------------------------------

@pytest.fixture
def passthrough_setup(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="passthrough", timeout=5)
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return cfg, engine, logger


@pytest.fixture
def intercept_generate_setup(tmp_path, monkeypatch):
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        timeout=5,
        rules=[{
            "match": {"path": "/api/generate", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return cfg, engine, logger


@pytest.fixture
def intercept_chat_setup(tmp_path, monkeypatch):
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        timeout=5,
        rules=[{
            "match": {"path": "/api/chat", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return cfg, engine, logger


async def test_handle_chat_http_status_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    mock_resp = MagicMock()
    mock_resp.status_code = 422

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.HTTPStatusError(
            "unprocessable", request=MagicMock(), response=mock_resp
        )),
    ):
        resp = await handle_chat_request(make_req({"model": "llama3"}), engine, logger)

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# handle_generate_request — modified body (line 185)
# ---------------------------------------------------------------------------

async def test_handle_generate_modified_body(intercept_generate_setup, monkeypatch):
    """Rule modifies body → body_json is replaced before forwarding."""
    cfg, engine, logger = intercept_generate_setup
    captured = {}

    async def _fake_fetch(target, method, path, headers, body, timeout):
        captured["body"] = json.loads(body)
        return 200, {"content-type": "application/json"}, b'{"done":true}'

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek", "prompt": "hi"})),
    )
    with patch("prompt_interceptor.proxy._fetch_from_ollama", new=_fake_fetch):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200
    assert captured["body"]["model"] == "deepseek"


# ---------------------------------------------------------------------------
# handle_generate_request — non-JSON response (lines 204-205)
# ---------------------------------------------------------------------------

async def test_handle_generate_non_json_response(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain text")),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# handle_generate_request — HTTPStatusError and generic Exception
# ---------------------------------------------------------------------------

async def test_handle_generate_http_status_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=mock_resp
        )),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 500


async def test_handle_generate_generic_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("unexpected")),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# handle_stream_chat — modified body and exception paths
# ---------------------------------------------------------------------------

async def test_handle_stream_chat_modified_body(intercept_chat_setup, monkeypatch):
    """When rule modifies body, body_json is updated before streaming."""
    cfg, engine, logger = intercept_chat_setup

    async def _mock_stream(*args, **kwargs):
        yield b'{"done":true}\n'

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek", "messages": []})),
    )
    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_stream_chat(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"done" in body


async def test_handle_stream_chat_timeout_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    async def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timeout")
        yield b''

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_raise_timeout):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_stream_chat(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "timeout" in json.loads(body)["error"].lower()


async def test_handle_stream_chat_connect_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    async def _raise_connect(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
        yield b''

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_raise_connect):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_stream_chat(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(body)
    assert "connect" in data["error"].lower() or "ollama" in data["error"].lower()


async def test_handle_stream_chat_generic_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    async def _raise_generic(*args, **kwargs):
        raise RuntimeError("boom")
        yield b''

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_raise_generic):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_stream_chat(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "boom" in json.loads(body)["error"]


# ---------------------------------------------------------------------------
# handle_stream_generate — modified body and exception paths
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_modified_body(tmp_path, monkeypatch):
    """Rule modifies body in stream_generate path."""
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        timeout=5,
        rules=[{
            "match": {"path": "/api/generate", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    async def _mock_stream(*args, **kwargs):
        yield b'{"done":true}\n'

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek", "prompt": "hi"})),
    )
    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_stream_generate(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"done" in body


async def test_handle_stream_generate_timeout_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    async def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timeout")
        yield b''

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_raise_timeout):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_stream_generate(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "timeout" in json.loads(body)["error"].lower()


async def test_handle_stream_generate_connect_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    async def _raise_connect(*args, **kwargs):
        raise httpx.ConnectError("refused")
        yield b''

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_raise_connect):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_stream_generate(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "error" in json.loads(body)


async def test_handle_stream_generate_generic_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    async def _raise_generic(*args, **kwargs):
        raise ValueError("bad value")
        yield b''

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_raise_generic):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_stream_generate(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "bad value" in json.loads(body)["error"]


# ---------------------------------------------------------------------------
# Invalid JSON body — all handlers must return 400
# ---------------------------------------------------------------------------

def _make_bad_json_req(path: str = "/api/chat"):
    """Request whose .json() raises an exception (malformed body)."""
    req = MagicMock()
    req.json = AsyncMock(side_effect=Exception("JSON decode error"))
    req.url.path = path
    req.headers = {"content-type": "application/json"}
    return req


async def test_handle_chat_request_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_chat_request returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_chat_request(_make_bad_json_req("/api/chat"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_generate_request_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_generate_request returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_generate_request(_make_bad_json_req("/api/generate"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_stream_chat_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_stream_chat returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_stream_chat(_make_bad_json_req("/api/chat"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_stream_generate_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_stream_generate returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_stream_generate(_make_bad_json_req("/api/generate"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


# ---------------------------------------------------------------------------
# handle_passthrough
# ---------------------------------------------------------------------------

def _make_passthrough_req(body: dict | bytes | None, path: str = "/v1/messages",
                           method: str = "POST", query: str = ""):
    req = MagicMock()
    raw = json.dumps(body).encode() if isinstance(body, dict) else (body or b"")
    req.body = AsyncMock(return_value=raw)
    req.method = method
    req.url.path = path
    req.query_params = query
    req.headers = {"content-type": "application/json"}
    return req


async def test_passthrough_non_streaming_forwards_and_returns(cfg, monkeypatch):
    """Non-streaming passthrough returns Ollama's response body and status."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'{"ok": true}'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "llama3"}, "/v1/messages"))

    assert resp.status_code == 200
    assert b'"ok"' in resp.body


async def test_passthrough_streaming_returns_streaming_response(cfg, monkeypatch):
    """Streaming passthrough returns a StreamingResponse with SSE media type for /v1/ paths."""
    from fastapi.responses import StreamingResponse
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")

    resp = await handle_passthrough(req)

    assert isinstance(resp, StreamingResponse)
    assert "text/event-stream" in resp.media_type


async def test_passthrough_streaming_ndjson_for_api_path(cfg, monkeypatch):
    """Streaming passthrough uses application/x-ndjson for /api/ paths."""
    from fastapi.responses import StreamingResponse
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/api/generate")

    resp = await handle_passthrough(req)

    assert isinstance(resp, StreamingResponse)
    assert "ndjson" in resp.media_type


async def test_passthrough_timeout_returns_408(cfg, monkeypatch):
    """Passthrough returns 408 when Ollama times out."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "llama3"}, "/v1/messages"))

    assert resp.status_code == 408


async def test_passthrough_connect_error_returns_502(cfg, monkeypatch):
    """Passthrough returns 502 when Ollama is unreachable."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req(None, "/api/tags", "GET"))

    assert resp.status_code == 502


async def test_passthrough_non_json_body_is_handled(cfg, monkeypatch):
    """Passthrough handles binary/non-JSON bodies without crashing."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b"pong"
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/plain"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req(b"not json", "/api/version", "GET"))

    assert resp.status_code == 200


async def test_passthrough_query_params_appended_to_url(cfg, monkeypatch):
    """Query params are appended to the forwarded URL."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    captured_urls = []

    fake_resp = MagicMock()
    fake_resp.content = b"{}"
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def capture_request(method, url, **kwargs):
        captured_urls.append(url)
        return fake_resp

    mock_client.request = capture_request

    req = _make_passthrough_req(None, "/api/tags", "GET", query="name=llama3")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)

    assert resp.status_code == 200
    assert "name=llama3" in captured_urls[0]


async def test_passthrough_generic_exception_returns_500(cfg, monkeypatch):
    """Passthrough returns 500 on unexpected errors."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=RuntimeError("unexpected"))

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "llama3"}, "/v1/messages"))

    assert resp.status_code == 500


async def test_passthrough_streaming_chunks_yielded(cfg, monkeypatch):
    """Streaming passthrough yields chunks from Ollama."""
    from fastapi.responses import StreamingResponse
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    chunks = [b'data: {"text": "hello"}', b'data: {"text": " world"}']

    async def fake_aiter_bytes():
        for c in chunks:
            yield c

    fake_stream_resp = MagicMock()
    fake_stream_resp.aiter_bytes = fake_aiter_bytes
    fake_stream_resp.__aenter__ = AsyncMock(return_value=fake_stream_resp)
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)
        assert isinstance(resp, StreamingResponse)
        collected = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"hello" in collected


async def test_passthrough_streaming_connect_error_yields_error_json(cfg, monkeypatch):
    """ConnectError inside _stream_gen yields an error JSON chunk."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)
        collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected)
    assert "error" in data


async def test_passthrough_streaming_generic_exception_yields_error_json(cfg, monkeypatch):
    """Generic exception inside _stream_gen yields an error JSON chunk."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)
        collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected)
    assert "error" in data
    assert "boom" in data["error"]


async def test_passthrough_streaming_logs_response_with_logger(cfg, monkeypatch, tmp_path):
    """Streaming passthrough calls logger.log_response after stream completes."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    async def fake_aiter_bytes():
        yield b'{"done":true}'

    fake_stream_resp = MagicMock()
    fake_stream_resp.aiter_bytes = fake_aiter_bytes
    fake_stream_resp.__aenter__ = AsyncMock(return_value=fake_stream_resp)
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    logger = TrafficLogger(cfg)
    logged = []
    original_log_response = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original_log_response(*a, **kw)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req, logger=logger)
        b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) == 1


async def test_passthrough_body_json_attribute_error_handled(cfg, monkeypatch):
    """When body_json.get raises AttributeError, is_stream defaults to False."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'{"ok": true}'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.json = MagicMock(return_value={"ok": True})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    # Craft a request whose body is a non-JSON string so body_json ends up None
    # then override body_json inside handle_passthrough via a body that json.loads returns a string
    req = _make_passthrough_req(b'"just a string"', "/api/test", "POST")

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)

    assert resp.status_code == 200


async def test_passthrough_resp_json_generic_exception_handled(cfg, monkeypatch):
    """When resp.json() raises a non-JSONDecodeError exception, resp_json is None."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'data'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/octet-stream"}
    fake_resp.json = MagicMock(side_effect=RuntimeError("not json at all"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    logger = TrafficLogger(cfg)
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "x"}, "/api/test"), logger=logger)

    assert resp.status_code == 200


async def test_passthrough_non_dict_json_response_handled(cfg, monkeypatch):
    """Non-dict resp.json() is treated as None when logging."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'"just a string"'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.json.return_value = "just a string"  # str, not dict/list

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    logger = TrafficLogger(cfg)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(
            _make_passthrough_req({"model": "llama3"}, "/v1/test"),
            logger=logger,
        )

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _parse_sse_response
# ---------------------------------------------------------------------------

def test_parse_sse_response_assembles_content():
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01","model":"qwen3:9b","role":"assistant","content":[]}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    result = _parse_sse_response(chunks)
    assert result is not None
    assert result["content"] == [{"type": "text", "text": "Hello world"}]


def test_parse_sse_response_preserves_message_start_metadata():
    chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_01","model":"qwen3:9b","role":"assistant"}}\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["id"] == "msg_01"
    assert result["model"] == "qwen3:9b"


def test_parse_sse_response_empty_chunks_returns_none():
    assert _parse_sse_response([]) is None


def test_parse_sse_response_ping_only_returns_none():
    chunks = [b'event: ping\ndata: {"type":"ping"}\n\n']
    assert _parse_sse_response(chunks) is None


def test_parse_sse_response_done_marker_skipped():
    """data: [DONE] line is skipped without crashing."""
    chunks = [
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["content"][0]["text"] == "hi"


def test_parse_sse_response_skips_invalid_json():
    chunks = [
        b'data: not-valid-json\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["content"] == [{"type": "text", "text": "ok"}]


def test_parse_sse_response_accepts_string_chunks():
    chunks = [
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["content"][0]["text"] == "x"


# ---------------------------------------------------------------------------
# _parse_openai_sse_response
# ---------------------------------------------------------------------------

def test_parse_openai_sse_response_assembles_content():
    chunks = [
        b'data: {"id":"cmp-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n',
        b'data: {"id":"cmp-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "Hello world"


def test_parse_openai_sse_response_empty_returns_none():
    assert _parse_openai_sse_response([]) is None


def test_parse_openai_sse_response_done_only_returns_none():
    chunks = [b'data: [DONE]\n']
    assert _parse_openai_sse_response(chunks) is None


def test_parse_openai_sse_response_skips_invalid_json():
    chunks = [
        b'data: bad json\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"y"},"finish_reason":null}]}\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result["choices"][0]["message"]["content"] == "y"


def test_parse_openai_sse_response_ollama_usage_chunk_last():
    """Ollama sends a final chunk with choices:[] and usage stats. Content must still be captured."""
    chunks = [
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n',
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["usage"]["completion_tokens"] == 2


# ---------------------------------------------------------------------------
# Helpers for v1 handler tests
# ---------------------------------------------------------------------------

def _make_v1_streaming_client(chunks: list):
    """Build a mock httpx.AsyncClient that streams the given byte chunks."""
    async def fake_aiter_bytes():
        for c in chunks:
            yield c

    fake_stream_resp = MagicMock()
    fake_stream_resp.aiter_bytes = fake_aiter_bytes
    fake_stream_resp.__aenter__ = AsyncMock(return_value=fake_stream_resp)
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)
    return mock_client


# ---------------------------------------------------------------------------
# handle_v1_messages — non-streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_non_streaming_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({
        "id": "msg_01", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Hi"}],
        "model": "qwen3:9b", "stop_reason": "end_turn",
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req(
            {"model": "qwen3:9b", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]},
            path="/v1/messages",
        )
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200
    assert json.loads(resp.body)["id"] == "msg_01"


async def test_handle_v1_messages_non_streaming_non_json_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200


async def test_handle_v1_messages_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 408


async def test_handle_v1_messages_connect_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 502


async def test_handle_v1_messages_generic_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 500


async def test_handle_v1_messages_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_v1_messages(_make_bad_json_req("/v1/messages"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_v1_messages_intercept_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    assert resp.status_code == 204


async def test_handle_v1_messages_intercept_forward(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    body = {"model": "qwen3:9b", "messages": []}
    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", body)),
    )
    response_bytes = json.dumps({"id": "msg_01", "type": "message"}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req(body, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# handle_v1_messages — streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_streaming_returns_event_stream(cfg, engine_and_logger, monkeypatch):
    from fastapi.responses import StreamingResponse
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]
    mock_client = _make_v1_streaming_client(sse_chunks)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_messages(req, engine, logger)
        assert isinstance(resp, StreamingResponse)
        assert "text/event-stream" in resp.media_type
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"Hi" in body


async def test_handle_v1_messages_streaming_logs_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_01","model":"qwen3:9b","role":"assistant"}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
    ]
    mock_client = _make_v1_streaming_client(sse_chunks)

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_messages(req, engine, logger)
        b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    # The response body logged should contain the assembled content
    logged_body = logged[-1][3]  # 4th arg to log_response is body
    assert logged_body is not None
    assert logged_body["content"][0]["text"] == "Hello"


async def test_handle_v1_messages_streaming_connect_error_yields_json(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_messages(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "error" in json.loads(body)


async def test_handle_v1_messages_streaming_timeout_yields_json(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_messages(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "timeout" in json.loads(body)["error"].lower()


# ---------------------------------------------------------------------------
# handle_v1_chat_completions — non-streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_non_streaming_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({
        "id": "chatcmpl-1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "model": "qwen3:9b",
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req(
            {"model": "qwen3:9b", "messages": [{"role": "user", "content": "Hi"}]},
            path="/v1/chat/completions",
        )
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 200
    assert json.loads(resp.body)["id"] == "chatcmpl-1"


async def test_handle_v1_chat_completions_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 408


async def test_handle_v1_chat_completions_connect_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 502


async def test_handle_v1_chat_completions_generic_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 500


async def test_handle_v1_chat_completions_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_v1_chat_completions(_make_bad_json_req("/v1/chat/completions"), engine, logger)
    assert resp.status_code == 400


async def test_handle_v1_chat_completions_intercept_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_v1_chat_completions — streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_streaming_returns_event_stream(cfg, engine_and_logger, monkeypatch):
    from fastapi.responses import StreamingResponse
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"id":"cmp-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n',
        b'data: [DONE]\n',
    ]
    mock_client = _make_v1_streaming_client(sse_chunks)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_chat_completions(req, engine, logger)
        assert isinstance(resp, StreamingResponse)
        assert "text/event-stream" in resp.media_type
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"Hi" in body


async def test_handle_v1_chat_completions_streaming_logs_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"id":"cmp-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        b'data: [DONE]\n',
    ]
    mock_client = _make_v1_streaming_client(sse_chunks)

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_chat_completions(req, engine, logger)
        b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    logged_body = logged[-1][3]
    assert logged_body is not None
    assert logged_body["choices"][0]["message"]["content"] == "Hello"


async def test_handle_v1_messages_streaming_generic_error_yields_json(cfg, engine_and_logger, monkeypatch):
    """Generic exception inside handle_v1_messages streaming yields error JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("unexpected"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_messages(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "unexpected" in json.loads(body)["error"]


async def test_handle_v1_chat_completions_streaming_timeout_yields_json(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_chat_completions(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "timeout" in json.loads(body)["error"].lower()


async def test_handle_v1_chat_completions_streaming_connect_error_yields_json(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_chat_completions(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "error" in json.loads(body)


async def test_handle_v1_chat_completions_streaming_generic_error_yields_json(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_stream_resp = MagicMock()
    fake_stream_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_v1_chat_completions(req, engine, logger)
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert "boom" in json.loads(body)["error"]


async def test_handle_v1_chat_completions_non_streaming_non_json_response(cfg, engine_and_logger, monkeypatch):
    """handle_v1_chat_completions handles non-JSON response body without crashing."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain text")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 200
