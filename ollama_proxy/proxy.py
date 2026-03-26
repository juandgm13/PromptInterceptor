"""
Core proxy module for PyProxy.

Handles HTTP proxying, async request forwarding, and response handling.
"""

import json
from typing import Optional, Dict, Any, AsyncIterator

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_config
from .interceptor import interceptor
from .logger import TrafficLogger
from .rules_engine import RuleEngine

# Headers that must not be forwarded verbatim (managed by the HTTP layer)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-encoding",  # httpx decompresses for us
    "content-length",    # will be recalculated
})


def _forward_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Strip hop-by-hop headers before forwarding."""
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


async def _fetch_from_ollama(
    target: str,
    method: str,
    path: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    timeout: int = 120,
) -> tuple:
    """
    Fetch data from Ollama backend (non-streaming).

    Returns:
        (status_code, headers, body_bytes)
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    ) as client:
        response = await client.request(
            method,
            target + path,
            headers=_forward_headers(headers),
            content=body,
        )
        return (
            response.status_code,
            dict(response.headers),
            await response.aread(),
        )


async def _stream_from_ollama(
    target: str,
    method: str,
    path: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    timeout: int = 120,
) -> AsyncIterator[bytes]:
    """
    Stream raw bytes from Ollama backend.

    Yields raw byte chunks as received.
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    ) as client:
        async with client.stream(
            method,
            target + path,
            headers=_forward_headers(headers),
            content=body,
        ) as response:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )


async def _apply_intercept(
    request_id: str,
    method: str,
    path: str,
    headers: Dict[str, str],
    body_json: Optional[Dict[str, Any]],
) -> tuple:
    """
    In intercept mode, pause the request until the dashboard resolves it.

    Returns:
        (should_drop: bool, body_json: dict|None)
    """
    action, resolved_body = await interceptor.intercept(
        request_id, method, path, headers, body_json
    )
    if action == "drop":
        return (True, None)
    return (False, resolved_body)


async def handle_chat_request(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger,
) -> Response:
    """Handle /api/chat request to Ollama (non-streaming)."""
    config = get_config()
    body_json = await request.json()

    modified, body, request_id = await rule_engine.process_request(
        "POST", request.url.path, dict(request.headers), body_json
    )
    if modified:
        body_json = body

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""

    try:
        response_code, response_headers, response_body = await _fetch_from_ollama(
            config.target, "POST", request.url.path,
            dict(request.headers), body_bytes, config.timeout,
        )

        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError:
            response_json = None

        logger.log_response(request_id, response_code, response_headers, response_json)

        return Response(
            content=response_body,
            status_code=response_code,
            headers=_forward_headers(response_headers),
            media_type=response_headers.get("content-type", "application/json"),
        )

    except httpx.TimeoutException:
        return _error_response(408, "Request timeout")
    except httpx.HTTPStatusError as e:
        return _error_response(e.response.status_code, str(e))
    except Exception as e:
        return _error_response(500, str(e))


async def handle_generate_request(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger,
) -> Response:
    """Handle /api/generate request to Ollama (non-streaming)."""
    config = get_config()
    body_json = await request.json()

    modified, body, request_id = await rule_engine.process_request(
        "POST", request.url.path, dict(request.headers), body_json
    )
    if modified:
        body_json = body

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""

    try:
        response_code, response_headers, response_body = await _fetch_from_ollama(
            config.target, "POST", request.url.path,
            dict(request.headers), body_bytes, config.timeout,
        )

        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError:
            response_json = None

        logger.log_response(request_id, response_code, response_headers, response_json)

        return Response(
            content=response_body,
            status_code=response_code,
            headers=_forward_headers(response_headers),
            media_type=response_headers.get("content-type", "application/json"),
        )

    except httpx.TimeoutException:
        return _error_response(408, "Request timeout")
    except httpx.HTTPStatusError as e:
        return _error_response(e.response.status_code, str(e))
    except Exception as e:
        return _error_response(500, str(e))


async def handle_stream_chat(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger,
) -> Response:
    """Handle /api/chat streaming request."""
    config = get_config()
    body_json = await request.json()

    modified, body, request_id = await rule_engine.process_request(
        "POST", request.url.path, dict(request.headers), body_json
    )
    if modified:
        body_json = body

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    forward_headers = dict(request.headers)

    async def generate() -> AsyncIterator[bytes]:
        try:
            async for chunk in _stream_from_ollama(
                config.target, "POST", request.url.path,
                forward_headers, body_bytes, config.timeout,
            ):
                yield chunk
        except httpx.TimeoutException:
            yield json.dumps({"error": "Request timeout"}).encode()
        except httpx.ConnectError:
            yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode()

        logger.log_response(request_id, 200, {}, None)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def handle_stream_generate(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger,
) -> Response:
    """Handle /api/generate streaming request."""
    config = get_config()
    body_json = await request.json()

    modified, body, request_id = await rule_engine.process_request(
        "POST", request.url.path, dict(request.headers), body_json
    )
    if modified:
        body_json = body

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    forward_headers = dict(request.headers)

    async def generate() -> AsyncIterator[bytes]:
        try:
            async for chunk in _stream_from_ollama(
                config.target, "POST", request.url.path,
                forward_headers, body_bytes, config.timeout,
            ):
                yield chunk
        except httpx.TimeoutException:
            yield json.dumps({"error": "Request timeout"}).encode()
        except httpx.ConnectError:
            yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode()

        logger.log_response(request_id, 200, {}, None)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
