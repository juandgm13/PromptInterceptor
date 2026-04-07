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


async def handle_passthrough(request: Request, logger: Optional[TrafficLogger] = None) -> Response:
    """Log and forward any unhandled request to Ollama as-is."""
    config = get_config()
    body = await request.body()
    method = request.method
    path = str(request.url.path)
    query = str(request.query_params)

    try:
        body_preview = json.loads(body) if body else {}
    except (json.JSONDecodeError, ValueError):
        body_preview = body.decode("utf-8", errors="replace")[:500]

    print(
        f"[PromptInterceptor] PASSTHROUGH {method} {path}"
        + (f"?{query}" if query else "")
        + f"\n  body: {json.dumps(body_preview, ensure_ascii=False)[:300]}"
    )

    body_json = body_preview if isinstance(body_preview, dict) else None
    request_id = logger.log_request(method, path, dict(request.headers), body_json) if logger else None

    forward_headers = _forward_headers(dict(request.headers))
    url = config.target + path
    if query:
        url += f"?{query}"

    try:
        is_stream = body_json.get("stream", False) if body_json else False
    except AttributeError:
        is_stream = False

    if is_stream:
        media = "text/event-stream" if path.startswith("/v1/") else "application/x-ndjson"

        async def _stream_gen():
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config.timeout), follow_redirects=True
                ) as client:
                    async with client.stream(method, url, headers=forward_headers, content=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk
            except httpx.ConnectError:
                yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
            except Exception as exc:
                yield json.dumps({"error": str(exc)}).encode()
            if logger and request_id:
                logger.log_response(request_id, 200, {}, None)

        return StreamingResponse(_stream_gen(), media_type=media)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout), follow_redirects=True
        ) as client:
            resp = await client.request(method, url, headers=forward_headers, content=body)
        print(f"[PromptInterceptor] PASSTHROUGH response {resp.status_code} from {path}")

        if logger and request_id:
            try:
                resp_json = resp.json()
                if not isinstance(resp_json, (dict, list)):
                    resp_json = None
            except Exception:
                resp_json = None
            logger.log_response(request_id, resp.status_code, dict(resp.headers), resp_json)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_forward_headers(dict(resp.headers)),
            media_type=resp.headers.get("content-type"),
        )
    except httpx.TimeoutException:
        return _error_response(408, "Request timeout")
    except httpx.ConnectError:
        return _error_response(502, "Cannot connect to Ollama")
    except Exception as exc:
        return _error_response(500, str(exc))


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
    try:
        body_json = await request.json()
    except Exception:
        return _error_response(400, "Invalid JSON in request body")

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
    try:
        body_json = await request.json()
    except Exception:
        return _error_response(400, "Invalid JSON in request body")

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
    try:
        body_json = await request.json()
    except Exception:
        return _error_response(400, "Invalid JSON in request body")

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
    try:
        body_json = await request.json()
    except Exception:
        return _error_response(400, "Invalid JSON in request body")

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
