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


def _inject_num_ctx(body_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Inject num_ctx into request options based on configured context_size."""
    if not body_json:
        return body_json
    config = get_config()
    if not config.context_size:
        return body_json
    options = dict(body_json.get("options") or {})
    options["num_ctx"] = config.context_size
    return {**body_json, "options": options}


def _parse_sse_response(chunks: list) -> Optional[Dict[str, Any]]:
    """Parse Anthropic SSE streaming chunks (/v1/messages) into a loggable response body."""
    full_content = ""
    full_thinking = ""
    final_message: Dict[str, Any] = {}

    for chunk in chunks:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        for line in text.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
                obj_type = obj.get("type", "")
                if obj_type == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        full_content += delta.get("text", "")
                    elif delta.get("type") == "thinking_delta":
                        full_thinking += delta.get("thinking", "")
                elif obj_type == "message_start":
                    final_message = dict(obj.get("message", {}))
            except (json.JSONDecodeError, AttributeError):
                pass

    if not full_content and not full_thinking and not final_message:
        return None
    result = dict(final_message)
    content_blocks = []
    if full_thinking:
        content_blocks.append({"type": "thinking", "thinking": full_thinking})
    if full_content:
        content_blocks.append({"type": "text", "text": full_content})
    if content_blocks:
        result["content"] = content_blocks
    return result


def _parse_openai_sse_response(chunks: list) -> Optional[Dict[str, Any]]:
    """Parse OpenAI-compatible SSE streaming chunks (/v1/chat/completions) into a loggable response body."""
    full_content = ""
    full_thinking = ""
    tool_calls_by_index: Dict[int, Dict] = {}
    final_obj: Dict[str, Any] = {}
    for chunk in chunks:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        for line in text.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    full_content += delta.get("content", "") or ""
                    # Ollama: 'reasoning' (Open Code / qwen3), 'thinking' (0.7+), 'reasoning_content' (DeepSeek)
                    full_thinking += delta.get("reasoning", "") or delta.get("thinking", "") or delta.get("reasoning_content", "") or ""
                    # Accumulate tool_calls deltas (OpenAI streaming format)
                    for tc in delta.get("tool_calls", []) or []:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        entry = tool_calls_by_index[idx]
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        if tc.get("type"):
                            entry["type"] = tc["type"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            entry["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += fn["arguments"]
                final_obj = obj
            except (json.JSONDecodeError, AttributeError):
                pass

    full_tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)] if tool_calls_by_index else []
    if not full_content and not full_thinking and not full_tool_calls and not final_obj:
        return None
    result = dict(final_obj)
    if full_content or full_thinking or full_tool_calls:
        base_choice = result["choices"][0] if result.get("choices") else {}
        msg: Dict[str, Any] = {"role": "assistant", "content": full_content}
        if full_thinking:
            msg["thinking"] = full_thinking
        if full_tool_calls:
            msg["tool_calls"] = full_tool_calls
        result["choices"] = [{**base_choice, "message": msg}]
    return result


def _parse_stream_response(chunks: list) -> Optional[Dict[str, Any]]:
    """Assemble accumulated NDJSON streaming chunks into a loggable response body."""
    full_content = ""
    full_thinking = ""
    full_tool_calls: list = []
    final_obj: Dict[str, Any] = {}
    for chunk in chunks:
        for line in chunk.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # /api/chat streaming: each chunk carries message.content and optionally message.thinking
                if "message" in obj:
                    msg_chunk = obj.get("message", {})
                    full_content += msg_chunk.get("content", "")
                    # Ollama 0.7+: thinking field; also check reasoning_content as fallback
                    full_thinking += msg_chunk.get("thinking", "") or msg_chunk.get("reasoning_content", "") or ""
                    # Ollama tool_calls: present in final chunk
                    if msg_chunk.get("tool_calls"):
                        full_tool_calls = msg_chunk["tool_calls"]
                # /api/generate streaming: each chunk carries response
                elif "response" in obj:
                    full_content += obj.get("response", "")
                if obj.get("done"):
                    final_obj = obj
            except (json.JSONDecodeError, AttributeError):
                pass
    if not full_content and not full_thinking and not full_tool_calls and not final_obj:
        return None
    result = dict(final_obj)
    if full_content or full_thinking or full_tool_calls:
        if "message" in final_obj:
            msg = {**final_obj.get("message", {}), "content": full_content}
            if full_thinking:
                msg["thinking"] = full_thinking
            if full_tool_calls:
                msg["tool_calls"] = full_tool_calls
            result["message"] = msg
        else:
            result["response"] = full_content
    return result


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
            accumulated = []
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config.timeout), follow_redirects=True
                ) as client:
                    async with client.stream(method, url, headers=forward_headers, content=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                accumulated.append(chunk)
                                yield chunk
            except httpx.ConnectError:
                yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
            except Exception as exc:
                yield json.dumps({"error": str(exc)}).encode()
            if logger and request_id:
                parsed = _parse_sse_response(accumulated) if path.startswith("/v1/") else _parse_stream_response(accumulated)
                logger.log_response(request_id, 200, {}, parsed)

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


async def handle_v1_messages(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger,
) -> Response:
    """Handle /v1/messages (Anthropic-compatible) request to Ollama."""
    config = get_config()
    try:
        body_json = await request.json()
    except Exception:
        return _error_response(400, "Invalid JSON in request body")

    _modified, body_json, request_id = await rule_engine.process_request(
        "POST", request.url.path, dict(request.headers), body_json
    )

    is_stream = bool(body_json.get("stream", False)) if body_json else False

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    forward_headers = _forward_headers(dict(request.headers))

    if is_stream:
        async def generate() -> AsyncIterator[bytes]:
            accumulated = []
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config.timeout), follow_redirects=True
                ) as client:
                    async with client.stream(
                        "POST", config.target + request.url.path,
                        headers=forward_headers, content=body_bytes,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                accumulated.append(chunk)
                                yield chunk
            except httpx.TimeoutException:
                yield json.dumps({"error": "Request timeout"}).encode()
            except httpx.ConnectError:
                yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
            except Exception as exc:
                yield json.dumps({"error": str(exc)}).encode()
            logger.log_response(request_id, 200, {}, _parse_sse_response(accumulated))

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        status_code, response_headers, response_body = await _fetch_from_ollama(
            config.target, "POST", request.url.path,
            dict(request.headers), body_bytes, config.timeout,
        )
        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError:
            response_json = None
        logger.log_response(request_id, status_code, response_headers, response_json)
        return Response(
            content=response_body,
            status_code=status_code,
            headers=_forward_headers(response_headers),
            media_type=response_headers.get("content-type", "application/json"),
        )
    except httpx.TimeoutException:
        return _error_response(408, "Request timeout")
    except httpx.ConnectError:
        return _error_response(502, "Cannot connect to Ollama")
    except Exception as exc:
        return _error_response(500, str(exc))


async def handle_v1_chat_completions(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger,
) -> Response:
    """Handle /v1/chat/completions (OpenAI-compatible) request to Ollama."""
    config = get_config()
    try:
        body_json = await request.json()
    except Exception:
        return _error_response(400, "Invalid JSON in request body")

    _modified, body_json, request_id = await rule_engine.process_request(
        "POST", request.url.path, dict(request.headers), body_json
    )

    is_stream = bool(body_json.get("stream", False)) if body_json else False

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    forward_headers = _forward_headers(dict(request.headers))

    if is_stream:
        async def generate() -> AsyncIterator[bytes]:
            accumulated = []
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config.timeout), follow_redirects=True
                ) as client:
                    async with client.stream(
                        "POST", config.target + request.url.path,
                        headers=forward_headers, content=body_bytes,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                accumulated.append(chunk)
                                yield chunk
            except httpx.TimeoutException:
                yield json.dumps({"error": "Request timeout"}).encode()
            except httpx.ConnectError:
                yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
            except Exception as exc:
                yield json.dumps({"error": str(exc)}).encode()
            logger.log_response(request_id, 200, {}, _parse_openai_sse_response(accumulated))

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        status_code, response_headers, response_body = await _fetch_from_ollama(
            config.target, "POST", request.url.path,
            dict(request.headers), body_bytes, config.timeout,
        )
        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError:
            response_json = None
        logger.log_response(request_id, status_code, response_headers, response_json)
        return Response(
            content=response_body,
            status_code=status_code,
            headers=_forward_headers(response_headers),
            media_type=response_headers.get("content-type", "application/json"),
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

    body_json = _inject_num_ctx(body_json)

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

    body_json = _inject_num_ctx(body_json)

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

    body_json = _inject_num_ctx(body_json)

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    forward_headers = dict(request.headers)

    async def generate() -> AsyncIterator[bytes]:
        accumulated = []
        try:
            async for chunk in _stream_from_ollama(
                config.target, "POST", request.url.path,
                forward_headers, body_bytes, config.timeout,
            ):
                accumulated.append(chunk)
                yield chunk
        except httpx.TimeoutException:
            yield json.dumps({"error": "Request timeout"}).encode()
        except httpx.ConnectError:
            yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode()

        logger.log_response(request_id, 200, {}, _parse_stream_response(accumulated))

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

    body_json = _inject_num_ctx(body_json)

    if config.mode == "intercept":
        drop, body_json = await _apply_intercept(
            request_id, "POST", request.url.path, dict(request.headers), body_json
        )
        if drop:
            return Response(status_code=204)

    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    forward_headers = dict(request.headers)

    async def generate() -> AsyncIterator[bytes]:
        accumulated = []
        try:
            async for chunk in _stream_from_ollama(
                config.target, "POST", request.url.path,
                forward_headers, body_bytes, config.timeout,
            ):
                accumulated.append(chunk)
                yield chunk
        except httpx.TimeoutException:
            yield json.dumps({"error": "Request timeout"}).encode()
        except httpx.ConnectError:
            yield json.dumps({"error": "Cannot connect to Ollama"}).encode()
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode()

        logger.log_response(request_id, 200, {}, _parse_stream_response(accumulated))

    return StreamingResponse(generate(), media_type="application/x-ndjson")
