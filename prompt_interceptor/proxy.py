"""
Core proxy module for PyProxy.

Handles HTTP proxying, async request forwarding, and response handling.
"""

import asyncio
import json
import time as _time
from typing import Optional, Dict, Any, AsyncIterator

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_config
from .interceptor import interceptor
from .logger import TrafficLogger, _get_ollama_ps_info
from .response_normalizer import (
    normalize_ollama_chat,
    normalize_openai_chat,
    normalize_anthropic_messages,
    emit_anthropic_sse,
    emit_openai_sse,
)
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
                elif "error" in obj:
                    # Ollama error chunk (e.g. context too long, model not found)
                    final_obj = obj
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


async def _start_streaming_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: bytes,
    timeout: int,
) -> tuple:
    """
    Initiate a streaming HTTP request and return (http_client, response).

    The response headers (including status_code) are available immediately.
    The caller is responsible for closing http_client and reading the body.
    Raises httpx exceptions on connection failure.
    """
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True)
    try:
        req = http_client.build_request(method, url, headers=headers, content=body)
        resp = await http_client.send(req, stream=True)
        return http_client, resp
    except Exception:
        await http_client.aclose()
        raise


async def _handle_error_stream(
    resp: httpx.Response,
    http_client: httpx.AsyncClient,
    status_code: int,
    logger: Optional[TrafficLogger],
    request_id: Optional[str],
) -> JSONResponse:
    """Read the full error body from a non-2xx stream, log it, and return a JSONResponse."""
    try:
        body = await resp.aread()
    except Exception:
        body = b""
    finally:
        await resp.aclose()
        await http_client.aclose()
    try:
        error_json = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        raw = body.decode("utf-8", errors="replace").strip()
        error_json = {"error": raw or f"HTTP {status_code}"}
    if logger and request_id:
        logger.log_response(request_id, status_code, {}, error_json)
    return JSONResponse(status_code=status_code, content=error_json)


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

    is_stream = body_json.get("stream", False) if body_json else False

    if is_stream:
        media = "text/event-stream" if path.startswith("/v1/") else "application/x-ndjson"
        try:
            http_client, resp = await _start_streaming_request(
                method, url, forward_headers, body, config.timeout
            )
        except httpx.ConnectError:
            if logger and request_id:
                logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
            return _error_response(502, "Cannot connect to Ollama")
        except Exception as exc:
            if logger and request_id:
                logger.log_response(request_id, 500, {}, {"error": str(exc)})
            return _error_response(500, str(exc))

        if resp.status_code >= 400:
            return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

        def _parse_passthrough(accumulated: list) -> Optional[Dict[str, Any]]:
            if path.startswith("/v1/"):
                return _parse_openai_sse_response(accumulated) if path == "/v1/chat/completions" else _parse_sse_response(accumulated)
            return _parse_stream_response(accumulated)

        async def _stream_gen():
            accumulated = []
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        accumulated.append(chunk)
                        yield chunk
            except Exception as exc:
                yield json.dumps({"error": str(exc)}).encode()
            finally:
                await resp.aclose()
                await http_client.aclose()
                if logger and request_id:
                    parsed = _parse_passthrough(accumulated)
                    if parsed is None and accumulated:
                        try:
                            parsed = json.loads(b"".join(accumulated))
                        except (json.JSONDecodeError, ValueError):
                            pass
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

    if config.context_size:
        # ── Native /api/chat path ────────────────────────────────────────────
        # Ollama's /v1/messages ignores options.num_ctx when deciding whether
        # to reload the model, so we translate to the native endpoint which
        # does honour it.
        native_body = _v1msg_to_api_chat_body(body_json, config.context_size)
        model_name = native_body.get("model", "")

        if config.mode == "intercept":
            drop, intercepted = await _apply_intercept(
                request_id, "POST", request.url.path, dict(request.headers), body_json
            )
            if drop:
                return Response(status_code=204)
            native_body = _v1msg_to_api_chat_body(intercepted or body_json, config.context_size)
            model_name = native_body.get("model", "")

        is_stream = bool(native_body.get("stream", False))
        body_bytes = json.dumps(native_body).encode("utf-8")
        forward_headers = _forward_headers(dict(request.headers))
        api_chat_url = config.target + "/api/chat"

        if is_stream:
            try:
                http_client, resp = await _start_streaming_request(
                    "POST", api_chat_url, forward_headers, body_bytes, config.timeout
                )
            except httpx.TimeoutException:
                logger.log_response(request_id, 408, {}, {"error": "Request timeout"})
                return _error_response(408, "Request timeout")
            except httpx.ConnectError:
                logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
                return _error_response(502, "Cannot connect to Ollama")
            except Exception as exc:
                logger.log_response(request_id, 500, {}, {"error": str(exc)})
                return _error_response(500, str(exc))

            if resp.status_code >= 400:
                return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

            resolved_ctx = await _resolve_context_size(model_name, config.target, config.context_size)

            async def generate_native_msg() -> AsyncIterator[bytes]:
                accumulated: list = []
                error_str = None
                try:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            accumulated.append(chunk)
                except Exception as exc:
                    error_str = str(exc)
                finally:
                    await resp.aclose()
                    await http_client.aclose()

                if error_str:
                    yield json.dumps({"error": error_str}).encode()
                    logger.log_response(request_id, 500, {}, {"error": error_str})
                    return

                parsed = _parse_stream_response(accumulated)
                if _detect_context_overflow(parsed, resolved_ctx):
                    yield json.dumps({"error": _CONTEXT_OVERFLOW_MSG}).encode() + b"\n"
                    logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                    return

                yield _api_chat_to_anthropic_sse(accumulated, model_name)
                if isinstance(parsed, dict):
                    logger.log_response(request_id, 200, {}, _api_chat_to_v1msg_response(parsed))
                else:
                    logger.log_response(request_id, 200, {}, parsed)

            return StreamingResponse(generate_native_msg(), media_type="text/event-stream")

        # Non-streaming native path
        try:
            status_code, response_headers, response_body = await _fetch_from_ollama(
                config.target, "POST", "/api/chat",
                dict(request.headers), body_bytes, config.timeout,
            )
            try:
                ollama_resp = json.loads(response_body)
            except json.JSONDecodeError:
                ollama_resp = None
            if isinstance(ollama_resp, dict):
                ctx = await _resolve_context_size(
                    ollama_resp.get("model") or model_name,
                    config.target, config.context_size,
                )
                if _detect_context_overflow(ollama_resp, ctx):
                    logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                    return _error_response(413, _CONTEXT_OVERFLOW_MSG)
                v1msg_resp = _api_chat_to_v1msg_response(ollama_resp)
                logger.log_response(request_id, status_code, response_headers, v1msg_resp)
                return JSONResponse(content=v1msg_resp, status_code=status_code)
            else:
                logger.log_response(request_id, status_code, response_headers, ollama_resp)
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

    # ── Legacy path: forward as-is to /v1/messages ──────────────────────────
    body_json = _inject_num_ctx(body_json)
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
        _v1msg_url = config.target + request.url.path
        try:
            http_client, resp = await _start_streaming_request(
                "POST", _v1msg_url, forward_headers, body_bytes, config.timeout
            )
        except httpx.TimeoutException:
            logger.log_response(request_id, 408, {}, {"error": "Request timeout"})
            return _error_response(408, "Request timeout")
        except httpx.ConnectError:
            logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
            return _error_response(502, "Cannot connect to Ollama")
        except Exception as exc:
            logger.log_response(request_id, 500, {}, {"error": str(exc)})
            return _error_response(500, str(exc))

        if resp.status_code >= 400:
            return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

        async def generate() -> AsyncIterator[bytes]:
            accumulated = []
            error_str = None
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        accumulated.append(chunk)
            except Exception as exc:
                error_str = str(exc)
            finally:
                await resp.aclose()
                await http_client.aclose()

            if error_str:
                yield json.dumps({"error": error_str}).encode()
                logger.log_response(request_id, 500, {}, {"error": error_str})
                return

            parsed = _parse_sse_response(accumulated)
            if parsed is None and accumulated:
                try:
                    parsed = json.loads(b"".join(accumulated))
                except (json.JSONDecodeError, ValueError):
                    pass

            corrected, was_corrected, fix_desc = (
                normalize_anthropic_messages(parsed) if isinstance(parsed, dict) else (parsed, False, '')
            )
            if was_corrected:
                yield emit_anthropic_sse(corrected)
                logger.log_response(request_id, 200, {}, corrected, correction_applied=fix_desc)
            else:
                for chunk in accumulated:
                    yield chunk
                logger.log_response(request_id, 200, {}, parsed)

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
        if isinstance(response_json, dict):
            response_json, was_corrected, fix_desc = normalize_anthropic_messages(response_json)
            if was_corrected:
                response_body = json.dumps(response_json).encode("utf-8")
                logger.log_response(request_id, status_code, response_headers, response_json, correction_applied=fix_desc)
            else:
                logger.log_response(request_id, status_code, response_headers, response_json)
        else:
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


# ---------------------------------------------------------------------------
# /v1/messages → /api/chat translation helpers
# ---------------------------------------------------------------------------

def _v1msg_to_api_chat_body(body_json: Dict[str, Any], num_ctx: Optional[int]) -> Dict[str, Any]:
    """Translate an Anthropic /v1/messages body to Ollama /api/chat format."""
    messages = list(body_json.get("messages", []))

    # Anthropic top-level system field → prepend as system message
    if body_json.get("system"):
        sys_content = body_json["system"]
        if isinstance(sys_content, list):
            sys_content = " ".join(
                b.get("text", "") for b in sys_content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        messages = [{"role": "system", "content": sys_content}] + messages

    # Normalize Anthropic content-block arrays to plain strings for Ollama
    normalized: list = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        inner = " ".join(
                            b.get("text", "") for b in inner
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    parts.append(str(inner))
            content = "".join(parts)
        normalized.append({**msg, "content": content})

    options: Dict[str, Any] = {}
    if num_ctx:
        options["num_ctx"] = num_ctx
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src in body_json:
            options.setdefault(dst, body_json[src])
    if body_json.get("max_tokens"):
        options.setdefault("num_predict", body_json["max_tokens"])

    result: Dict[str, Any] = {
        "model": body_json.get("model", ""),
        "messages": normalized,
        "stream": body_json.get("stream", False),
    }
    if options:
        result["options"] = options
    if "tools" in body_json:
        result["tools"] = body_json["tools"]
    return result


def _api_chat_to_v1msg_response(ollama_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a non-streaming /api/chat response to Anthropic /v1/messages format."""
    model = ollama_resp.get("model", "")
    message = dict(ollama_resp.get("message") or {})
    content_text = message.get("content", "")
    thinking = message.get("thinking", "") or message.get("reasoning", "") or ""
    done_reason = ollama_resp.get("done_reason") or "stop"
    stop_reason = "max_tokens" if done_reason == "length" else "end_turn"
    if message.get("tool_calls"):
        stop_reason = "tool_use"
    input_tokens = ollama_resp.get("prompt_eval_count") or 0
    output_tokens = ollama_resp.get("eval_count") or 0
    content_blocks: list = []
    if thinking:
        content_blocks.append({"type": "thinking", "thinking": thinking})
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            content_blocks.append({
                "type": "tool_use",
                "id": f"toolu_{int(_time.time())}",
                "name": fn.get("name", ""),
                "input": args,
            })
    return {
        "id": f"msg_{int(_time.time())}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _api_chat_to_anthropic_sse(accumulated: list, model: str) -> bytes:
    """Convert accumulated Ollama /api/chat NDJSON chunks to Anthropic SSE format."""
    full_content = ""
    full_thinking = ""
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason = "end_turn"
    for chunk in accumulated:
        for line in chunk.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msg = obj.get("message") or {}
                full_content += msg.get("content", "")
                full_thinking += msg.get("thinking", "") or msg.get("reasoning", "") or ""
                if obj.get("done"):
                    prompt_tokens = obj.get("prompt_eval_count", 0) or 0
                    completion_tokens = obj.get("eval_count", 0) or 0
                    done_reason = obj.get("done_reason") or "stop"
                    stop_reason = "max_tokens" if done_reason == "length" else "end_turn"
            except (json.JSONDecodeError, AttributeError):
                pass

    msg_id = f"msg_{int(_time.time())}"

    def _ev(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    events: list = [
        _ev("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
            },
        }),
    ]
    idx = 0
    if full_thinking:
        events.append(_ev("content_block_start", {
            "type": "content_block_start", "index": idx,
            "content_block": {"type": "thinking", "thinking": ""},
        }))
        events.append(_ev("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "thinking_delta", "thinking": full_thinking},
        }))
        events.append(_ev("content_block_stop", {"type": "content_block_stop", "index": idx}))
        idx += 1
    events.append(_ev("content_block_start", {
        "type": "content_block_start", "index": idx,
        "content_block": {"type": "text", "text": ""},
    }))
    events.append(_ev("ping", {"type": "ping"}))
    if full_content:
        events.append(_ev("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "text_delta", "text": full_content},
        }))
    events.append(_ev("content_block_stop", {"type": "content_block_stop", "index": idx}))
    events.append(_ev("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": completion_tokens},
    }))
    events.append(_ev("message_stop", {"type": "message_stop"}))
    return "".join(events).encode("utf-8")


# ---------------------------------------------------------------------------
# /v1/chat/completions → /api/chat translation helpers
# ---------------------------------------------------------------------------

def _v1cc_to_api_chat_body(body_json: Dict[str, Any], num_ctx: Optional[int]) -> Dict[str, Any]:
    """Translate an OpenAI /v1/chat/completions body to Ollama /api/chat format."""
    options: Dict[str, Any] = dict(body_json.get("options") or {})
    if num_ctx:
        options["num_ctx"] = num_ctx
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed")):
        if src in body_json:
            options.setdefault(dst, body_json[src])
    for max_f in ("max_tokens", "max_completion_tokens"):
        if max_f in body_json:
            options.setdefault("num_predict", body_json[max_f])
            break
    result: Dict[str, Any] = {
        "model": body_json.get("model", ""),
        "messages": body_json.get("messages", []),
        "stream": body_json.get("stream", False),
    }
    if options:
        result["options"] = options
    if "tools" in body_json:
        result["tools"] = body_json["tools"]
    if "format" in body_json:
        result["format"] = body_json["format"]
    return result


def _api_chat_to_v1cc_response(ollama_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a non-streaming /api/chat response to OpenAI /v1/chat/completions format."""
    model = ollama_resp.get("model", "")
    message = dict(ollama_resp.get("message") or {})
    done_reason = ollama_resp.get("done_reason") or "stop"
    finish_reason = "length" if done_reason == "length" else "stop"
    if message.get("tool_calls"):
        finish_reason = "tool_calls"
    prompt_tokens = ollama_resp.get("prompt_eval_count") or 0
    completion_tokens = ollama_resp.get("eval_count") or 0
    return {
        "id": f"chatcmpl-{int(_time.time())}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _api_chat_chunk_to_openai_sse(chunk: Dict[str, Any], chat_id: str, created: int) -> str:
    """Convert a single Ollama /api/chat NDJSON chunk to an OpenAI SSE data line."""
    base: Dict[str, Any] = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": chunk.get("model", ""),
    }
    msg = chunk.get("message") or {}
    if not chunk.get("done"):
        content = msg.get("content", "")
        thinking = msg.get("thinking", "") or msg.get("reasoning", "") or ""
        delta: Dict[str, Any] = {}
        if thinking:
            delta["thinking"] = thinking
        if content:
            delta["content"] = content
        choice: Dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": None}
    else:
        done_reason = chunk.get("done_reason") or "stop"
        finish_reason = "length" if done_reason == "length" else "stop"
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            finish_reason = "tool_calls"
            choice = {"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": finish_reason}
        else:
            choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    return f"data: {json.dumps({**base, 'choices': [choice]})}\n\n"


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

    if config.context_size:
        # ── Native /api/chat path ────────────────────────────────────────────
        # Ollama's /v1/ endpoints ignore options.num_ctx when deciding whether
        # to reload the model, so we translate to the native endpoint which
        # does honour it.
        native_body = _v1cc_to_api_chat_body(body_json, config.context_size)
        model_name = native_body.get("model", "")

        if config.mode == "intercept":
            drop, intercepted = await _apply_intercept(
                request_id, "POST", request.url.path, dict(request.headers), body_json
            )
            if drop:
                return Response(status_code=204)
            native_body = _v1cc_to_api_chat_body(intercepted or body_json, config.context_size)
            model_name = native_body.get("model", "")

        is_stream = bool(native_body.get("stream", False))
        body_bytes = json.dumps(native_body).encode("utf-8")
        forward_headers = _forward_headers(dict(request.headers))
        api_chat_url = config.target + "/api/chat"

        if is_stream:
            try:
                http_client, resp = await _start_streaming_request(
                    "POST", api_chat_url, forward_headers, body_bytes, config.timeout
                )
            except httpx.TimeoutException:
                logger.log_response(request_id, 408, {}, {"error": "Request timeout"})
                return _error_response(408, "Request timeout")
            except httpx.ConnectError:
                logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
                return _error_response(502, "Cannot connect to Ollama")
            except Exception as exc:
                logger.log_response(request_id, 500, {}, {"error": str(exc)})
                return _error_response(500, str(exc))

            if resp.status_code >= 400:
                return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

            resolved_ctx = await _resolve_context_size(model_name, config.target, config.context_size)
            chat_id = f"chatcmpl-{int(_time.time())}"
            created = int(_time.time())

            async def generate_native() -> AsyncIterator[bytes]:
                accumulated: list = []
                error_str = None
                try:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            accumulated.append(chunk)
                except Exception as exc:
                    error_str = str(exc)
                finally:
                    await resp.aclose()
                    await http_client.aclose()

                if error_str:
                    yield json.dumps({"error": error_str}).encode()
                    logger.log_response(request_id, 500, {}, {"error": error_str})
                    return

                parsed = _parse_stream_response(accumulated)
                if _detect_context_overflow(parsed, resolved_ctx):
                    yield json.dumps({"error": _CONTEXT_OVERFLOW_MSG}).encode() + b"\n"
                    logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                    return

                for raw in accumulated:
                    for line in raw.split(b"\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            yield _api_chat_chunk_to_openai_sse(obj, chat_id, created).encode("utf-8")
                        except (json.JSONDecodeError, AttributeError):
                            pass
                yield b"data: [DONE]\n\n"

                if isinstance(parsed, dict):
                    logger.log_response(request_id, 200, {}, _api_chat_to_v1cc_response(parsed))
                else:
                    logger.log_response(request_id, 200, {}, parsed)

            return StreamingResponse(generate_native(), media_type="text/event-stream")

        # Non-streaming native path
        try:
            status_code, response_headers, response_body = await _fetch_from_ollama(
                config.target, "POST", "/api/chat",
                dict(request.headers), body_bytes, config.timeout,
            )
            try:
                ollama_resp = json.loads(response_body)
            except json.JSONDecodeError:
                ollama_resp = None
            if isinstance(ollama_resp, dict):
                ctx = await _resolve_context_size(
                    ollama_resp.get("model") or model_name,
                    config.target, config.context_size,
                )
                if _detect_context_overflow(ollama_resp, ctx):
                    logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                    return _error_response(413, _CONTEXT_OVERFLOW_MSG)
                v1_resp = _api_chat_to_v1cc_response(ollama_resp)
                logger.log_response(request_id, status_code, response_headers, v1_resp)
                return JSONResponse(content=v1_resp, status_code=status_code)
            else:
                logger.log_response(request_id, status_code, response_headers, ollama_resp)
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

    # ── Legacy path: forward as-is to /v1/chat/completions ──────────────────
    body_json = _inject_num_ctx(body_json)

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
        _v1cc_url = config.target + request.url.path
        try:
            http_client, resp = await _start_streaming_request(
                "POST", _v1cc_url, forward_headers, body_bytes, config.timeout
            )
        except httpx.TimeoutException:
            logger.log_response(request_id, 408, {}, {"error": "Request timeout"})
            return _error_response(408, "Request timeout")
        except httpx.ConnectError:
            logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
            return _error_response(502, "Cannot connect to Ollama")
        except Exception as exc:
            logger.log_response(request_id, 500, {}, {"error": str(exc)})
            return _error_response(500, str(exc))

        if resp.status_code >= 400:
            return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

        model_name = (body_json or {}).get("model")
        resolved_ctx = await _resolve_context_size(model_name, config.target, config.context_size)

        async def generate() -> AsyncIterator[bytes]:
            accumulated = []
            error_str = None
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        accumulated.append(chunk)
            except Exception as exc:
                error_str = str(exc)
            finally:
                await resp.aclose()
                await http_client.aclose()

            if error_str:
                yield json.dumps({"error": error_str}).encode()
                logger.log_response(request_id, 500, {}, {"error": error_str})
                return

            parsed = _parse_openai_sse_response(accumulated)
            if parsed is None and accumulated:
                try:
                    parsed = json.loads(b"".join(accumulated))
                except (json.JSONDecodeError, ValueError):
                    pass

            if _detect_context_overflow(parsed, resolved_ctx):
                yield json.dumps({"error": _CONTEXT_OVERFLOW_MSG}).encode()
                logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                return

            corrected, was_corrected, fix_desc = (
                normalize_openai_chat(parsed) if isinstance(parsed, dict) else (parsed, False, '')
            )
            if was_corrected:
                yield emit_openai_sse(corrected)
                logger.log_response(request_id, 200, {}, corrected, correction_applied=fix_desc)
            else:
                for chunk in accumulated:
                    yield chunk
                logger.log_response(request_id, 200, {}, parsed)

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
        if isinstance(response_json, dict):
            ctx = await _resolve_context_size(
                response_json.get("model") or (body_json or {}).get("model"),
                config.target, config.context_size,
            )
            if _detect_context_overflow(response_json, ctx):
                logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                return _error_response(413, _CONTEXT_OVERFLOW_MSG)
            response_json, was_corrected, fix_desc = normalize_openai_chat(response_json)
            if was_corrected:
                response_body = json.dumps(response_json).encode("utf-8")
                logger.log_response(request_id, status_code, response_headers, response_json, correction_applied=fix_desc)
            else:
                logger.log_response(request_id, status_code, response_headers, response_json)
        else:
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


_CONTEXT_OVERFLOW_MSG = (
    "Contexto agotado: el modelo truncó su respuesta al alcanzar el límite de contexto. "
    "Reduce la longitud de la conversación o aumenta el tamaño de contexto."
)


async def _preload_model_with_ctx(model: str, target: str, num_ctx: int) -> None:
    """
    Ensure Ollama loads the model with the correct num_ctx via the native API.

    Ollama's /v1/ (OpenAI/Anthropic-compat) endpoints do not trigger a model
    reload when options.num_ctx changes, so we force it here by sending a
    no-op request to /api/generate (the native endpoint that does respect it).
    Skipped if the model is already loaded with the correct context size.
    """
    if not model or not num_ctx:
        return
    try:
        ps_info = await asyncio.to_thread(_get_ollama_ps_info, model, target)
        if ps_info.get("context_size") == num_ctx:
            return
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            await client.post(
                target + "/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "options": {"num_ctx": num_ctx},
                },
            )
    except Exception:
        pass


async def _resolve_context_size(
    model: Optional[str],
    target: str,
    config_ctx: Optional[int],
) -> Optional[int]:
    """Query /api/ps for the model's actual loaded context size; falls back to config."""
    if model:
        try:
            ps_info = await asyncio.to_thread(_get_ollama_ps_info, model, target)
            if ps_info.get("context_size"):
                return ps_info["context_size"]
        except Exception:
            pass
    return config_ctx


def _detect_context_overflow(
    parsed: Optional[Dict[str, Any]],
    context_size: Optional[int] = None,
) -> bool:
    """Return True if Ollama indicates the response was cut off due to context limit.

    Two signals are checked:
    - done_reason/finish_reason == "length" (model stopped at context boundary)
    - prompt_tokens >= context_size (input already fills 100% of the window)
    """
    if not isinstance(parsed, dict):
        return False
    # Ollama native /api/chat and /api/generate
    if parsed.get("done_reason") == "length":
        return True
    # OpenAI-compat /v1/chat/completions
    choices = parsed.get("choices") or []
    if choices and isinstance(choices[0], dict) and choices[0].get("finish_reason") == "length":
        return True
    # Context utilization >= 100% (mirrors the dashboard calculation)
    if context_size:
        prompt_tokens = parsed.get("prompt_eval_count")
        if prompt_tokens is None and isinstance(parsed.get("usage"), dict):
            u = parsed["usage"]
            prompt_tokens = u.get("prompt_tokens") or u.get("input_tokens")
        if prompt_tokens is not None and prompt_tokens >= context_size:
            return True
    return False


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

        if isinstance(response_json, dict):
            ctx = await _resolve_context_size(
                response_json.get("model") or (body_json or {}).get("model"),
                config.target, config.context_size,
            )
            if _detect_context_overflow(response_json, ctx):
                logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                return _error_response(413, _CONTEXT_OVERFLOW_MSG)
            response_json, was_corrected, fix_desc = normalize_ollama_chat(response_json)
            if was_corrected:
                response_body = json.dumps(response_json).encode("utf-8")
                logger.log_response(request_id, response_code, response_headers, response_json, correction_applied=fix_desc)
            else:
                logger.log_response(request_id, response_code, response_headers, response_json)
        else:
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

        if isinstance(response_json, dict):
            ctx = await _resolve_context_size(
                response_json.get("model") or (body_json or {}).get("model"),
                config.target, config.context_size,
            )
            if _detect_context_overflow(response_json, ctx):
                logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
                return _error_response(413, _CONTEXT_OVERFLOW_MSG)
            response_json, was_corrected, fix_desc = normalize_ollama_chat(response_json)
            if was_corrected:
                response_body = json.dumps(response_json).encode("utf-8")
                logger.log_response(request_id, response_code, response_headers, response_json, correction_applied=fix_desc)
            else:
                logger.log_response(request_id, response_code, response_headers, response_json)
        else:
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
    forward_headers = _forward_headers(dict(request.headers))

    try:
        http_client, resp = await _start_streaming_request(
            "POST", config.target + request.url.path,
            forward_headers, body_bytes, config.timeout,
        )
    except httpx.TimeoutException:
        logger.log_response(request_id, 408, {}, {"error": "Request timeout"})
        return _error_response(408, "Request timeout")
    except httpx.ConnectError:
        logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
        return _error_response(502, "Cannot connect to Ollama")
    except Exception as exc:
        logger.log_response(request_id, 500, {}, {"error": str(exc)})
        return _error_response(500, str(exc))

    if resp.status_code >= 400:
        return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

    model_name = (body_json or {}).get("model")
    resolved_ctx = await _resolve_context_size(model_name, config.target, config.context_size)

    async def generate() -> AsyncIterator[bytes]:
        accumulated = []
        error_str = None
        try:
            async for chunk in resp.aiter_bytes():
                if chunk:
                    accumulated.append(chunk)
        except Exception as exc:
            error_str = str(exc)
        finally:
            await resp.aclose()
            await http_client.aclose()

        if error_str:
            yield json.dumps({"error": error_str}).encode()
            logger.log_response(request_id, 500, {}, {"error": error_str})
            return

        parsed = _parse_stream_response(accumulated)
        if parsed is None and accumulated:
            try:
                parsed = json.loads(b"".join(accumulated))
            except (json.JSONDecodeError, ValueError):
                pass

        if _detect_context_overflow(parsed, resolved_ctx):
            yield json.dumps({"error": _CONTEXT_OVERFLOW_MSG}).encode() + b"\n"
            logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
            return

        corrected, was_corrected, fix_desc = (
            normalize_ollama_chat(parsed) if isinstance(parsed, dict) else (parsed, False, '')
        )
        if was_corrected:
            yield json.dumps(corrected).encode() + b"\n"
            logger.log_response(request_id, 200, {}, corrected, correction_applied=fix_desc)
        else:
            for chunk in accumulated:
                yield chunk
            logger.log_response(request_id, 200, {}, parsed)

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
    forward_headers = _forward_headers(dict(request.headers))

    try:
        http_client, resp = await _start_streaming_request(
            "POST", config.target + request.url.path,
            forward_headers, body_bytes, config.timeout,
        )
    except httpx.TimeoutException:
        logger.log_response(request_id, 408, {}, {"error": "Request timeout"})
        return _error_response(408, "Request timeout")
    except httpx.ConnectError:
        logger.log_response(request_id, 502, {}, {"error": "Cannot connect to Ollama"})
        return _error_response(502, "Cannot connect to Ollama")
    except Exception as exc:
        logger.log_response(request_id, 500, {}, {"error": str(exc)})
        return _error_response(500, str(exc))

    if resp.status_code >= 400:
        return await _handle_error_stream(resp, http_client, resp.status_code, logger, request_id)

    model_name = (body_json or {}).get("model")
    resolved_ctx = await _resolve_context_size(model_name, config.target, config.context_size)

    async def generate() -> AsyncIterator[bytes]:
        accumulated = []
        error_str = None
        try:
            async for chunk in resp.aiter_bytes():
                if chunk:
                    accumulated.append(chunk)
        except Exception as exc:
            error_str = str(exc)
        finally:
            await resp.aclose()
            await http_client.aclose()

        if error_str:
            yield json.dumps({"error": error_str}).encode()
            logger.log_response(request_id, 500, {}, {"error": error_str})
            return

        parsed = _parse_stream_response(accumulated)
        if parsed is None and accumulated:
            try:
                parsed = json.loads(b"".join(accumulated))
            except (json.JSONDecodeError, ValueError):
                pass

        if _detect_context_overflow(parsed, resolved_ctx):
            yield json.dumps({"error": _CONTEXT_OVERFLOW_MSG}).encode() + b"\n"
            logger.log_response(request_id, 413, {}, {"error": _CONTEXT_OVERFLOW_MSG})
            return

        corrected, was_corrected, fix_desc = (
            normalize_ollama_chat(parsed) if isinstance(parsed, dict) else (parsed, False, '')
        )
        if was_corrected:
            yield json.dumps(corrected).encode() + b"\n"
            logger.log_response(request_id, 200, {}, corrected, correction_applied=fix_desc)
        else:
            for chunk in accumulated:
                yield chunk
            logger.log_response(request_id, 200, {}, parsed)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
