"""
Core proxy module for PyProxy.

Handles HTTP proxying, async request forwarding, and response handling.
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, AsyncIterator
from pathlib import Path

import httpx
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from .config import get_config
from .logger import TrafficLogger
from .models.request_model import (
    ChatRequest, GenerateRequest, ChatResponse,
    GenerateResponse, parse_chat_request, parse_generate_request,
    create_chat_response, create_generate_response
)
from .rules_engine import RuleEngine


async def _parse_stream_response(
    async_iterable: Any
) -> AsyncIterator[tuple[str, int, Dict[str, str]]]:
    """
    Parse streaming response chunks into events.

    Args:
        async_iterable: Async generator from Ollama

    Yields:
        (data, content_type, status) tuples
    """
    content_type = "application/x-ndjson"
    status_code = 200
    chunk_num = 0

    async for chunk in async_iterable:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        elif not chunk:
            continue

        try:
            if isinstance(chunk, str):
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    # Try raw chunk
                    pass
                else:
                    chunk_num += 1
                    yield chunk, content_type, status_code

        except Exception:
            pass

    yield "", content_type, status_code


async def _fetch_from_ollama(
    target: str,
    method: str,
    path: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    timeout: int = 120
) -> tuple:
    """
    Fetch data from Ollama backend.

    Args:
        target: Ollama target URL
        method: HTTP method
        path: Request path
        headers: Request headers
        body: Request body
        timeout: Timeout in seconds

    Returns:
        (status_code, headers, body)
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True
    ) as client:
        response = await client.request(
            method,
            target + path,
            headers=headers,
            content=body
        )
        return (
            response.status_code,
            response.headers,
            await response.aread()
        )


async def _fetch_streaming(
    target: str,
    method: str,
    path: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    timeout: int = 120
) -> AsyncIterator[tuple[str, int, Dict[str, str]]]:
    """
    Fetch streaming response from Ollama.

    Args:
        target: Ollama target URL
        method: HTTP method
        path: Request path
        headers: Request headers
        body: Request body
        timeout: Timeout in seconds

    Yields:
        (data, content_type, status) tuples
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True
        ) as client:
            async with client.stream(
                method,
                target + path,
                headers=headers,
                content=body
            ) as response:
                status_code = response.status_code
                content_type = response.headers.get("content-type", "")
                yield "", content_type, status_code

                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunk = chunk.decode("utf-8")
                        yield chunk, content_type, status_code

    except httpx.ConnectError as e:
        yield "", "text/plain", 502


async def handle_chat_request(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger
) -> Response:
    """Handle chat request to Ollama."""
    config = get_config()
    target = config.target
    body_json = await request.json()

    # Apply rules
    modified, body, request_id = await rule_engine.process_request(
        "POST",
        request.url.path,
        dict(request.headers),
        body_json
    )

    if modified:
        body_json = body

    try:
        # Forward request
        response_code, response_headers, body_bytes = await _fetch_from_ollama(
            target,
            "POST",
            request.url.path,
            dict(request.headers),
            body_json.encode("utf-8") if body_json else b"",
            config.timeout
        )

        # Log response
        logger.log_response(
            request_id,
            response_code,
            response_headers,
            json.loads(body_bytes.decode("utf-8")) if body_bytes else None
        )

        # Build response
        return JSONResponse(
            status_code=response_code,
            content=body_bytes,
            headers=dict(response_headers)
        )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=408,
            content=json.dumps({"error": "Request timeout"}),
            headers={"content-type": "application/json"}
        )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content=e.response.json() if e.response.content else {"error": str(e)},
            headers=dict(e.response.headers)
        )

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content=json.dumps({"error": "Invalid JSON"}),
            headers={"content-type": "application/json"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=json.dumps({"error": str(e)}),
            headers={"content-type": "application/json"}
        )


async def handle_generate_request(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger
) -> Response:
    """Handle generate request to Ollama."""
    config = get_config()
    body_json = await request.json()

    # Apply rules
    modified, body, request_id = await rule_engine.process_request(
        "POST",
        request.url.path,
        dict(request.headers),
        body_json
    )

    if modified:
        body_json = body

    try:
        response_code, response_headers, body_bytes = await _fetch_from_ollama(
            config.target,
            "POST",
            request.url.path,
            dict(request.headers),
            body_json.encode("utf-8") if body_json else b"",
            config.timeout
        )

        logger.log_response(
            request_id,
            response_code,
            response_headers,
            json.loads(body_bytes.decode("utf-8")) if body_bytes else None
        )

        return JSONResponse(
            status_code=response_code,
            content=body_bytes,
            headers=dict(response_headers)
        )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=408,
            content=json.dumps({"error": "Request timeout"}),
            headers={"content-type": "application/json"}
        )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content=e.response.json() if e.response.content else {"error": str(e)},
            headers=dict(e.response.headers)
        )

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content=json.dumps({"error": "Invalid JSON"}),
            headers={"content-type": "application/json"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=json.dumps({"error": str(e)}),
            headers={"content-type": "application/json"}
        )


async def handle_stream_chat(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger
) -> Response:
    """Handle streaming chat request."""
    config = get_config()
    body_json = await request.json()

    # Apply rules
    modified, body, request_id = await rule_engine.process_request(
        "POST",
        request.url.path,
        dict(request.headers),
        body_json
    )

    if modified:
        body_json = body

    try:
        # Stream response
        async for chunk, content_type, status_code in _fetch_streaming(
            config.target,
            "POST",
            request.url.path,
            dict(request.headers),
            body_json.encode("utf-8") if body_json else b"",
            config.timeout
        ):
            logger.log_raw_request(
                request_id,
                chunk if chunk else "",
                dict(request.headers)
            )

        logger.log_response(
            request_id,
            status_code,
            dict(request.headers),
            None
        )

        return Response(
            content=chunk if chunk else b"",
            status_code=status_code,
            media_type=content_type
        )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=408,
            content=json.dumps({"error": "Request timeout"}),
            headers={"content-type": "application/json"}
        )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content=e.response.json() if e.response.content else {"error": str(e)},
            headers=dict(e.response.headers)
        )

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content=json.dumps({"error": "Invalid JSON"}),
            headers={"content-type": "application/json"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=json.dumps({"error": str(e)}),
            headers={"content-type": "application/json"}
        )


async def handle_stream_generate(
    request: Request,
    rule_engine: RuleEngine,
    logger: TrafficLogger
) -> Response:
    """Handle streaming generate request."""
    config = get_config()
    body_json = await request.json()

    # Apply rules
    modified, body, request_id = await rule_engine.process_request(
        "POST",
        request.url.path,
        dict(request.headers),
        body_json
    )

    if modified:
        body_json = body

    try:
        async for chunk, content_type, status_code in _fetch_streaming(
            config.target,
            "POST",
            request.url.path,
            dict(request.headers),
            body_json.encode("utf-8") if body_json else b"",
            config.timeout
        ):
            logger.log_raw_request(
                request_id,
                chunk if chunk else "",
                dict(request.headers)
            )

        logger.log_response(
            request_id,
            status_code,
            dict(request.headers),
            None
        )

        return Response(
            content=chunk if chunk else b"",
            status_code=status_code,
            media_type=content_type
        )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=408,
            content=json.dumps({"error": "Request timeout"}),
            headers={"content-type": "application/json"}
        )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content=e.response.json() if e.response.content else {"error": str(e)},
            headers=dict(e.response.headers)
        )

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content=json.dumps({"error": "Invalid JSON"}),
            headers={"content-type": "application/json"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=json.dumps({"error": str(e)}),
            headers={"content-type": "application/json"}
        )
