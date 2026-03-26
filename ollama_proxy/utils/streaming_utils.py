"""
Streaming utilities for PyProxy.

Helpers for chunked transfer, NDJSON parsing, and stream composition.
"""

import json
from typing import Any, AsyncIterator, Callable, Optional


async def aiter_lines(stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """
    Yield newline-delimited UTF-8 lines from a raw byte stream.

    Handles chunks that span multiple lines or contain partial lines.
    """
    buffer = b""
    async for chunk in stream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                yield decoded

    # Flush any remaining bytes without a trailing newline
    if buffer.strip():
        yield buffer.decode("utf-8", errors="replace").strip()


async def aiter_ndjson(stream: AsyncIterator[bytes]) -> AsyncIterator[Any]:
    """
    Parse newline-delimited JSON objects from a raw byte stream.

    Skips lines that are not valid JSON.
    """
    async for line in aiter_lines(stream):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass


async def collect_stream(stream: AsyncIterator[bytes]) -> bytes:
    """
    Fully consume an async byte stream and return the concatenated bytes.

    Use only for small/bounded responses; for large streams prefer
    forwarding chunks directly.
    """
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)


async def tee_stream(
    stream: AsyncIterator[bytes],
    callback: Callable[[bytes], Any],
) -> AsyncIterator[bytes]:
    """
    Yield each chunk from *stream* while also calling *callback* with it.

    Useful for logging or accumulating chunks without buffering the whole
    response.
    """
    async for chunk in stream:
        callback(chunk)
        yield chunk


async def limit_stream(
    stream: AsyncIterator[bytes],
    max_bytes: int,
) -> AsyncIterator[bytes]:
    """
    Yield chunks from *stream* up to *max_bytes* total, then stop.
    """
    total = 0
    async for chunk in stream:
        if total + len(chunk) > max_bytes:
            yield chunk[: max_bytes - total]
            return
        yield chunk
        total += len(chunk)
        if total >= max_bytes:
            return
