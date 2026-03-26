"""Tests for utils/streaming_utils.py."""

import pytest
from ollama_proxy.utils.streaming_utils import (
    aiter_lines,
    aiter_ndjson,
    collect_stream,
    tee_stream,
    limit_stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def bytes_stream(*chunks: bytes):
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# aiter_lines
# ---------------------------------------------------------------------------

async def test_aiter_lines_complete_newlines():
    lines = [l async for l in aiter_lines(bytes_stream(b"line1\nline2\nline3\n"))]
    assert lines == ["line1", "line2", "line3"]


async def test_aiter_lines_split_across_chunks():
    lines = [l async for l in aiter_lines(bytes_stream(b"lin", b"e1\nline2\n"))]
    assert lines == ["line1", "line2"]


async def test_aiter_lines_no_trailing_newline():
    lines = [l async for l in aiter_lines(bytes_stream(b"only\nline"))]
    assert lines == ["only", "line"]


async def test_aiter_lines_empty_stream():
    lines = [l async for l in aiter_lines(bytes_stream())]
    assert lines == []


async def test_aiter_lines_skips_blank_lines():
    lines = [l async for l in aiter_lines(bytes_stream(b"a\n\nb\n"))]
    assert lines == ["a", "b"]


# ---------------------------------------------------------------------------
# aiter_ndjson
# ---------------------------------------------------------------------------

async def test_aiter_ndjson_parses_objects():
    items = [i async for i in aiter_ndjson(bytes_stream(b'{"a":1}\n{"b":2}\n'))]
    assert items == [{"a": 1}, {"b": 2}]


async def test_aiter_ndjson_skips_invalid_json():
    items = [i async for i in aiter_ndjson(bytes_stream(b'{"ok":1}\nnot-json\n{"ok":2}\n'))]
    assert items == [{"ok": 1}, {"ok": 2}]


async def test_aiter_ndjson_empty():
    items = [i async for i in aiter_ndjson(bytes_stream())]
    assert items == []


# ---------------------------------------------------------------------------
# collect_stream
# ---------------------------------------------------------------------------

async def test_collect_stream_concatenates():
    result = await collect_stream(bytes_stream(b"hello", b" ", b"world"))
    assert result == b"hello world"


async def test_collect_stream_empty():
    assert await collect_stream(bytes_stream()) == b""


# ---------------------------------------------------------------------------
# tee_stream
# ---------------------------------------------------------------------------

async def test_tee_stream_forwards_and_calls_callback():
    received = []
    forwarded = []
    async for chunk in tee_stream(bytes_stream(b"a", b"b", b"c"), received.append):
        forwarded.append(chunk)
    assert forwarded == [b"a", b"b", b"c"]
    assert received == [b"a", b"b", b"c"]


async def test_tee_stream_empty():
    calls = []
    result = [c async for c in tee_stream(bytes_stream(), calls.append)]
    assert result == []
    assert calls == []


# ---------------------------------------------------------------------------
# limit_stream
# ---------------------------------------------------------------------------

async def test_limit_stream_within_limit():
    result = await collect_stream(limit_stream(bytes_stream(b"12345"), 10))
    assert result == b"12345"


async def test_limit_stream_exact_limit():
    result = await collect_stream(limit_stream(bytes_stream(b"12345"), 5))
    assert result == b"12345"


async def test_limit_stream_truncates():
    result = await collect_stream(limit_stream(bytes_stream(b"1234567890"), 5))
    assert result == b"12345"


async def test_limit_stream_across_chunks():
    result = await collect_stream(
        limit_stream(bytes_stream(b"123", b"456", b"789"), 5)
    )
    assert result == b"12345"


async def test_limit_stream_empty():
    assert await collect_stream(limit_stream(bytes_stream(), 100)) == b""
