"""Tests for interceptor.py."""

import asyncio
import pytest

from prompt_interceptor.interceptor import Interceptor, InterceptedRequest


# ---------------------------------------------------------------------------
# InterceptedRequest
# ---------------------------------------------------------------------------

async def test_forward_returns_original_body():
    req = InterceptedRequest("rid1", "POST", "/api/chat", {}, {"model": "llama3"})

    async def resolve():
        await asyncio.sleep(0.01)
        req.forward()

    asyncio.create_task(resolve())
    action, body = await req.wait(timeout=2.0)
    assert action == "forward"
    assert body == {"model": "llama3"}


async def test_edit_returns_modified_body():
    req = InterceptedRequest("rid2", "POST", "/api/chat", {}, {"model": "llama3"})

    async def resolve():
        await asyncio.sleep(0.01)
        req.edit({"model": "deepseek"})

    asyncio.create_task(resolve())
    action, body = await req.wait(timeout=2.0)
    assert action == "edit"
    assert body == {"model": "deepseek"}


async def test_drop_returns_drop_action():
    req = InterceptedRequest("rid3", "POST", "/api/chat", {}, {"model": "llama3"})

    async def resolve():
        await asyncio.sleep(0.01)
        req.drop()

    asyncio.create_task(resolve())
    action, body = await req.wait(timeout=2.0)
    assert action == "drop"
    assert body == {"model": "llama3"}  # original body on drop


async def test_timeout_auto_forwards():
    req = InterceptedRequest("rid4", "POST", "/", {}, {"k": "v"})
    action, body = await req.wait(timeout=0.05)
    assert action == "forward"
    assert body == {"k": "v"}


def test_to_dict_structure():
    req = InterceptedRequest("rid5", "GET", "/health", {"h": "1"}, None)
    d = req.to_dict()
    assert d["request_id"] == "rid5"
    assert d["method"] == "GET"
    assert d["path"] == "/health"
    assert "created_at" in d


def test_edit_deep_copies_body():
    original = {"model": "llama3"}
    req = InterceptedRequest("rid6", "POST", "/", {}, original)
    # The stored body should be a deep copy — mutating original doesn't affect it
    original["model"] = "changed"
    assert req.body["model"] == "llama3"


# ---------------------------------------------------------------------------
# Interceptor singleton helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def interceptor():
    return Interceptor(intercept_timeout=0.1)


async def test_get_pending_empty(interceptor):
    assert interceptor.get_pending() == []


async def test_resolve_forward_not_found(interceptor):
    assert interceptor.resolve_forward("unknown-id") is False


async def test_resolve_edit_not_found(interceptor):
    assert interceptor.resolve_edit("unknown-id", {}) is False


async def test_resolve_drop_not_found(interceptor):
    assert interceptor.resolve_drop("unknown-id") is False


async def test_len_reflects_pending(interceptor):
    async def _intercept():
        await interceptor.intercept("r1", "POST", "/", {}, {})

    t = asyncio.create_task(_intercept())
    await asyncio.sleep(0.01)
    assert len(interceptor) == 1
    interceptor.resolve_forward("r1")
    await t
    assert len(interceptor) == 0


async def test_intercept_forward(interceptor):
    async def _intercept():
        return await interceptor.intercept("r2", "POST", "/api/chat", {}, {"model": "x"})

    t = asyncio.create_task(_intercept())
    await asyncio.sleep(0.01)

    pending = interceptor.get_pending()
    assert len(pending) == 1
    assert pending[0]["request_id"] == "r2"

    interceptor.resolve_forward("r2")
    action, body = await t
    assert action == "forward"
    assert body == {"model": "x"}


async def test_intercept_edit(interceptor):
    async def _intercept():
        return await interceptor.intercept("r3", "POST", "/", {}, {"model": "old"})

    t = asyncio.create_task(_intercept())
    await asyncio.sleep(0.01)
    interceptor.resolve_edit("r3", {"model": "new"})
    action, body = await t
    assert action == "edit"
    assert body == {"model": "new"}


async def test_intercept_drop(interceptor):
    async def _intercept():
        return await interceptor.intercept("r4", "POST", "/", {}, {})

    t = asyncio.create_task(_intercept())
    await asyncio.sleep(0.01)
    interceptor.resolve_drop("r4")
    action, _ = await t
    assert action == "drop"


async def test_intercept_timeout_auto_forward(interceptor):
    action, body = await interceptor.intercept("r5", "POST", "/", {}, {"k": "v"})
    assert action == "forward"


async def test_pending_cleaned_after_resolution(interceptor):
    async def _intercept():
        await interceptor.intercept("r6", "POST", "/", {}, {})

    t = asyncio.create_task(_intercept())
    await asyncio.sleep(0.01)
    interceptor.resolve_forward("r6")
    await t
    assert len(interceptor) == 0


# ---------------------------------------------------------------------------
# debug_intercept tracing
# ---------------------------------------------------------------------------

async def test_intercept_logs_when_debug_enabled(interceptor, monkeypatch):
    """intercept() emits info logs when debug_intercept is True."""
    from unittest.mock import MagicMock, patch
    from prompt_interceptor import interceptor as interceptor_mod

    cfg = MagicMock()
    cfg.debug_intercept = True
    monkeypatch.setattr(interceptor_mod, "get_config", lambda: cfg)

    log_calls = []
    with patch.object(interceptor_mod._log, "info", side_effect=lambda *a, **kw: log_calls.append(a[0] % a[1:])):
        async def _intercept():
            return await interceptor.intercept("dbg1", "POST", "/api/chat", {}, {})

        t = asyncio.create_task(_intercept())
        await asyncio.sleep(0.01)
        interceptor.resolve_forward("dbg1")
        await t

    assert any("paused" in c for c in log_calls)
    assert any("resolved" in c for c in log_calls)


def test_resolve_forward_logs_when_debug_enabled(monkeypatch):
    """resolve_forward() logs the action when debug_intercept is True."""
    from unittest.mock import MagicMock, patch
    from prompt_interceptor import interceptor as interceptor_mod
    from prompt_interceptor.interceptor import Interceptor, InterceptedRequest

    cfg = MagicMock()
    cfg.debug_intercept = True
    monkeypatch.setattr(interceptor_mod, "get_config", lambda: cfg)

    ic = Interceptor()
    req = InterceptedRequest("dbg2", "POST", "/", {}, {})
    ic._pending["dbg2"] = req

    log_calls = []
    with patch.object(interceptor_mod._log, "info", side_effect=lambda *a, **kw: log_calls.append(a[0] % a[1:])):
        ic.resolve_forward("dbg2")

    assert any("forward" in c for c in log_calls)


def test_resolve_drop_logs_when_debug_enabled(monkeypatch):
    """resolve_drop() logs the action when debug_intercept is True."""
    from unittest.mock import MagicMock, patch
    from prompt_interceptor import interceptor as interceptor_mod
    from prompt_interceptor.interceptor import Interceptor, InterceptedRequest

    cfg = MagicMock()
    cfg.debug_intercept = True
    monkeypatch.setattr(interceptor_mod, "get_config", lambda: cfg)

    ic = Interceptor()
    req = InterceptedRequest("dbg3", "POST", "/", {}, {})
    ic._pending["dbg3"] = req

    log_calls = []
    with patch.object(interceptor_mod._log, "info", side_effect=lambda *a, **kw: log_calls.append(a[0] % a[1:])):
        ic.resolve_drop("dbg3")

    assert any("drop" in c for c in log_calls)
