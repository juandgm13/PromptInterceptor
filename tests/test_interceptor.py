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


# ---------------------------------------------------------------------------
# InterceptedRequest.pause / resume (lines 92, 96)
# ---------------------------------------------------------------------------

def test_intercepted_request_pause_sets_flag():
    """pause() sets _paused to True."""
    req = InterceptedRequest("p1", "POST", "/", {}, {})
    assert req._paused is False
    req.pause()
    assert req._paused is True


def test_intercepted_request_resume_clears_flag():
    """resume() sets _paused back to False."""
    req = InterceptedRequest("p2", "POST", "/", {}, {})
    req.pause()
    assert req._paused is True
    req.resume()
    assert req._paused is False


# ---------------------------------------------------------------------------
# InterceptedRequest.wait() debug timeout path (lines 120, 123)
# ---------------------------------------------------------------------------

async def test_wait_debug_timeout_logs_info(monkeypatch):
    """When debug_intercept=True and the request times out, the timeout log is emitted."""
    from unittest.mock import MagicMock, patch
    from prompt_interceptor import interceptor as interceptor_mod

    cfg = MagicMock()
    cfg.debug_intercept = True
    monkeypatch.setattr(interceptor_mod, "get_config", lambda: cfg)

    req = InterceptedRequest("dbg_timeout", "POST", "/api/chat", {}, {"k": "v"})

    log_calls = []
    with patch.object(interceptor_mod._log, "info", side_effect=lambda *a, **kw: log_calls.append(a[0] % a[1:])):
        action, body = await req.wait(timeout=0.05)

    assert action == "forward"
    assert any("timeout" in c for c in log_calls)


async def test_wait_paused_rolls_deadline_then_resumes():
    """While _paused=True, the deadline keeps rolling forward (line 120); after resume it times out."""
    req = InterceptedRequest("pause_roll", "POST", "/", {}, {"k": "v"})
    req._paused = True  # start paused

    async def _unpause_after():
        await asyncio.sleep(0.15)
        req._paused = False

    asyncio.create_task(_unpause_after())
    # timeout=0.05 so it would fire immediately when not paused; pausing for 0.15s first
    action, body = await req.wait(timeout=0.05)
    # After resume, it will timeout quickly and auto-forward
    assert action == "forward"


# ---------------------------------------------------------------------------
# Interceptor.resolve_edit debug logging (line 221)
# ---------------------------------------------------------------------------

def test_resolve_edit_logs_when_debug_enabled(monkeypatch):
    """resolve_edit() logs the action when debug_intercept is True."""
    from unittest.mock import MagicMock, patch
    from prompt_interceptor import interceptor as interceptor_mod
    from prompt_interceptor.interceptor import Interceptor, InterceptedRequest

    cfg = MagicMock()
    cfg.debug_intercept = True
    monkeypatch.setattr(interceptor_mod, "get_config", lambda: cfg)

    ic = Interceptor()
    req = InterceptedRequest("dbg_edit", "POST", "/", {}, {})
    ic._pending["dbg_edit"] = req

    log_calls = []
    with patch.object(interceptor_mod._log, "info", side_effect=lambda *a, **kw: log_calls.append(a[0] % a[1:])):
        ic.resolve_edit("dbg_edit", {"model": "new"})

    assert any("edit" in c for c in log_calls)


# ---------------------------------------------------------------------------
# Interceptor.pause_request / resume_request (lines 238-242, 246-250)
# ---------------------------------------------------------------------------

def test_pause_request_returns_true_for_existing():
    """pause_request() returns True when the request exists."""
    ic = Interceptor()
    req = InterceptedRequest("pr1", "POST", "/", {}, {})
    ic._pending["pr1"] = req
    result = ic.pause_request("pr1")
    assert result is True
    assert req._paused is True


def test_pause_request_returns_false_for_missing():
    """pause_request() returns False when request_id is not found."""
    ic = Interceptor()
    assert ic.pause_request("nonexistent") is False


def test_resume_request_returns_true_for_existing():
    """resume_request() returns True and clears _paused."""
    ic = Interceptor()
    req = InterceptedRequest("rr1", "POST", "/", {}, {})
    req._paused = True
    ic._pending["rr1"] = req
    result = ic.resume_request("rr1")
    assert result is True
    assert req._paused is False


def test_resume_request_returns_false_for_missing():
    """resume_request() returns False when request_id is not found."""
    ic = Interceptor()
    assert ic.resume_request("nonexistent") is False
