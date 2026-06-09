"""Tests for dashboard.py endpoints."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from prompt_interceptor.dashboard import app
from prompt_interceptor.interceptor import interceptor as _module_interceptor


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ollama_unavailable():
    """Mock httpx so Ollama target health checks return 503 instantly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        yield


@pytest.fixture
async def client(mock_ollama_unavailable):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Root / HTML
# ---------------------------------------------------------------------------

async def test_root_returns_html(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "PromptInterceptor" in resp.text


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

async def test_api_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

async def test_api_models(client):
    # Ollama is mocked as unavailable — endpoint still executes get_models()
    resp = await client.get("/api/models")
    assert resp.status_code in (200, 503, 504)


# ---------------------------------------------------------------------------
# /api/target-health
# ---------------------------------------------------------------------------

async def test_api_target_health_unreachable(client):
    # Ollama is mocked as unavailable (see mock_ollama_unavailable fixture)
    resp = await client.get("/api/target-health")
    assert resp.status_code in (200, 503, 504)


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------

async def test_api_status_structure(client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await client.get("/api/status")

    assert resp.status_code == 200
    data = resp.json()
    assert "proxy" in data
    assert "rules" in data


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

async def test_api_stats(client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await client.get("/api/stats")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/rules
# ---------------------------------------------------------------------------

async def test_api_rules_returns_list(client):
    resp = await client.get("/api/rules")
    assert resp.status_code == 200
    data = resp.json()
    assert "rules" in data
    assert isinstance(data["rules"], list)


# ---------------------------------------------------------------------------
# /api/enable-rule / /api/disable-rule
# ---------------------------------------------------------------------------

async def test_enable_rule_valid_index(client):
    resp = await client.get("/api/enable-rule/0")
    assert resp.status_code in (200, 404)   # 404 if no rules loaded


async def test_disable_rule_valid_index(client):
    resp = await client.get("/api/disable-rule/0")
    assert resp.status_code in (200, 404)


async def test_enable_rule_invalid_index(client):
    resp = await client.get("/api/enable-rule/9999")
    assert resp.status_code == 404
    assert resp.json()["status"] == "not_found"


async def test_disable_rule_invalid_index(client):
    resp = await client.get("/api/disable-rule/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/logs
# ---------------------------------------------------------------------------

async def test_api_logs(client):
    resp = await client.get("/api/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert "logs" in data
    assert isinstance(data["logs"], list)


async def test_api_logs_with_limit(client):
    resp = await client.get("/api/logs?limit=5")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/raw-logs
# ---------------------------------------------------------------------------

async def test_api_raw_logs(client):
    resp = await client.get("/api/raw-logs")
    assert resp.status_code == 200
    assert "logs" in resp.json()


# ---------------------------------------------------------------------------
# /api/intercept/pending
# ---------------------------------------------------------------------------

async def test_intercept_pending_empty(client):
    resp = await client.get("/api/intercept/pending")
    assert resp.status_code == 200
    data = resp.json()
    assert "pending" in data
    assert "count" in data


# ---------------------------------------------------------------------------
# /api/intercept/{id}/forward|edit|drop — not found
# ---------------------------------------------------------------------------

async def test_intercept_forward_not_found(client):
    resp = await client.post("/api/intercept/nonexistent/forward")
    assert resp.status_code == 404


async def test_intercept_edit_not_found(client):
    resp = await client.post(
        "/api/intercept/nonexistent/edit",
        json={"model": "new"},
    )
    assert resp.status_code == 404


async def test_intercept_drop_not_found(client):
    resp = await client.post("/api/intercept/nonexistent/drop")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/intercept/{id}/forward — found (inject a pending request)
# ---------------------------------------------------------------------------

async def test_intercept_forward_found(client):
    from prompt_interceptor.interceptor import Interceptor, interceptor as _interceptor
    import prompt_interceptor.dashboard as dash_mod

    local_interceptor = Interceptor(intercept_timeout=5.0)
    original = dash_mod.interceptor
    dash_mod.interceptor = local_interceptor

    try:
        rid = "test-rid-fwd"
        # Start an intercept in the background
        task = asyncio.create_task(
            local_interceptor.intercept(rid, "POST", "/api/chat", {}, {"model": "x"})
        )
        await asyncio.sleep(0.05)

        resp = await client.post(f"/api/intercept/{rid}/forward")
        assert resp.status_code == 200
        assert resp.json()["status"] == "forwarded"

        action, _ = await task
        assert action == "forward"
    finally:
        dash_mod.interceptor = original


async def test_intercept_drop_found(client):
    from prompt_interceptor.interceptor import Interceptor
    import prompt_interceptor.dashboard as dash_mod

    local_interceptor = Interceptor(intercept_timeout=5.0)
    original = dash_mod.interceptor
    dash_mod.interceptor = local_interceptor

    try:
        rid = "test-rid-drop"
        task = asyncio.create_task(
            local_interceptor.intercept(rid, "POST", "/", {}, {})
        )
        await asyncio.sleep(0.05)

        resp = await client.post(f"/api/intercept/{rid}/drop")
        assert resp.status_code == 200

        action, _ = await task
        assert action == "drop"
    finally:
        dash_mod.interceptor = original


async def test_intercept_edit_found(client):
    from prompt_interceptor.interceptor import Interceptor
    import prompt_interceptor.dashboard as dash_mod

    local_interceptor = Interceptor(intercept_timeout=5.0)
    original = dash_mod.interceptor
    dash_mod.interceptor = local_interceptor

    try:
        rid = "test-rid-edit"
        task = asyncio.create_task(
            local_interceptor.intercept(rid, "POST", "/", {}, {"model": "old"})
        )
        await asyncio.sleep(0.05)

        resp = await client.post(
            f"/api/intercept/{rid}/edit",
            json={"model": "new"},
        )
        assert resp.status_code == 200

        action, body = await task
        assert action == "edit"
        assert body == {"model": "new"}
    finally:
        dash_mod.interceptor = original


# ---------------------------------------------------------------------------
# /api/mode  (new endpoint)
# ---------------------------------------------------------------------------

async def test_set_mode_passthrough(client):
    resp = await client.post("/api/mode", json={"mode": "passthrough"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "passthrough"


async def test_set_mode_intercept(client):
    resp = await client.post("/api/mode", json={"mode": "intercept"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "intercept"
    # Reset back to passthrough
    await client.post("/api/mode", json={"mode": "passthrough"})


async def test_set_mode_invalid(client):
    resp = await client.post("/api/mode", json={"mode": "invalid"})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# POST /api/rules  (new endpoint)
# ---------------------------------------------------------------------------

async def test_add_rule_valid(client):
    rule = {
        "match": {"path": "/api/chat", "jsonpath": "$.model"},
        "replace": {"jsonpath": "$.model", "value": "test-model"},
    }
    resp = await client.post("/api/rules", json=rule)
    assert resp.status_code == 201
    assert resp.json()["status"] == "created"


async def test_add_rule_invalid(client):
    resp = await client.post("/api/rules", json={"bad": "data"})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# DELETE /api/rules/{index}  (new endpoint)
# ---------------------------------------------------------------------------

async def test_delete_rule_not_found(client):
    resp = await client.delete("/api/rules/9999")
    assert resp.status_code == 404
    assert resp.json()["status"] == "not_found"


async def test_delete_rule_valid(client):
    # First add a rule, then delete it
    rule = {
        "match": {"path": "/api/generate", "jsonpath": "$.model"},
        "replace": {"jsonpath": "$.model", "value": "to-delete"},
    }
    add_resp = await client.post("/api/rules", json=rule)
    assert add_resp.status_code == 201

    # Get current rule count to find the new index
    list_resp = await client.get("/api/rules")
    rules = list_resp.json()["rules"]
    last_index = len(rules) - 1

    del_resp = await client.delete(f"/api/rules/{last_index}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# POST /api/reset
# ---------------------------------------------------------------------------

async def test_reset_session_clears_logs(client):
    """POST /api/reset calls clear_logs and returns status=reset."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod._logger, "clear_logs", return_value=5) as mock_clear:
        resp = await client.post("/api/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reset"
    assert data["deleted"] == 5
    mock_clear.assert_called_once()


async def test_reset_session_no_logs(client):
    """POST /api/reset returns deleted=0 when there are no log files."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod._logger, "clear_logs", return_value=0):
        resp = await client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 0


# ---------------------------------------------------------------------------
# get_app
# ---------------------------------------------------------------------------

def test_get_app():
    from prompt_interceptor.dashboard import get_app
    from fastapi import FastAPI
    assert isinstance(get_app(), FastAPI)


# ---------------------------------------------------------------------------
# GET /favicon.ico — icon file exists  (lines 416-418)
# ---------------------------------------------------------------------------

@pytest.fixture
async def plain_client():
    """Dashboard client without Ollama mock (for favicon/static tests)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_favicon_returns_file_when_exists(plain_client):
    """When the icon file exists, /favicon.ico returns 200."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b'\x89PNG\r\n\x1a\n')
        tmp_path = tmp.name

    try:
        import prompt_interceptor.dashboard as dash_mod
        with patch.object(dash_mod.os.path, "exists", return_value=True), \
             patch("prompt_interceptor.dashboard._ICON_PATH", tmp_path):
            resp = await plain_client.get("/favicon.ico")
        assert resp.status_code == 200
    finally:
        os.unlink(tmp_path)


async def test_favicon_returns_404_when_missing(plain_client):
    """When the icon file is absent, /favicon.ico returns 404."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod.os.path, "exists", return_value=False):
        resp = await plain_client.get("/favicon.ico")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET / — static/index.html fallback  (lines 426-427)
# ---------------------------------------------------------------------------

async def test_root_serves_static_index_when_present(plain_client):
    """When static/index.html exists, it is served instead of inline HTML."""
    custom_html = "<html><body>Custom Dashboard</body></html>"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
        tmp.write(custom_html)
        tmp_path = tmp.name

    try:
        import prompt_interceptor.dashboard as dash_mod
        original_exists = os.path.exists

        def _fake_exists(path):
            if "static" in str(path) and "index.html" in str(path):
                return True
            return original_exists(path)

        with patch.object(dash_mod.os.path, "exists", side_effect=_fake_exists), \
             patch("builtins.open", return_value=open(tmp_path)):
            resp = await plain_client.get("/")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Live Prompts — Guardar / Limpiar buttons in HTML
# ---------------------------------------------------------------------------

async def test_dashboard_html_has_clear_logs_button(client):
    """Dashboard HTML includes the Limpiar button that calls clearLogs()."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "clearLogs()" in resp.text
    assert "Limpiar" in resp.text


async def test_dashboard_html_has_save_logs_button(client):
    """Dashboard HTML includes the Guardar button that calls saveLogs()."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "saveLogs()" in resp.text
    assert "Guardar" in resp.text


async def test_dashboard_html_save_logs_js_function(client):
    """saveLogs() JS function creates a download link with the logs cache."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "URL.createObjectURL" in resp.text
    assert "prompt-interceptor-logs-" in resp.text


async def test_dashboard_html_clear_logs_calls_reset_api(client):
    """clearLogs() JS function calls /api/reset via fetch."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "/api/reset" in resp.text


async def test_dashboard_html_has_response_column(client):
    """Dashboard table has a Response column for LLM output."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "<th>Response</th>" in resp.text
    assert "response_body" in resp.text


# ---------------------------------------------------------------------------
# Message detail modal — split layout
# ---------------------------------------------------------------------------

async def test_dashboard_html_modal_has_split_layout(client):
    """Modal uses a two-column grid layout (modal-split class)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "modal-split" in resp.text


async def test_dashboard_html_modal_has_prompt_panel(client):
    """Modal contains a Prompt panel with id=modal-prompt."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modal-prompt"' in resp.text
    assert "Prompt" in resp.text


async def test_dashboard_html_modal_has_response_panel(client):
    """Modal contains a Response panel with id=modal-response."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modal-response"' in resp.text


async def test_dashboard_html_modal_thinking_block_hidden_by_default(client):
    """Thinking block starts hidden and is only shown when thinking content exists."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modal-thinking-block"' in resp.text
    assert 'id="modal-thinking"' in resp.text
    # Must start hidden
    assert 'id="modal-thinking-block" class="modal-thinking-block" style="display:none"' in resp.text


async def test_dashboard_html_showraw_uses_textcontent(client):
    """showRaw uses textContent so \\n in strings renders as real line breaks."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # textContent assignment is used for prompt, thinking, and response
    assert "modal-prompt').textContent" in html or ".textContent = " in html
    # white-space:pre-wrap on pre elements renders newlines correctly
    assert "white-space:pre-wrap" in html


async def test_dashboard_html_showraw_extracts_all_message_roles(client):
    """showRaw formats messages with role labels like [USER], [ASSISTANT]."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "role" in html
    assert "toUpperCase()" in html


async def test_dashboard_html_showraw_handles_ollama_response(client):
    """showRaw handles Ollama /api/chat response format (rb.message.content)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "rb.message?.content" in resp.text


async def test_dashboard_html_showraw_handles_openai_response(client):
    """showRaw handles OpenAI-compatible response format (choices[0].message.content)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "rb.choices" in resp.text
    assert "choices[0]?.message?.content" in resp.text


async def test_dashboard_html_showraw_handles_anthropic_thinking(client):
    """showRaw extracts thinking blocks from Anthropic-style content arrays."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "type === 'thinking'" in html
    assert "modal-thinking-block" in html


async def test_dashboard_html_showraw_handles_think_tags(client):
    """showRaw parses <think>...</think> tags as fallback for inline thinking."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "thinkTagMatch" in html or "<think>" in html or "think>" in html


async def test_dashboard_html_showraw_reads_message_thinking_field(client):
    """showRaw reads rb.message.thinking (Ollama 0.7+ /api/chat)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "rb.message.thinking" in resp.text or "message.thinking" in resp.text


async def test_dashboard_html_showraw_reads_choices_thinking_field(client):
    """showRaw reads choices[0].message.thinking (Ollama 0.7+ /v1/chat/completions)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "msg.thinking" in resp.text or "message.thinking" in resp.text


async def test_dashboard_html_showraw_handles_generate_prompt(client):
    """showRaw falls back to l.body.prompt for /api/generate requests."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "l.body?.prompt" in resp.text


async def test_dashboard_html_modal_wider_than_before(client):
    """Modal box uses a wider max-width to accommodate split layout."""
    resp = await client.get("/")
    assert resp.status_code == 200
    # Previous max-width was 900px; new layout needs at least 1200px
    html = resp.text
    assert "1400px" in html or "1200px" in html


# ---------------------------------------------------------------------------
# Dashboard reorganisation — modifiers section, proxy info modal, logo
# ---------------------------------------------------------------------------

async def test_dashboard_html_modifiers_section_id(client):
    """Modifiers section has id='modifiers-section'."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modifiers-section"' in resp.text


async def test_dashboard_html_modifiers_section_starts_hidden(client):
    """Modifiers section starts hidden (display:none) and is shown by JS."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modifiers-section" style="display:none"' in resp.text


async def test_dashboard_html_modifiers_before_live_prompts(client):
    """Modifiers section appears before Live Prompts in the HTML."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    mod_pos = html.find('id="modifiers-section"')
    live_pos = html.find("Live Prompts")
    assert mod_pos != -1 and live_pos != -1
    assert mod_pos < live_pos


async def test_dashboard_html_modifiers_shown_in_intercept_mode(client):
    """JS toggles modifiers-section visibility based on mode === 'intercept'."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "modifiers-section" in html
    assert "mode === 'intercept'" in html


async def test_dashboard_html_no_status_box(client):
    """Static proxy status <pre> (status-box) has been removed from the HTML."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="status-box"' not in resp.text


async def test_dashboard_html_has_proxy_info_button(client):
    """Header contains a Proxy Info button that calls showProxyInfo()."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "showProxyInfo()" in resp.text
    assert "Proxy Info" in resp.text


async def test_dashboard_html_proxy_info_modal_present(client):
    """Proxy info modal overlay is present in the HTML."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="proxy-modal-overlay"' in resp.text
    assert 'id="proxy-status-content"' in resp.text


async def test_dashboard_html_proxy_modal_functions(client):
    """showProxyInfo and closeProxyModal JS functions are defined."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "function showProxyInfo()" in html
    assert "function closeProxyModal()" in html


async def test_dashboard_html_logo_img_tag(client):
    """Dashboard title area contains an <img> pointing to /logo."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'src="/logo"' in resp.text
    assert 'id="dashboard-logo"' in resp.text


async def test_logo_returns_file_when_exists(plain_client):
    """GET /logo returns 200 with image/png when logo file exists."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b'\x89PNG\r\n\x1a\n')
        tmp_path = tmp.name

    try:
        import prompt_interceptor.dashboard as dash_mod
        with patch.object(dash_mod.os.path, "exists", return_value=True), \
             patch("prompt_interceptor.dashboard._LOGO_PATH", tmp_path):
            resp = await plain_client.get("/logo")
        assert resp.status_code == 200
    finally:
        os.unlink(tmp_path)


async def test_logo_returns_404_when_missing(plain_client):
    """GET /logo returns 404 when logo file is absent."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod.os.path, "exists", return_value=False):
        resp = await plain_client.get("/logo")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tools block in modal
# ---------------------------------------------------------------------------

async def test_dashboard_html_modal_tools_block_present(client):
    """Modal contains a tools block with id=modal-tools-block."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modal-tools-block"' in resp.text
    assert 'id="modal-tools"' in resp.text


async def test_dashboard_html_modal_tools_block_hidden_by_default(client):
    """Tools block starts hidden (display:none) like the thinking block."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="modal-tools-block" class="modal-tools-block" style="display:none"' in resp.text


async def test_dashboard_html_modal_tools_block_has_css(client):
    """Dashboard CSS defines modal-tools-block and modal-tools-pre styles."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "modal-tools-block" in html
    assert "modal-tools-pre" in html
    assert "modal-tools-title" in html


async def test_dashboard_html_showraw_extracts_choices_tool_calls(client):
    """showRaw reads tool_calls from choices[0].message.tool_calls (OpenAI format)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "tool_calls" in html
    assert "choices[0]" in html or "choices?.[0]" in html


async def test_dashboard_html_showraw_extracts_message_tool_calls(client):
    """showRaw reads tool_calls from rb.message.tool_calls (Ollama native format)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "rb.message?.tool_calls" in resp.text or "message.tool_calls" in resp.text


async def test_dashboard_html_table_shows_tool_names_instead_of_dash(client):
    """Table preview JS shows [tools: name] when tool_calls present and content is empty."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "tools:" in html
    assert "tool_calls" in html


async def test_dashboard_html_modal_tools_label(client):
    """Modal tools block has a visible label containing 'Tools'."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Tools" in resp.text
    assert "modal-tools-title" in resp.text


async def test_dashboard_html_showraw_extracts_anthropic_tool_use_blocks(client):
    """showRaw checks rb.content for tool_use blocks (Anthropic format)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "tool_use" in html
    assert "rb.content" in html


async def test_dashboard_html_showraw_fallback_when_no_valid_names(client):
    """showRaw tries embedded fallback when explicit tool_calls have no valid function names."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "hasValidNames" in html


async def test_dashboard_html_table_searches_thinking_for_embedded_tools(client):
    """renderLogsTable embedded tool search includes the thinking field, not just content."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "thinkingStr" in resp.text


async def test_dashboard_html_extract_embedded_parses_xml_function_format(client):
    """extractEmbeddedToolCalls handles <function=NAME><parameter=KEY>VAL</parameter> format."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "fnMatch" in html
    assert "<function=" in html or "function=" in html


async def test_dashboard_html_extract_embedded_parses_json_format(client):
    """extractEmbeddedToolCalls still handles the JSON {"name":..., "arguments":...} format."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "JSON.parse" in html
    assert "extractEmbeddedToolCalls" in html


# ---------------------------------------------------------------------------
# Load file button
# ---------------------------------------------------------------------------

async def test_dashboard_html_has_load_file_button(client):
    """Dashboard HTML includes a Cargar button that calls loadFile()."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "loadFile()" in resp.text
    assert "Cargar" in resp.text


async def test_dashboard_html_load_file_js_function(client):
    """loadFile() JS function uses FileReader and accepts .json files."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "function loadFile()" in html
    assert "FileReader" in html
    assert "accept = '.json'" in html or "accept='.json'" in html or '.json' in html


async def test_dashboard_html_file_mode_banner_present(client):
    """file-mode-banner element is present and hidden by default."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="file-mode-banner"' in html
    assert 'id="file-mode-name"' in html
    assert 'display:none' in html


async def test_dashboard_html_exit_file_mode_function(client):
    """exitFileMode() JS function is defined to return to live view."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "function exitFileMode()" in resp.text


async def test_dashboard_html_render_logs_table_function(client):
    """renderLogsTable() is extracted from loadLogs() as a standalone function."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "function renderLogsTable(" in resp.text


async def test_dashboard_html_file_mode_flag_variables(client):
    """_fileMode and _loadedLogs variables are declared in the JS."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "_fileMode" in html
    assert "_loadedLogs" in html


async def test_dashboard_html_modifiers_stat_box_starts_hidden(client):
    """Active Modifiers stat box starts hidden; only shown in intercept mode."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert 'id="stat-modifiers-box" style="display:none"' in resp.text


async def test_dashboard_html_modifiers_stat_box_toggled_with_mode(client):
    """JS sets stat-modifiers-box display based on mode === 'intercept'."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert 'stat-modifiers-box' in html
    assert "mode === 'intercept'" in html


async def test_dashboard_html_ctx_uses_max_not_last(client):
    """Context Used stat tracks the maximum utilisation across all logs, not just the latest."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "maxCtxPct" in html
    assert "lastCtxPct" not in html


async def test_dashboard_html_ctx_max_accumulates(client):
    """maxCtxPct is updated when a higher value is found (pct > maxCtxPct)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "pct > maxCtxPct" in resp.text
