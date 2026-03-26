"""
Dashboard module for PyProxy.

Serves a web UI for monitoring traffic, managing rules, and
handling intercepted requests.
"""

import json
import os

from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from .config import get_config
from .logger import TrafficLogger
from .rules_engine import RuleEngine
from .health import check_proxy_health, check_target_health, get_status

# ---------------------------------------------------------------------------
# App & shared singletons
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PyProxy Dashboard",
    description="Dashboard for PyProxy - Ollama Traffic Interceptor & Model Switcher",
    version="0.1.0",
)

_config = get_config()
_logger = TrafficLogger()
_rule_engine = RuleEngine(_logger)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Dashboard HTML (inline fallback when no static/index.html exists)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyProxy Dashboard</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#1a1a2e;color:#eee;padding:20px}
  .container{max-width:1200px;margin:0 auto}
  h1{color:#fff;margin-bottom:20px}
  h2{color:#ccc;margin-bottom:12px}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.8em}
  .green{background:#1a6b1a;color:#0f0}
  .red{background:#6b1a1a;color:#f66}
  .section{background:#16213e;border-radius:8px;padding:20px;margin-bottom:20px}
  table{width:100%;border-collapse:collapse}
  th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #2a2a4e}
  th{color:#888;font-weight:normal}
  button{padding:5px 12px;border:none;border-radius:4px;cursor:pointer;font-size:.85em}
  .btn-enable{background:#1a6b1a;color:#0f0}
  .btn-disable{background:#6b1a1a;color:#f66}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
  .stat{background:#16213e;border-radius:8px;padding:16px;text-align:center}
  .stat-value{font-size:2em;color:#4fc3f7}
  .stat-label{color:#888;font-size:.85em;margin-top:4px}
  .mode-badge{font-size:.9em}
  footer{text-align:center;margin-top:30px;color:#555;font-size:.85em}
  pre{background:#0d0d1e;padding:12px;border-radius:6px;font-size:.8em;
      overflow:auto;max-height:300px}
</style>
</head>
<body>
<div class="container">
  <h1>PyProxy Dashboard</h1>

  <div class="stats">
    <div class="stat">
      <div class="stat-value" id="stat-requests">-</div>
      <div class="stat-label">Total Requests</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-rules">-</div>
      <div class="stat-label">Active Rules</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-mode">-</div>
      <div class="stat-label">Mode</div>
    </div>
  </div>

  <div class="section">
    <h2>Proxy Status</h2>
    <pre id="status-box">Loading...</pre>
  </div>

  <div class="section">
    <h2>Rules</h2>
    <table>
      <thead><tr><th>#</th><th>Match</th><th>Replace</th><th>Status</th><th>Action</th></tr></thead>
      <tbody id="rules-body"><tr><td colspan="5">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="section">
    <h2>Recent Logs</h2>
    <pre id="logs-box">Loading...</pre>
  </div>

  <footer>PyProxy Dashboard v0.1.0 &mdash; <a href="/api/docs" style="color:#4fc3f7">API Docs</a></footer>
</div>

<script>
const BASE = '';

async function fetchJSON(url) {
  const r = await fetch(BASE + url);
  return r.json();
}

async function loadStatus() {
  try {
    const data = await fetchJSON('/api/status');
    document.getElementById('status-box').textContent = JSON.stringify(data, null, 2);
    document.getElementById('stat-mode').textContent = data.proxy?.mode ?? '-';
    document.getElementById('stat-rules').textContent = data.rules?.enabled_count ?? '-';
    document.getElementById('stat-requests').textContent = data.logs ? '?' : '-';
  } catch(e) {
    document.getElementById('status-box').textContent = 'Error: ' + e.message;
  }
}

async function loadRules() {
  try {
    const data = await fetchJSON('/api/rules');
    const tbody = document.getElementById('rules-body');
    if (!data.rules || data.rules.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5">No rules configured</td></tr>';
      return;
    }
    tbody.innerHTML = data.rules.map((r, i) => `
      <tr>
        <td>${i}</td>
        <td><code>${JSON.stringify(r.match)}</code></td>
        <td><code>${JSON.stringify(r.replace)}</code></td>
        <td><span class="badge ${r.enabled ? 'green' : 'red'}">${r.enabled ? 'enabled' : 'disabled'}</span></td>
        <td>
          <button class="btn-enable" onclick="toggleRule(${i}, true)">Enable</button>
          <button class="btn-disable" onclick="toggleRule(${i}, false)">Disable</button>
        </td>
      </tr>`).join('');
  } catch(e) {
    document.getElementById('rules-body').innerHTML = '<tr><td colspan="5">Error: ' + e.message + '</td></tr>';
  }
}

async function toggleRule(index, enable) {
  const action = enable ? 'enable-rule' : 'disable-rule';
  await fetch(`${BASE}/api/${action}/${index}`);
  loadRules();
  loadStatus();
}

async function loadLogs() {
  try {
    const data = await fetchJSON('/api/logs?limit=20');
    document.getElementById('logs-box').textContent =
      data.logs.length ? JSON.stringify(data.logs, null, 2) : 'No logs yet.';
  } catch(e) {
    document.getElementById('logs-box').textContent = 'Error: ' + e.message;
  }
}

function refresh() {
  loadStatus();
  loadRules();
  loadLogs();
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve dashboard HTML."""
    static_index = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_index):
        with open(static_index) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=_DASHBOARD_HTML)


@router.get("/status")
async def status():
    return await get_status()


@router.get("/health")
async def health():
    return await check_proxy_health()


@router.get("/target-health")
async def target_health():
    return await check_target_health()


@router.get("/stats")
async def stats():
    return await get_status()


@router.get("/logs")
async def logs(limit: int = 20):
    return {"logs": _logger.get_logs(limit=limit)}


@router.get("/raw-logs")
async def raw_logs():
    return {"logs": _logger.get_all_logs()}


@router.get("/rules")
async def get_rules():
    return {"rules": _rule_engine.get_rules()}


@router.get("/enable-rule/{index}")
async def enable_rule(index: int):
    if _rule_engine.enable_rule(index):
        return JSONResponse(status_code=200, content={"status": "enabled"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


@router.get("/disable-rule/{index}")
async def disable_rule(index: int):
    if _rule_engine.disable_rule(index):
        return JSONResponse(status_code=200, content={"status": "disabled"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


# Include the API router
app.include_router(router)

# Mount static files directory (created lazily)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


def get_app() -> FastAPI:
    """Return the dashboard app instance (for uvicorn factory)."""
    return app


__all__ = ["app", "get_app"]
