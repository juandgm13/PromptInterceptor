"""
Dashboard module for PromptInterceptor.

Serves a web UI for monitoring traffic, managing modifiers, and
handling intercepted requests.
"""

import json
import os

from fastapi import FastAPI, APIRouter, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response

from .config import get_config, save_config
from .interceptor import interceptor
from .logger import TrafficLogger
from .rules_engine import RuleEngine
from .health import check_proxy_health, check_target_health, get_status

# ---------------------------------------------------------------------------
# App & shared singletons
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PromptInterceptor Dashboard",
    description="Dashboard for PromptInterceptor - Ollama Traffic Interceptor for AI Clients",
    version="0.1.0",
)

_config = get_config()
_logger = TrafficLogger()
_rule_engine = RuleEngine(_logger)

router = APIRouter(prefix="/api")

_ICON_PATH = os.path.join(os.path.dirname(__file__), "..", "res", "PromptInterceptor_Icon.png")

# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PromptInterceptor Dashboard</title>
<link rel="icon" type="image/png" href="/favicon.ico">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#1a1a2e;color:#eee;padding:20px}
  .container{max-width:1280px;margin:0 auto}
  h1{color:#fff;margin-bottom:4px;font-size:1.6em}
  h2{color:#ccc;margin-bottom:12px;font-size:1.1em}
  .subtitle{color:#888;font-size:.85em;margin-bottom:20px}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.8em}
  .green{background:#1a6b1a;color:#0f0}
  .red{background:#6b1a1a;color:#f66}
  .blue{background:#0d3b66;color:#4fc3f7}
  .section{background:#16213e;border-radius:8px;padding:20px;margin-bottom:20px}
  table{width:100%;border-collapse:collapse;font-size:.9em}
  th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #2a2a4e}
  th{color:#888;font-weight:normal;font-size:.85em}
  td code{font-size:.8em;background:#0d0d1e;padding:2px 6px;border-radius:3px;word-break:break-all}
  button{padding:5px 10px;border:none;border-radius:4px;cursor:pointer;font-size:.82em;margin:1px}
  .btn-enable{background:#1a4d1a;color:#5f5;border:1px solid #2a6b2a}
  .btn-disable{background:#4d1a1a;color:#f88;border:1px solid #6b2a2a}
  .btn-delete{background:#3a1a1a;color:#f55;border:1px solid #5a2a2a}
  .btn-forward{background:#0d3b66;color:#4fc3f7;border:1px solid #1a5b8a}
  .btn-drop{background:#4d1a1a;color:#f88;border:1px solid #6b2a2a}
  .btn-mode{padding:6px 14px;border:1px solid #333;border-radius:4px;cursor:pointer;
            font-size:.85em;background:#16213e;color:#888;margin-right:6px}
  .btn-mode.active{background:#0d3b66;color:#4fc3f7;border-color:#4fc3f7}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
  .stat{background:#16213e;border-radius:8px;padding:16px;text-align:center}
  .stat-value{font-size:2em;color:#4fc3f7}
  .stat-label{color:#888;font-size:.82em;margin-top:4px}
  footer{text-align:center;margin-top:30px;color:#555;font-size:.82em}
  pre{background:#0d0d1e;padding:12px;border-radius:6px;font-size:.78em;
      overflow:auto;max-height:280px;line-height:1.5}
  textarea{width:100%;background:#0d0d1e;color:#eee;border:1px solid #2a2a4e;
           border-radius:6px;padding:10px;font-size:.8em;font-family:monospace;resize:vertical}
  input[type=text]{background:#0d0d1e;color:#eee;border:1px solid #2a2a4e;
                   border-radius:4px;padding:6px 10px;font-size:.85em;width:100%}
  .form-grid{display:grid;grid-template-columns:repeat(5,1fr) auto;gap:8px;align-items:end;
             margin-top:14px}
  .form-grid label{color:#888;font-size:.8em;display:block;margin-bottom:4px}
  .btn-add{padding:7px 14px;background:#0d3b66;color:#4fc3f7;border:1px solid #4fc3f7;
           border-radius:4px;cursor:pointer;font-size:.85em;white-space:nowrap}
  .pending-card{background:#0d0d1e;border:1px solid #2a2a4e;border-radius:6px;
                padding:16px;margin-bottom:12px}
  .pending-card h3{color:#4fc3f7;font-size:.95em;margin-bottom:8px}
  .pending-meta{color:#888;font-size:.8em;margin-bottom:8px}
  .pending-actions{display:flex;gap:8px;margin-top:10px}
  #pending-section{display:none}
  .header-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
  .mode-controls{display:flex;align-items:center;gap:8px}
  .mode-label{color:#888;font-size:.85em;margin-right:4px}
  a{color:#4fc3f7;text-decoration:none}
  a:hover{text-decoration:underline}
  .empty{color:#555;font-style:italic}
</style>
</head>
<body>
<div class="container">

  <div class="header-row">
    <div>
      <h1>PromptInterceptor Dashboard</h1>
      <div class="subtitle">Ollama Traffic Interceptor for AI Clients</div>
    </div>
    <div class="mode-controls">
      <span class="mode-label">Mode:</span>
      <button class="btn-mode" id="btn-passthrough" onclick="setMode('passthrough')">Passthrough</button>
      <button class="btn-mode" id="btn-intercept" onclick="setMode('intercept')">Intercept</button>
    </div>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-value" id="stat-requests">-</div>
      <div class="stat-label">Total Requests</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-modifiers">-</div>
      <div class="stat-label">Active Modifiers</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-mode">-</div>
      <div class="stat-label">Current Mode</div>
    </div>
  </div>

  <!-- Pending Intercepts -->
  <div class="section" id="pending-section">
    <h2>&#9888; Pending Intercepts</h2>
    <div id="pending-list"></div>
  </div>

  <!-- Live Prompts -->
  <div class="section">
    <h2>Live Prompts</h2>
    <table>
      <thead><tr>
        <th>Time</th><th>Method</th><th>Path</th><th>Model</th><th>Preview</th><th>Status</th>
      </tr></thead>
      <tbody id="logs-body"><tr><td colspan="6" class="empty">Loading...</td></tr></tbody>
    </table>
  </div>

  <!-- Modifiers -->
  <div class="section">
    <h2>Modifiers</h2>
    <table>
      <thead><tr>
        <th>#</th><th>Match Path</th><th>Match JSONPath</th><th>Match Value</th>
        <th>Replace JSONPath</th><th>Replace Value</th><th>Status</th><th>Actions</th>
      </tr></thead>
      <tbody id="rules-body"><tr><td colspan="8" class="empty">Loading...</td></tr></tbody>
    </table>

    <div class="form-grid">
      <div>
        <label>Match Path</label>
        <input type="text" id="f-match-path" placeholder="/api/chat">
      </div>
      <div>
        <label>Match JSONPath</label>
        <input type="text" id="f-match-jp" placeholder="$.model">
      </div>
      <div>
        <label>Match Value (comma-sep.)</label>
        <input type="text" id="f-match-val" placeholder="llama3,mistral">
      </div>
      <div>
        <label>Replace JSONPath</label>
        <input type="text" id="f-replace-jp" placeholder="$.model">
      </div>
      <div>
        <label>Replace Value</label>
        <input type="text" id="f-replace-val" placeholder="deepseek-coder">
      </div>
      <div>
        <label>&nbsp;</label>
        <button class="btn-add" onclick="addModifier()">+ Add Modifier</button>
      </div>
    </div>
  </div>

  <!-- Proxy Status -->
  <div class="section">
    <h2>Proxy Status</h2>
    <pre id="status-box">Loading...</pre>
  </div>

  <footer>
    PromptInterceptor Dashboard v0.1.0 &mdash;
    <a href="/docs">API Docs</a>
  </footer>
</div>

<script>
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

async function loadStatus() {
  try {
    const data = await fetchJSON('/api/status');
    document.getElementById('status-box').textContent = JSON.stringify(data, null, 2);
    const mode = data.proxy?.mode ?? '-';
    document.getElementById('stat-mode').textContent = mode;
    document.getElementById('stat-modifiers').textContent = data.rules?.enabled_count ?? '-';
    document.getElementById('btn-passthrough').classList.toggle('active', mode === 'passthrough');
    document.getElementById('btn-intercept').classList.toggle('active', mode === 'intercept');
  } catch(e) {
    document.getElementById('status-box').textContent = 'Error: ' + e.message;
  }
}

async function loadLogs() {
  try {
    const data = await fetchJSON('/api/logs?limit=50');
    const tbody = document.getElementById('logs-body');
    const logs = data.logs || [];
    document.getElementById('stat-requests').textContent = logs.length;
    if (!logs.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">No requests yet.</td></tr>';
      return;
    }
    tbody.innerHTML = logs.slice().reverse().map(l => {
      const ts = l.timestamp ? l.timestamp.slice(11,19) : '-';
      const method = l.method || '-';
      const path = l.path || '-';
      const model = l.body?.model || '-';
      const msgs = l.body?.messages;
      let preview = '-';
      if (msgs && msgs.length) {
        const last = msgs[msgs.length - 1];
        const txt = (last.content || '').toString().slice(0, 80);
        preview = txt.length < (last.content||'').length ? txt + '…' : txt;
      } else if (l.body?.prompt) {
        preview = l.body.prompt.toString().slice(0, 80);
      }
      const status = l.type === 'response' ?
        '<span class="badge green">resp</span>' :
        '<span class="badge blue">req</span>';
      return `<tr>
        <td>${ts}</td>
        <td>${method}</td>
        <td><code>${path}</code></td>
        <td><code>${model}</code></td>
        <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${preview}</td>
        <td>${status}</td>
      </tr>`;
    }).join('');
  } catch(e) {
    document.getElementById('logs-body').innerHTML =
      '<tr><td colspan="6">Error: ' + e.message + '</td></tr>';
  }
}

async function loadRules() {
  try {
    const data = await fetchJSON('/api/rules');
    const tbody = document.getElementById('rules-body');
    const rules = data.rules || [];
    if (!rules.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">No modifiers configured.</td></tr>';
      return;
    }
    tbody.innerHTML = rules.map((r, i) => {
      const mv = r.match?.value ? [].concat(r.match.value).join(', ') : '-';
      return `<tr>
        <td>${i}</td>
        <td><code>${r.match?.path || '-'}</code></td>
        <td><code>${r.match?.jsonpath || '-'}</code></td>
        <td><code>${mv}</code></td>
        <td><code>${r.replace?.jsonpath || '-'}</code></td>
        <td><code>${r.replace?.value ?? '-'}</code></td>
        <td><span class="badge ${r.enabled ? 'green' : 'red'}">${r.enabled ? 'on' : 'off'}</span></td>
        <td>
          <button class="btn-enable" onclick="toggleRule(${i},true)">Enable</button>
          <button class="btn-disable" onclick="toggleRule(${i},false)">Disable</button>
          <button class="btn-delete" onclick="deleteRule(${i})">Delete</button>
        </td>
      </tr>`;
    }).join('');
  } catch(e) {
    document.getElementById('rules-body').innerHTML =
      '<tr><td colspan="8">Error: ' + e.message + '</td></tr>';
  }
}

async function loadPending() {
  try {
    const data = await fetchJSON('/api/intercept/pending');
    const section = document.getElementById('pending-section');
    const list = document.getElementById('pending-list');
    if (!data.count) {
      section.style.display = 'none';
      return;
    }
    section.style.display = '';
    list.innerHTML = (data.pending || []).map(p => `
      <div class="pending-card">
        <h3>${p.method} ${p.path}</h3>
        <div class="pending-meta">ID: ${p.request_id}</div>
        <textarea id="ta-${p.request_id}" rows="6">${JSON.stringify(p.body, null, 2)}</textarea>
        <div class="pending-actions">
          <button class="btn-forward" onclick="forwardRequest('${p.request_id}')">Forward</button>
          <button class="btn-forward" onclick="editRequest('${p.request_id}')">Edit &amp; Forward</button>
          <button class="btn-drop" onclick="dropRequest('${p.request_id}')">Drop</button>
        </div>
      </div>`).join('');
  } catch(e) { /* silent */ }
}

async function setMode(mode) {
  await fetch('/api/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode})
  });
  loadStatus();
}

async function toggleRule(index, enable) {
  const action = enable ? 'enable-rule' : 'disable-rule';
  await fetch(`/api/${action}/${index}`);
  loadRules();
  loadStatus();
}

async function deleteRule(index) {
  await fetch(`/api/rules/${index}`, {method: 'DELETE'});
  loadRules();
  loadStatus();
}

async function addModifier() {
  const matchPath = document.getElementById('f-match-path').value.trim();
  const matchJp   = document.getElementById('f-match-jp').value.trim();
  const matchVal  = document.getElementById('f-match-val').value.trim();
  const replJp    = document.getElementById('f-replace-jp').value.trim();
  const replVal   = document.getElementById('f-replace-val').value.trim();

  if (!matchJp || !replJp || !replVal) {
    alert('Match JSONPath, Replace JSONPath and Replace Value are required.');
    return;
  }

  const match = {};
  if (matchPath) match.path = matchPath;
  if (matchJp)   match.jsonpath = matchJp;
  if (matchVal)  match.value = matchVal.split(',').map(s => s.trim()).filter(Boolean);

  let replaceValue = replVal;
  try { replaceValue = JSON.parse(replVal); } catch(_) {}

  await fetch('/api/rules', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({match, replace: {jsonpath: replJp, value: replaceValue}})
  });

  ['f-match-path','f-match-jp','f-match-val','f-replace-jp','f-replace-val']
    .forEach(id => document.getElementById(id).value = '');

  loadRules();
  loadStatus();
}

async function forwardRequest(id) {
  await fetch(`/api/intercept/${id}/forward`, {method: 'POST'});
  loadPending();
}

async function editRequest(id) {
  const ta = document.getElementById('ta-' + id);
  let body;
  try { body = JSON.parse(ta.value); } catch(e) { alert('Invalid JSON: ' + e.message); return; }
  await fetch(`/api/intercept/${id}/edit`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  loadPending();
}

async function dropRequest(id) {
  await fetch(`/api/intercept/${id}/drop`, {method: 'POST'});
  loadPending();
}

function refreshAll() { loadStatus(); loadRules(); loadLogs(); }
function refreshPending() { loadPending(); }

refreshAll();
setInterval(refreshAll, 5000);
setInterval(refreshPending, 3000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if os.path.exists(_ICON_PATH):
        return FileResponse(_ICON_PATH, media_type="image/png")
    return Response(status_code=404)


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


@router.post("/mode")
async def set_mode(body: dict = Body(...)):
    """Change proxy mode (passthrough or intercept)."""
    new_mode = body.get("mode")
    if new_mode not in ("passthrough", "intercept"):
        return JSONResponse(status_code=400, content={"error": "invalid mode, use 'passthrough' or 'intercept'"})
    config = get_config()
    config.mode = new_mode
    save_config(config)
    return JSONResponse(status_code=200, content={"mode": new_mode})


@router.post("/rules")
async def add_rule(body: dict = Body(...)):
    """Add a new modifier rule."""
    if _rule_engine.add_rule(body):
        return JSONResponse(status_code=201, content={"status": "created"})
    return JSONResponse(status_code=400, content={"error": "invalid rule, must contain 'match' and 'replace' keys"})


@router.delete("/rules/{index}")
async def delete_rule(index: int):
    """Delete a modifier rule by index."""
    if _rule_engine.delete_rule(index):
        return JSONResponse(status_code=200, content={"status": "deleted"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


# ---------------------------------------------------------------------------
# Interceptor endpoints
# ---------------------------------------------------------------------------

@router.get("/intercept/pending")
async def intercept_pending():
    """List all requests currently waiting for a decision."""
    return {"pending": interceptor.get_pending(), "count": len(interceptor)}


@router.post("/intercept/{request_id}/forward")
async def intercept_forward(request_id: str):
    """Forward a paused request unchanged."""
    if interceptor.resolve_forward(request_id):
        return JSONResponse(status_code=200, content={"status": "forwarded"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


@router.post("/intercept/{request_id}/edit")
async def intercept_edit(request_id: str, body: dict = Body(...)):
    """Forward a paused request with a modified body."""
    if interceptor.resolve_edit(request_id, body):
        return JSONResponse(status_code=200, content={"status": "edited"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


@router.post("/intercept/{request_id}/drop")
async def intercept_drop(request_id: str):
    """Drop a paused request (returns 204 to the original client)."""
    if interceptor.resolve_drop(request_id):
        return JSONResponse(status_code=200, content={"status": "dropped"})
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
