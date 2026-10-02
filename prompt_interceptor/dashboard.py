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

from ._version import __version__
from .config import get_config, save_config
from .interceptor import interceptor
from .logger import TrafficLogger, extract_images, strip_images
from .rules_engine import RuleEngine
from .health import check_proxy_health, check_target_health, get_models, get_status

# ---------------------------------------------------------------------------
# App & shared singletons
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PromptInterceptor Dashboard",
    description="Dashboard for PromptInterceptor - Ollama Traffic Interceptor for AI Clients",
    version=__version__,
)

_config = get_config()
_logger = TrafficLogger()
_rule_engine = RuleEngine(_logger)

router = APIRouter(prefix="/api")

_ICON_PATH = os.path.join(os.path.dirname(__file__), "..", "res", "PromptInterceptor_Icon.png")
_LOGO_PATH = os.path.join(os.path.dirname(__file__), "..", "res", "PromptInterceptor_Logo.png")

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
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
  .stat{background:#16213e;border-radius:8px;padding:16px;text-align:center}
  .stat-value{font-size:2em;color:#4fc3f7}
  .stat-label{color:#888;font-size:.82em;margin-top:4px}
  .ctx-warn{display:none;background:#3a1a00;border:1px solid #7a4a00;border-radius:6px;padding:10px 16px;margin-bottom:10px;color:#ffa040;font-size:.88em}
  .ctx-alert-banner{display:none;background:#3a0000;border:1px solid #8a1a1a;border-radius:6px;padding:10px 16px;margin-bottom:10px;color:#ff6060;font-size:.88em;font-weight:600;align-items:center;justify-content:space-between}
  #ctx-toast{position:fixed;top:18px;right:18px;z-index:9999;background:#3a0000;border:1px solid #8a1a1a;border-radius:8px;padding:14px 20px;color:#ff6060;font-size:.9em;font-weight:600;box-shadow:0 4px 18px rgba(0,0,0,.6);max-width:320px;display:none;animation:ctx-slide-in .25s ease}
  @keyframes ctx-slide-in{from{opacity:0;transform:translateX(40px)}to{opacity:1;transform:none}}
  .ctx-alert-controls{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:8px;padding-top:8px;border-top:1px solid #2a2a4e}
  .ctx-alert-controls span{color:#888;font-size:.75em}
  input[type=number].threshold-input{width:70px;background:#0d0d1e;color:#eee;border:1px solid #2a2a4e;border-radius:4px;padding:5px 8px;font-size:.85em}
  .toggle-switch{position:relative;display:inline-block;width:36px;height:20px}
  .toggle-switch input{opacity:0;width:0;height:0}
  .toggle-slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#333;border-radius:20px;transition:.3s}
  .toggle-slider:before{position:absolute;content:"";height:14px;width:14px;left:3px;bottom:3px;background:#888;border-radius:50%;transition:.3s}
  input:checked+.toggle-slider{background:#0d3b66}
  input:checked+.toggle-slider:before{transform:translateX(16px);background:#4fc3f7}
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
  .pending-card{background:#0d0d1e;border:1px solid #6a4a1a;border-left:4px solid #ffa040;
                border-radius:6px;padding:16px;margin-bottom:12px}
  .pending-card h3{color:#ffa040;font-size:.95em;margin-bottom:8px}
  .pending-meta{color:#888;font-size:.8em;margin-bottom:8px}
  .pending-actions{display:flex;gap:8px;margin-top:10px}
  #pending-section{display:none}
  #pending-section h2{color:#ffa040}
  .header-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
  .mode-controls{display:flex;align-items:center;gap:8px}
  .mode-label{color:#888;font-size:.85em;margin-right:4px}
  a{color:#4fc3f7;text-decoration:none}
  a:hover{text-decoration:underline}
  .empty{color:#555;font-style:italic}
  .link-show{color:#4fc3f7;cursor:pointer;font-size:.8em;text-decoration:underline;
    background:none;border:none;padding:0}
  .link-del{color:#f66;cursor:pointer;font-size:.9em;background:none;border:none;
    padding:0 2px;opacity:.65}
  .link-del:hover{opacity:1}
  .modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
    background:rgba(0,0,0,.75);z-index:1000;overflow-y:auto}
  .modal-box{background:#16213e;border-radius:8px;padding:24px;max-width:1400px;
    margin:40px auto;position:relative;border:1px solid #2a2a4e}
  .modal-close{position:absolute;top:10px;right:14px;background:none;border:none;
    color:#888;font-size:1.3em;cursor:pointer}
  .modal-close:hover{color:#eee}
  .modal-pre{background:#0d0d1e;padding:14px;border-radius:6px;font-size:.78em;
    overflow:auto;line-height:1.5;white-space:pre-wrap;word-break:break-word}
  .modal-title{color:#4fc3f7;margin-bottom:14px;padding-right:24px}
  .modal-split{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .modal-panel{display:flex;flex-direction:column;gap:8px;min-width:0}
  .modal-panel-title{color:#4fc3f7;font-size:.82em;font-weight:600;
    text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
  .modal-thinking-block{display:flex;flex-direction:column;gap:4px}
  .modal-thinking-title{color:#f0a500;font-size:.78em;font-weight:600;
    text-transform:uppercase;letter-spacing:.05em}
  .modal-thinking-pre{background:#1a1200;border:1px solid #3a2800;padding:10px;
    border-radius:6px;font-size:.75em;overflow:auto;max-height:200px;
    line-height:1.5;white-space:pre-wrap;word-break:break-word;color:#f0c060}
  .modal-tools-block{display:flex;flex-direction:column;gap:4px}
  .modal-tools-title{color:#4caf88;font-size:.78em;font-weight:600;
    text-transform:uppercase;letter-spacing:.05em}
  .modal-tools-pre{background:#0a1a14;border:1px solid #1a3a28;padding:10px;
    border-radius:6px;font-size:.75em;overflow:auto;max-height:200px;
    line-height:1.5;white-space:pre-wrap;word-break:break-word;color:#7dcc9a}
  .modal-resp-pre{background:#0d0d1e;padding:14px;border-radius:6px;font-size:.78em;
    overflow:auto;flex:1;min-height:120px;max-height:calc(80vh - 80px);
    line-height:1.5;white-space:pre-wrap;word-break:break-word}
  .modal-prompt-pre{background:#0d0d1e;padding:14px;border-radius:6px;font-size:.78em;
    overflow:auto;max-height:calc(80vh - 40px);line-height:1.5;
    white-space:pre-wrap;word-break:break-word}
  .img-badge{background:#2a1a3a;color:#c89bff;border:1px solid #4a2a6a;font-size:.75em;margin-right:4px}
  .ctx-badge{background:#2a2000;color:#ffa040;border:1px solid #5a4000;font-size:.75em}
  .modal-images-block{display:flex;flex-direction:column;gap:6px;margin-bottom:8px}
  .modal-images-title{color:#c89bff;font-size:.78em;font-weight:600}
  .modal-images-grid{display:flex;flex-wrap:wrap;gap:8px}
  .modal-thumb{display:flex;flex-direction:column;align-items:center;gap:3px;
    background:#0d0d1e;border:1px solid #2a2a4a;border-radius:6px;padding:6px;cursor:zoom-in}
  .modal-thumb:hover{border-color:#c89bff}
  .modal-thumb img{max-height:160px;max-width:220px;object-fit:contain;display:block}
  .modal-thumb span{color:#9a9ac0;font-size:.72em}
  .img-lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:2000;
    align-items:center;justify-content:center;flex-direction:column;gap:10px;cursor:zoom-out}
  .img-lightbox img{max-width:95vw;max-height:88vh;object-fit:contain;background:#fff1}
  .img-lightbox a{color:#4fc3f7;font-size:.85em}
  .crash-hint{margin-top:8px;padding-top:8px;border-top:1px solid #4a1a1a;color:#ffa040}
</style>
</head>
<body>
<div class="container">

  <div class="header-row">
    <div style="display:flex;align-items:center;gap:14px">
      <img src="/logo" alt="PromptInterceptor" id="dashboard-logo"
           style="height:48px;width:auto" onerror="this.style.display='none'">
      <div>
        <h1>PromptInterceptor Dashboard</h1>
        <div class="subtitle">Ollama Traffic Interceptor for AI Clients</div>
      </div>
    </div>
    <div class="mode-controls">
      <span class="mode-label">Mode:</span>
      <button class="btn-mode" id="btn-passthrough" onclick="setMode('passthrough')">Passthrough</button>
      <button class="btn-mode" id="btn-intercept" onclick="setMode('intercept')">Intercept</button>
      <button class="btn-mode" onclick="showProxyInfo()" style="margin-left:10px">&#x1F4CB; Proxy Info</button>
    </div>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-value" id="stat-requests">-</div>
      <div class="stat-label">Total Requests</div>
    </div>
    <div class="stat" id="stat-modifiers-box" style="display:none">
      <div class="stat-value" id="stat-modifiers">-</div>
      <div class="stat-label">Active Modifiers</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-mode">-</div>
      <div class="stat-label">Current Mode</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-ctx">-</div>
      <div class="stat-label">Context Used</div>
      <div class="ctx-alert-controls">
        <span>Alert:</span>
        <label class="toggle-switch" title="Enable context alert">
          <input type="checkbox" id="ctx-alert-enabled" onchange="saveAlertSettings()" checked>
          <span class="toggle-slider"></span>
        </label>
        <input type="number" class="threshold-input" id="ctx-alert-threshold"
          min="0" max="100" value="80" onchange="saveAlertSettings()">
        <span>%</span>
      </div>
    </div>
  </div>

  <!-- Pending Intercepts -->
  <div class="section" id="pending-section">
    <h2>&#9888; Pending Intercepts</h2>
    <div id="pending-list"></div>
  </div>

  <!-- Modifiers — only visible in intercept mode -->
  <div class="section" id="modifiers-section" style="display:none">
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

  <!-- Live Prompts -->
  <div class="section">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <h2 style="margin-bottom:0">Live Prompts</h2>
      <div style="display:flex;gap:6px">
        <button class="btn-add" onclick="loadFile()" style="background:#0d3b66;border-color:#1a5b8a;color:#4fc3f7">&#128193; Load</button>
        <button class="btn-add" onclick="saveLogs()" style="background:#1a4d1a;border-color:#2a6b2a;color:#5f5">&#8595; Save</button>
        <button class="btn-add" onclick="clearLogs()" style="background:#4d1a1a;border-color:#6b2a2a;color:#f88">&#x2715; Clear</button>
      </div>
    </div>
    <div id="file-mode-banner" style="display:none;background:#0a1e0a;border:1px solid #1a4b1a;border-radius:6px;padding:8px 14px;margin-bottom:10px;color:#5f5;font-size:.85em;align-items:center;justify-content:space-between">
      <span>&#128196; Archivo: <strong id="file-mode-name"></strong></span>
      <button onclick="exitFileMode()" style="background:#4d1a1a;color:#f88;border:1px solid #6b2a2a;border-radius:4px;padding:3px 8px;cursor:pointer;font-size:.8em">&#x2715; Volver a live</button>
    </div>
    <div class="ctx-alert-banner" id="ctx-alert-banner">
      <span>&#x26A0; Context at <strong id="ctx-alert-pct">-</strong> &mdash; exceeds alert threshold. Response quality may degrade.</span>
      <button onclick="dismissAlert()" style="background:none;border:none;color:#ff6060;cursor:pointer;font-size:1em;padding:0 4px">&#x2715;</button>
    </div>
    <table>
      <thead><tr>
        <th>Time</th><th>Method</th><th>Path</th><th>Model</th><th>Tokens / Ctx%</th><th>Prompt</th><th>Response</th><th>Duration</th><th>Status</th><th></th>
      </tr></thead>
      <tbody id="logs-body"><tr><td colspan="10" class="empty">Loading...</td></tr></tbody>
    </table>
  </div>

  <footer>
    PromptInterceptor Dashboard {APP_VERSION} &mdash;
    <a href="/docs">API Docs</a>
  </footer>
</div>

<!-- Proxy Info modal -->
<div class="modal-overlay" id="proxy-modal-overlay" onclick="if(event.target===this)closeProxyModal()">
  <div class="modal-box" style="max-width:700px">
    <button class="modal-close" onclick="closeProxyModal()">&#x2715;</button>
    <h3 class="modal-title">Proxy Status</h3>
    <pre class="modal-pre" id="proxy-status-content">Loading...</pre>
  </div>
</div>

<!-- Full-size image viewer -->
<div class="img-lightbox" id="img-lightbox" onclick="closeLightbox()">
  <img id="img-lightbox-img" alt="">
  <a id="img-lightbox-link" target="_blank" rel="noopener" onclick="event.stopPropagation()">Open in new tab</a>
</div>

<!-- Raw data modal -->
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal()">&#x2715;</button>
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;padding-right:36px">
      <h3 class="modal-title" id="modal-title" style="margin:0;flex:1">Message Detail</h3>
      <button id="modal-raw-btn" class="link-show" onclick="toggleModalRaw()" style="font-size:.85em;padding:3px 10px;border:1px solid #4fc3f7;border-radius:4px;background:transparent;white-space:nowrap;cursor:pointer">Raw</button>
    </div>
    <div id="modal-correction-banner" style="display:none;background:#0a2a1a;border:1px solid #1a4a2a;border-radius:5px;padding:7px 12px;margin-bottom:12px;color:#5dba8a;font-size:.82em">
      &#x2714; Auto-corrected by proxy: <span id="modal-correction-text" style="font-weight:600"></span>
    </div>
    <div id="modal-token-banner" style="display:none;background:#0a1222;border:1px solid #1a2a4a;border-radius:5px;padding:7px 12px;margin-bottom:12px;color:#4fc3f7;font-size:.82em">
      &#x1F4CA; Tokens: <span id="modal-token-text"></span>
    </div>
    <div id="modal-error-banner" style="display:none;background:#1a0a0a;border:1px solid #4a1a1a;border-radius:5px;padding:10px 14px;margin-bottom:12px;color:#f55;font-size:.83em">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <span style="font-size:1.1em;font-weight:700" id="modal-error-code"></span>
        <span style="color:#c77;font-weight:600" id="modal-error-label"></span>
      </div>
      <div id="modal-error-msg" style="color:#daa;margin-top:2px;word-break:break-word"></div>
      <div id="modal-error-hint" class="crash-hint" style="display:none"></div>
    </div>
    <div id="modal-dropped-banner" style="display:none;background:#3a0808;border:1px solid #6a1818;border-radius:5px;padding:10px 14px;margin-bottom:12px;color:#f55;font-size:.83em">
      &#x2715; Request dropped &mdash; Ollama never received it (204 No Content)
    </div>
    <div class="modal-split">
      <div class="modal-panel">
        <div class="modal-panel-title">Prompt</div>
        <div id="modal-images" class="modal-images-block" style="display:none">
          <div class="modal-images-title" id="modal-images-title"></div>
          <div class="modal-images-grid" id="modal-images-grid"></div>
        </div>
        <pre class="modal-prompt-pre" id="modal-prompt"></pre>
      </div>
      <div class="modal-panel">
        <div id="modal-thinking-block" class="modal-thinking-block" style="display:none">
          <div class="modal-thinking-title">&#x1F9E0; Thinking</div>
          <pre class="modal-thinking-pre" id="modal-thinking"></pre>
        </div>
        <div id="modal-tools-block" class="modal-tools-block" style="display:none">
          <div class="modal-tools-title">&#x1F527; Tools</div>
          <pre class="modal-tools-pre" id="modal-tools"></pre>
        </div>
        <div class="modal-panel-title">Response</div>
        <pre class="modal-resp-pre" id="modal-response"></pre>
      </div>
    </div>
    <div id="modal-raw-panel" style="display:none">
      <pre id="modal-raw-pre" style="background:#0d1117;color:#c9d1d9;padding:16px;border-radius:6px;overflow:auto;max-height:70vh;font-size:.8em;white-space:pre-wrap;word-break:break-all;margin:0"></pre>
    </div>
  </div>
</div>

<script>
const _logsCache = {};
let _pendingCache = {};
let _editModalId = null;
let _proxyStatusData = null;
let _fileMode = false;
let _loadedLogs = null;
let _currentModalLog = null;
let _currentModalId = null;
let _alertSettings = { context_alert_enabled: true, context_alert_threshold: 80 };
let _alertDismissed = false;
let _alertToastTimer = null;
let _modalRawMode = false;

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// --- Images (vision models) ---
// Mirrors extract_images() in logger.py: Ollama "images" lists (bare base64),
// OpenAI "image_url" blocks and Anthropic "image" blocks.
const _IMG_SIGS = [['iVBOR', 'image/png'], ['/9j/', 'image/jpeg'], ['R0lG', 'image/gif'], ['UklG', 'image/webp'], ['Qk', 'image/bmp']];
const _IMG_ICON = String.fromCodePoint(0x1F5BC);

function sniffImageMime(data) {
  for (const [prefix, mime] of _IMG_SIGS) if (data.startsWith(prefix)) return mime;
  return 'image/png';
}

// Only image data URIs and http(s) URLs are ever assigned to <img>/<a>.
function safeImageSrc(src) {
  return typeof src === 'string' && (/^data:image\\/[\\w.+-]+;base64,/.test(src) || /^https?:\\/\\//.test(src));
}

// Server placeholder that replaces a base64 payload in /api/logs bodies.
function isImagePlaceholder(v) { return typeof v === 'string' && v.startsWith('<image #'); }

// Images whose payload is present in the body (logs loaded from a file, image URLs).
function extractImagesClient(body) {
  const out = [];
  if (!body || typeof body !== 'object') return out;
  const addStr = (v, location, bare) => {
    if (typeof v !== 'string' || isImagePlaceholder(v)) return;
    if (v.startsWith('data:') || /^https?:/.test(v)) out.push({src: v, location});
    else if (bare) out.push({src: 'data:' + sniffImageMime(v) + ';base64,' + v, location});
  };
  if (Array.isArray(body.images)) body.images.forEach(v => addStr(v, 'prompt', true));
  (Array.isArray(body.messages) ? body.messages : []).forEach((m, i) => {
    if (!m || typeof m !== 'object') return;
    const location = 'messages[' + i + ']';
    if (Array.isArray(m.images)) m.images.forEach(v => addStr(v, location, true));
    if (!Array.isArray(m.content)) return;
    m.content.forEach(b => {
      if (!b || typeof b !== 'object') return;
      if (b.type === 'image_url') {
        addStr(typeof b.image_url === 'string' ? b.image_url : b.image_url?.url, location, false);
      } else if (b.type === 'image' && b.source) {
        const src = b.source;
        if (src.type === 'base64' && typeof src.data === 'string' && !isImagePlaceholder(src.data)) {
          out.push({src: 'data:' + (src.media_type || sniffImageMime(src.data)) + ';base64,' + src.data, location});
        } else if (src.type === 'url') {
          addStr(src.url, location, false);
        }
      }
    });
  });
  return out.filter(im => safeImageSrc(im.src));
}

function messageImageCount(m) {
  if (!m || typeof m !== 'object') return 0;
  let n = Array.isArray(m.images) ? m.images.length : 0;
  if (Array.isArray(m.content)) n += m.content.filter(b => b && (b.type === 'image_url' || b.type === 'image')).length;
  return n;
}

function logImageCount(l) {
  if (l._image_count != null) return l._image_count;
  const b = l.body;
  if (!b || typeof b !== 'object') return 0;
  let n = Array.isArray(b.images) ? b.images.length : 0;
  (Array.isArray(b.messages) ? b.messages : []).forEach(m => { n += messageImageCount(m); });
  return n;
}

function renderModalImages(images) {
  const block = document.getElementById('modal-images');
  const grid = document.getElementById('modal-images-grid');
  grid.textContent = '';
  const safe = (images || []).filter(im => safeImageSrc(im.src));
  if (!safe.length) { block.style.display = 'none'; return; }
  document.getElementById('modal-images-title').textContent = _IMG_ICON + ' Images (' + safe.length + ')';
  safe.forEach((im, i) => {
    const card = document.createElement('div');
    card.className = 'modal-thumb';
    card.title = 'Click to enlarge';
    const img = document.createElement('img');
    img.alt = 'image #' + (i + 1);
    const cap = document.createElement('span');
    cap.textContent = '#' + (i + 1) + (im.location ? ' · ' + im.location : '');
    img.onload = () => { cap.textContent += ' · ' + img.naturalWidth + '×' + img.naturalHeight; };
    img.src = im.src;
    card.append(img, cap);
    card.onclick = () => openLightbox(im.src);
    grid.appendChild(card);
  });
  block.style.display = '';
}

// Inline payloads are shown at once; base64 stripped from live logs is fetched on demand.
function loadModalImages(l, id) {
  renderModalImages([]);
  const total = logImageCount(l);
  if (!total) return;
  const inline = extractImagesClient(l.body);
  renderModalImages(inline);
  if (_fileMode || !l.request_id || inline.length >= total) return;
  fetchJSON('/api/logs/' + encodeURIComponent(l.request_id) + '/images')
    .then(d => { if (_currentModalId === id && Array.isArray(d.images)) renderModalImages(d.images); })
    .catch(() => {});
}

function openLightbox(src) {
  if (!safeImageSrc(src)) return;
  document.getElementById('img-lightbox-img').src = src;
  document.getElementById('img-lightbox-link').href = src;
  document.getElementById('img-lightbox').style.display = 'flex';
}

function closeLightbox() {
  const box = document.getElementById('img-lightbox');
  box.style.display = 'none';
  document.getElementById('img-lightbox-img').removeAttribute('src');
  document.getElementById('img-lightbox-link').removeAttribute('href');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && document.getElementById('img-lightbox').style.display === 'flex') closeLightbox();
});

function numCtxOverrideText(ov) {
  return 'num_ctx: client ' + (ov.client ?? 'not set') + ' → sent ' + ov.sent;
}

// Extract <tool_call>...</tool_call> blocks embedded in text (e.g. inside <think>)
function extractEmbeddedToolCalls(text) {
  if (!text) return [];
  const re = /<tool_call>([\\s\\S]*?)<\\/tool_call>/g;
  const calls = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const inner = m[1].trim();
    // Format 1: JSON  {"name": "...", "arguments": {...}}
    try { calls.push(JSON.parse(inner)); continue; }
    catch (_) {}
    // Format 2: XML  <function=NAME><parameter=KEY>VALUE</parameter>...</function>
    const fnMatch = inner.match(/<function=([\\w.:-]+)>/);
    if (fnMatch) {
      const name = fnMatch[1];
      const args = {};
      const paramRe = /<parameter=([\\w.:-]+)>([\\s\\S]*?)<\\/parameter>/g;
      let pm;
      while ((pm = paramRe.exec(inner)) !== null) {
        const val = pm[2].trim();
        try { args[pm[1]] = JSON.parse(val); } catch (_) { args[pm[1]] = val; }
      }
      calls.push({ name, arguments: args });
    } else {
      calls.push({ _raw: inner });
    }
  }
  return calls;
}

// Strip <tool_call>...</tool_call> blocks from text
function stripEmbeddedToolCalls(text) {
  return text ? text.replace(/<tool_call>[\\s\\S]*?<\\/tool_call>/g, '').trim() : text;
}

function showRaw(id) {
  const l = _logsCache[id];
  if (!l) return;
  _currentModalLog = l;
  _currentModalId = id;
  _modalRawMode = false;
  document.getElementById('modal-raw-panel').style.display = 'none';
  document.getElementById('modal-raw-btn').textContent = 'Raw';
  document.querySelector('.modal-split').style.display = '';
  document.getElementById('modal-title').textContent =
    (l.method || '') + ' ' + (l.path || '') + ' — ' + (l.timestamp || l.response_timestamp || '');

  // --- Extract prompt ---
  let promptText = '';
  let imgNo = 0;
  const imgMarkers = n => Array.from({length: n}, () => '[' + _IMG_ICON + ' image #' + (++imgNo) + ']').join(' ');
  const msgs = l.body?.messages;
  if (msgs && msgs.length) {
    promptText = msgs.map(m => {
      const role = (m.role || 'user').toUpperCase();
      let content = '';
      if (typeof m.content === 'string') {
        content = m.content;
      } else if (Array.isArray(m.content)) {
        content = m.content
          .filter(b => b.type === 'text')
          .map(b => b.text || '')
          .join('\\n');
      }
      const nImg = messageImageCount(m);
      if (nImg) content = imgMarkers(nImg) + (content ? '\\n' + content : '');
      return `[${role}]\\n${content}`;
    }).join('\\n\\n---\\n\\n');
  } else if (l.body?.prompt) {
    const nImg = Array.isArray(l.body.images) ? l.body.images.length : 0;
    promptText = (nImg ? imgMarkers(nImg) + '\\n' : '') + String(l.body.prompt);
  }
  loadModalImages(l, id);
  document.getElementById('modal-prompt').textContent = promptText || '(sin prompt)';

  // --- Extract response and thinking ---
  let thinkingText = '';
  let respText = '';
  const rb = l.response_body;
  if (rb) {
    if (rb.choices && rb.choices[0]?.message) {
      const msg = rb.choices[0].message;
      respText = msg.content || '';
      thinkingText = msg.thinking || '';
    } else if (rb.message) {
      respText = rb.message.content || '';
      thinkingText = rb.message.thinking || '';
    } else if (typeof rb.response === 'string') {
      respText = rb.response;
    } else if (Array.isArray(rb.content)) {
      // Anthropic-style: separate thinking and text blocks
      thinkingText = rb.content.filter(b => b.type === 'thinking').map(b => b.thinking || '').join('\\n\\n');
      respText = rb.content.filter(b => b.type === 'text').map(b => b.text || '').join('\\n');
    }
  }

  // Fallback: <think>...</think> inline tags (older Ollama / deepseek-r1)
  if (!thinkingText && respText) {
    const thinkTagMatch = respText.match(/<think>([\\s\\S]*?)<\\/think>([\\s\\S]*)/);
    if (thinkTagMatch) {
      thinkingText = thinkTagMatch[1].trim();
      respText = thinkTagMatch[2].trim();
    }
  }

  // --- Extract tool_calls (explicit fields or embedded <tool_call> tags) ---
  let toolsText = '';
  let embeddedCalls = [];
  if (rb) {
    let toolCalls = null;
    if (rb.choices && rb.choices[0]?.message?.tool_calls) {
      toolCalls = rb.choices[0].message.tool_calls;
    } else if (rb.message?.tool_calls) {
      toolCalls = rb.message.tool_calls;
    } else if (Array.isArray(rb.content)) {
      const toolUse = rb.content.filter(b => b.type === 'tool_use');
      if (toolUse.length) toolCalls = toolUse;
    }
    if (toolCalls && toolCalls.length) {
      toolsText = toolCalls.map((tc, i) => {
        const fn = tc.function || {};
        let args = fn.arguments !== undefined ? fn.arguments : (tc.input || tc.arguments || '');
        try {
          if (typeof args === 'string' && args) args = JSON.stringify(JSON.parse(args), null, 2);
          else if (typeof args === 'object') args = JSON.stringify(args, null, 2);
        } catch (_) {}
        const id = tc.id ? ` (${tc.id})` : '';
        return `[${i + 1}]${id} ${fn.name || tc.name || tc.type || 'tool'}(\n${args}\n)`;
      }).join('\\n\\n');
    }
    // Fallback: parse <tool_call> blocks embedded in thinking or response text.
    // Also used when explicit tool_calls exist but have no valid function names
    // (can happen when streaming assembly produces empty deltas).
    const hasValidNames = toolCalls?.some(tc => tc.function?.name || tc.name);
    if (!toolsText || !hasValidNames) {
      embeddedCalls = extractEmbeddedToolCalls(thinkingText + '\\n' + respText);
      if (embeddedCalls.length) {
        toolsText = embeddedCalls.map((tc, i) => {
          let args = tc.arguments || tc.input || tc.parameters || '';
          try {
            if (typeof args === 'object') args = JSON.stringify(args, null, 2);
            else if (typeof args === 'string' && args) args = JSON.stringify(JSON.parse(args), null, 2);
          } catch (_) {}
          return `[${i + 1}] ${tc.name || 'tool'}(\n${args}\n)`;
        }).join('\\n\\n');
      }
    }
  }

  // Strip <tool_call> XML from thinking text so it doesn't appear raw in the panel
  if (thinkingText) thinkingText = stripEmbeddedToolCalls(thinkingText);

  const thinkingBlock = document.getElementById('modal-thinking-block');
  if (thinkingText) {
    document.getElementById('modal-thinking').textContent = thinkingText;
    thinkingBlock.style.display = '';
  } else {
    thinkingBlock.style.display = 'none';
  }
  const toolsBlock = document.getElementById('modal-tools-block');
  if (toolsText) {
    document.getElementById('modal-tools').textContent = toolsText;
    toolsBlock.style.display = '';
  } else {
    toolsBlock.style.display = 'none';
  }

  document.getElementById('modal-response').textContent = respText || '(no response)';

  const corrBanner = document.getElementById('modal-correction-banner');
  if (l._correction_applied) {
    document.getElementById('modal-correction-text').textContent = l._correction_applied;
    corrBanner.style.display = '';
  } else {
    corrBanner.style.display = 'none';
  }
  const tokenBanner = document.getElementById('modal-token-banner');
  const ttu = l._tokens_usage;
  const ctxOv = l._num_ctx_override;
  const ctxOvHtml = ctxOv ? '<span style="color:#ffa040">' + esc(numCtxOverrideText(ctxOv)) + ' &#x26A0;</span>' : '';
  if (ttu && ttu.total_tokens != null) {
    const parts = ['Total: ' + ttu.total_tokens];
    if (ttu.prompt_tokens != null) parts.push('Prompt: ' + ttu.prompt_tokens);
    if (ttu.completion_tokens != null) parts.push('Completion: ' + ttu.completion_tokens);
    if (ttu.context_size != null) parts.push('Ctx: ' + ttu.context_size);
    if (ttu.context_utilization_pct != null) {
      const pc = ttu.context_utilization_pct >= 90 ? '#f55' : ttu.context_utilization_pct >= 70 ? '#ffa040' : '#4fc3f7';
      parts.push('<span style="color:' + pc + '">' + ttu.context_utilization_pct + '% of context</span>');
    }
    if (ttu.offload) {
      const offloadLabel = ttu.offload === 'gpu' ? '&#x1F7E2; GPU' : ttu.offload === 'cpu' ? '&#x1F7E1; CPU' : '&#x1F7E0; GPU+CPU';
      const vramMB = ttu.size_vram ? Math.round(ttu.size_vram / 1024 / 1024) + ' MB VRAM' : '';
      parts.push(offloadLabel + (vramMB ? ' (' + vramMB + ')' : ''));
    }
    if (ctxOvHtml) parts.push(ctxOvHtml);
    document.getElementById('modal-token-text').innerHTML = parts.join(' &nbsp;|&nbsp; ');
    tokenBanner.style.display = '';
  } else if (ctxOvHtml) {
    document.getElementById('modal-token-text').innerHTML = ctxOvHtml;
    tokenBanner.style.display = '';
  } else {
    tokenBanner.style.display = 'none';
  }

  // --- Error banner ---
  const errorBanner = document.getElementById('modal-error-banner');
  const sc = l.status_code;
  const errMsg = l.response_body?.error || '';
  if (sc && sc >= 400) {
    const labels = {
      400: 'Bad Request',
      408: 'Request Timeout — the request took too long',
      413: 'Context Window Exceeded — context window full',
      500: 'Internal Server Error — proxy failure',
      502: 'Bad Gateway — cannot connect to Ollama',
      503: 'Service Unavailable — Ollama is not available',
      504: 'Gateway Timeout — Ollama took too long',
    };
    // Ollama's runner dies (e.g. out of VRAM) and Ollama relays it as a 500
    const isCrash = /llama-server process has terminated|CUDA error/i.test(String(errMsg));
    document.getElementById('modal-error-code').textContent = 'HTTP ' + sc;
    document.getElementById('modal-error-label').textContent =
      isCrash ? 'Ollama crashed while loading the model' : (labels[sc] || 'Error');
    document.getElementById('modal-error-msg').textContent = errMsg || '';
    const hintEl = document.getElementById('modal-error-hint');
    if (isCrash) {
      const sent = ctxOv?.sent ?? l.body?.options?.num_ctx;
      let hint = 'Ollama crashed while loading the model (usually out of VRAM).';
      if (sent != null) {
        hint += ' num_ctx sent: ' + sent;
        if (ctxOv) hint += ' (client requested ' + (ctxOv.client ?? 'not set') + ')';
        hint += '.';
      }
      hint += ' Try a smaller Context Size in the launcher or a smaller model.';
      hintEl.textContent = hint;
      hintEl.style.display = '';
    } else {
      hintEl.style.display = 'none';
    }
    errorBanner.style.display = '';
  } else {
    errorBanner.style.display = 'none';
  }

  const droppedBanner = document.getElementById('modal-dropped-banner');
  if (sc === 204 && l.response_body?.status === 'dropped') {
    droppedBanner.style.display = '';
  } else {
    droppedBanner.style.display = 'none';
  }

  document.getElementById('modal-overlay').style.display = 'block';
}

function toggleModalRaw() {
  _modalRawMode = !_modalRawMode;
  const split = document.querySelector('.modal-split');
  const rawPanel = document.getElementById('modal-raw-panel');
  const rawBtn = document.getElementById('modal-raw-btn');
  const bannerIds = ['modal-correction-banner','modal-token-banner','modal-error-banner','modal-dropped-banner'];
  if (_modalRawMode) {
    split.style.display = 'none';
    bannerIds.forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });
    document.getElementById('modal-raw-pre').textContent = JSON.stringify(_currentModalLog, null, 2).replace(/\\\\n/g, '\\n').replace(/\\\\t/g, '\\t');
    rawPanel.style.display = 'block';
    rawBtn.textContent = 'Show';
  } else {
    rawPanel.style.display = 'none';
    split.style.display = '';
    rawBtn.textContent = 'Raw';
    if (_currentModalId) showRaw(_currentModalId);
  }
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
  _modalRawMode = false;
  _currentModalLog = null;
  _currentModalId = null;
  document.getElementById('modal-raw-panel').style.display = 'none';
  document.getElementById('modal-raw-btn').textContent = 'Raw';
  document.querySelector('.modal-split').style.display = '';
  document.getElementById('modal-dropped-banner').style.display = 'none';
  renderModalImages([]);
  closeLightbox();
}

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

async function loadStatus() {
  try {
    const data = await fetchJSON('/api/status');
    _proxyStatusData = data;
    const mode = data.proxy?.mode ?? '-';
    document.getElementById('stat-mode').textContent = mode;
    document.getElementById('stat-modifiers').textContent = data.rules?.enabled_count ?? '-';
    document.getElementById('btn-passthrough').classList.toggle('active', mode === 'passthrough');
    document.getElementById('btn-intercept').classList.toggle('active', mode === 'intercept');
    document.getElementById('modifiers-section').style.display = mode === 'intercept' ? '' : 'none';
    document.getElementById('stat-modifiers-box').style.display = mode === 'intercept' ? '' : 'none';
  } catch(e) { /* silent */ }
}

async function showProxyInfo() {
  const el = document.getElementById('proxy-status-content');
  document.getElementById('proxy-modal-overlay').style.display = 'block';
  el.textContent = 'Loading...';
  try {
    const [status, modelsData] = await Promise.all([
      fetchJSON('/api/status'),
      fetchJSON('/api/models'),
    ]);
    const combined = Object.assign({}, status, { models: modelsData.models ?? [] });
    el.textContent = JSON.stringify(combined, null, 2);
  } catch(e) {
    el.textContent = _proxyStatusData ? JSON.stringify(_proxyStatusData, null, 2) : 'Error loading status';
  }
}

function closeProxyModal() {
  document.getElementById('proxy-modal-overlay').style.display = 'none';
}

function renderLogsTable(logs) {
  const tbody = document.getElementById('logs-body');
  document.getElementById('stat-requests').textContent = logs.length;
  if (!logs.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty">No requests yet.</td></tr>';
    document.getElementById('stat-ctx').textContent = '-';
    document.getElementById('stat-ctx').style.color = '';
    document.getElementById('ctx-warn-banner').style.display = 'none';
    return;
  }
  let maxCtxPct = null;
  tbody.innerHTML = logs.slice().reverse().map((l, i) => {
    const cacheKey = l.request_id || i;
    _logsCache[cacheKey] = l;
    const ts = (l.timestamp || l.response_timestamp || '').slice(11,19) || '-';
    const method = l.method || '-';
    const path = l.path || '-';
    const model = l.body?.model || l.response_body?.model || '-';
    const modelModified = l._intercepted_modified || (l._rules_applied && l._rules_applied.length > 0);
    const msgs = l.body?.messages;
    let preview = '-';
    if (msgs && msgs.length) {
      const last = msgs[msgs.length - 1];
      const content = last.content;
      let txt = '';
      if (typeof content === 'string') {
        txt = content.slice(0, 80);
      } else if (Array.isArray(content)) {
        const textBlock = content.find(b => b.type === 'text');
        txt = (textBlock?.text || '').slice(0, 80);
      }
      preview = txt || '-';
      if (txt.length === 80) preview += '…';
    } else if (l.body?.prompt) {
      preview = l.body.prompt.toString().slice(0, 80);
    }
    const imgCount = logImageCount(l);
    if (imgCount && preview === '-') preview = '[image]';
    const imgBadge = imgCount
      ? `<span class="badge img-badge" title="${imgCount} image(s)">&#x1F5BC; ${imgCount}</span>`
      : '';
    // Extract LLM response text from response_body
    let respText = '';
    const rb = l.response_body;
    if (rb) {
      // /v1/chat/completions (OpenAI-compatible)
      if (rb.choices && rb.choices[0]?.message?.content) {
        respText = rb.choices[0].message.content;
      // /api/chat (Ollama native) — strip <think> and <tool_call> tags for clean preview
      } else if (rb.message?.content) {
        respText = rb.message.content
          .replace(/<think>[\\s\\S]*?<\\/think>/g, '')
          .replace(/<tool_call>[\\s\\S]*?<\\/tool_call>/g, '')
          .trim();
      // /api/generate (Ollama native)
      } else if (typeof rb.response === 'string') {
        respText = rb.response;
      // /v1/messages (Anthropic-compatible): first text block (may come after thinking)
      } else if (Array.isArray(rb.content)) {
        const textBlock = rb.content.find(b => b.type === 'text');
        respText = textBlock?.text || '';
      }
      // Fallback: tool_calls (explicit or embedded <tool_call> tags in content/thinking)
      if (!respText) {
        let toolCalls = rb.choices?.[0]?.message?.tool_calls || rb.message?.tool_calls || null;
        if (!toolCalls?.length && Array.isArray(rb.content)) {
          const toolUse = rb.content.filter(b => b.type === 'tool_use');
          if (toolUse.length) toolCalls = toolUse;
        }
        if (!toolCalls?.length) {
          const thinkingStr = rb.message?.thinking || rb.choices?.[0]?.message?.thinking || '';
          const contentStr = rb.message?.content || rb.choices?.[0]?.message?.content || '';
          const embedded = extractEmbeddedToolCalls(thinkingStr + '\\n' + contentStr);
          if (embedded.length) toolCalls = embedded;
        }
        if (toolCalls?.length) {
          const names = toolCalls.map(tc => tc.function?.name || tc.name || 'tool').join(', ');
          respText = `[tools: ${names}]`;
        }
      }
    }
    const isDropped = l.status_code === 204 && rb?.status === 'dropped';
    const respPreview = isDropped
      ? '<span style="color:#f55;font-weight:600">[dropped]</span>'
      : (respText ? esc(respText.slice(0, 100)) + (respText.length > 100 ? '…' : '') : '<span class="empty">-</span>');
    const statusCode = l.status_code ? `<span class="badge ${isDropped || l.status_code >= 400 ? 'red' : 'green'}">${l.status_code}</span>` : '';
    const typeTag = l.type === 'response'
      ? `<span class="badge green">resp</span>`
      : `<span class="badge blue">req</span>`;
    const corrBadge = l._correction_applied
      ? `<span class="badge" style="background:#0a2a1a;color:#5dba8a;border:1px solid #1a4a2a;font-size:.75em" title="Auto-corrected: ${esc(l._correction_applied)}">&#x2714; fixed</span>`
      : '';
    const ctxBadge = l._num_ctx_override
      ? `<span class="badge ctx-badge" title="${esc('num_ctx overridden by proxy: ' + numCtxOverrideText(l._num_ctx_override).replace('num_ctx: ', ''))}">&#x26A0; ctx</span>`
      : '';
    let interceptBadge = '';
    if (_pendingCache[l.request_id]) {
      interceptBadge = '<span class="badge" style="background:#2a2000;color:#ffa040;border:1px solid #5a4000;font-size:.75em">&#x23F3; intercepting</span>';
    } else if (isDropped) {
      interceptBadge = '<span class="badge" style="background:#3a0808;color:#f55;border:1px solid #6a1818;font-size:.75em">&#x2715; dropped</span>';
    } else if (l._intercepted_modified) {
      interceptBadge = '<span class="badge" style="background:#0d3b66;color:#4fc3f7;border:1px solid #1a5b8a;font-size:.75em">&#x270E; edited</span>';
    } else if (l._intercepted_forwarded) {
      interceptBadge = '<span class="badge" style="background:#1a2a1a;color:#7dcc7d;border:1px solid #2a4a2a;font-size:.75em">&#x2713; forwarded</span>';
    }
    const tu = l._tokens_usage;
    let tokenCell = '<span class="empty">-</span>';
    if (tu && tu.total_tokens != null) {
      const tot = tu.total_tokens;
      const totStr = tot >= 1000 ? (tot/1000).toFixed(1) + 'K' : String(tot);
      const pct = tu.context_utilization_pct;
      if (pct != null) {
        if (maxCtxPct === null || pct > maxCtxPct) maxCtxPct = pct;
        const pctColor = pct >= 90 ? '#f55' : pct >= 70 ? '#ffa040' : '#4fc3f7';
        tokenCell = totStr + '<br><small style="color:' + pctColor + '">' + pct + '%</small>';
      } else {
        tokenCell = totStr;
      }
    }
    // Hidden while an intercept is pending: log_response would recreate the file.
    const delBtn = (l.request_id && !_pendingCache[l.request_id])
      ? ` <button class="link-del" title="Delete this message" onclick="deleteLog('${esc(String(l.request_id))}')">&#x2715;</button>`
      : '';
    const dur = l._duration_ms;
    let durCell = '<span class="empty">-</span>';
    if (dur != null) {
      const durStr = dur < 1000 ? dur + ' ms' : (dur/1000).toFixed(1) + ' s';
      const durColor = dur >= 20000 ? '#f55' : dur >= 5000 ? '#ffa040' : '#9a9ac0';
      durCell = '<span style="color:' + durColor + '">' + durStr + '</span>';
    }
    return `<tr>
      <td>${ts}</td>
      <td>${esc(method)}</td>
      <td><code>${esc(path)}</code></td>
      <td><code>${esc(model)}</code>${modelModified ? ' <span class="badge blue">&#x270E; edited</span>' : ''}</td>
      <td style="white-space:nowrap;text-align:right;font-size:.85em">${tokenCell}</td>
      <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${imgBadge}${esc(preview)}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${respPreview}</td>
      <td style="white-space:nowrap;text-align:right;font-size:.85em">${durCell}</td>
      <td>${typeTag} ${statusCode} ${corrBadge} ${ctxBadge} ${interceptBadge}</td>
      <td style="white-space:nowrap"><button class="link-show" onclick="showRaw('${esc(String(cacheKey))}')">show</button>${delBtn}</td>
    </tr>`;
  }).join('');
  if (maxCtxPct !== null) {
    const ctxEl = document.getElementById('stat-ctx');
    const pctColor = maxCtxPct >= 90 ? '#f55' : maxCtxPct >= 70 ? '#ffa040' : '#4fc3f7';
    ctxEl.textContent = maxCtxPct + '%';
    ctxEl.style.color = pctColor;
    if (_alertSettings.context_alert_enabled && maxCtxPct >= _alertSettings.context_alert_threshold) {
      triggerContextAlert(maxCtxPct);
    } else {
      document.getElementById('ctx-alert-banner').style.display = 'none';
      if (maxCtxPct < _alertSettings.context_alert_threshold) _alertDismissed = false;
    }
  } else {
    document.getElementById('stat-ctx').textContent = '-';
    document.getElementById('stat-ctx').style.color = '';
    document.getElementById('ctx-alert-banner').style.display = 'none';
  }
}

async function loadLogs() {
  if (_fileMode) { renderLogsTable(_loadedLogs || []); return; }
  try {
    const data = await fetchJSON('/api/logs?limit=50');
    renderLogsTable(data.logs || []);
  } catch(e) {
    const tb = document.getElementById('logs-body');
    tb.innerHTML = '<tr><td colspan="10" id="_logs-err"></td></tr>';
    document.getElementById('_logs-err').textContent = 'Error: ' + e.message;
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
        <td><code>${esc(r.match?.path || '-')}</code></td>
        <td><code>${esc(r.match?.jsonpath || '-')}</code></td>
        <td><code>${esc(mv)}</code></td>
        <td><code>${esc(r.replace?.jsonpath || '-')}</code></td>
        <td><code>${esc(r.replace?.value ?? '-')}</code></td>
        <td><span class="badge ${r.enabled ? 'green' : 'red'}">${r.enabled ? 'on' : 'off'}</span></td>
        <td>
          <button class="btn-enable" onclick="toggleRule(${i},true)">Enable</button>
          <button class="btn-disable" onclick="toggleRule(${i},false)">Disable</button>
          <button class="btn-delete" onclick="deleteRule(${i})">Delete</button>
        </td>
      </tr>`;
    }).join('');
  } catch(e) {
    const tb = document.getElementById('rules-body');
    tb.innerHTML = '<tr><td colspan="8" id="_rules-err"></td></tr>';
    document.getElementById('_rules-err').textContent = 'Error: ' + e.message;
  }
}

async function loadPending() {
  try {
    const data = await fetchJSON('/api/intercept/pending');
    const section = document.getElementById('pending-section');
    const list = document.getElementById('pending-list');
    _pendingCache = {};
    if (!data.count) {
      section.style.display = 'none';
      return;
    }
    section.style.display = 'block';
    list.innerHTML = (data.pending || []).map(p => {
      _pendingCache[p.request_id] = p;
      const ts = p.created_at ? new Date(p.created_at).toLocaleTimeString() : '';
      const meta = [ts ? `received ${ts}` : '', `id: ${esc(p.request_id)}`].filter(Boolean).join(' · ');
      return `
      <div class="pending-card">
        <h3>&#9654; ${esc(p.method)} ${esc(p.path)}</h3>
        <div class="pending-meta">${meta}</div>
        <pre style="max-height:180px;overflow:auto;margin:0 0 8px">${esc(JSON.stringify(p.body, null, 2))}</pre>
        <div class="pending-actions">
          <button class="btn-forward" onclick="forwardRequest('${esc(p.request_id)}')">&#x2713; Forward</button>
          <button class="btn-forward" onclick="openEditModal('${esc(p.request_id)}')">&#x270E; Edit &amp; Forward</button>
          <button class="btn-drop" onclick="dropRequest('${esc(p.request_id)}')">&#x2715; Drop</button>
        </div>
      </div>`;
    }).join('');
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

async function deleteLog(requestId) {
  await fetch(`/api/logs/${requestId}`, {method: 'DELETE'});
  delete _logsCache[requestId];
  loadLogs();
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

function openEditModal(id) {
  const p = _pendingCache[id];
  if (!p) return;
  _editModalId = id;
  document.getElementById('edit-modal-title').textContent = p.method + ' ' + p.path;
  document.getElementById('edit-modal-ta').value = JSON.stringify(p.body, null, 2);
  document.getElementById('edit-modal-overlay').style.display = 'block';
  fetch(`/api/intercept/${id}/pause`, {method: 'POST'});
}

function closeEditModal() {
  document.getElementById('edit-modal-overlay').style.display = 'none';
  if (_editModalId) fetch(`/api/intercept/${_editModalId}/resume`, {method: 'POST'});
  _editModalId = null;
}

async function submitEditModal() {
  if (!_editModalId) return;
  const ta = document.getElementById('edit-modal-ta');
  let body;
  try { body = JSON.parse(ta.value); } catch(e) { alert('Invalid JSON: ' + e.message); return; }
  await fetch(`/api/intercept/${_editModalId}/edit`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  closeEditModal();
  loadPending();
}

async function dropRequest(id) {
  await fetch(`/api/intercept/${id}/drop`, {method: 'POST'});
  loadPending();
}

async function clearLogs() {
  if (!confirm('Clear all logs from the current session?')) return;
  await fetch('/api/reset', {method: 'POST'});
  Object.keys(_logsCache).forEach(k => delete _logsCache[k]);
  loadLogs();
  loadStatus();
}

async function saveLogs() {
  const cached = Object.values(_logsCache);
  if (!cached.length) { alert('No hay logs para guardar.'); return; }
  // Live logs carry image placeholders; fetch the full entries so the export keeps the images.
  const logs = await Promise.all(cached.map(l =>
    (!_fileMode && l._image_count && l.request_id)
      ? fetchJSON('/api/logs/' + encodeURIComponent(l.request_id))
          .then(full => (full && full.request_id ? full : l))
          .catch(() => l)
      : l));
  const blob = new Blob([JSON.stringify(logs, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const ts = new Date().toISOString().slice(0,19).replace(/[T:]/g, '-');
  a.href = url; a.download = `prompt-interceptor-logs-${ts}.json`;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

function loadFile() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        const logs = Array.isArray(data) ? data : (data.logs || []);
        if (!logs.length) { alert('No se encontraron logs en el archivo.'); return; }
        _loadedLogs = logs;
        _fileMode = true;
        Object.keys(_logsCache).forEach(k => delete _logsCache[k]);
        logs.forEach((l, i) => { _logsCache[l.request_id || i] = l; });
        document.getElementById('file-mode-name').textContent = file.name;
        document.getElementById('file-mode-banner').style.display = 'flex';
        renderLogsTable(logs);
      } catch(err) {
        alert('Error al leer el archivo: ' + err.message);
      }
    };
    reader.readAsText(file);
  };
  input.click();
}

function exitFileMode() {
  _fileMode = false;
  _loadedLogs = null;
  document.getElementById('file-mode-banner').style.display = 'none';
  Object.keys(_logsCache).forEach(k => delete _logsCache[k]);
  loadLogs();
}

async function loadAlertSettings() {
  try {
    const data = await fetchJSON('/api/alert-settings');
    _alertSettings = data;
    document.getElementById('ctx-alert-enabled').checked = data.context_alert_enabled;
    document.getElementById('ctx-alert-threshold').value = data.context_alert_threshold;
  } catch(e) { /* usar defaults */ }
}

async function saveAlertSettings() {
  const enabled = document.getElementById('ctx-alert-enabled').checked;
  const threshold = parseInt(document.getElementById('ctx-alert-threshold').value, 10);
  if (isNaN(threshold) || threshold < 0 || threshold > 100) return;
  try {
    const data = await fetchJSON('/api/alert-settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ context_alert_enabled: enabled, context_alert_threshold: threshold })
    });
    _alertSettings = data;
    _alertDismissed = false;
  } catch(e) { /* silent */ }
}

function triggerContextAlert(pct) {
  document.getElementById('ctx-alert-pct').textContent = pct + '%';
  document.getElementById('ctx-alert-banner').style.display = 'flex';
  if (!_alertDismissed) {
    const toast = document.getElementById('ctx-toast');
    document.getElementById('ctx-toast-pct').textContent = pct + '%';
    toast.style.display = 'block';
    if (_alertToastTimer) clearTimeout(_alertToastTimer);
    _alertToastTimer = setTimeout(() => { toast.style.display = 'none'; }, 6000);
  }
}

function dismissAlert() {
  document.getElementById('ctx-alert-banner').style.display = 'none';
  document.getElementById('ctx-toast').style.display = 'none';
  if (_alertToastTimer) clearTimeout(_alertToastTimer);
  _alertDismissed = true;
}

function refreshAll() { loadStatus(); loadRules(); loadLogs(); loadPending(); }

loadAlertSettings();
refreshAll();
setInterval(refreshAll, 5000);
setInterval(loadPending, 1000);
</script>

<div id="ctx-toast">
  &#x26A0; Context alert: <strong id="ctx-toast-pct">-</strong> used
</div>

<!-- Edit Intercept modal -->
<div class="modal-overlay" id="edit-modal-overlay" onclick="if(event.target===this)closeEditModal()">
  <div class="modal-box" style="max-width:800px">
    <button class="modal-close" onclick="closeEditModal()">&#x2715;</button>
    <h3 class="modal-title" id="edit-modal-title">Edit Request</h3>
    <textarea id="edit-modal-ta" rows="20" style="width:100%;background:#0d0d1e;color:#eee;border:1px solid #2a2a4e;border-radius:6px;padding:10px;font-size:.8em;font-family:monospace;resize:vertical;min-height:220px"></textarea>
    <div style="display:flex;gap:8px;margin-top:12px;justify-content:flex-end">
      <button class="btn-disable" onclick="closeEditModal()" style="padding:7px 18px">Cancelar</button>
      <button class="btn-forward" onclick="submitEditModal()" style="padding:7px 18px">&#x2713; Forward</button>
    </div>
  </div>
</div>
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


@app.get("/logo", include_in_schema=False)
async def logo():
    if os.path.exists(_LOGO_PATH):
        return FileResponse(_LOGO_PATH, media_type="image/png")
    return Response(status_code=404)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve dashboard HTML."""
    static_index = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_index):
        with open(static_index) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=_DASHBOARD_HTML.replace("{APP_VERSION}", f"v{__version__}"))


@router.get("/status")
async def status():
    return await get_status()


@router.get("/health")
async def health():
    return await check_proxy_health()


@router.get("/target-health")
async def target_health():
    return await check_target_health()


@router.get("/models")
async def models():
    return await get_models()


@router.get("/stats")
async def stats():
    return await get_status()


@router.post("/reset")
async def reset_session():
    """Clear all logs from the current session (today's log files)."""
    deleted = _logger.clear_logs()
    return {"status": "reset", "deleted": deleted}


def _without_image_payloads(entry):
    """Replace base64 images in a log's request body with placeholders.

    The table polls /api/logs every few seconds; shipping each image again
    (hundreds of KB apiece) would make the dashboard crawl. The modal fetches
    them on demand from /api/logs/{request_id}/images.
    """
    body, count = strip_images(entry.get("body"))
    if not count:
        return entry
    return {**entry, "body": body, "_image_count": count}


@router.get("/logs")
async def logs(limit: int = 20):
    return {"logs": [_without_image_payloads(e) for e in _logger.get_logs(limit=limit)]}


@router.get("/logs/{request_id}")
async def get_log(request_id: str):
    """Full log entry, images included (used by Save, since /api/logs strips them)."""
    entry = _logger.get_log(request_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    return entry


@router.get("/logs/{request_id}/images")
async def log_images(request_id: str):
    """Images sent in a logged request, as data URIs or URLs ready for <img src>."""
    entry = _logger.get_log(request_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    images = []
    for img in extract_images(entry.get("body")):
        src = img.get("url") or f"data:{img['mime']};base64,{img['data']}"
        images.append({"mime": img.get("mime"), "src": src, "location": img["location"]})
    return {"images": images}


@router.delete("/logs/{request_id}")
async def delete_log(request_id: str):
    """Delete a single traffic log entry by request id."""
    if _logger.delete_log(request_id):
        return JSONResponse(status_code=200, content={"status": "deleted"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


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


@router.get("/alert-settings")
async def get_alert_settings():
    """Return current context alert configuration."""
    config = get_config()
    return JSONResponse(status_code=200, content={
        "context_alert_enabled": config.context_alert_enabled,
        "context_alert_threshold": config.context_alert_threshold,
    })


@router.post("/alert-settings")
async def set_alert_settings(body: dict = Body(...)):
    """Update context alert configuration."""
    enabled = body.get("context_alert_enabled")
    threshold = body.get("context_alert_threshold")
    if threshold is not None and not (0 <= int(threshold) <= 100):
        return JSONResponse(status_code=400, content={"error": "threshold must be 0-100"})
    config = get_config()
    if enabled is not None:
        config.context_alert_enabled = bool(enabled)
    if threshold is not None:
        config.context_alert_threshold = int(threshold)
    save_config(config)
    return JSONResponse(status_code=200, content={
        "context_alert_enabled": config.context_alert_enabled,
        "context_alert_threshold": config.context_alert_threshold,
    })


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


@router.post("/intercept/{request_id}/pause")
async def intercept_pause(request_id: str):
    """Freeze the auto-forward timer (e.g. edit modal is open)."""
    if interceptor.pause_request(request_id):
        return JSONResponse(status_code=200, content={"status": "paused"})
    return JSONResponse(status_code=404, content={"status": "not_found"})


@router.post("/intercept/{request_id}/resume")
async def intercept_resume(request_id: str):
    """Resume the auto-forward timer."""
    if interceptor.resume_request(request_id):
        return JSONResponse(status_code=200, content={"status": "resumed"})
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
