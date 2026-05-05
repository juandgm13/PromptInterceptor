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
from .health import check_proxy_health, check_target_health, get_models, get_status

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
  .link-show{color:#4fc3f7;cursor:pointer;font-size:.8em;text-decoration:underline;
    background:none;border:none;padding:0}
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
        <button class="btn-add" onclick="loadFile()" style="background:#0d3b66;border-color:#1a5b8a;color:#4fc3f7">&#128193; Cargar</button>
        <button class="btn-add" onclick="saveLogs()" style="background:#1a4d1a;border-color:#2a6b2a;color:#5f5">&#8595; Guardar</button>
        <button class="btn-add" onclick="clearLogs()" style="background:#4d1a1a;border-color:#6b2a2a;color:#f88">&#x2715; Limpiar</button>
      </div>
    </div>
    <div id="file-mode-banner" style="display:none;background:#0a1e0a;border:1px solid #1a4b1a;border-radius:6px;padding:8px 14px;margin-bottom:10px;color:#5f5;font-size:.85em;align-items:center;justify-content:space-between">
      <span>&#128196; Archivo: <strong id="file-mode-name"></strong></span>
      <button onclick="exitFileMode()" style="background:#4d1a1a;color:#f88;border:1px solid #6b2a2a;border-radius:4px;padding:3px 8px;cursor:pointer;font-size:.8em">&#x2715; Volver a live</button>
    </div>
    <table>
      <thead><tr>
        <th>Time</th><th>Method</th><th>Path</th><th>Model</th><th>Prompt</th><th>Response</th><th>Status</th><th>Raw</th>
      </tr></thead>
      <tbody id="logs-body"><tr><td colspan="8" class="empty">Loading...</td></tr></tbody>
    </table>
  </div>

  <footer>
    PromptInterceptor Dashboard v0.1.0 &mdash;
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

<!-- Raw data modal -->
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal()">&#x2715;</button>
    <h3 class="modal-title" id="modal-title">Message Detail</h3>
    <div class="modal-split">
      <div class="modal-panel">
        <div class="modal-panel-title">Prompt</div>
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
  </div>
</div>

<script>
const _logsCache = {};
let _proxyStatusData = null;
let _fileMode = false;
let _loadedLogs = null;

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
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
  document.getElementById('modal-title').textContent =
    (l.method || '') + ' ' + (l.path || '') + ' — ' + (l.timestamp || l.response_timestamp || '');

  // --- Extract prompt ---
  let promptText = '';
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
      return `[${role}]\\n${content}`;
    }).join('\\n\\n---\\n\\n');
  } else if (l.body?.prompt) {
    promptText = String(l.body.prompt);
  }
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

  document.getElementById('modal-response').textContent = respText || '(sin respuesta)';

  document.getElementById('modal-overlay').style.display = 'block';
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
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
    tbody.innerHTML = '<tr><td colspan="8" class="empty">No requests yet.</td></tr>';
    return;
  }
  tbody.innerHTML = logs.slice().reverse().map((l, i) => {
    const cacheKey = l.request_id || i;
    _logsCache[cacheKey] = l;
    const ts = (l.timestamp || l.response_timestamp || '').slice(11,19) || '-';
    const method = l.method || '-';
    const path = l.path || '-';
    const model = l.body?.model || l.response_body?.model || '-';
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
    const respPreview = respText ? esc(respText.slice(0, 100)) + (respText.length > 100 ? '…' : '') : '<span class="empty">-</span>';
    const statusCode = l.status_code ? `<span class="badge ${l.status_code < 400 ? 'green' : 'red'}">${l.status_code}</span>` : '';
    const typeTag = l.type === 'response'
      ? `<span class="badge green">resp</span>`
      : `<span class="badge blue">req</span>`;
    return `<tr>
      <td>${ts}</td>
      <td>${esc(method)}</td>
      <td><code>${esc(path)}</code></td>
      <td><code>${esc(model)}</code></td>
      <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(preview)}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${respPreview}</td>
      <td>${typeTag} ${statusCode}</td>
      <td><button class="link-show" onclick="showRaw('${esc(String(cacheKey))}')">show</button></td>
    </tr>`;
  }).join('');
}

async function loadLogs() {
  if (_fileMode) { renderLogsTable(_loadedLogs || []); return; }
  try {
    const data = await fetchJSON('/api/logs?limit=50');
    renderLogsTable(data.logs || []);
  } catch(e) {
    const tb = document.getElementById('logs-body');
    tb.innerHTML = '<tr><td colspan="8" id="_logs-err"></td></tr>';
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
    if (!data.count) {
      section.style.display = 'none';
      return;
    }
    section.style.display = '';
    list.innerHTML = (data.pending || []).map(p => `
      <div class="pending-card">
        <h3>${esc(p.method)} ${esc(p.path)}</h3>
        <div class="pending-meta">ID: ${esc(p.request_id)}</div>
        <textarea id="ta-${esc(p.request_id)}" rows="6">${esc(JSON.stringify(p.body, null, 2))}</textarea>
        <div class="pending-actions">
          <button class="btn-forward" onclick="forwardRequest('${esc(p.request_id)}')">Forward</button>
          <button class="btn-forward" onclick="editRequest('${esc(p.request_id)}')">Edit &amp; Forward</button>
          <button class="btn-drop" onclick="dropRequest('${esc(p.request_id)}')">Drop</button>
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

async function clearLogs() {
  if (!confirm('¿Limpiar todos los logs de la sesión actual?')) return;
  await fetch('/api/reset', {method: 'POST'});
  Object.keys(_logsCache).forEach(k => delete _logsCache[k]);
  loadLogs();
  loadStatus();
}

function saveLogs() {
  const logs = Object.values(_logsCache);
  if (!logs.length) { alert('No hay logs para guardar.'); return; }
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
