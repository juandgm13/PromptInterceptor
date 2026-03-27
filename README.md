<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PromptInterceptor</title>
<style>
  body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0 20px 60px;
    line-height: 1.6;
  }
  .container { max-width: 900px; margin: 0 auto; }
  header { text-align: center; padding: 48px 0 32px; }
  header img { max-width: 420px; width: 100%; }
  h2 {
    color: #ffffff;
    border-bottom: 1px solid #222;
    padding-bottom: 8px;
    margin-top: 40px;
  }
  h3 { color: #cccccc; margin-top: 24px; }
  a { color: #4fc3f7; text-decoration: none; }
  a:hover { text-decoration: underline; }
  ul, ol { padding-left: 24px; }
  li { margin: 6px 0; }
  pre {
    background: #1a1a2e;
    border-radius: 6px;
    padding: 14px 16px;
    overflow-x: auto;
    font-size: .9em;
    line-height: 1.5;
  }
  code {
    background: #1a1a2e;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: .88em;
    color: #4fc3f7;
  }
  pre code {
    background: transparent;
    padding: 0;
    color: #e0e0e0;
  }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th, td { text-align: left; padding: 8px 14px; border-bottom: 1px solid #222; font-size: .92em; }
  th { color: #888; font-weight: normal; }
  .badge {
    display: inline-block;
    background: #0d3b66;
    color: #4fc3f7;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: .8em;
    margin: 2px;
  }
  .docs-links { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
  .docs-links a {
    background: #16213e;
    border: 1px solid #2a2a4e;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: .9em;
  }
  footer { text-align: center; margin-top: 60px; color: #444; font-size: .82em; }
</style>
</head>
<body>
<div class="container">

<header>
  <img src="res/PromptInterceptor_Logo.png" alt="PromptInterceptor">
</header>

<p style="text-align:center;color:#888;font-size:1.05em;margin-top:0">
  Ollama Traffic Interceptor for AI Clients
</p>

<div class="docs-links">
  <a href="docs/architecture.md">Architecture</a>
  <a href="docs/api.md">API Reference</a>
  <a href="docs/configuration.md">Configuration</a>
  <a href="docs/rules.md">Modifiers / Rules</a>
</div>

<h2>Description</h2>

<p>
  PromptInterceptor is a local HTTP proxy that sits between your AI client
  (Claude Code, Open Code) and an Ollama server. It lets you inspect, modify,
  and intercept every prompt and response in real time — without touching the
  client or the model.
</p>

<h2>Features</h2>

<ul>
  <li><strong>Model switching</strong> — automatically reroute requests from one model to another using JSONPath rules</li>
  <li><strong>Temperature tuning</strong> — override temperature or any other option in every request</li>
  <li><strong>Live prompt viewer</strong> — see every request and response as it flows through the proxy</li>
  <li><strong>Intercept mode</strong> — pause any request, edit the JSON body, then forward or drop it</li>
  <li><strong>Modifier management</strong> — create, enable, disable and delete rules from the dashboard without restarting</li>
  <li><strong>Context size control</strong> — set <code>OLLAMA_NUM_CTX</code> on the Ollama server at launch time (4k&#8211;256k)</li>
  <li><strong>Full traffic logging</strong> — all requests and responses saved as JSON files with rotation</li>
  <li><strong>Desktop launcher</strong> — tkinter window that starts Ollama, your AI client, and the proxy in one click</li>
  <li><strong>Streaming support</strong> — transparent proxying of NDJSON streaming responses</li>
</ul>

<h2>Installation</h2>

<pre><code># Clone the repository
git clone https://github.com/your-user/PromptInterceptor.git
cd PromptInterceptor

# Install dependencies
pip install -r prompt_interceptor/requirements.txt</code></pre>

<p>Requirements: Python 3.9+, Ollama installed and available in PATH.</p>

<h2>Usage</h2>

<h3>Launch with the desktop window</h3>

<pre><code>python -m prompt_interceptor</code></pre>

<p>
  A configuration window will appear. Select your context size and AI client,
  then click <strong>Start</strong>. Three things happen automatically:
</p>

<ol>
  <li>A terminal opens running <code>ollama serve</code> with the selected context size</li>
  <li>A terminal opens running your chosen AI client (Claude Code or Open Code)</li>
  <li>The proxy starts and the dashboard opens in your browser at <code>http://localhost:9090</code></li>
</ol>

<h3>Use case: Ollama + Claude Code</h3>

<ol>
  <li>Make sure Ollama is installed: <code>ollama --version</code></li>
  <li>Make sure Claude Code is installed: <code>claude --version</code></li>
  <li>Run <code>python -m prompt_interceptor</code></li>
  <li>In the launcher: select <em>Ollama</em> as server, choose context size, select <em>Claude Code</em> as client</li>
  <li>Click <strong>Start</strong></li>
  <li>In Claude Code, configure the Ollama endpoint to <code>http://localhost:8080</code> instead of <code>http://localhost:11434</code></li>
  <li>All prompts now flow through PromptInterceptor &#8212; open the dashboard to see them</li>
</ol>

<h2>Configuration</h2>

<p>Edit <code>prompt_interceptor/config.json</code>:</p>

<pre><code>{
  "proxy_port": 8080,
  "target": "http://localhost:11434",
  "mode": "passthrough",
  "context_size": 4096,
  "dashboard_enabled": true,
  "dashboard_port": 9090,
  "rules": []
}</code></pre>

<p>See <a href="docs/configuration.md">docs/configuration.md</a> for all available fields.</p>

<h2>API Endpoints</h2>

<h3>Proxy (port 8080)</h3>

<table>
  <tr><th>Method</th><th>Path</th><th>Description</th></tr>
  <tr><td><span class="badge">GET</span></td><td><code>/health</code></td><td>Health check</td></tr>
  <tr><td><span class="badge">GET</span></td><td><code>/status</code></td><td>System status</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/chat</code></td><td>Chat completion</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/generate</code></td><td>Text generation</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/chat/stream</code></td><td>Streaming chat</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/generate/stream</code></td><td>Streaming generate</td></tr>
</table>

<h3>Dashboard (port 9090)</h3>

<table>
  <tr><th>Method</th><th>Path</th><th>Description</th></tr>
  <tr><td><span class="badge">GET</span></td><td><code>/</code></td><td>Dashboard UI</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/mode</code></td><td>Change proxy mode</td></tr>
  <tr><td><span class="badge">GET</span></td><td><code>/api/rules</code></td><td>List modifiers</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/rules</code></td><td>Create modifier</td></tr>
  <tr><td><span class="badge">DEL</span></td><td><code>/api/rules/{index}</code></td><td>Delete modifier</td></tr>
  <tr><td><span class="badge">GET</span></td><td><code>/api/logs</code></td><td>Recent traffic logs</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/intercept/{id}/forward</code></td><td>Forward paused request</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/intercept/{id}/edit</code></td><td>Edit and forward</td></tr>
  <tr><td><span class="badge">POST</span></td><td><code>/api/intercept/{id}/drop</code></td><td>Drop request</td></tr>
</table>

<p>See <a href="docs/api.md">docs/api.md</a> for full details and examples.</p>

<footer>
  PromptInterceptor &#8212; MIT License
</footer>

</div>
</body>
</html>
