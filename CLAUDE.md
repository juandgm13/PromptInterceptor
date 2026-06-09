# PromptInterceptor - Ollama Traffic Interceptor for AI Clients

This is a FastAPI-based proxy that intercepts traffic between AI applications and Ollama, enabling model switching, temperature tuning, and real-time prompt inspection.

## Architecture

```
AI Client (Claude Code / Open Code)
    ↓ (proxy_port: 8080)
PromptInterceptor [FastAPI]
    ↓ (redirects/forwards)
Ollama (target: 11434)
```

## Core Modules

### `prompt_interceptor/launcher.py`
- tkinter desktop launcher
- Server/client/context-size selection
- Starts Ollama with OLLAMA_NUM_CTX env var
- Launches AI client terminal
- Starts proxy + opens dashboard in browser

### `prompt_interceptor/__main__.py`
- Entry point for `python -m prompt_interceptor`
- Calls `launcher.launch()`

### `prompt_interceptor/main.py`
- Main entry point
- Creates FastAPI app
- Registers routes
- Starts proxy & dashboard servers

### `prompt_interceptor/proxy.py`
- Core proxy logic
- Request handling (chat, generate)
- Streaming support
- Response handling

### `prompt_interceptor/rules_engine.py`
- Rule management (add, enable, disable, delete)
- Request/response transformation
- Model switching logic
- Temperature tuning

### `prompt_interceptor/config.py`
- Configuration management
- Loads config.json
- Provides config getter/setter
- Fields include: proxy_port, target, mode, context_size, rules, dashboard_*

### `prompt_interceptor/health.py`
- Health check endpoints
- Target health checks
- Status API

### `prompt_interceptor/cors_middleware.py`
- CORS configuration
- Origin/headers/metrics

### `prompt_interceptor/dashboard.py`
- Web dashboard UI (port 9090)
- Modifier management (create/enable/disable/delete)
- Mode switching (passthrough/intercept)
- Live prompt viewer
- Intercept queue (forward/edit/drop)
- Statistics display

### `prompt_interceptor/logger.py`
- Request/response logging
- Log file management
- Statistics collection

## Key Files

- `prompt_interceptor/config.json` - Configuration
- `prompt_interceptor/main.py` - Proxy entry point
- `prompt_interceptor/launcher.py` - Desktop launcher
- `prompt_interceptor/requirements.txt` - Dependencies
- `pyproject.toml` - Project metadata
- `docs/` - Technical documentation

## Running

```bash
pip install -r prompt_interceptor/requirements.txt

# Desktop launcher (recommended)
python -m prompt_interceptor

# Proxy server only (headless)
python prompt_interceptor/main.py
```

## Configuration

Edit `prompt_interceptor/config.json`:
- `proxy_port`: Port for proxy server (default: 8080)
- `target`: Ollama target URL (default: http://localhost:11434)
- `mode`: `"passthrough"` or `"intercept"`
- `context_size`: Context window for Ollama server (passed as OLLAMA_NUM_CTX)
- `rules`: List of matching/replacement modifiers
- `dashboard_enabled`: Enable web dashboard
- `dashboard_port`: Dashboard port (default: 9090)

## API Endpoints

### Proxy (port 8080)
- `/api/chat` - Chat completion
- `/api/generate` - Text generation
- `/api/chat/stream` - Stream chat
- `/api/generate/stream` - Stream generate
- `/v1/messages` - Anthropic-compatible messages
- `/v1/chat/completions` - OpenAI-compatible chat completions
- `/health` - Health check
- `/status` - Get status
- `/dashboard` - Redirect to dashboard

### Dashboard (port 9090)
- `/` - Web UI
- `/api/status` - Full status JSON
- `/api/health` - Proxy health
- `/api/target-health` - Ollama target health
- `/api/mode` - Change proxy mode (POST)
- `/api/rules` - List/create modifiers
- `/api/rules/{index}` - Delete modifier (DELETE)
- `/api/enable-rule/{index}` - Enable modifier
- `/api/disable-rule/{index}` - Disable modifier
- `/api/logs` - Recent traffic logs
- `/api/raw-logs` - All logs (no limit)
- `/api/reset` - Clear logs and reset session (POST)
- `/api/intercept/pending` - Pending intercepts
- `/api/intercept/{id}/forward|edit|drop` - Resolve intercept

## Modifier Example

```json
{
  "match": {"path": "/api/chat", "jsonpath": "$.model", "value": ["llama3"]},
  "replace": {"jsonpath": "$.model", "value": "deepseek-coder"}
}
```

## Documentation

- `docs/architecture.md` - System architecture and request flow
- `docs/api.md` - Full API reference with examples
- `docs/configuration.md` - All configuration fields
- `docs/rules.md` - Modifier/rule creation guide
