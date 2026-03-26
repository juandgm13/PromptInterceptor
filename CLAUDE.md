# PyProxy - Ollama Traffic Interceptor & Model Switcher

This is a FastAPI-based proxy that intercepts traffic between AI applications and Ollama, enabling model switching and temperature tuning.

## Architecture

```
AI Application
    ↓ (proxy_port)
PyProxy [FastAPI]
    ↓ (redirects/forwards)
Ollama (target)
```

## Core Modules

### `ollama_proxy/main.py`
- Main entry point
- Creates FastAPI app
- Registers routes
- Starts proxy & dashboard servers

### `ollama_proxy/proxy.py`
- Core proxy logic
- Request handling (chat, generate)
- Streaming support
- Response handling

### `ollama_proxy/rules_engine.py`
- Rule management
- Request/response transformation
- Model switching logic
- Temperature tuning

### `ollama_proxy/config.py`
- Configuration management
- Loads config.json
- Provides config getter/setter

### `ollama_proxy/health.py`
- Health check endpoints
- Target health checks
- Status API

### `ollama_proxy/cors_middleware.py`
- CORS configuration
- Origin/headers/metrics

### `ollama_proxy/dashboard.py`
- Web dashboard UI
- Rule management
- Statistics display

### `ollama_proxy/logger.py`
- Request/response logging
- Log file management
- Statistics collection

## Key Files

- `ollama_proxy/config.json` - Configuration
- `ollama_proxy/main.py` - Entry point
- `ollama_proxy/requirements.txt` - Dependencies
- `pyproject.toml` - Project metadata

## Running

```bash
pip install -r ollama_proxy/requirements.txt
python ollama_proxy/main.py
```

## Configuration

Edit `ollama_proxy/config.json`:
- `proxy_port`: Port for proxy server
- `target`: Ollama target URL
- `rules`: List of matching/replacement rules
- `dashboard_enabled`: Enable dashboard
- `dashboard_port`: Dashboard port

## API Endpoints

- `/api/chat` - Chat completion
- `/api/generate` - Text generation
- `/api/chat/stream` - Stream chat
- `/api/generate/stream` - Stream generate
- `/health` - Health check
- `/status` - Get status
- `/dashboard` - Web dashboard

## Rules Example

```json
[
  {
    "match": {"value": "mistral"},
    "replace": {"jsonpath": "$.model", "value": "mistral"}
  }
]
```
