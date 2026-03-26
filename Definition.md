# PyProxy - Definition & Requirements

## What is PyProxy?

**PyProxy** is a FastAPI-based proxy service that intercepts HTTP traffic between AI applications and Ollama, enabling:
- Model switching (changing which model handles each request)
- Temperature tuning (adjusting response randomness)
- System prompt injection
- Full request/response logging
- Streaming support
- CORS configuration

## Architecture

```
AI Application → PyProxy (proxy_port) → Ollama (target)
```

PyProxy sits between the client and Ollama:
1. Client sends request to `localhost:proxy_port`
2. PyProxy intercepts, applies rules (if enabled)
3. Redirects to Ollama at `target:11434`
4. Ollama responds
5. PyProxy applies rules (if enabled)
6. Redirects response to client

## Core Features

### Model Switcher
- Match incoming request (by path/header)
- Replace model name in request/response
- Support for streaming responses

### Temperature Tuner
- Match temperature values in request
- Replace with custom value
- Apply on-the-fly during proxy

### System Prompt Injection
- Add system prompts to requests
- Customizable per-rule
- Optional/forced toggles

### Logging
- Full request/response logging
- JSON format for easy parsing
- File-based storage
- Configurable max files

### CORS
- Configurable allowed origins
- Support for credentials
- Custom headers support

### Dashboard
- Web UI for rule management
- Status monitoring
- Statistics display
- Rule enable/disable

## Requirements

### Python
- Python 3.9+

### Dependencies
```
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
httpx>=0.23.0
pydantic>=2.5.0
python-multipart>=0.0.6
aiofiles>=23.0.0
orjson>=3.9.0
```

### Ollama
- Running Ollama server at `http://localhost:11434` (or custom port)
- At least one model loaded

## Configuration

Create `ollama_proxy/config.json`:

```json
{
  "proxy_port": 8080,
  "target": "http://localhost:11434",
  "mode": "intercept",
  "rules": [],
  "dashboard_enabled": true,
  "dashboard_port": 9090
}
```

### Config Fields

| Field | Type | Description |
|-------|------|-------------|
| proxy_port | int | Port for proxy server |
| target | str | Ollama target URL |
| mode | str | Intercept or forward |
| rules | list | List of rules |
| dashboard_enabled | bool | Enable dashboard |
| dashboard_port | int | Dashboard port |
| log_dir | str | Log directory |
| max_log_files | int | Max log files |

### Rules Format

```json
{
  "match": {
    "path": "/api/chat",
    "value": ["mistral", "llama"]
  },
  "replace": {
    "jsonpath": "$.model",
    "value": "mistral"
  }
}
```

### Match Options

| Path | Description |
|------|-------------|
| path | Match request path |
| header | Match request header |
| value | Value to match |
| operation | Equals/Contains/NotEquals/NotContains |

### Replace Options

| Path | Description |
|------|-------------|
| jsonpath | JSONPath expression |
| value | Value to replace |
| optional | Whether rule is optional |
| forced | Whether to force replacement |

## Usage

### Start Proxy

```bash
pip install -r ollama_proxy/requirements.txt
python ollama_proxy/main.py
```

Or with uvicorn:

```bash
uvicorn ollama_proxy.main:app --host 0.0.0.0 --port 8080
```

### API Endpoints

#### Proxy Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Chat completion |
| POST | `/api/generate` | Text generation |
| POST | `/api/chat/stream` | Stream chat |
| POST | `/api/generate/stream` | Stream generate |

#### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Get proxy status |
| GET | `/api/models` | List models |
| GET | `/dashboard` | Dashboard |

## Use Cases

### Model Routing
Route different clients to different models based on header/path.

### A/B Testing
Test different models or configurations.

### Load Balancing
Distribute load across multiple Ollama instances.

### Content Filtering
Block or modify responses based on rules.

### Quality Control
Enforce temperature settings for critical applications.

## Project Structure

```
PyProxy/
├── .gitignore
├── CLAUDE.md           # Project documentation
├── Definition.md       # This file
├── README.md           # Public README
└── ollama_proxy/       # Main application
    ├── __init__.py
    ├── config.py       # Configuration management
    ├── models.py       # Pydantic models
    ├── rules_engine.py # Rule engine
    ├── proxy.py        # Core proxy logic
    ├── health.py       # Health checks
    ├── cors_middleware.py
    ├── dashboard.py    # Web dashboard
    ├── logger.py       # Logging module
    ├── main.py         # Entry point
    ├── requirements.txt
    └── config.json     # Configuration
```

## License

MIT
