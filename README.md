# PyProxy

**PyProxy** - Ollama Traffic Interceptor & Model Switcher

A proxy service that intercepts traffic between AI applications and Ollama, enabling:
- **Rule-based model switching** between requests
- **Temperature tuning** on responses
- **System prompt injection**
- **Full request/response logging**
- **Streaming support**
- **CORS configuration**

---

## Features

- 🔀 **Model switching** - Inject different model names per request
- 🌡️ **Temperature tuning** - Adjust temperature on responses
- 📝 **System prompts** - Add system prompts automatically
- 📊 **Full logging** - Track all requests/responses
- 🔄 **Streaming** - Supports streaming chat responses
- 🛡️ **CORS support** - Configurable CORS headers
- 🎛️ **Dashboard** - Web UI for rule management

---

## Installation

```bash
# Install dependencies
pip install -r ollama_proxy/requirements.txt

# Or with uv
uv pip install -r ollama_proxy/requirements.txt
```

---

## Configuration

Edit `ollama_proxy/config.json`:

```json
{
  "proxy_port": 8080,
  "target": "http://localhost:11434",
  "mode": "intercept",
  "rules": [
    {
      "match": { "value": "mistral" },
      "replace": { "jsonpath": "$.model", "value": "mistral" }
    },
    {
      "match": { "value": "0.7" },
      "replace": { "jsonpath": "$.options.temperature", "value": "0.5" }
    }
  ],
  "dashboard_enabled": true,
  "dashboard_port": 9090
}
```

---

## Usage

```bash
# Run the proxy
python ollama_proxy/main.py

# Or with uvicorn
uvicorn ollama_proxy.main:app --host 0.0.0.0 --port 8080
```

---

## API Endpoints

### Proxy Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Chat completion |
| POST | `/api/generate` | Text generation |
| POST | `/api/chat/stream` | Stream chat |
| POST | `/api/generate/stream` | Stream generate |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Get proxy status |
| GET | `/api/models` | List models |
| GET | `/dashboard` | Dashboard |

---

## Rule Format

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

---

## License

MIT
