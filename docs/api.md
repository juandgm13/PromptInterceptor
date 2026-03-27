# API Reference

## Proxy API (port 8080)

These endpoints replicate the Ollama API. Point your AI client to `http://localhost:8080` instead of `http://localhost:11434`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Proxy health check |
| `GET` | `/status` | Full system status (proxy + target + rules) |
| `GET` | `/dashboard` | Redirect to dashboard UI |
| `GET` | `/api/models` | List available models from Ollama |
| `POST` | `/api/chat` | Chat completion (non-streaming) |
| `POST` | `/api/generate` | Text generation (non-streaming) |
| `POST` | `/api/chat/stream` | Chat completion with streaming |
| `POST` | `/api/generate/stream` | Text generation with streaming |

### Example: Chat Request

```http
POST http://localhost:8080/api/chat
Content-Type: application/json

{
  "model": "llama3",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "options": {
    "temperature": 0.7
  }
}
```

### Example: Generate Request

```http
POST http://localhost:8080/api/generate
Content-Type: application/json

{
  "model": "mistral",
  "prompt": "Explain recursion in one sentence.",
  "stream": false
}
```

---

## Dashboard API (port 9090)

The dashboard web UI uses these endpoints internally. You can also call them directly.

### Status & Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Full status JSON |
| `GET` | `/api/health` | Proxy health |
| `GET` | `/api/target-health` | Ollama target health |
| `GET` | `/api/stats` | Alias for `/api/status` |

### Mode

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/api/mode` | `{"mode": "passthrough"}` | Change proxy mode |

Mode values: `"passthrough"` or `"intercept"`.

### Modifiers (Rules)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/rules` | List all modifiers |
| `POST` | `/api/rules` | Create a new modifier |
| `DELETE` | `/api/rules/{index}` | Delete a modifier by index |
| `GET` | `/api/enable-rule/{index}` | Enable a modifier |
| `GET` | `/api/disable-rule/{index}` | Disable a modifier |

#### Create modifier example

```http
POST http://localhost:9090/api/rules
Content-Type: application/json

{
  "match": {
    "path": "/api/chat",
    "jsonpath": "$.model",
    "value": ["llama3", "mistral"]
  },
  "replace": {
    "jsonpath": "$.model",
    "value": "deepseek-coder"
  }
}
```

### Logs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/logs?limit=20` | Recent traffic logs |
| `GET` | `/api/raw-logs` | All available logs |

### Intercept

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/intercept/pending` | List requests waiting for decision |
| `POST` | `/api/intercept/{id}/forward` | Forward unchanged |
| `POST` | `/api/intercept/{id}/edit` | Forward with modified body |
| `POST` | `/api/intercept/{id}/drop` | Drop the request |

#### Edit example

```http
POST http://localhost:9090/api/intercept/abc123/edit
Content-Type: application/json

{
  "model": "deepseek-coder",
  "messages": [{"role": "user", "content": "Modified content"}]
}
```
