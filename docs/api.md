# API Reference

## Proxy API (port 8080)

These endpoints replicate the Ollama API. Point your AI client to `http://localhost:8080` instead of `http://localhost:11434`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Proxy health check |
| `GET` | `/status` | Full system status (proxy + target + rules) |
| `GET` | `/dashboard` | Redirect to dashboard UI |
| `GET` | `/api/models` | List available models from Ollama |
| `POST` | `/api/chat` | Chat completion (non-streaming) — rules applied in intercept mode |
| `POST` | `/api/generate` | Text generation (non-streaming) — rules applied in intercept mode |
| `POST` | `/api/chat/stream` | Chat completion with streaming — rules applied in intercept mode |
| `POST` | `/api/generate/stream` | Text generation with streaming — rules applied in intercept mode |
| `POST` | `/v1/messages` | Anthropic-compatible messages endpoint — forwarded to Ollama |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions — forwarded to Ollama |
| `ANY` | `/{path}` | Pass-through: any other path (e.g. `/api/tags`, `/api/ps`) is forwarded to Ollama as-is and logged to disk |

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
| `GET` | `/api/logs?limit=20` | Recent traffic logs (default limit: 20). Base64 images in request bodies are replaced by placeholders such as `"<image #1, 356 KB>"` and the entry gets `_image_count` |
| `GET` | `/api/logs/{request_id}` | One full log entry, images included |
| `GET` | `/api/logs/{request_id}/images` | Images sent in that request, ready for `<img src>` |
| `DELETE` | `/api/logs/{request_id}` | Delete one log entry |
| `GET` | `/api/raw-logs` | All available logs (no limit), images included |
| `POST` | `/api/reset` | Clear all logs and reset session statistics |

#### Images (vision models)

Images are detected in the three request formats the proxy handles:

| Format | Location |
|--------|----------|
| Ollama native (`/api/chat`, `/api/generate`) | `messages[].images[]` / `images[]` (bare base64; the MIME type is sniffed) |
| OpenAI-compatible (`/v1/chat/completions`) | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` |
| Anthropic-compatible (`/v1/messages`) | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}` |

```json
GET /api/logs/69f41bc4992e/images

{"images": [{"mime": "image/png", "src": "data:image/png;base64,iVBOR...", "location": "messages[1]"}]}
```

Image URLs (`http(s)://`) are returned as-is in `src`, with `mime: null`.

#### Log entry fields

| Field | Description |
|-------|-------------|
| `status_code` | Response status. Missing while the request is still in flight. Proxy-side failures also set it: `408` (Ollama read timeout), `502` (cannot connect), `500` (other error), `499` (client disconnected mid-stream), with `response_body: {"error": "..."}` |
| `_image_count` | Number of images in the request (only in `/api/logs`, when the request has images) |
| `_num_ctx_override` | `{"client": 8192, "sent": 32768}` when the proxy replaced the client's `options.num_ctx` with `context_size` (`client` is `null` if the client sent none). `body` then shows the body actually forwarded |

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
