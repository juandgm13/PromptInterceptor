# Configuration Reference

Configuration is stored in `prompt_interceptor/config.json`. All fields are optional — missing fields use their default values.

## Full Example

```json
{
  "proxy_port": 8080,
  "proxy_host": "0.0.0.0",
  "target": "http://localhost:11434",
  "target_port": 11434,
  "mode": "passthrough",
  "timeout": 120,
  "log_dir": "logs",
  "log_size_limit": 1048576,
  "max_log_files": 10,
  "allow_origins": ["*"],
  "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  "allow_headers": ["*"],
  "dashboard_enabled": true,
  "dashboard_port": 9090,
  "model_names": ["llama3", "mistral", "deepseek-coder"],
  "context_size": 32768,
  "default_model": "qwen3.5",
  "rules": []
}
```

## Fields

### Server

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `proxy_port` | int | `8080` | Port the proxy listens on |
| `proxy_host` | string | `"0.0.0.0"` | Host to bind the proxy |
| `target` | string | `"http://localhost:11434"` | Ollama server URL |
| `target_port` | int | `11434` | Ollama port (informational) |
| `timeout` | int | `120` | Request timeout in seconds |

### Mode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"passthrough"` | `"passthrough"` or `"intercept"` |

### Context Size

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context_size` | int | `32768` | Context window size passed as `OLLAMA_NUM_CTX` when launching Ollama from the desktop launcher. Does not affect the proxy itself. |

Available options: `4096`, `8192`, `16384`, `32768`, `65536`, `131072`, `262144` (4k–256k).

### Logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_dir` | string | `"logs"` | Directory for log files |
| `log_size_limit` | int | `1048576` | Max log file size in bytes (1 MB) |
| `max_log_files` | int | `10` | Max number of log files to keep |

### CORS

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_origins` | list | `["*"]` | Allowed CORS origins |
| `allow_methods` | list | `["GET","POST","PUT","DELETE","OPTIONS"]` | Allowed HTTP methods |
| `allow_headers` | list | `["*"]` | Allowed request headers |

### Dashboard

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dashboard_enabled` | bool | `true` | Enable the web dashboard |
| `dashboard_port` | int | `9090` | Dashboard port |

### Models

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_names` | list | `["llama3","mistral"]` | Known model names (used as fallback list in the launcher when Ollama is not reachable) |
| `default_model` | string | `""` | Model pre-selected in the launcher. Saved automatically when you click Start. |

### Rules

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rules` | list | `[]` | List of modifier rules. See [rules.md](rules.md) |
