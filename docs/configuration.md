# Configuration Reference

Configuration is stored in `prompt_interceptor/config.json`. All fields are optional — missing fields use their default values.

## Full Example

```json
{
  "debug": false,
  "debug_intercept": false,
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
  "model_names": ["llama3", "mistral"],
  "context_size": 32768,
  "default_model": "",
  "client_work_dir": "",
  "python_app_path": "",
  "python_app_command": "python main.py",
  "python_app_use_venv": false,
  "python_app_env_var": "OLLAMA_HOST",
  "windows_cmd_mode": false,
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
| `health_timeout` | float | `5.0` | Timeout for health-check requests to Ollama |

### Mode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"passthrough"` | `"passthrough"` (no rules) or `"intercept"` (rules applied, requests can be paused) |

### Debug

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `debug` | bool | `false` | Enable uvicorn debug/reload mode and FastAPI debug output |
| `debug_intercept` | bool | `false` | Enable detailed `INFO`-level traces for the intercept pipeline (mode checks, rule evaluation, pause/resolve events). Useful to verify that intercept mode is active and rules are being applied. |

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
| `max_log_files` | int | `10` | Max number of log files to keep before rotating |

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
| `model_names` | list | `["llama3","mistral"]` | Fallback model list used in the launcher when Ollama is not reachable |
| `default_model` | string | `""` | Model pre-selected in the launcher. Saved automatically when you open the dashboard. |

### Launcher — Python App

These fields are saved by the desktop launcher when you configure a Python App client. You can also set them directly in `config.json`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `client_work_dir` | string | `""` | Working directory for the Open Code client terminal |
| `python_app_path` | string | `""` | Working directory for the Python app |
| `python_app_command` | string | `"python main.py"` | Command used to launch the Python app |
| `python_app_use_venv` | bool | `false` | If true, auto-detects `.venv` or `venv` in `python_app_path` and uses that interpreter |
| `python_app_env_var` | string | `"OLLAMA_HOST"` | Env var name the app uses for the Ollama host URL |

### Windows CMD mode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `windows_cmd_mode` | bool | `false` | Translate Unix bash commands in LLM tool calls to Windows CMD equivalents before forwarding to the client |

When `windows_cmd_mode` is `true`, the proxy intercepts every `Bash` tool call in LLM responses and rewrites the `command` field so it runs on Windows CMD. The translation is transparent — neither the model nor the AI client needs any changes.

#### Command translation table

| Bash | Windows CMD |
|------|-------------|
| `ls` | `dir` |
| `ls -l` | `dir` |
| `ls -la` / `ls -al` | `dir /a` |
| `ls -a path/` | `dir /a path/` |
| `pwd` | `cd` |
| `clear` | `cls` |
| `which cmd` | `where cmd` |
| `cat file` | `type file` |
| `touch file` | `type nul > file` |
| `rm file` | `del file` |
| `rm -rf dir/` | `rd /s /q dir/` |
| `rm -r dir/` | `rd /s /q dir/` |
| `cp src dst` | `copy src dst` |
| `mv src dst` | `move src dst` |
| `mkdir -p a/b/c` | `mkdir a/b/c` |
| `grep pattern file` | `findstr pattern file` |
| `grep -r pattern dir/` | `findstr /s pattern dir/` |
| `grep -i pattern file` | `findstr /i pattern file` |
| `find . -name '*.py'` | `dir /s /b *.py` |
| `export VAR=val` | `set VAR=val` |
| `echo $VAR` | `echo %VAR%` |
| `python3 script.py` | `python script.py` |
| `pip3 install pkg` | `pip install pkg` |
| `sudo <cmd>` | `<cmd>` (prefix stripped) |
| `chmod …` / `chown …` | `rem chmod …` (commented out) |

Compound operators (`&&`, `\|\|`, `\|`) are preserved as-is. The semicolon separator (`;`) is converted to `&&`. Multi-line commands are joined with `&&`.

### Rules

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rules` | list | `[]` | List of modifier rules. See [rules.md](rules.md) |
