<p align="center">
  <img src="res/PromptInterceptor_Logo.png" alt="PromptInterceptor" width="420">
</p>

<p align="center">Ollama Traffic Interceptor for AI Clients</p>

[Architecture](docs/architecture.md) · [API Reference](docs/api.md) · [Configuration](docs/configuration.md) · [Modifiers / Rules](docs/rules.md)

## Description

PromptInterceptor is a local HTTP proxy that sits between your AI client (Claude Code, Open Code) and an Ollama server. It lets you inspect, modify, and intercept every prompt and response in real time — without touching the client or the model.

## Features

- **Model switching** — automatically reroute requests from one model to another using JSONPath rules
- **Temperature tuning** — override temperature or any other option in every request
- **Live prompt viewer** — see every request and response as it flows through the proxy
- **Intercept mode** — pause any request, edit the JSON body, then forward or drop it
- **Modifier management** — create, enable, disable and delete rules from the dashboard without restarting
- **Context size control** — set `OLLAMA_NUM_CTX` on the Ollama server at launch time (4k–256k)
- **Model selector** — pick from models already downloaded in Ollama, populated automatically after launch
- **Python app support** — connect any Python app that uses Ollama by specifying its entry point and the Ollama host env var
- **Full traffic logging** — all requests and responses saved as JSON files with rotation
- **Desktop launcher** — guided 3-step tkinter window (Ollama → Client → Proxy)
- **Streaming support** — transparent proxying of NDJSON streaming responses

## Installation

```bash
# Clone the repository
git clone https://github.com/your-user/PromptInterceptor.git
cd PromptInterceptor

# Install dependencies
pip install -r prompt_interceptor/requirements.txt
```

Requirements: Python 3.9+, Ollama installed and available in PATH.

## Usage

### Launch with the desktop window

```bash
python -m prompt_interceptor
```

The launcher uses a three-step sequential flow. Each step unlocks the next.

### Step 1 — Ollama Server

| Field / Button | Description |
|---|---|
| **Context Size** | Sets `OLLAMA_NUM_CTX` for the Ollama server (4k–256k) |
| **Model** | Populated automatically from Ollama's downloaded models after launch |
| **Launch Ollama Server** | Opens a terminal running `ollama serve`. Fetches available models. Unlocks Step 2. |

### Step 2 — AI Client

| Field / Button | Description |
|---|---|
| **AI Client** | Auto-detected clients (`claude`, `opencode`) plus _Python App (Ollama)_ for any Python app that uses Ollama |
| **Work Dir** _(Claude Code / Open Code)_ | Directory where the client terminal opens. Use **Browse…** to pick a folder. |
| **App Path** _(Python App)_ | Path to the Python app's entry point (`.py` or executable). Use **Browse…** to pick a file. |
| **Ollama Env Var** _(Python App)_ | Name of the environment variable the app uses to configure the Ollama host (e.g. `OLLAMA_HOST`). Set to `http://localhost:8080` so the app goes through the proxy. |
| **Launch Client** | Opens a terminal running the selected client with the correct configuration. Unlocks Step 3. |

### Step 3 — Proxy

Click **Start** to launch the proxy server and open the dashboard in your browser at `http://localhost:9090`.

### Use case: Ollama + Claude Code

1. Make sure Ollama is installed: `ollama --version`
2. Make sure Claude Code is installed: `claude --version`
3. Run `python -m prompt_interceptor`
4. Step 1: choose context size → click **Launch Ollama Server**
5. Step 2: select model, select _Claude Code_, choose a work directory → click **Launch Client**
6. Step 3: click **Start**
7. The launcher sets `ANTHROPIC_BASE_URL=http://localhost:8080` and passes `--model <model>` automatically. All prompts now flow through PromptInterceptor — open the dashboard to see them.

### Use case: Ollama + custom Python app

1. Run `python -m prompt_interceptor`
2. Step 1: choose context size → click **Launch Ollama Server**
3. Step 2: select _Python App (Ollama)_, fill in **App Path** and **Ollama Env Var** → click **Launch Client**
4. Step 3: click **Start**
5. The app will use `http://localhost:8080` as its Ollama host, routing all traffic through the proxy

## Configuration

Edit `prompt_interceptor/config.json`:

```json
{
  "proxy_port": 8080,
  "target": "http://localhost:11434",
  "mode": "passthrough",
  "context_size": 32768,
  "default_model": "qwen3.5",
  "dashboard_enabled": true,
  "dashboard_port": 9090,
  "rules": []
}
```

See [docs/configuration.md](docs/configuration.md) for all available fields.

## API Endpoints

### Proxy (port 8080)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/status` | System status |
| POST | `/api/chat` | Chat completion |
| POST | `/api/generate` | Text generation |
| POST | `/api/chat/stream` | Streaming chat |
| POST | `/api/generate/stream` | Streaming generate |
| ANY | `/{path}` | Pass-through: any unmatched path is forwarded to Ollama and logged |

### Dashboard (port 9090)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| POST | `/api/mode` | Change proxy mode |
| GET | `/api/rules` | List modifiers |
| POST | `/api/rules` | Create modifier |
| DELETE | `/api/rules/{index}` | Delete modifier |
| GET | `/api/logs` | Recent traffic logs |
| POST | `/api/intercept/{id}/forward` | Forward paused request |
| POST | `/api/intercept/{id}/edit` | Edit and forward |
| POST | `/api/intercept/{id}/drop` | Drop request |

See [docs/api.md](docs/api.md) for full details and examples.

---

PromptInterceptor — MIT License
