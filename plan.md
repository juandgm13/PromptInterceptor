# PyProxy Implementation Plan

## Context

Implement a complete HTTP Interceptor Proxy for Ollama from scratch. The proxy will sit between clients and an Ollama model server, providing passthrough mode, intercept mode with manual editing, automatic rule-based modifications, and complete traffic logging.

---

## Phase 1: Project Setup & Core Configuration

### Task 1.1: Create Project Directory Structure
- Create `ollama_proxy/` directory
- Create `ollama_proxy/models/` subdirectory
- Create `ollama_proxy/utils/` subdirectory
- Create `logs/` directory for traffic logs

### Task 1.2: Implement `config.py`
- Define configuration dataclass/models
- Load settings from `config.json`
- Support configuration for:
  - `proxy_port`: Default 8080
  - `target`: Default http://localhost:11434
  - `mode`: "passthrough" or "intercept"
  - `rules`: Array of rule configurations
  - `timeout`: Default 120 seconds

### Task 1.3: Create `config.json`
- Initial configuration file with default values
- Example rules for model switching and parameter tuning
- Enable/disable options for logging

---

## Phase 2: Logging Infrastructure

### Task 2.1: Implement `logger.py`
- Create log directory structure (`logs/YYYY-MM-DD/`)
- Generate unique request IDs for tracking
- Log requests with: timestamp, method, path, headers, body
- Log responses with same structure
- Support JSON logging format
- Handle large payloads efficiently (truncation if needed)

---

## Phase 3: Request/Response Models

### Task 3.1: Create `models/request_model.py`
- Define Pydantic models for request validation
- Models for:
  - `ChatRequest`: Ollama /api/chat payload
  - `GenerateRequest`: Ollama /api/generate payload
  - `Response` models for responses

---

## Phase 4: Rule Engine

### Task 4.1: Implement `rules_engine.py`
- Parse JSONPath expressions using `jsonpath-ng`
- Match requests based on path and JSON field patterns
- Apply replacements (model switching, parameter tuning)
- Support deep JSON path targeting (e.g., `$.options.temperature`)
- Handle both request and response rules
- Process rules asynchronously (non-blocking)

---

## Phase 5: Core Proxy Logic

### Task 5.1: Implement `proxy.py`
- HTTP server using FastAPI
- Request reception and body parsing
- Apply rule engine transformations
- Forward requests to target Ollama server
- Stream responses back to client
- Handle concurrent requests with async/await
- Preserve `Transfer-Encoding: chunked` for streaming
- Implement configurable timeouts

---

## Phase 6: Interceptor (Manual Editing)

### Task 6.1: Implement `interceptor.py`
- Pause request flow in intercept mode
- Display request to user (blocking until response)
- Options:
  - `[f] forward`: Send original request
  - `[e] edit`: Modify request
  - `[d] drop`: Cancel request
- Edit interface for modifying JSON body and headers
- Log intercepted requests separately

---

## Phase 7: CLI Interface

### Task 7.1: Create CLI using Typer
- Commands:
  - `proxy start`: Start proxy server
  - `proxy stop`: Stop proxy server
  - `proxy mode <mode>`: Switch between passthrough/intercept
  - `proxy reload-rules`: Reload configuration rules
  - `proxy logs`: Display logged traffic
  - `proxy status`: Show current proxy status

---

## Phase 8: Dashboard (Web UI)

### Task 8.1: Create Dashboard Endpoint `/dashboard`
- Display captured requests with filtering
- Form to edit rules (save to `config.json`)
- Request replay functionality
- Current mode and configuration display
- Simple HTML/JavaScript interface

---

## Phase 9: Streaming Support

### Task 9.1: Implement `utils/streaming_utils.py`
- Handle chunked transfer encoding
- Stream responses without blocking
- Preserve stream flow and order
- Handle streaming timeouts

---

## Phase 10: JSON Utility Functions

### Task 10.1: Implement `utils/json_utils.py`
- JSON parsing helpers
- JSON modification utilities
- Large payload handling
- Safe JSONPath evaluation

---

## Phase 11: Main Application

### Task 11.1: Implement `main.py`
- FastAPI app initialization
- Route definitions
- Integrate all components:
  - Config loader
  - Rule engine
  - Logger
  - Interceptor
- CORS configuration
- Health check endpoint
- Dashboard routing

---

## Phase 12: Testing

### Task 12.1: Create Basic Tests
- Test passthrough mode with curl
- Test rule application
- Test intercept mode flow
- Test streaming support
- Test concurrent requests
- Test logging functionality

---

## Verification Steps

1. **Setup**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create config.json**:
   ```bash
   copy config.example.json config.json
   ```

3. **Start proxy**:
   ```bash
   uvicorn ollama_proxy.main:app --host 0.0.0.0 --port 8080
   ```

4. **Test with curl**:
   ```bash
   curl -X POST http://localhost:8080/api/chat \
     -H "Content-Type: application/json" \
     -d '{"model": "llama3", "messages": [{"role": "user", "content": "hello"}]}'
   ```

5. **Verify rule application**:
   - Check if model was changed to "deepseek-coder"
   - Check logs directory for captured traffic

---

## Critical Files Summary

| File | Purpose | Phase |
|------|---------|-------|
| `config.py` | Configuration management | 1.2 |
| `config.json` | Configuration file | 1.3 |
| `logger.py` | Traffic logging | 2.1 |
| `models/request_model.py` | Pydantic models | 3.1 |
| `rules_engine.py` | Rule processing | 4.1 |
| `proxy.py` | Core proxy logic | 5.1 |
| `interceptor.py` | Manual editing | 6.1 |
| `main.py` | FastAPI app | 11.1 |
| `cli.py` | CLI interface | 7.1 |
| `utils/json_utils.py` | JSON helpers | 10.1 |
| `utils/streaming_utils.py` | Streaming support | 9.1 |
| `dashboard.html` | Web UI | 8.1 |
| `requirements.txt` | Dependencies | - |

---

## Dependencies

```txt
fastapi
uvicorn
httpx
pydantic
python-json-logger
jsonpath-ng
rich
typer
```

---

## End-to-End Test Flow

1. Start proxy in passthrough mode
2. Send request to `/api/chat` via curl
3. Verify request appears in `logs/2026-03-26/`
4. Switch to intercept mode
5. Send another request
6. Verify manual editing interface appears
7. Configure rules in `config.json`
8. Verify automatic model switching works
9. Test streaming response handling
10. Access `/dashboard` to view requests

---

## Notes

- Start with simple, working implementation
- Add streaming support early (critical requirement)
- Ensure non-blocking behavior throughout
- Handle edge cases: large JSON, timeouts, malformed requests
- Keep code clean and well-documented
- Use async/await for all HTTP operations
