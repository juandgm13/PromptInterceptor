# Modifiers (Rules)

Modifiers are JSONPath-based transformation rules applied to every request passing through the proxy. They can be defined in `config.json` or created at runtime via the dashboard.

## Rule Structure

```json
{
  "match": {
    "path": "/api/chat",
    "jsonpath": "$.model",
    "value": ["llama3", "mistral"]
  },
  "replace": {
    "jsonpath": "$.model",
    "value": "deepseek-coder"
  },
  "enabled": true
}
```

### `match` fields

| Field | Required | Description |
|-------|----------|-------------|
| `path` | No | Request path filter. Rule only applies to this endpoint (e.g. `"/api/chat"`). Omit to match all paths. |
| `jsonpath` | No | JSONPath expression to extract a value from the request body. If the path resolves to no value, the rule does not match. |
| `value` | No | One or more values the extracted field must equal. If omitted, any non-empty value matches. Can be a string or list. |

### `replace` fields

| Field | Required | Description |
|-------|----------|-------------|
| `jsonpath` | Yes | JSONPath expression pointing to the field to overwrite. |
| `value` | Yes | The new value to write. Can be any JSON type (string, number, object, array). |

## Examples

### Model Switching

Route specific models through a different backend model:

```json
{
  "match": {
    "path": "/api/chat",
    "jsonpath": "$.model",
    "value": ["llama3", "mistral", "qwen2.5"]
  },
  "replace": {
    "jsonpath": "$.model",
    "value": "deepseek-coder"
  }
}
```

### Temperature Override

Force a fixed temperature regardless of what the client requests:

```json
{
  "match": {
    "path": "/api/chat",
    "jsonpath": "$.options.temperature"
  },
  "replace": {
    "jsonpath": "$.options.temperature",
    "value": 0.5
  }
}
```

### System Prompt Injection

Prepend a system message to every chat request (requires the messages array to exist):

```json
{
  "match": {
    "path": "/api/chat",
    "jsonpath": "$.messages"
  },
  "replace": {
    "jsonpath": "$.messages[0].content",
    "value": "You are a helpful coding assistant. Always respond in English."
  }
}
```

## Rule Evaluation Order

Rules are evaluated in array order. Multiple rules can match the same request — each matching rule is applied in sequence. Later rules see the body as modified by earlier rules.

## Runtime Management

Rules can be managed without restarting via:

- **Dashboard UI** — use the Modifiers section to create, toggle, or delete rules
- **API** — `POST /api/rules`, `DELETE /api/rules/{index}`, `GET /api/enable-rule/{index}`

> **Note:** Rules added or deleted at runtime are not automatically persisted to `config.json`. They reset on the next server restart. To persist a rule permanently, add it to `config.json`.
