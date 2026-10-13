# Supported version ranges for AI client binaries.
# This file is the single source of truth — launcher.py imports from here,
# and documentation references these values by citing this file path.

OPENCODE_MIN_VERSION = "1.17.4"
# Any 2.x release is supported; 3.0+ triggers the compatibility warning.
OPENCODE_MAX_VERSION = "2.999.999"

# First OpenCode release that discovers Ollama models on its own (via /api/tags
# and /api/show), so the launcher hides its model selector from this version on.
OPENCODE_AUTO_MODELS_VERSION = "2.0.0"
