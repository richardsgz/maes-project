# MAES

MAES (Memory-Augmented Enterprise Swarm) is a Python 3.12+ foundation for coordinating specialized agents over enterprise data.

## Architecture

- **Supervisor/Worker orchestration:** `core/orchestrator.py` defines a LangGraph `StateGraph`. The Supervisor inspects the current request, extraction, and validation state, then routes to the Extractor or Validator. Workers return state updates to the Supervisor; failed validation can trigger another extraction attempt up to the configured retry limit.
- **MCP policy validation:** `mcp_servers/data_server.py` exposes `query_enterprise_rules` with typed query and response models from `core/policy.py`. The demo server has an in-memory certification-validity rule; `agents/policy_client.py` calls it over stdio, and `agents/validator.py` compares returned requirements with extracted evidence. MCP failures become structured state errors. Replace the demo rule catalog with an authorized enterprise policy source before production use. Graph-level error routing is planned for the orchestration phase.
- **Persistent memory graph:** `memory/` is the home for a NetworkX-backed knowledge graph. Persisting entities, relationships, provenance, and validated corrections there will let later runs retrieve prior outcomes and improve extraction. The storage and retrieval adapter is an extension point, not implemented in this initial scaffold.
- **Provider-neutral model access:** Agent workers depend on the `StructuredModel` protocol in `agents/model_provider.py`. `core/model_factory.py` selects a registered adapter from explicit configuration, while concrete integrations live under `agents/providers/`. Gemini is an optional adapter, not a core dependency.

## Getting started

Install dependencies and development tools with `uv`:

```bash
uv sync --dev
```

The core project and tests do not require a model provider or API credentials. To enable Gemini, install its optional dependency and configure provider selection:

```bash
uv sync --dev --extra google
export MAES_MODEL_PROVIDER=google
export MAES_MODEL_NAME=gemini-2.5-flash
```

The application composition layer can then construct a provider with `create_structured_model()`. Alternative adapters can implement `StructuredModel` and register a factory with `register_model_provider()`.

Run the test suite:

```bash
uv run pytest
```

Run the MCP server over stdio:

```bash
uv run python -m mcp_servers.data_server
```

Invoke the graph from Python:

```python
from agents.extractor import VendorRiskExtractor
from core.model_factory import create_structured_model
from core.state import VendorRiskState

extractor = VendorRiskExtractor(create_structured_model())
state = VendorRiskState(original_document="Vendor assessment document text")
result = extractor.run(state)
```
