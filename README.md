# MAES

MAES (Memory-Augmented Enterprise Swarm) is a Python 3.12+ foundation for coordinating specialized agents over enterprise data.

## Architecture

- **Supervisor/Worker orchestration:** `core/orchestrator.py` defines a LangGraph `StateGraph`. The Supervisor inspects the current request, extraction, and validation state, then routes to the Extractor or Validator. Workers return state updates to the Supervisor; failed validation can trigger another extraction attempt up to the configured retry limit.
- **MCP tool execution:** `mcp_servers/data_server.py` exposes `query_enterprise_rules` through the Model Context Protocol. The current tool is a safe placeholder; connect it to an authorized rules source before using it for real enterprise decisions.
- **Persistent memory graph:** `memory/` is the home for a NetworkX-backed knowledge graph. Persisting entities, relationships, provenance, and validated corrections there will let later runs retrieve prior outcomes and improve extraction. The storage and retrieval adapter is an extension point, not implemented in this initial scaffold.
- **Model access:** `google-genai` is included for Gemini-backed worker implementations. The scaffold does not make API calls or require credentials to run its smoke test.

## Getting started

Install dependencies and development tools with `uv`:

```bash
uv sync --dev
```

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
from core.orchestrator import build_graph

graph = build_graph()
result = graph.invoke({"request": "Extract the key requirements from this document."})
```
