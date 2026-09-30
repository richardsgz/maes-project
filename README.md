# MAES

MAES (Memory-Augmented Enterprise Swarm) is a Python 3.12+ foundation for coordinating specialized agents over enterprise data.

## Architecture

- **Supervisor/Worker orchestration:** `core/orchestrator.py` defines a Pydantic-backed async LangGraph `StateGraph`. The Supervisor routes to the Extractor, Validator, Error node, or completion. Failed policy validation can trigger a bounded extraction retry with the prior findings as context; MCP errors and ungrounded evidence route to the terminal Error node and remain in structured state.
- **MCP policy validation:** `mcp_servers/data_server.py` exposes `query_enterprise_rules` with typed query and response models from `core/policy.py`. The demo server has an in-memory certification-validity rule; `agents/policy_client.py` calls it over stdio, and `agents/validator.py` compares returned requirements with extracted evidence. Replace the demo rule catalog with an authorized enterprise policy source before production use.
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

Invoke the graph asynchronously with the configured model provider and MCP stdio client:

```python
import asyncio
import sys

from mcp.client.stdio import StdioServerParameters

from agents.extractor import VendorRiskExtractor
from agents.policy_client import StdioMCPPolicyClient
from agents.validator import VendorRiskValidator
from core.model_factory import create_structured_model
from core.state import VendorRiskState
from core.orchestrator import build_graph

graph = build_graph(
	extractor=VendorRiskExtractor(create_structured_model()),
	validator=VendorRiskValidator(
		StdioMCPPolicyClient(
			StdioServerParameters(
				command=sys.executable,
				args=["-m", "mcp_servers.data_server"],
			)
		)
	),
)

async def main():
	state = VendorRiskState(
		original_document="Vendor security certification valid until 2027."
	)
	result = await graph.ainvoke(state.model_dump())
	print(VendorRiskState.model_validate(result).model_dump())

asyncio.run(main())
```
