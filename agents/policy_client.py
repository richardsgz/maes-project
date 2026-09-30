import json
from collections.abc import Sequence
from typing import Protocol

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from core.policy import PolicyRuleQuery, PolicyRuleResponse


class MCPPolicyClientError(RuntimeError):
    """Raised when the MCP policy tool cannot return a valid response."""


class PolicyRuleClient(Protocol):
    async def lookup_many(
        self,
        queries: Sequence[PolicyRuleQuery],
    ) -> list[PolicyRuleResponse]: ...


class StdioMCPPolicyClient:
    """Call query_enterprise_rules over an MCP stdio connection."""

    def __init__(self, server: StdioServerParameters):
        self._server = server

    async def lookup_many(
        self,
        queries: Sequence[PolicyRuleQuery],
    ) -> list[PolicyRuleResponse]:
        if not queries:
            return []

        try:
            async with (
                stdio_client(self._server) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                responses = []
                for query in queries:
                    result = await session.call_tool(
                        "query_enterprise_rules",
                        arguments=query.model_dump(),
                    )
                    responses.append(self._parse_response(result))
                return responses
        except MCPPolicyClientError:
            raise
        except Exception as error:
            raise MCPPolicyClientError(f"MCP policy lookup failed: {error}") from error

    @staticmethod
    def _parse_response(result: object) -> PolicyRuleResponse:
        if getattr(result, "isError", False):
            raise MCPPolicyClientError("MCP query_enterprise_rules returned an error")

        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            return PolicyRuleResponse.model_validate(structured_content)

        text_blocks = [
            block.text
            for block in getattr(result, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        for text in text_blocks:
            try:
                return PolicyRuleResponse.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValueError):
                continue
        raise MCPPolicyClientError("MCP response did not contain a valid policy rule payload")