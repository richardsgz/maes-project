from mcp.server.fastmcp import FastMCP

mcp = FastMCP("maes-data-server")


@mcp.tool()
def query_enterprise_rules(query: str) -> str:
    """Look up enterprise rules relevant to a natural-language query.

    Replace this placeholder with an authorized enterprise rules data source.
    """
    return f"Enterprise rules lookup is not configured yet. Received query: {query}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
