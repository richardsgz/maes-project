from mcp.server.fastmcp import FastMCP

from core.policy import EnterprisePolicyRule, PolicyRuleResponse

mcp = FastMCP("maes-data-server")

_ENTERPRISE_RULES = [
    EnterprisePolicyRule(
        policy_id="VR-CERT-001",
        description="Security certifications must include evidence of current validity.",
        entity_type="certification",
        required_evidence_terms=["valid until", "expires", "expiration", "current"],
        match_mode="any",
    ),
]


@mcp.tool()
def query_enterprise_rules(
    query: str,
    entity_type: str,
    entity_value: str,
    evidence: str,
) -> PolicyRuleResponse:
    """Return policy rules applicable to an extracted vendor-risk entity."""
    del entity_value, evidence
    rules = [
        rule
        for rule in _ENTERPRISE_RULES
        if rule.entity_type.casefold() == entity_type.casefold()
    ]
    return PolicyRuleResponse(query=query, rules=rules)


if __name__ == "__main__":
    mcp.run(transport="stdio")
