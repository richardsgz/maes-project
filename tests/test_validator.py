import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from mcp.client.stdio import StdioServerParameters

from agents.policy_client import MCPPolicyClientError, StdioMCPPolicyClient
from agents.validator import VendorRiskValidator
from core.policy import EnterprisePolicyRule, PolicyRuleQuery, PolicyRuleResponse
from core.state import (
    AgentError,
    ComplianceEntity,
    ValidationStatus,
    VendorRiskState,
)


class FakePolicyClient:
    def __init__(self, responses: list[PolicyRuleResponse] | Exception):
        self.responses = responses
        self.queries: Sequence[PolicyRuleQuery] = []

    async def lookup_many(
        self,
        queries: Sequence[PolicyRuleQuery],
    ) -> list[PolicyRuleResponse]:
        self.queries = queries
        if isinstance(self.responses, Exception):
            raise self.responses
        return self.responses


def build_state(evidence: str = "ISO 27001 certification is current.") -> VendorRiskState:
    return VendorRiskState(
        original_document=f"The vendor's {evidence}",
        extracted_entities=[
            ComplianceEntity(
                entity_type="certification",
                value="ISO 27001",
                evidence=evidence,
                confidence=0.98,
            )
        ],
        needs_policy_lookup=True,
        policy_queries=["Check certification validity"],
    )


def certification_response() -> PolicyRuleResponse:
    return PolicyRuleResponse(
        query="Check certification validity",
        rules=[
            EnterprisePolicyRule(
                policy_id="VR-CERT-001",
                description="Security certifications must include evidence of current validity.",
                entity_type="certification",
                required_evidence_terms=["valid until", "expires", "expiration", "current"],
                match_mode="any",
            )
        ],
    )


def test_validator_evaluates_evidence_against_returned_policy() -> None:
    client = FakePolicyClient([certification_response()])

    result = asyncio.run(VendorRiskValidator(client).run(build_state()))

    assert result.validation_status is ValidationStatus.PASSED
    assert result.policy_findings[0].policy_id == "VR-CERT-001"
    assert result.policy_findings[0].compliant is True
    assert client.queries[0].entity_value == "ISO 27001"


def test_validator_marks_missing_required_evidence_non_compliant() -> None:
    client = FakePolicyClient([certification_response()])

    result = asyncio.run(
        VendorRiskValidator(client).run(build_state("ISO 27001 certification"))
    )

    assert result.validation_status is ValidationStatus.FAILED
    assert result.policy_findings[0].compliant is False


def test_validator_records_no_matching_policy_as_inconclusive() -> None:
    response = PolicyRuleResponse(query="Check certification validity", rules=[])
    result = asyncio.run(VendorRiskValidator(FakePolicyClient([response])).run(build_state()))

    assert result.validation_status is ValidationStatus.PENDING
    assert result.policy_findings[0].compliant is None
    assert result.error_logs[-1].code == "no_applicable_policy"


def test_validator_normalizes_mcp_failure_into_state_error() -> None:
    client = FakePolicyClient(MCPPolicyClientError("server unavailable"))

    result = asyncio.run(VendorRiskValidator(client).run(build_state()))

    assert result.validation_status is ValidationStatus.ERROR
    assert result.error_logs[-1] == AgentError(
        stage="mcp",
        code="policy_lookup_failed",
        message="Enterprise policy lookup failed",
        details="server unavailable",
    )


def test_validator_skips_mcp_when_lookup_is_not_needed() -> None:
    client = FakePolicyClient([])
    state = build_state().model_copy(update={"needs_policy_lookup": False})

    result = asyncio.run(VendorRiskValidator(client).run(state))

    assert result.validation_status is ValidationStatus.PASSED
    assert client.queries == []


def test_stdio_mcp_policy_client_round_trip() -> None:
    query = PolicyRuleQuery(
        query="Check certification validity",
        entity_type="certification",
        entity_value="ISO 27001",
        evidence="ISO 27001 certification is current",
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.data_server"],
        cwd=Path.cwd(),
    )

    result = asyncio.run(StdioMCPPolicyClient(server).lookup_many([query]))

    assert result[0].query == query.query
    assert result[0].rules[0].policy_id == "VR-CERT-001"