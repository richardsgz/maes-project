import asyncio
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from agents.extractor import VendorRiskExtractor
from agents.policy_client import MCPPolicyClientError
from agents.validator import VendorRiskValidator
from core.orchestrator import build_graph
from core.policy import EnterprisePolicyRule, PolicyRuleQuery, PolicyRuleResponse
from core.state import (
    ComplianceEntity,
    ExtractionPlan,
    ValidationStatus,
    VendorRiskState,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class SequenceModel:
    def __init__(self, plans: list[ExtractionPlan]):
        self.plans = plans
        self.prompts: list[str] = []

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        self.prompts.append(prompt)
        return response_model.model_validate(self.plans.pop(0))


class FixedPolicyClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.queries: Sequence[PolicyRuleQuery] = []

    async def lookup_many(
        self,
        queries: Sequence[PolicyRuleQuery],
    ) -> list[PolicyRuleResponse]:
        self.queries = queries
        if self.fail:
            raise MCPPolicyClientError("MCP server unavailable")
        return [
            PolicyRuleResponse(
                query=query.query,
                rules=[
                    EnterprisePolicyRule(
                        policy_id="VR-CERT-001",
                        description="Certification evidence must include an expiry date.",
                        entity_type="certification",
                        required_evidence_terms=["valid until"],
                    )
                ],
            )
            for query in queries
        ]


def make_plan(evidence: str) -> ExtractionPlan:
    return ExtractionPlan(
        observation_summary="The source lists a vendor security certification.",
        entities=[
            ComplianceEntity(
                entity_type="certification",
                value="ISO 27001",
                evidence=evidence,
                confidence=0.98,
            )
        ],
        needs_policy_lookup=True,
        policy_queries=["Check security certification validity"],
    )


def make_graph(
    plans: list[ExtractionPlan],
    *,
    policy_client: FixedPolicyClient | None = None,
):
    return build_graph(
        extractor=VendorRiskExtractor(SequenceModel(plans)),
        validator=VendorRiskValidator(policy_client or FixedPolicyClient()),
    )


def invoke(graph: object, state: VendorRiskState) -> VendorRiskState:
    result = asyncio.run(graph.ainvoke(state.model_dump()))
    return VendorRiskState.model_validate(result)


def test_graph_extracts_and_validates_assessment() -> None:
    state = VendorRiskState(
        original_document="ISO 27001 certification valid until 2027.",
    )
    graph = make_graph([make_plan("ISO 27001 certification valid until 2027")])

    result = invoke(graph, state)

    assert result.validation_status is ValidationStatus.PASSED
    assert result.policy_findings[0].compliant is True
    assert result.retry_count == 0
    assert result.error_state_reached is False


def test_graph_retries_failed_validation_with_finding_context() -> None:
    document = "ISO 27001 certification, valid until 2027."
    model = SequenceModel(
        [
            make_plan("ISO 27001 certification"),
            make_plan("ISO 27001 certification, valid until 2027"),
        ]
    )
    graph = build_graph(
        extractor=VendorRiskExtractor(model),
        validator=VendorRiskValidator(FixedPolicyClient()),
    )

    result = invoke(graph, VendorRiskState(original_document=document))

    assert result.validation_status is ValidationStatus.PASSED
    assert result.retry_count == 1
    assert "Prior policy findings" in model.prompts[1]


def test_graph_routes_exhausted_retries_to_error_state() -> None:
    graph = make_graph([make_plan("ISO 27001 certification")] * 2)
    state = VendorRiskState(
        original_document="ISO 27001 certification, valid until 2027.",
        max_retries=1,
    )

    result = invoke(graph, state)

    assert result.validation_status is ValidationStatus.FAILED
    assert result.retry_count == 1
    assert result.error_state_reached is True
    assert result.error_logs[-1].code == "max_retries_exhausted"


def test_graph_routes_hallucinated_evidence_to_error_state() -> None:
    graph = make_graph([make_plan("ISO 27001 valid until 2030")])
    state = VendorRiskState(original_document="ISO 27001 certification.")

    result = invoke(graph, state)

    assert result.validation_status is ValidationStatus.ERROR
    assert result.error_state_reached is True
    assert result.error_logs[-1].code == "unverified_evidence"
    assert result.error_logs[-1].recoverable is False


def test_graph_routes_mcp_failure_to_error_state() -> None:
    graph = make_graph(
        [make_plan("ISO 27001 certification valid until 2027")],
        policy_client=FixedPolicyClient(fail=True),
    )
    state = VendorRiskState(
        original_document="ISO 27001 certification valid until 2027.",
    )

    result = invoke(graph, state)

    assert result.validation_status is ValidationStatus.ERROR
    assert result.error_state_reached is True
    assert result.error_logs[-1].code == "policy_lookup_failed"