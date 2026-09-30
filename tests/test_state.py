import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from core.state import (
    AgentError,
    ComplianceEntity,
    ValidationStatus,
    VendorRiskState,
)


def test_vendor_risk_state_has_isolated_defaults() -> None:
    first = VendorRiskState(original_document="Vendor assessment")
    second = VendorRiskState(original_document="Another assessment")

    first.error_logs.append(
        AgentError(stage="mcp", code="timeout", message="Policy lookup timed out")
    )

    assert first.validation_status is ValidationStatus.PENDING
    assert len(first.error_logs) == 1
    assert second.error_logs == []


def test_compliance_entity_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ComplianceEntity(
            entity_type="certification",
            value="ISO 27001",
            evidence="Listed in the assessment",
            confidence=1.2,
        )


def test_vendor_risk_state_works_as_langgraph_schema() -> None:
    graph = StateGraph(VendorRiskState)
    graph.add_node("noop", lambda _state: {})
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)

    result = graph.compile().invoke({"original_document": "Vendor assessment"})

    assert VendorRiskState.model_validate(result).original_document == "Vendor assessment"