import json

import pytest

from core.state import (
    ComplianceEntity,
    PolicyFinding,
    ValidationStatus,
    VendorRiskState,
)
from memory.graph import NetworkXMemoryGraph


def build_state(
    *,
    status: ValidationStatus,
    evidence: str = "ISO 27001 certification valid until 2027",
) -> VendorRiskState:
    return VendorRiskState(
        original_document=f"Acme security record: {evidence}",
        extracted_entities=[
            ComplianceEntity(
                entity_type="certification",
                value="ISO 27001",
                evidence=evidence,
                confidence=0.95,
            )
        ],
        extraction_summary="A vendor certification was identified.",
        policy_findings=[
            PolicyFinding(
                policy_id="VR-CERT-001",
                description="Evidence must state the certification expiry.",
                entity_value="ISO 27001",
                compliant=status is ValidationStatus.PASSED,
            )
        ],
        validation_status=status,
    )


def test_memory_persists_and_reloads_without_storing_source_document(tmp_path) -> None:
    path = tmp_path / "memory" / "assessments.json"
    memory = NetworkXMemoryGraph(path)
    state = build_state(status=ValidationStatus.PASSED)

    memory.record_assessment(state)
    restored = NetworkXMemoryGraph(path)
    context = restored.retrieve_for_document("New vendor ISO 27001 file")

    assert restored.assessment_count == 1
    assert len(context) == 1
    assert context[0].validation_status is ValidationStatus.PASSED
    assert context[0].policy_findings[0].compliant is True
    assert state.original_document not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 1


def test_later_pass_links_to_failed_assessment_as_correction(tmp_path) -> None:
    path = tmp_path / "assessments.json"
    memory = NetworkXMemoryGraph(path)
    memory.record_assessment(
        build_state(
            status=ValidationStatus.FAILED,
            evidence="ISO 27001 certification",
        )
    )
    memory.record_assessment(build_state(status=ValidationStatus.PASSED))

    restored = NetworkXMemoryGraph(path)
    context = restored.retrieve_for_document("ISO 27001 vendor")

    assert restored.assessment_count == 2
    assert restored.correction_count == 1
    assert context[0].validation_status is ValidationStatus.PASSED
    assert "prior failed assessment" in context[0].correction_note


def test_memory_rejects_unvalidated_outcomes(tmp_path) -> None:
    memory = NetworkXMemoryGraph(tmp_path / "assessments.json")

    with pytest.raises(ValueError, match="Only completed policy assessments"):
        memory.record_assessment(build_state(status=ValidationStatus.ERROR))

    assert memory.assessment_count == 0