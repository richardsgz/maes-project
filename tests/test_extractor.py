from types import SimpleNamespace
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from agents.extractor import UnverifiedEvidenceError, VendorRiskExtractor
from agents.model_provider import GeminiStructuredModel
from core.state import ComplianceEntity, ExtractionPlan, VendorRiskState

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class FakeStructuredModel:
    def __init__(self, response: BaseModel):
        self.response = response
        self.prompt = ""

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        self.prompt = prompt
        return response_model.model_validate(self.response)


class FakeGeminiModels:
    def __init__(self, parsed: BaseModel | None):
        self.parsed = parsed
        self.request: dict[str, object] = {}

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(parsed=self.parsed)


class FakeGeminiClient:
    def __init__(self, parsed: BaseModel | None):
        self.models = FakeGeminiModels(parsed)


def test_extractor_records_grounded_entities_and_policy_action() -> None:
    source = "The vendor holds ISO 27001 certification."
    plan = ExtractionPlan(
        observation_summary="The source states the vendor holds a certification.",
        entities=[
            ComplianceEntity(
                entity_type="certification",
                value="ISO 27001",
                evidence="holds ISO 27001 certification",
                confidence=0.98,
            )
        ],
        needs_policy_lookup=True,
        policy_queries=["What evidence is required for ISO 27001 certification?"],
    )
    model = FakeStructuredModel(plan)

    result = VendorRiskExtractor(model).run(VendorRiskState(original_document=source))

    assert result.extracted_entities == plan.entities
    assert result.extraction_summary == plan.observation_summary
    assert result.needs_policy_lookup is True
    assert result.policy_queries == plan.policy_queries
    assert "untrusted evidence" in model.prompt
    assert "private chain-of-thought" in model.prompt


def test_extractor_rejects_evidence_not_in_source() -> None:
    plan = ExtractionPlan(
        observation_summary="A certification was reported.",
        entities=[
            ComplianceEntity(
                entity_type="certification",
                value="SOC 2",
                evidence="SOC 2 Type II certified",
                confidence=0.9,
            )
        ],
        needs_policy_lookup=False,
    )

    with pytest.raises(UnverifiedEvidenceError):
        VendorRiskExtractor(FakeStructuredModel(plan)).run(
            VendorRiskState(original_document="No certification is listed.")
        )


def test_extraction_plan_requires_queries_when_lookup_is_needed() -> None:
    with pytest.raises(ValidationError):
        ExtractionPlan(
            observation_summary="An assessment claim needs policy review.",
            needs_policy_lookup=True,
        )


def test_gemini_adapter_uses_injected_client_and_requested_schema() -> None:
    expected = ExtractionPlan(
        observation_summary="No compliance claims were found.",
        needs_policy_lookup=False,
    )
    client = FakeGeminiClient(expected)
    model = GeminiStructuredModel(model_name="test-model", client=client)

    result = model.generate_structured(
        prompt="Extract the claims.",
        response_model=ExtractionPlan,
    )

    assert result == expected
    assert client.models.request["model"] == "test-model"
    assert client.models.request["contents"] == "Extract the claims."
    assert client.models.request["config"].response_schema is ExtractionPlan


def test_gemini_adapter_rejects_missing_structured_output() -> None:
    model = GeminiStructuredModel(client=FakeGeminiClient(None))

    with pytest.raises(RuntimeError, match="no parsed structured response"):
        model.generate_structured(prompt="Extract the claims.", response_model=ExtractionPlan)