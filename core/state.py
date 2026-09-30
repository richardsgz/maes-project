from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ValidationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ComplianceEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    entity_type: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class PolicyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    policy_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    compliant: bool | None


class ExtractionPlan(BaseModel):
    """Evidence-backed extraction and proposed next action from the Extractor."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    observation_summary: str = Field(min_length=1)
    entities: list[ComplianceEntity] = Field(default_factory=list)
    needs_policy_lookup: bool
    policy_queries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lookup_plan(self) -> Self:
        if self.needs_policy_lookup and not self.policy_queries:
            raise ValueError("Policy lookup requires at least one query")
        if not self.needs_policy_lookup and self.policy_queries:
            raise ValueError("Policy queries require needs_policy_lookup=true")
        return self


class AgentError(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stage: Literal["extractor", "mcp", "validator", "orchestrator"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    recoverable: bool = True
    details: str | None = None


class VendorRiskState(BaseModel):
    """Validated state shared between Vendor Risk Assessment graph nodes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    original_document: str = Field(min_length=1)
    extracted_entities: list[ComplianceEntity] = Field(default_factory=list)
    extraction_summary: str | None = None
    needs_policy_lookup: bool = False
    policy_queries: list[str] = Field(default_factory=list)
    policy_findings: list[PolicyFinding] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    error_logs: list[AgentError] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=2, ge=0)
    next_step: Literal["Supervisor", "Extractor", "Validator", "Error", "END"] = "Supervisor"
    error_state_reached: bool = False