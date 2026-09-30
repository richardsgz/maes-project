from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    compliant: bool


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
    policy_findings: list[PolicyFinding] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    error_logs: list[AgentError] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=2, ge=0)