from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyRuleQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class EnterprisePolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    policy_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    required_evidence_terms: list[str] = Field(min_length=1)
    match_mode: Literal["all", "any"] = "all"


class PolicyRuleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    rules: list[EnterprisePolicyRule] = Field(default_factory=list)