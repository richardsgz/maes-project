from core.policy import EnterprisePolicyRule, PolicyRuleQuery
from core.state import AgentError, PolicyFinding, ValidationStatus, VendorRiskState

from .policy_client import MCPPolicyClientError, PolicyRuleClient


class VendorRiskValidator:
    """Retrieve applicable policy rules and compare them to extracted evidence."""

    def __init__(self, policy_client: PolicyRuleClient):
        self._policy_client = policy_client

    async def run(self, state: VendorRiskState) -> VendorRiskState:
        if not state.needs_policy_lookup:
            return self._updated_state(state, validation_status=ValidationStatus.PASSED)

        if not state.extracted_entities:
            return self._with_error(
                state,
                AgentError(
                    stage="validator",
                    code="no_entities_to_validate",
                    message="Policy lookup was requested but no entities were extracted",
                ),
                ValidationStatus.ERROR,
            )

        query_text = "; ".join(state.policy_queries)
        queries = [
            PolicyRuleQuery(
                query=query_text,
                entity_type=entity.entity_type,
                entity_value=entity.value,
                evidence=entity.evidence,
            )
            for entity in state.extracted_entities
        ]

        try:
            responses = await self._policy_client.lookup_many(queries)
            if len(responses) != len(queries):
                raise ValueError("MCP returned a different number of results than requested")
        except (MCPPolicyClientError, ValueError) as error:
            return self._with_error(
                state,
                AgentError(
                    stage="mcp",
                    code="policy_lookup_failed",
                    message="Enterprise policy lookup failed",
                    details=str(error),
                ),
                ValidationStatus.ERROR,
            )

        findings: list[PolicyFinding] = []
        errors: list[AgentError] = []
        for entity, response in zip(state.extracted_entities, responses, strict=True):
            applicable_rules = [
                rule
                for rule in response.rules
                if rule.entity_type.casefold() == entity.entity_type.casefold()
            ]
            if not applicable_rules:
                findings.append(
                    PolicyFinding(
                        policy_id="NO_APPLICABLE_POLICY",
                        description="No matching enterprise rule was returned; manual review is needed.",
                        entity_value=entity.value,
                        compliant=None,
                    )
                )
                errors.append(
                    AgentError(
                        stage="validator",
                        code="no_applicable_policy",
                        message=f"No enterprise policy matched entity type {entity.entity_type!r}",
                    )
                )
                continue

            findings.extend(
                self._evaluate_rule(rule, entity.value, entity.evidence)
                for rule in applicable_rules
            )

        if any(finding.compliant is False for finding in findings):
            status = ValidationStatus.FAILED
        elif any(finding.compliant is None for finding in findings):
            status = ValidationStatus.PENDING
        else:
            status = ValidationStatus.PASSED

        return self._updated_state(
            state,
            policy_findings=findings,
            validation_status=status,
            error_logs=[*state.error_logs, *errors],
        )

    @staticmethod
    def _evaluate_rule(
        rule: EnterprisePolicyRule,
        entity_value: str,
        evidence: str,
    ) -> PolicyFinding:
        evidence_text = evidence.casefold()
        matches = [term.casefold() in evidence_text for term in rule.required_evidence_terms]
        compliant = all(matches) if rule.match_mode == "all" else any(matches)
        return PolicyFinding(
            policy_id=rule.policy_id,
            description=rule.description,
            entity_value=entity_value,
            compliant=compliant,
        )

    @staticmethod
    def _updated_state(state: VendorRiskState, **updates: object) -> VendorRiskState:
        values = state.model_dump()
        values.update(updates)
        return VendorRiskState.model_validate(values)

    @classmethod
    def _with_error(
        cls,
        state: VendorRiskState,
        error: AgentError,
        status: ValidationStatus,
    ) -> VendorRiskState:
        return cls._updated_state(
            state,
            validation_status=status,
            error_logs=[*state.error_logs, error],
        )