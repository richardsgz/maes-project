from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.extractor import UnverifiedEvidenceError, VendorRiskExtractor
from agents.validator import VendorRiskValidator
from core.state import AgentError, ValidationStatus, VendorRiskState

NextStep = Literal["Extractor", "Validator", "Error", "END"]


class GraphUpdate(TypedDict, total=False):
    extracted_entities: list[Any]
    extraction_summary: str | None
    needs_policy_lookup: bool
    policy_queries: list[str]
    policy_findings: list[Any]
    validation_status: ValidationStatus
    error_logs: list[AgentError]
    retry_count: int
    next_step: NextStep
    error_state_reached: bool


def _validated_state(state: VendorRiskState | dict[str, Any]) -> VendorRiskState:
    return VendorRiskState.model_validate(state)


def _next_step(state: VendorRiskState | dict[str, Any]) -> NextStep:
    return _validated_state(state).next_step


def supervisor(state: VendorRiskState | dict[str, Any]) -> GraphUpdate:
    """Route graph progress, retries, and terminal failures from typed state."""
    current = _validated_state(state)

    if current.validation_status is ValidationStatus.ERROR:
        return {"next_step": "Error"}
    if current.validation_status is ValidationStatus.PASSED:
        return {"next_step": "END"}
    if current.validation_status is ValidationStatus.FAILED:
        if current.retry_count < current.max_retries:
            return {
                "next_step": "Extractor",
                "retry_count": current.retry_count + 1,
                "validation_status": ValidationStatus.PENDING,
            }
        exhausted_error = AgentError(
            stage="orchestrator",
            code="max_retries_exhausted",
            message="Validation still fails after the configured retry limit",
            recoverable=False,
        )
        return {
            "next_step": "Error",
            "error_logs": [*current.error_logs, exhausted_error],
        }

    if current.extraction_summary is None:
        return {"next_step": "Extractor"}
    return {"next_step": "Validator"}


def build_graph(
    *,
    extractor: VendorRiskExtractor,
    validator: VendorRiskValidator,
):
    """Build the async Supervisor/Worker graph with injected model and MCP workers."""

    def run_extractor(state: VendorRiskState | dict[str, Any]) -> GraphUpdate:
        current = _validated_state(state)
        try:
            result = extractor.run(current)
        except UnverifiedEvidenceError as error:
            return _record_failure(
                current,
                stage="extractor",
                code="unverified_evidence",
                message="Extractor returned evidence not present in the source document",
                details=str(error),
                recoverable=False,
            )
        except Exception as error:  # noqa: BLE001 - worker failures must enter graph error state
            return _record_failure(
                current,
                stage="extractor",
                code="extraction_failed",
                message="Extractor failed to produce a valid assessment",
                details=str(error),
            )
        return result.model_dump()

    async def run_validator(state: VendorRiskState | dict[str, Any]) -> GraphUpdate:
        current = _validated_state(state)
        try:
            result = await validator.run(current)
        except Exception as error:  # noqa: BLE001 - worker failures must enter graph error state
            return _record_failure(
                current,
                stage="validator",
                code="validation_failed",
                message="Validator failed to complete policy assessment",
                details=str(error),
            )
        return result.model_dump()

    def error_node(_state: VendorRiskState | dict[str, Any]) -> GraphUpdate:
        return {"error_state_reached": True}

    graph = StateGraph(VendorRiskState)
    graph.add_node("Supervisor", supervisor)
    graph.add_node("Extractor", run_extractor)
    graph.add_node("Validator", run_validator)
    graph.add_node("Error", error_node)

    graph.add_edge(START, "Supervisor")
    graph.add_conditional_edges(
        "Supervisor",
        _next_step,
        {
            "Extractor": "Extractor",
            "Validator": "Validator",
            "Error": "Error",
            "END": END,
        },
    )
    graph.add_edge("Extractor", "Supervisor")
    graph.add_edge("Validator", "Supervisor")
    graph.add_edge("Error", END)
    return graph.compile()


def _record_failure(
    state: VendorRiskState,
    *,
    stage: Literal["extractor", "validator"],
    code: str,
    message: str,
    details: str,
    recoverable: bool = True,
) -> GraphUpdate:
    error = AgentError(
        stage=stage,
        code=code,
        message=message,
        details=details,
        recoverable=recoverable,
    )
    return {
        "validation_status": ValidationStatus.ERROR,
        "error_logs": [*state.error_logs, error],
    }
