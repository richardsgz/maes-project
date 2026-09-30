from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class MAESState(TypedDict, total=False):
    request: str
    extracted_data: dict[str, Any]
    validation_passed: bool
    retry_count: int
    max_retries: int
    feedback: list[str]
    next_step: Literal["Extractor", "Validator", "END"]


def supervisor(state: MAESState) -> dict[str, str]:
    """Choose the next worker from the current extraction and validation state."""
    if state.get("validation_passed") is True:
        return {"next_step": "END"}

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if state.get("validation_passed") is False:
        if retry_count >= max_retries:
            return {"next_step": "END"}
        return {"next_step": "Extractor"}

    if not state.get("extracted_data"):
        return {"next_step": "Extractor"}
    return {"next_step": "Validator"}


def extractor(state: MAESState) -> dict[str, dict[str, str]]:
    """Placeholder worker; replace with domain-specific or Gemini-backed extraction."""
    return {"extracted_data": {"source_text": state.get("request", "")}}


def validator(state: MAESState) -> dict[str, Any]:
    """Perform a minimal structural check and retain feedback for future retries."""
    extracted_data = state.get("extracted_data", {})
    feedback = [] if extracted_data.get("source_text") else ["Missing source_text"]
    retry_count = state.get("retry_count", 0)
    return {
        "validation_passed": not feedback,
        "feedback": feedback,
        "retry_count": retry_count + bool(feedback),
    }


def build_graph():
    """Compile the Supervisor/Worker graph for invocation by an application."""
    graph = StateGraph(MAESState)
    graph.add_node("Supervisor", supervisor)
    graph.add_node("Extractor", extractor)
    graph.add_node("Validator", validator)

    graph.add_edge(START, "Supervisor")
    graph.add_conditional_edges(
        "Supervisor",
        lambda state: state["next_step"],
        {"Extractor": "Extractor", "Validator": "Validator", "END": END},
    )
    graph.add_edge("Extractor", "Supervisor")
    graph.add_edge("Validator", "Supervisor")
    return graph.compile()
