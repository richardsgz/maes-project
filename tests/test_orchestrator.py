from core.orchestrator import build_graph


def test_graph_extracts_and_validates_request() -> None:
    result = build_graph().invoke({"request": "Check vendor retention requirements."})

    assert result["extracted_data"] == {
        "source_text": "Check vendor retention requirements."
    }
    assert result["validation_passed"] is True
    assert result["retry_count"] == 0
