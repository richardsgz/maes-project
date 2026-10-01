import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import networkx as nx

from core.state import MemoryContext, ValidationStatus, VendorRiskState


class AssessmentMemory(Protocol):
    def retrieve_for_document(self, document: str) -> list[MemoryContext]: ...

    def record_assessment(self, state: VendorRiskState) -> str: ...


class NetworkXMemoryGraph:
    """Persist validated vendor assessments and correction links in a JSON-backed graph."""

    format_version = 1

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._graph = nx.MultiDiGraph()
        if self.path.exists():
            self._load()

    @property
    def assessment_count(self) -> int:
        return sum(
            1
            for _, attributes in self._graph.nodes(data=True)
            if attributes.get("kind") == "assessment"
        )

    @property
    def correction_count(self) -> int:
        return sum(
            1
            for _, _, attributes in self._graph.edges(data=True)
            if attributes.get("relation") == "CORRECTS"
        )

    def record_assessment(self, state: VendorRiskState) -> str:
        if state.validation_status not in (ValidationStatus.PASSED, ValidationStatus.FAILED):
            raise ValueError("Only completed policy assessments can be stored in memory")

        assessment_id = f"assessment:{uuid4().hex}"
        created_at = datetime.now(UTC).isoformat()
        self._graph.add_node(
            assessment_id,
            kind="assessment",
            created_at=created_at,
            validation_status=state.validation_status.value,
            source_sha256=hashlib.sha256(state.original_document.encode("utf-8")).hexdigest(),
            extraction_summary=state.extraction_summary,
        )

        findings_by_entity: dict[str, list[dict[str, object]]] = {}
        for finding in state.policy_findings:
            findings_by_entity.setdefault(finding.entity_value.casefold(), []).append(
                finding.model_dump(mode="json")
            )

        for entity in state.extracted_entities:
            concept_id = self._concept_id(entity.entity_type, entity.value)
            self._graph.add_node(
                concept_id,
                kind="concept",
                entity_type=entity.entity_type,
                entity_value=entity.value,
            )
            self._graph.add_edge(
                assessment_id,
                concept_id,
                relation="EXTRACTED",
                evidence=entity.evidence,
                confidence=entity.confidence,
                policy_findings=findings_by_entity.get(entity.value.casefold(), []),
            )

            if state.validation_status is ValidationStatus.PASSED:
                self._link_corrections(
                    assessment_id,
                    concept_id,
                    entity.evidence,
                    created_at,
                )

        self._persist()
        return assessment_id

    def retrieve_for_document(self, document: str, *, limit: int = 10) -> list[MemoryContext]:
        document_tokens = self._tokens(document)
        latest: dict[str, tuple[str, MemoryContext]] = {}

        for assessment_id, concept_id, edge in self._graph.edges(data=True):
            if edge.get("relation") != "EXTRACTED":
                continue
            concept = self._graph.nodes[concept_id]
            entity_tokens = self._tokens(str(concept["entity_value"]))
            if not entity_tokens or not entity_tokens.issubset(document_tokens):
                continue

            assessment = self._graph.nodes[assessment_id]
            correction_note = self._correction_note(assessment_id, concept_id)
            context = MemoryContext(
                entity_type=concept["entity_type"],
                entity_value=concept["entity_value"],
                evidence=edge["evidence"],
                validation_status=assessment["validation_status"],
                policy_findings=edge.get("policy_findings", []),
                correction_note=correction_note,
            )
            existing = latest.get(concept_id)
            created_at = assessment["created_at"]
            if existing is None or created_at > existing[0]:
                latest[concept_id] = (created_at, context)

        ordered = sorted(latest.values(), key=lambda item: item[0], reverse=True)
        return [context for _, context in ordered[:limit]]

    def _link_corrections(
        self,
        assessment_id: str,
        concept_id: str,
        evidence: str,
        created_at: str,
    ) -> None:
        for previous_id, _, edge in self._graph.in_edges(concept_id, data=True):
            if previous_id == assessment_id or edge.get("relation") != "EXTRACTED":
                continue
            previous_status = self._graph.nodes[previous_id]["validation_status"]
            if (
                previous_status == ValidationStatus.FAILED.value
                and edge["evidence"] != evidence
            ):
                self._graph.add_edge(
                    assessment_id,
                    previous_id,
                    relation="CORRECTS",
                    entity=self._graph.nodes[concept_id]["entity_value"],
                    previous_evidence=edge["evidence"],
                    corrected_at=created_at,
                )

    def _correction_note(self, assessment_id: str, concept_id: str) -> str | None:
        for _, previous_id, attributes in self._graph.out_edges(assessment_id, data=True):
            if (
                attributes.get("relation") == "CORRECTS"
                and attributes.get("entity") == self._graph.nodes[concept_id]["entity_value"]
            ):
                previous_status = self._graph.nodes[previous_id]["validation_status"]
                return f"A prior {previous_status} assessment for this entity was corrected."
        return None

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            payload = {
                "format_version": self.format_version,
                "graph": nx.node_link_data(self._graph),
            }
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=True, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("format_version") != self.format_version:
            raise ValueError("Unsupported MAES memory graph format version")
        graph = nx.node_link_graph(payload["graph"], edges="edges")
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("Persisted MAES memory graph is not a directed multigraph")
        self._graph = graph

    @staticmethod
    def _concept_id(entity_type: str, value: str) -> str:
        normalized = f"{entity_type.casefold()}\0{value.casefold()}"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"concept:{digest}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.casefold()))