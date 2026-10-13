from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["low", "medium", "high"]
PriorityLevel = Literal["low", "medium", "high"]


class DiagnosisAnalyzeRequest(BaseModel):
    machine_id: str | None = None
    incident_id: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.75, ge=-1.0, le=1.0)


class DiagnosisEvidenceItem(BaseModel):
    source: str
    document_id: str
    document_title: str
    chunk_id: str
    record_type: str | None = None
    section_title: str | None = None
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RootCauseHypothesis(BaseModel):
    cause: str
    confidence: ConfidenceLevel
    reasoning: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class RecommendedCheck(BaseModel):
    check: str
    priority: PriorityLevel
    related_cause: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class DiagnosisTraceStep(BaseModel):
    step: str
    description: str


class DiagnosisAnalyzeResponse(BaseModel):
    machine_id: str | None = None
    incident_id: str | None = None
    summary: str
    possible_causes: list[RootCauseHypothesis]
    recommended_checks: list[RecommendedCheck]
    safety_notes: list[str]
    evidence: list[DiagnosisEvidenceItem]
    trace: list[DiagnosisTraceStep]
