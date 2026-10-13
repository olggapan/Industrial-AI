from __future__ import annotations

from app.rag.runtime import search
from app.rag.schemas import RetrievalQuery, RetrievedChunk
from app.schemas.diagnosis import (
    DiagnosisAnalyzeRequest,
    DiagnosisAnalyzeResponse,
    DiagnosisEvidenceItem,
    DiagnosisTraceStep,
    RecommendedCheck,
    RootCauseHypothesis,
)


CAUSE_LIBRARY = [
    {
        "cause": "Clogged return filter",
        "keywords": [
            "clogged return filter",
            "return filter",
            "filter differential pressure",
            "filter was partially clogged",
        ],
        "checks": [
            "Inspect return filter condition.",
            "Check filter differential pressure.",
            "Replace the return filter if contamination or blockage is confirmed.",
        ],
    },
    {
        "cause": "Pump wear or pump degradation",
        "keywords": [
            "pump wear",
            "pump degradation",
            "reduced pump flow",
            "pump vibration",
            "pump noise",
        ],
        "checks": [
            "Review pump flow under load.",
            "Check pump vibration and noise.",
            "Compare current pump performance against baseline readings.",
        ],
    },
    {
        "cause": "Pressure sensor drift or sensor mismatch",
        "keywords": [
            "pressure sensor drift",
            "sensor drift",
            "sensor mismatch",
            "pressure sensor readings",
        ],
        "checks": [
            "Verify pressure sensor readings against a calibrated reference.",
            "Inspect sensor wiring and connector condition.",
            "Check whether redundant sensor values disagree.",
        ],
    },
    {
        "cause": "Accumulator pre-charge issue",
        "keywords": [
            "accumulator",
            "pre-charge",
            "precharge",
            "accumulator pre-charge issue",
        ],
        "checks": [
            "Verify accumulator pre-charge pressure.",
            "Inspect accumulator isolation valve state.",
            "Check whether pressure fluctuations reduce after accumulator service.",
        ],
    },
    {
        "cause": "Air in the hydraulic circuit",
        "keywords": [
            "air in the hydraulic circuit",
            "air ingress",
            "foaming",
            "aeration",
        ],
        "checks": [
            "Inspect suction line for air ingress.",
            "Check oil foaming or aeration.",
            "Bleed the hydraulic circuit according to service procedure.",
        ],
    },
]


DEFAULT_SAFETY_NOTES = [
    "Depressurize the hydraulic circuit before inspecting filters, hoses, sensors, or accumulator components.",
    "Apply lockout/tagout before maintenance work.",
    "Wear proper protective equipment when working with high-pressure hydraulic systems.",
]


async def analyze_incident(
    request: DiagnosisAnalyzeRequest,
) -> DiagnosisAnalyzeResponse:
    trace: list[DiagnosisTraceStep] = []

    base_query = _build_base_query(request)

    trace.append(
        DiagnosisTraceStep(
            step="build_query",
            description=f"Built diagnosis query from symptoms and error codes: {base_query}",
        )
    )

    evidence: list[DiagnosisEvidenceItem] = []

    if request.error_codes:
        error_code_result = await search(
            RetrievalQuery(
                query=base_query,
                top_k=request.top_k,
                record_types=["error_code"],
                error_codes=request.error_codes,
            )
        )
        evidence.extend(
            _to_evidence_items(
                chunks=error_code_result.results,
                source="error_code_lookup",
                min_score=request.min_score,
            )
        )
        trace.append(
            DiagnosisTraceStep(
                step="retrieve_error_codes",
                description=f"Retrieved error code evidence for: {', '.join(request.error_codes)}",
            )
        )

    repair_case_result = await search(
        RetrievalQuery(
            query=base_query,
            top_k=request.top_k,
            record_types=["repair_case"],
            machine_id=request.machine_id,
            error_codes=request.error_codes or None,
        )
    )
    evidence.extend(
        _to_evidence_items(
            chunks=repair_case_result.results,
            source="similar_repair_cases",
            min_score=request.min_score,
        )
    )
    trace.append(
        DiagnosisTraceStep(
            step="retrieve_repair_cases",
            description="Retrieved similar historical repair cases.",
        )
    )

    general_result = await search(
        RetrievalQuery(
            query=base_query,
            top_k=request.top_k,
            machine_id=request.machine_id,
            error_codes=request.error_codes or None,
        )
    )
    evidence.extend(
        _to_evidence_items(
            chunks=general_result.results,
            source="general_rag_search",
            min_score=request.min_score,
        )
    )
    trace.append(
        DiagnosisTraceStep(
            step="retrieve_general_evidence",
            description="Retrieved general RAG evidence from the local knowledge base.",
        )
    )

    evidence = _deduplicate_evidence(evidence)

    possible_causes = _infer_possible_causes(evidence)
    recommended_checks = _build_recommended_checks(possible_causes)
    safety_notes = _extract_safety_notes(evidence)

    if not safety_notes:
        safety_notes = DEFAULT_SAFETY_NOTES

    summary = _build_summary(request, possible_causes, evidence)

    trace.append(
        DiagnosisTraceStep(
            step="generate_structured_diagnosis",
            description="Generated rule-based hypotheses, recommended checks, safety notes, and evidence trace.",
        )
    )

    return DiagnosisAnalyzeResponse(
        machine_id=request.machine_id,
        incident_id=request.incident_id,
        summary=summary,
        possible_causes=possible_causes,
        recommended_checks=recommended_checks,
        safety_notes=safety_notes,
        evidence=evidence,
        trace=trace,
    )


def _build_base_query(request: DiagnosisAnalyzeRequest) -> str:
    parts: list[str] = []

    if request.error_codes:
        parts.extend(request.error_codes)

    if request.symptoms:
        parts.extend(request.symptoms)

    if request.machine_id:
        parts.append(f"machine {request.machine_id}")

    return " ".join(parts).strip() or "hydraulic troubleshooting"


def _to_evidence_items(
    chunks: list[RetrievedChunk],
    source: str,
    min_score: float,
) -> list[DiagnosisEvidenceItem]:
    items: list[DiagnosisEvidenceItem] = []

    for chunk in chunks:
        if chunk.score < min_score:
            continue

        items.append(
            DiagnosisEvidenceItem(
                source=source,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                chunk_id=chunk.chunk_id,
                record_type=chunk.metadata.get("record_type"),
                section_title=chunk.section_title,
                content=chunk.content,
                score=chunk.score,
                metadata=chunk.metadata,
            )
        )

    return items


def _deduplicate_evidence(
    evidence: list[DiagnosisEvidenceItem],
) -> list[DiagnosisEvidenceItem]:
    by_chunk_id: dict[str, DiagnosisEvidenceItem] = {}

    for item in evidence:
        existing = by_chunk_id.get(item.chunk_id)

        if existing is None or item.score > existing.score:
            by_chunk_id[item.chunk_id] = item

    return sorted(
        by_chunk_id.values(),
        key=lambda item: item.score,
        reverse=True,
    )


def _infer_possible_causes(
    evidence: list[DiagnosisEvidenceItem],
) -> list[RootCauseHypothesis]:
    hypotheses: list[RootCauseHypothesis] = []

    combined_text_by_chunk = {
        item.chunk_id: f"{item.document_title}\n{item.section_title or ''}\n{item.content}".lower()
        for item in evidence
    }

    for cause_def in CAUSE_LIBRARY:
        matched_chunk_ids: list[str] = []

        for chunk_id, text in combined_text_by_chunk.items():
            if any(keyword.lower() in text for keyword in cause_def["keywords"]):
                matched_chunk_ids.append(chunk_id)

        if not matched_chunk_ids:
            continue

        confidence = _confidence_from_evidence_count(len(matched_chunk_ids))

        hypotheses.append(
            RootCauseHypothesis(
                cause=cause_def["cause"],
                confidence=confidence,
                reasoning=(
                    f"Matched {len(matched_chunk_ids)} evidence chunk(s) mentioning "
                    f"signals related to {cause_def['cause']}."
                ),
                evidence_chunk_ids=matched_chunk_ids,
            )
        )

    if hypotheses:
        return hypotheses

    return [
        RootCauseHypothesis(
            cause="Insufficient evidence for a specific root cause",
            confidence="low",
            reasoning=(
                "The retrieved evidence was relevant to the incident, but no known "
                "cause pattern matched strongly enough."
            ),
            evidence_chunk_ids=[item.chunk_id for item in evidence[:3]],
        )
    ]


def _confidence_from_evidence_count(count: int) -> str:
    if count >= 3:
        return "high"

    if count == 2:
        return "medium"

    return "low"


def _build_recommended_checks(
    causes: list[RootCauseHypothesis],
) -> list[RecommendedCheck]:
    checks: list[RecommendedCheck] = []

    for cause in causes:
        cause_def = next(
            (item for item in CAUSE_LIBRARY if item["cause"] == cause.cause),
            None,
        )

        if cause_def is None:
            continue

        priority = "high" if cause.confidence == "high" else "medium"

        for check in cause_def["checks"]:
            checks.append(
                RecommendedCheck(
                    check=check,
                    priority=priority,
                    related_cause=cause.cause,
                    evidence_chunk_ids=cause.evidence_chunk_ids,
                )
            )

    if checks:
        return checks

    return [
        RecommendedCheck(
            check="Review the top retrieved evidence and collect additional telemetry before making a repair decision.",
            priority="medium",
            related_cause=None,
            evidence_chunk_ids=[],
        )
    ]


def _extract_safety_notes(
    evidence: list[DiagnosisEvidenceItem],
) -> list[str]:
    safety_notes: list[str] = []

    for item in evidence:
        text = item.content.lower()

        if "depressurize" in text:
            safety_notes.append("Depressurize the hydraulic circuit before inspection.")

        if "lockout" in text or "tagout" in text:
            safety_notes.append("Apply lockout/tagout before maintenance work.")

        if "protective" in text or "ppe" in text:
            safety_notes.append("Wear proper protective equipment.")

        if "high pressure" in text:
            safety_notes.append(
                "Treat high-pressure hydraulic fluid as a serious injury risk."
            )

    deduped: list[str] = []

    for note in safety_notes:
        if note not in deduped:
            deduped.append(note)

    return deduped


def _build_summary(
    request: DiagnosisAnalyzeRequest,
    causes: list[RootCauseHypothesis],
    evidence: list[DiagnosisEvidenceItem],
) -> str:
    incident_part = (
        f"Incident {request.incident_id}"
        if request.incident_id
        else "The reported incident"
    )

    machine_part = f" on machine {request.machine_id}" if request.machine_id else ""

    if not evidence:
        return (
            f"{incident_part}{machine_part} could not be diagnosed because no "
            "retrieval evidence passed the configured score threshold."
        )

    top_cause = causes[0].cause if causes else "an unresolved hydraulic issue"

    return (
        f"{incident_part}{machine_part} is most consistent with {top_cause}. "
        f"The workflow used {len(evidence)} evidence chunk(s) from the local knowledge base."
    )
