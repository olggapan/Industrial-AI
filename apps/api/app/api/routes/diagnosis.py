from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.diagnosis import DiagnosisAnalyzeRequest, DiagnosisAnalyzeResponse
from app.services.diagnosis_service import analyze_incident


router = APIRouter()


@router.post("/analyze", response_model=DiagnosisAnalyzeResponse)
async def diagnosis_analyze(
    payload: DiagnosisAnalyzeRequest,
) -> DiagnosisAnalyzeResponse:
    try:
        return await analyze_incident(payload)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
