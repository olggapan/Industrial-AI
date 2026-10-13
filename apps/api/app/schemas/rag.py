from typing import Any

from pydantic import BaseModel, Field


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float | None = Field(default=0.80, ge=-1.0, le=1.0)
    record_types: list[str] | None = None
    machine_id: str | None = None
    error_codes: list[str] | None = None


class RagSearchResult(BaseModel):
    document_id: str
    document_title: str
    chunk_id: str
    chunk_index: int | None = None
    section_title: str | None = None
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_placeholder: bool = False


class RagSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[RagSearchResult] = Field(default_factory=list)
    message: str
    is_placeholder: bool = False
