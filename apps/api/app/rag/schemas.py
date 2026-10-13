from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentLoadRequest(BaseModel):
    """
    Input for a source-specific document loader.

    The file loader currently accepts local Markdown. Synthetic JSON adapters
    create LoadedDocument objects directly.
    """

    source_uri: str
    source_type: Literal["markdown", "json_record"] = "markdown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    """
    A raw document loaded from disk before chunking.

    Example:
    - hydraulic-system-overview.md
    - pressure-instability-troubleshooting.md
    """

    id: str
    title: str
    source_path: str
    source_type: Literal["markdown", "json_record"] = "markdown"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """
    A smaller text unit created from a document.

    Chunks are embedded and stored through the configured vector-store boundary.
    """

    id: str
    document_id: str
    chunk_index: int
    content: str
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    """
    Output of the chunking step for a single document.
    """

    document_id: str
    chunks: list[DocumentChunk]


class EmbeddedChunk(DocumentChunk):
    """
    A document chunk with its embedding vector.

    The embedding is a list of floats returned by an embedding model.
    """

    embedding: list[float]


class RetrievedChunk(BaseModel):
    """
    A search result returned from vector search or hybrid search.
    """

    chunk_id: str
    document_id: str
    document_title: str
    content: str
    section_title: str | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """
    Input for RAG search.
    """

    query: str
    top_k: int = Field(default=5, ge=1, le=20)

    record_types: list[str] | None = None
    machine_id: str | None = None
    error_codes: list[str] | None = None


class RetrievalResult(BaseModel):
    """
    Output of a RAG search.
    """

    query: str
    results: list[RetrievedChunk]
