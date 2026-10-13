from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.rag.embeddings import EmbeddingInput, EmbeddingProvider
from app.rag.schemas import (
    EmbeddedChunk,
    RetrievedChunk,
    RetrievalQuery,
    RetrievalResult,
)
from app.rag.vector_store import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreRecord,
    VectorStoreSearchResult,
)


class RagRetriever(Protocol):
    async def index_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Index embedded chunks for retrieval."""

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""


class VectorRagRetriever:
    """
    RAG retriever based on embeddings and vector search.

    Responsibilities:
    - index embedded chunks into the vector store
    - embed user query text
    - search the vector store
    - convert vector store results into RetrievalResult
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def index_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        records = [embedded_chunk_to_vector_store_record(chunk) for chunk in chunks]
        await self.vector_store.upsert_chunks(records)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        embedding_vectors = await self.embedding_provider.embed(
            [
                EmbeddingInput(
                    text=query.query,
                    metadata={"source": "retrieval_query", "input_type": "query"},
                )
            ]
        )

        if not embedding_vectors:
            raise ValueError("Embedding provider returned no query embedding.")

        query_embedding = embedding_vectors[0].vector

        search_results = await self.vector_store.search(
            VectorStoreQuery(
                query=query.query,
                embedding=query_embedding,
                top_k=query.top_k,
                record_types=query.record_types,
                machine_id=query.machine_id,
                error_codes=query.error_codes,
            )
        )

        return RetrievalResult(
            query=query.query,
            results=[
                vector_store_result_to_retrieved_chunk(result)
                for result in search_results
            ],
        )


def embedded_chunk_to_vector_store_record(chunk: EmbeddedChunk) -> VectorStoreRecord:
    metadata = {
        **chunk.metadata,
        "document_title": chunk.metadata.get("document_title", chunk.document_id),
        "section_title": chunk.section_title or "",
        "chunk_index": chunk.chunk_index,
    }

    return VectorStoreRecord(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        content=chunk.content,
        embedding=chunk.embedding,
        metadata=metadata,
    )


def vector_store_result_to_retrieved_chunk(
    result: VectorStoreSearchResult,
) -> RetrievedChunk:
    metadata = result.metadata

    section_title = metadata.get("section_title")
    if section_title == "":
        section_title = None

    return RetrievedChunk(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        document_title=str(metadata.get("document_title", result.document_id)),
        content=result.content,
        section_title=section_title,
        score=result.score,
        metadata=metadata,
    )


class PlaceholderRagRetriever:
    async def index_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        raise NotImplementedError(
            "Manual RAG retriever indexing is not implemented yet."
        )

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError("Manual RAG retrieval is not implemented yet.")
