"""Semantic search over document chunks, shared by the REST API and the MCP server."""
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.models import Document, DocumentChunk
from backend.services.embeddings import EmbeddingService


async def search_chunks(
    db: AsyncSession,
    query: str,
    *,
    limit: int = 10,
    project_id: Optional[int] = None,
    lens_type: Optional[str] = None,
    allowed_project_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Closest chunks to the query, most similar first.

    project_id narrows to one project. allowed_project_ids is the caller's access list;
    None means no limit (admins). Raises EmbeddingError if the embedding model is unreachable.
    """
    embedding_service = await run_in_threadpool(EmbeddingService)
    query_embedding = await run_in_threadpool(embedding_service.get_embedding, query)

    statement = select(
        DocumentChunk,
        Document,
        (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label("similarity"),
    ).join(Document)

    if project_id:
        statement = statement.filter(Document.project_id == project_id)
    if allowed_project_ids is not None:
        statement = statement.filter(Document.project_id.in_(allowed_project_ids))
    if lens_type:
        statement = statement.filter(DocumentChunk.lens_type == lens_type)

    # Smallest cosine distance first, so the most similar chunks come first
    statement = statement.order_by(DocumentChunk.embedding.cosine_distance(query_embedding)).limit(limit)

    result = await db.execute(statement)
    results = []
    for chunk, doc, similarity in result.all():
        results.append({
            "document": {
                "id": doc.id,
                "title": doc.title,
                "source_type": doc.source_type,
                "file_type": doc.file_type,
            },
            "chunk": {
                "id": chunk.id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "lens_type": chunk.lens_type,
                "chunk_index": chunk.chunk_index,
            },
            "similarity": float(similarity),
        })
    return results
