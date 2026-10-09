#!/usr/bin/env python3
"""Resize the chunk embedding column to EMBEDDING_DIMENSION, then re-embed every chunk.

Run this after changing EMBEDDING_DIMENSION, EMBEDDING_PROVIDER, or the embedding model:

    python backend/scripts/reembed.py

Old vectors come from another model, so they are cleared and recomputed. Nothing else changes.
Needs the embedding model running (for LOCAL: `ollama pull nomic-embed-text`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text

from backend.config import settings
from backend.database import SessionLocal, sync_engine
from backend.models import DocumentChunk
from backend.services.embeddings import EmbeddingError, EmbeddingService

BATCH_SIZE = 32


def column_dimension(conn):
    return conn.execute(text(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'"
    )).scalar()


def main() -> int:
    dim = settings.embedding_dimension
    with sync_engine.begin() as conn:
        current = column_dimension(conn)
        if current != dim:
            print(f"Resizing embedding column from {current} to {dim} dimensions")
        conn.execute(text(f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({dim}) USING NULL"))

    try:
        service = EmbeddingService()
        service.check_ready()
    except EmbeddingError as exc:
        print(f"Cannot re-embed: {exc}")
        return 1

    with SessionLocal() as session:
        chunk_ids = [row[0] for row in session.execute(text("SELECT id FROM document_chunks ORDER BY id")).all()]
    print(f"Re-embedding {len(chunk_ids)} chunks with {service.provider} ({dim} dimensions)")

    done = 0
    for start in range(0, len(chunk_ids), BATCH_SIZE):
        batch_ids = chunk_ids[start:start + BATCH_SIZE]
        with SessionLocal() as session:
            chunks = session.query(DocumentChunk).filter(DocumentChunk.id.in_(batch_ids)).all()
            try:
                vectors = service.get_embeddings_batch([c.text for c in chunks])
            except EmbeddingError as exc:
                print(f"Stopped after {done} chunks: {exc}")
                return 1
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            session.commit()
        done += len(chunks)
        print(f"  {done}/{len(chunk_ids)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
