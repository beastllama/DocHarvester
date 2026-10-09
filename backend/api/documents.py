from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db
from backend.models import Document, DocumentChunk, Project, User
from backend.api.auth import get_current_user
from backend.api.deps import accessible_project_ids, get_project_for_user, is_project_member
from backend.services.embeddings import EmbeddingError
from backend.services.semantic_search import search_chunks


router = APIRouter()


class DocumentResponse(BaseModel):
    id: int
    project_id: int
    doc_id: str
    title: str
    source_type: Optional[str]
    source_url: Optional[str]
    source_meta: dict
    file_type: Optional[str]
    last_modified: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = 0
    
    class Config:
        from_attributes = True


class DocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    text: str
    lens_type: str
    confidence_score: Optional[float]
    importance_score: float
    is_generated: bool
    generation_status: str
    tokens: Optional[int]
    chunk_metadata: dict
    
    class Config:
        from_attributes = True


class DocumentSearchResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    pages: int


@router.get("/", response_model=DocumentSearchResponse)
async def search_documents(
    q: Optional[str] = Query(None, description="Search query"),
    project_id: Optional[int] = Query(None, description="Filter by project"),
    lens_type: Optional[str] = Query(None, description="Filter by lens type"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    is_generated: Optional[bool] = Query(None, description="Filter by generation status"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search and filter documents"""
    # Base query
    query = select(Document, func.count(DocumentChunk.id).label('chunk_count'))\
        .outerjoin(DocumentChunk)\
        .group_by(Document.id)
    
    # Apply filters
    filters = []
    
    if project_id:
        await get_project_for_user(project_id, db, current_user)
        filters.append(Document.project_id == project_id)
    
    # Only return documents from projects the user can access (None = admin, no filter)
    allowed_ids = await accessible_project_ids(db, current_user)
    if allowed_ids is not None:
        filters.append(Document.project_id.in_(allowed_ids))
    
    if file_type:
        filters.append(Document.file_type == file_type)
    
    if q:
        # Text search in title and content
        search_filter = or_(
            Document.title.ilike(f"%{q}%"),
            Document.raw_text.ilike(f"%{q}%")
        )
        filters.append(search_filter)
    
    if lens_type:
        # Need to join with chunks for lens filtering
        query = query.filter(DocumentChunk.lens_type == lens_type)
    
    if is_generated is not None:
        query = query.filter(DocumentChunk.is_generated == is_generated)
    
    if filters:
        query = query.filter(and_(*filters))
    
    # Count total results
    count_query = select(func.count(func.distinct(Document.id))).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    documents = []
    
    for row in result:
        doc = row[0]
        chunk_count = row[1]
        
        doc_dict = {
            **doc.__dict__,
            "chunk_count": chunk_count
        }
        documents.append(DocumentResponse(**doc_dict))
    
    # Calculate pages
    pages = (total + limit - 1) // limit
    
    return DocumentSearchResponse(
        documents=documents,
        total=total,
        page=page,
        pages=pages
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get document details"""
    query = select(Document, func.count(DocumentChunk.id).label('chunk_count'))\
        .outerjoin(DocumentChunk)\
        .where(Document.id == document_id)\
        .group_by(Document.id)
    
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = row[0]
    chunk_count = row[1]
    await get_project_for_user(doc.project_id, db, current_user)
    
    return DocumentResponse(
        **doc.__dict__,
        chunk_count=chunk_count
    )


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the raw content of a document for viewing"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    await get_project_for_user(document.project_id, db, current_user)
    
    return {
        "id": document.id,
        "title": document.title,
        "file_type": document.file_type,
        "raw_text": document.raw_text,
        "source_meta": document.source_meta,
        "created_at": document.created_at,
        "last_modified": document.last_modified
    }


@router.get("/{document_id}/chunks", response_model=List[DocumentChunkResponse])
async def get_document_chunks(
    document_id: int,
    lens_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all chunks for a document"""
    # Verify document exists
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    await get_project_for_user(document.project_id, db, current_user)
    
    # Get chunks
    query = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    
    if lens_type:
        query = query.filter(DocumentChunk.lens_type == lens_type)
    
    query = query.order_by(DocumentChunk.chunk_index)
    
    result = await db.execute(query)
    chunks = result.scalars().all()
    
    return [DocumentChunkResponse(**chunk.__dict__) for chunk in chunks]


@router.post("/search/semantic")
async def semantic_search(
    query: str,
    project_id: Optional[int] = None,
    lens_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Semantic search using embeddings"""
    if project_id:
        await get_project_for_user(project_id, db, current_user)
    allowed_ids = await accessible_project_ids(db, current_user)

    try:
        results = await search_chunks(
            db,
            query,
            limit=limit,
            project_id=project_id,
            lens_type=lens_type,
            allowed_project_ids=allowed_ids,
        )
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "query": query,
        "results": results,
        "count": len(results)
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a document"""
    # Get document with project info
    result = await db.execute(
        select(Document).options(selectinload(Document.project))
        .where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check authorization
    if not is_project_member(document.project, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to delete this document")
    
    await db.delete(document)
    await db.commit()
    
    return {"message": "Document deleted successfully"}


@router.put("/{document_id}/reclassify")
async def reclassify_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger reclassification of document chunks"""
    # Verify document exists
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    await get_project_for_user(document.project_id, db, current_user)
    
    # Queue reclassification task
    # TODO: Implement reclassification task
    
    return {
        "message": "Reclassification queued",
        "document_id": document_id
    }


@router.get("/stats/by-lens")
async def get_lens_statistics(
    project_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get document statistics grouped by lens type"""
    query = select(
        DocumentChunk.lens_type,
        func.count(func.distinct(DocumentChunk.document_id)).label('document_count'),
        func.count(DocumentChunk.id).label('chunk_count'),
        func.avg(DocumentChunk.confidence_score).label('avg_confidence')
    ).select_from(DocumentChunk).join(Document).group_by(DocumentChunk.lens_type)
    
    if project_id:
        await get_project_for_user(project_id, db, current_user)
        query = query.filter(Document.project_id == project_id)
    
    # Only count documents from projects the user can access (None = admin, no filter)
    allowed_ids = await accessible_project_ids(db, current_user)
    if allowed_ids is not None:
        query = query.filter(Document.project_id.in_(allowed_ids))
    
    result = await db.execute(query)
    
    stats = []
    for row in result:
        stats.append({
            "lens_type": row[0],
            "document_count": row[1],
            "chunk_count": row[2],
            "avg_confidence": float(row[3]) if row[3] else 0.0
        })
    
    return {"stats": stats} 