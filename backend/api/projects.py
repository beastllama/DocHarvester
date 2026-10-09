from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db
from backend.models import Project, Document, DocumentChunk, User
from backend.workers.ingest_tasks import discover_and_ingest_project
from backend.api.auth import get_current_user
from backend.api.deps import accessible_project_ids, get_project_for_user, is_project_member


router = APIRouter()


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = []
    owners: List[str] = []
    connector_configs: Optional[dict] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    owners: Optional[List[str]] = None
    connector_configs: Optional[dict] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    tags: List[str]
    owners: List[str]
    created_at: datetime
    updated_at: datetime
    document_count: Optional[int] = 0
    coverage_percentage: Optional[float] = 0.0
    
    class Config:
        from_attributes = True


class ProjectStats(BaseModel):
    total_documents: int
    total_chunks: int
    documents_by_type: dict
    coverage_by_lens: dict
    recent_activity: List[dict]


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all projects with stats"""
    # Build query with document count
    ids = await accessible_project_ids(db, current_user)
    query = select(
        Project,
        func.count(Document.id).label('document_count')
    ).outerjoin(Document)
    if ids is not None:
        query = query.where(Project.id.in_(ids))
    query = query.group_by(Project.id)

    result = await db.execute(query.offset(skip).limit(limit))
    projects = []
    
    for row in result:
        project = row[0]
        doc_count = row[1]
        
        # Calculate coverage percentage (simplified for now)
        coverage = 0.0
        if doc_count > 0:
            # This is a simplified calculation - in production, 
            # you'd check against coverage requirements
            coverage = min(doc_count * 10, 100)  # Mock calculation
        
        project_dict = {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "tags": project.tags,
            "owners": project.owners,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "document_count": doc_count,
            "coverage_percentage": coverage
        }
        projects.append(ProjectResponse(**project_dict))
    
    return projects


@router.post("/", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new project"""
    # Check if project already exists
    result = await db.execute(
        select(Project).where(Project.name == project.name)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail="Project already exists")
    
    # Create new project
    db_project = Project(
        name=project.name,
        description=project.description,
        tags=project.tags,
        owners=project.owners if project.owners else [current_user.email],
        connector_configs=project.connector_configs or {}
    )
    
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    
    # Return with default stats
    return ProjectResponse(
        **db_project.__dict__,
        document_count=0,
        coverage_percentage=0.0
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get project details"""
    await get_project_for_user(project_id, db, current_user)

    # Get project with document count
    query = select(
        Project,
        func.count(Document.id).label('document_count')
    ).outerjoin(Document).where(
        Project.id == project_id
    ).group_by(Project.id)
    
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = row[0]
    doc_count = row[1]
    
    # Calculate coverage
    coverage = min(doc_count * 10, 100) if doc_count > 0 else 0.0
    
    return ProjectResponse(
        **project.__dict__,
        document_count=doc_count,
        coverage_percentage=coverage
    )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update project details"""
    project = await get_project_for_user(project_id, db, current_user)

    # Update fields
    update_data = project_update.dict(exclude_unset=True)
    if not current_user.is_admin and ("owners" in update_data or "connector_configs" in update_data):
        raise HTTPException(status_code=403, detail="Only admins can change owners or connector configs")
    for field, value in update_data.items():
        setattr(project, field, value)
    
    await db.commit()
    await db.refresh(project)
    
    # Get document count
    doc_count_result = await db.execute(
        select(func.count(Document.id)).where(Document.project_id == project_id)
    )
    doc_count = doc_count_result.scalar() or 0
    
    return ProjectResponse(
        **project.__dict__,
        document_count=doc_count,
        coverage_percentage=min(doc_count * 10, 100) if doc_count > 0 else 0.0
    )


@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a project"""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if user is owner or admin
    if not is_project_member(project, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to delete this project")
    
    await db.delete(project)
    await db.commit()
    
    return {"message": "Project deleted successfully"}


@router.get("/{project_id}/stats", response_model=ProjectStats)
async def get_project_stats(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed project statistics"""
    # Verify project exists and the user may access it
    await get_project_for_user(project_id, db, current_user)
    
    # Get document stats
    doc_stats = await db.execute(
        select(
            func.count(Document.id).label('total_docs'),
            func.count(DocumentChunk.id).label('total_chunks')
        ).select_from(Document).outerjoin(DocumentChunk).where(
            Document.project_id == project_id
        )
    )
    stats_row = doc_stats.first()
    
    # Get documents by type
    type_stats = await db.execute(
        select(
            Document.file_type,
            func.count(Document.id).label('count')
        ).where(
            Document.project_id == project_id
        ).group_by(Document.file_type)
    )
    
    docs_by_type = {row[0]: row[1] for row in type_stats if row[0]}
    
    # Get coverage by lens
    lens_stats = await db.execute(
        select(
            DocumentChunk.lens_type,
            func.count(DocumentChunk.id).label('count')
        ).select_from(DocumentChunk).join(Document).where(
            Document.project_id == project_id
        ).group_by(DocumentChunk.lens_type)
    )
    
    coverage_by_lens = {row[0]: row[1] for row in lens_stats}
    
    # Get recent activity (last 10 documents)
    recent_docs = await db.execute(
        select(Document).where(
            Document.project_id == project_id
        ).order_by(Document.created_at.desc()).limit(10)
    )
    
    recent_activity = [
        {
            "action": "Document added",
            "document": doc.title,
            "time": doc.created_at.isoformat(),
            "type": doc.file_type
        }
        for doc in recent_docs.scalars()
    ]
    
    return ProjectStats(
        total_documents=stats_row[0] or 0,
        total_chunks=stats_row[1] or 0,
        documents_by_type=docs_by_type,
        coverage_by_lens=coverage_by_lens,
        recent_activity=recent_activity
    )


@router.post("/{project_id}/ingest")
async def start_ingestion(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start document ingestion for a project"""
    # Verify project exists and the user may access it
    await get_project_for_user(project_id, db, current_user)
    
    # Queue ingestion task
    task = discover_and_ingest_project.delay(project_id)
    
    return {
        "message": "Ingestion started",
        "project_id": project_id,
        "status": "queued",
        "task_id": task.id
    }


@router.get("/{project_id}/ingestion-status")
async def get_ingestion_status(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the status of recent ingestion tasks"""
    # Verify project exists and the user may access it
    await get_project_for_user(project_id, db, current_user)
    
    # For now, return a simple status based on recent document activity
    # In a full implementation, you'd track task status in Redis or database
    recent_docs = await db.execute(
        select(Document).where(
            Document.project_id == project_id
        ).order_by(Document.created_at.desc()).limit(5)
    )
    
    docs = recent_docs.scalars().all()
    
    return {
        "project_id": project_id,
        "status": "completed" if docs else "idle",
        "recent_documents": [
            {
                "title": doc.title,
                "created_at": doc.created_at.isoformat(),
                "file_type": doc.file_type
            }
            for doc in docs
        ],
        "last_activity": docs[0].created_at.isoformat() if docs else None
    }


@router.post("/{project_id}/upload")
async def upload_documents(
    project_id: int,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload documents directly to a project"""
    import os
    from pathlib import Path

    # Verify project exists and the user may access it
    await get_project_for_user(project_id, db, current_user)
    
    # Create upload directory if it doesn't exist
    upload_dir = Path("/app/uploads") / str(project_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Keep only the base name so a client path like ../x cannot leave the upload folder
        base_name = Path(file.filename or "upload").name or "upload"
        safe_filename = f"{timestamp}_{base_name}"
        file_path = upload_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Create document record
        doc = Document(
            project_id=project_id,
            doc_id=f"upload_{project_id}_{safe_filename}",
            title=file.filename,
            source_type="upload",
            source_url=str(file_path),
            file_type=base_name.split('.')[-1].lower() if '.' in base_name else 'unknown',
            source_meta={
                "original_name": file.filename,
                "size": len(content),
                "upload_time": datetime.now().isoformat()
            }
        )
        
        db.add(doc)
        uploaded_files.append(file.filename)
    
    await db.commit()
    
    # Trigger ingestion task using Celery
    discover_and_ingest_project.delay(project_id)
    
    return {
        "message": f"Uploaded {len(uploaded_files)} files successfully. Processing started.",
        "files": uploaded_files,
        "project_id": project_id
    } 