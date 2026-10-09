"""Wiki API endpoints"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.database import get_db
from backend.models import WikiPage, WikiStructure
from backend.services.wiki_generator import WikiGenerator
from backend.api.auth import get_current_user, User
from backend.api.deps import get_project_for_user


router = APIRouter(tags=["wiki"])


@router.post("/generate/{project_id}")
async def generate_wiki(
    project_id: int,
    force_regenerate: bool = Query(False, description="Force regeneration even if wiki exists"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate wiki for a project with progress tracking"""
    try:
        # Check project exists and user has access
        await get_project_for_user(project_id, db, current_user)

        print(f"� API: Starting wiki generation for project {project_id}")
        
        # Generate wiki with progress tracking - pass user_id
        generator = WikiGenerator()
        result = await generator.generate_wiki_for_project(
            db, project_id, current_user.id, force_regenerate
        )
        
        if "error" in result:
            print(f"❌ API: Wiki generation failed: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        print(f"✅ API: Wiki generation completed successfully")
        
        # Add task_id to response for progress tracking
        return {
            **result,
            "message": "Wiki generation started successfully",
            "progress_endpoint": f"/api/v1/progress/projects/{project_id}/active-operations"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ API: Unexpected error in wiki generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to rollback the database session
        try:
            await db.rollback()
        except Exception as rollback_error:
            print(f"⚠️ API: Rollback failed: {rollback_error}")
        
        # Return a more user-friendly error
        error_msg = "Wiki generation failed due to a system error"
        if "greenlet" in str(e).lower():
            error_msg = "Wiki generation failed due to async database handling. Please try again."
        elif "timeout" in str(e).lower():
            error_msg = "Wiki generation timed out. Please try again with a smaller project."
        elif "openai" in str(e).lower():
            error_msg = "Wiki generation failed. Please check your OpenAI API key configuration."
        
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/structure/{project_id}")
async def get_wiki_structure(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get wiki structure for a project"""
    await get_project_for_user(project_id, db, current_user)

    result = await db.execute(
        select(WikiStructure).where(WikiStructure.project_id == project_id)
    )
    structure = result.scalar_one_or_none()
    
    if not structure:
        raise HTTPException(status_code=404, detail="Wiki structure not found")
    
    return {
        "project_id": project_id,
        "structure": structure.structure,
        "generation_status": structure.generation_status,
        "last_generated_at": structure.last_generated_at
    }


@router.get("/pages/{project_id}")
async def get_wiki_pages(
    project_id: int,
    parent_id: Optional[int] = Query(None, description="Filter by parent page ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get wiki pages for a project"""
    await get_project_for_user(project_id, db, current_user)

    query = select(WikiPage).where(WikiPage.project_id == project_id)
    
    if parent_id is not None:
        query = query.where(WikiPage.parent_id == parent_id)
    else:
        # Get root pages if no parent specified
        query = query.where(WikiPage.parent_id.is_(None))
    
    query = query.order_by(WikiPage.order_index)
    
    result = await db.execute(query)
    pages = result.scalars().all()
    
    return {
        "project_id": project_id,
        "pages": [
            {
                "id": page.id,
                "title": page.title,
                "slug": page.slug,
                "summary": page.summary,
                "parent_id": page.parent_id,
                "order_index": page.order_index,
                "status": page.status,
                "tags": page.tags,
                "has_children": await _has_children(db, page.id)
            }
            for page in pages
        ]
    }


@router.get("/page/{project_id}/{page_slug}")
async def get_wiki_page(
    project_id: int,
    page_slug: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific wiki page by slug"""
    await get_project_for_user(project_id, db, current_user)

    result = await db.execute(
        select(WikiPage).where(
            and_(
                WikiPage.project_id == project_id,
                WikiPage.slug == page_slug
            )
        )
    )
    page = result.scalar_one_or_none()
    
    if not page:
        raise HTTPException(status_code=404, detail="Wiki page not found")
    
    # Get children
    children_result = await db.execute(
        select(WikiPage)
        .where(WikiPage.parent_id == page.id)
        .order_by(WikiPage.order_index)
    )
    children = children_result.scalars().all()
    
    # Update view count safely
    try:
        page.view_count += 1
        await db.commit()
    except Exception as e:
        print(f"Warning: Failed to update view count: {e}")
        await db.rollback()
    
    return {
        "id": page.id,
        "project_id": page.project_id,
        "title": page.title,
        "slug": page.slug,
        "content": page.content,
        "summary": page.summary,
        "parent_id": page.parent_id,
        "order_index": page.order_index,
        "is_generated": page.is_generated,
        "confidence_score": page.confidence_score,
        "tags": page.tags,
        "status": page.status,
        "view_count": page.view_count,
        "created_at": page.created_at,
        "updated_at": page.updated_at,
        "published_at": page.published_at,
        "children": [
            {
                "id": child.id,
                "title": child.title,
                "slug": child.slug,
                "summary": child.summary,
                "order_index": child.order_index
            }
            for child in children
        ]
    }


@router.get("/search/{project_id}")
async def search_wiki(
    project_id: int,
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search wiki pages"""
    await get_project_for_user(project_id, db, current_user)

    # Simple text search - in production use full-text search or embeddings
    result = await db.execute(
        select(WikiPage)
        .where(
            and_(
                WikiPage.project_id == project_id,
                WikiPage.status == "published",
                WikiPage.content.ilike(f"%{q}%")
            )
        )
        .limit(limit)
    )
    pages = result.scalars().all()
    
    return {
        "query": q,
        "results": [
            {
                "id": page.id,
                "title": page.title,
                "slug": page.slug,
                "summary": page.summary,
                "excerpt": _extract_excerpt(page.content, q),
                "tags": page.tags
            }
            for page in pages
        ]
    }


@router.put("/page/{page_id}")
async def update_wiki_page(
    page_id: int,
    content: str,
    summary: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a wiki page (manual edit)"""
    result = await db.execute(
        select(WikiPage).where(WikiPage.id == page_id)
    )
    page = result.scalar_one_or_none()
    
    if not page:
        raise HTTPException(status_code=404, detail="Wiki page not found")

    await get_project_for_user(page.project_id, db, current_user)

    # Update page
    page.content = content
    if summary:
        page.summary = summary
    page.is_generated = False  # Mark as manually edited
    page.confidence_score = 1.0  # Human-edited content has full confidence
    
    await db.commit()
    
    return {"message": "Wiki page updated successfully", "page_id": page_id}


async def _has_children(db: AsyncSession, page_id: int) -> bool:
    """Check if a wiki page has children"""
    result = await db.execute(
        select(WikiPage.id)
        .where(WikiPage.parent_id == page_id)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


def _extract_excerpt(content: str, query: str, context_length: int = 150) -> str:
    """Extract a relevant excerpt from content containing the query"""
    content_lower = content.lower()
    query_lower = query.lower()
    
    pos = content_lower.find(query_lower)
    if pos == -1:
        return content[:context_length] + "..."
    
    start = max(0, pos - context_length // 2)
    end = min(len(content), pos + len(query) + context_length // 2)
    
    excerpt = content[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt = excerpt + "..."
    
    return excerpt


@router.get("/generation-status/{project_id}")
async def get_wiki_generation_status(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the current status of wiki generation for a project"""
    await get_project_for_user(project_id, db, current_user)

    # Import here to avoid circular imports
    from backend.services.progress_tracker import progress_tracker
    
    # Get active wiki generation tasks for this project
    tasks = await progress_tracker.get_project_tasks(db, project_id, active_only=True)
    wiki_tasks = [task for task in tasks if task["task_type"] == "wiki_generation"]
    
    if not wiki_tasks:
        # Check if wiki already exists
        result = await db.execute(
            select(WikiStructure).where(WikiStructure.project_id == project_id)
        )
        structure = result.scalar_one_or_none()
        
        if structure and structure.generation_status == "completed":
            return {
                "status": "completed",
                "has_wiki": True,
                "wiki_structure_id": structure.id,
                "last_generated": structure.last_generated_at.isoformat() if structure.last_generated_at else None
            }
        else:
            return {
                "status": "not_started",
                "has_wiki": False
            }
    
    # Return status of the most recent task
    current_task = wiki_tasks[0]
    return {
        "status": current_task["status"],
        "progress": current_task["progress_percentage"],
        "current_step": current_task["current_step"],
        "estimated_duration": current_task["estimated_duration_seconds"],
        "remaining_time": current_task["remaining_time_seconds"],
        "started_at": current_task["started_at"],
        "task_id": current_task["id"]
    } 