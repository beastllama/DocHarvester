"""Progress tracking API endpoints"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import ProcessingTask
from backend.services.progress_tracker import progress_tracker
from backend.api.auth import get_current_user, User
from backend.api.deps import get_project_for_user


router = APIRouter(tags=["progress"])


async def _authorize_task(task_id: int, db: AsyncSession, user: User) -> None:
    """Raise 404 if the task does not exist, or 403 if the user cannot access its project"""
    result = await db.execute(select(ProcessingTask.project_id).where(ProcessingTask.id == task_id))
    project_id = result.scalar_one_or_none()
    if project_id is None:
        raise HTTPException(status_code=404, detail="Task not found")
    await get_project_for_user(project_id, db, user)


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the status of a specific task"""
    await _authorize_task(task_id, db, current_user)
    
    task_status = await progress_tracker.get_task_status(db, task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status


@router.get("/projects/{project_id}/tasks")
async def get_project_tasks(
    project_id: int,
    active_only: bool = Query(True, description="Only return active tasks"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all tasks for a project"""
    await get_project_for_user(project_id, db, current_user)
    
    tasks = await progress_tracker.get_project_tasks(db, project_id, active_only)
    
    return {
        "project_id": project_id,
        "tasks": tasks,
        "active_count": len([t for t in tasks if t["status"] in ["pending", "running"]])
    }


@router.get("/projects/{project_id}/active-operations")
async def get_active_operations(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all currently active operations for a project with detailed progress"""
    await get_project_for_user(project_id, db, current_user)
    
    tasks = await progress_tracker.get_project_tasks(db, project_id, active_only=True)
    
    # Enrich with additional details for frontend
    operations = []
    for task in tasks:
        operation = {
            "id": task["id"],
            "type": task["task_type"],
            "status": task["status"],
            "progress": task["progress_percentage"],
            "current_step": task["current_step"],
            "estimated_duration": task["estimated_duration_seconds"],
            "remaining_time": task["remaining_time_seconds"],
            "started_at": task["started_at"],
            "title": _get_operation_title(task["task_type"]),
            "description": _get_operation_description(task["task_type"], task["current_step"]),
            "icon": _get_operation_icon(task["task_type"])
        }
        operations.append(operation)
    
    return {
        "project_id": project_id,
        "operations": operations,
        "has_active_operations": len(operations) > 0
    }


def _get_operation_title(task_type: str) -> str:
    """Get user-friendly title for operation type"""
    titles = {
        "wiki_generation": "Generating Wiki",
        "entity_extraction": "Extracting Entities",
        "knowledge_graph_refresh": "Refreshing Knowledge Graph"
    }
    return titles.get(task_type, "Processing")


def _get_operation_description(task_type: str, current_step: str) -> str:
    """Get user-friendly description for current step"""
    descriptions = {
        "wiki_generation": {
            "analyzing_project": "Analyzing project structure and content",
            "extracting_entities": "Extracting entities from documents using AI",
            "generating_structure": "Creating wiki structure and navigation",
            "creating_pages": "Generating wiki pages with AI content",
            "finalizing": "Finalizing wiki and saving to database"
        },
        "entity_extraction": {
            "initializing": "Preparing entity extraction pipeline",
            "processing_chunks": "Processing document chunks for entities",
            "storing_entities": "Storing extracted entities in knowledge graph",
            "creating_relationships": "Mapping relationships between entities"
        },
        "knowledge_graph_refresh": {
            "analyzing_documents": "Analyzing documents for entity updates",
            "extracting_entities": "Re-extracting entities with latest models",
            "mapping_relationships": "Updating entity relationships",
            "updating_graph": "Saving updates to knowledge graph"
        }
    }
    
    task_descriptions = descriptions.get(task_type, {})
    return task_descriptions.get(current_step, f"Processing {current_step}")


def _get_operation_icon(task_type: str) -> str:
    """Get icon name for operation type"""
    icons = {
        "wiki_generation": "article",
        "entity_extraction": "psychology",
        "knowledge_graph_refresh": "account_tree"
    }
    return icons.get(task_type, "settings")


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel a running task (if possible)"""
    await _authorize_task(task_id, db, current_user)
    
    # Note: This is a placeholder - actual cancellation would need to be implemented
    # in the task execution logic (Celery workers, etc.)
    
    task_status = await progress_tracker.get_task_status(db, task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_status["status"] not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")
    
    # For now, just mark as failed - in production this would signal the worker
    success = await progress_tracker.complete_task(
        db, task_id, "failed", error_message="Cancelled by user"
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel task")
    
    return {"message": "Task cancelled successfully", "task_id": task_id}