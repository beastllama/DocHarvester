"""Shared access checks for project-scoped routes"""
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Project, User


def is_project_member(project: Project, user: User) -> bool:
    """True if the user is an owner of the project or an admin"""
    return user.is_admin or user.email in (project.owners or [])


async def get_project_for_user(project_id: int, db: AsyncSession, user: User) -> Project:
    """Load a project the user may access. Raises 404 if missing, 403 if not allowed"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if not is_project_member(project, user):
        raise HTTPException(status_code=403, detail="You do not have access to this project")
    return project


async def accessible_project_ids(db: AsyncSession, user: User) -> Optional[List[int]]:
    """Project ids the user can see. None means no filter (admins see everything)"""
    if user.is_admin:
        return None
    result = await db.execute(select(Project.id, Project.owners))
    return [pid for pid, owners in result.all() if user.email in (owners or [])]
