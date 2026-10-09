from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.config import settings
from backend.database import get_db
from backend.models import User
from backend.api.auth import get_current_user
from backend.api.deps import get_project_for_user
from backend.connectors.local_folder import LocalFolderConnector


router = APIRouter()

SECRET_KEY_MARKERS = ("password", "secret", "token", "api_key")


class ConnectorConfig(BaseModel):
    connector_type: str
    config: Dict[str, Any]


class ConnectorTestResponse(BaseModel):
    success: bool
    message: str


def _ensure_folder_path_in_ingest_root(config: Dict[str, Any]) -> None:
    """Reject a folder_path that does not resolve inside settings.ingest_root"""
    if "folder_path" not in config:
        return
    try:
        resolved = Path(config["folder_path"]).resolve()
    except (TypeError, ValueError, OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid folder_path")
    if not resolved.is_relative_to(Path(settings.ingest_root).resolve()):
        raise HTTPException(status_code=400, detail="folder_path must be inside the ingest root")


def _mask_secrets(value: Any) -> Any:
    """Replace values whose key looks like a credential with a fixed mask"""
    if isinstance(value, dict):
        return {
            k: "********" if isinstance(k, str) and any(m in k.lower() for m in SECRET_KEY_MARKERS)
            else _mask_secrets(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_mask_secrets(v) for v in value]
    return value


@router.get("/available")
async def list_available_connectors(
    current_user: User = Depends(get_current_user)
):
    """List all available connector types"""
    return {
        "connectors": [
            {
                "type": "local_folder",
                "name": "Local Folder",
                "description": "Ingest documents from a local folder",
                "config_schema": {
                    "folder_path": {
                        "type": "string",
                        "description": "Path to the folder to scan",
                        "required": True
                    },
                    "allowed_extensions": {
                        "type": "array",
                        "description": "List of file extensions to include",
                        "required": False,
                        "default": [".txt", ".md", ".pdf", ".docx", ".html", ".json", ".yml", ".yaml"]
                    }
                }
            },
            {
                "type": "sharepoint",
                "name": "SharePoint",
                "description": "Ingest documents from SharePoint (Coming Soon)",
                "config_schema": {
                    "site_url": {"type": "string", "required": True},
                    "client_id": {"type": "string", "required": True},
                    "client_secret": {"type": "string", "required": True, "secret": True}
                },
                "available": False
            },
            {
                "type": "git",
                "name": "Git Repository",
                "description": "Ingest documents from Git repositories (Coming Soon)",
                "config_schema": {
                    "repo_url": {"type": "string", "required": True},
                    "branch": {"type": "string", "default": "main"},
                    "access_token": {"type": "string", "required": False, "secret": True}
                },
                "available": False
            }
        ]
    }


@router.post("/test", response_model=ConnectorTestResponse)
async def test_connector(
    connector: ConnectorConfig,
    current_user: User = Depends(get_current_user)
):
    """Test a connector configuration"""
    # Raised outside the try below so the generic handler does not turn the 400 into a 200
    _ensure_folder_path_in_ingest_root(connector.config)
    try:
        if connector.connector_type == "local_folder":
            conn = LocalFolderConnector(connector.config)
            success = await conn.test_connection()
            
            if success:
                return ConnectorTestResponse(
                    success=True,
                    message="Successfully connected to local folder"
                )
            else:
                return ConnectorTestResponse(
                    success=False,
                    message="Folder does not exist or is not accessible"
                )
        else:
            return ConnectorTestResponse(
                success=False,
                message=f"Connector type '{connector.connector_type}' is not yet implemented"
            )
            
    except Exception as e:
        return ConnectorTestResponse(
            success=False,
            message=f"Error testing connector: {str(e)}"
        )


@router.post("/project/{project_id}/configure")
async def configure_project_connector(
    project_id: int,
    connector: ConnectorConfig,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Configure a connector for a project"""
    project = await get_project_for_user(project_id, db, current_user)
    _ensure_folder_path_in_ingest_root(connector.config)

    # Keep stored values where the client sent back the masked placeholder
    existing = dict(project.connector_configs or {})
    stored = dict(existing.get(connector.connector_type) or {})
    incoming = {
        key: (stored[key] if value == "********" and key in stored else value)
        for key, value in connector.config.items()
    }

    # Assign a new dict. In-place edits to a JSON column are not saved.
    project.connector_configs = {**existing, connector.connector_type: incoming}

    await db.commit()
    await db.refresh(project)
    
    return {
        "message": f"Connector '{connector.connector_type}' configured successfully",
        "project_id": project_id
    }


@router.get("/project/{project_id}/configurations")
async def get_project_connectors(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get configured connectors for a project"""
    project = await get_project_for_user(project_id, db, current_user)

    return {
        "project_id": project_id,
        "connectors": _mask_secrets(project.connector_configs or {})
    }


@router.delete("/project/{project_id}/connector/{connector_type}")
async def remove_project_connector(
    project_id: int,
    connector_type: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a connector configuration from a project"""
    project = await get_project_for_user(project_id, db, current_user)

    # Remove connector config. Assign a new dict so the change is saved.
    if project.connector_configs and connector_type in project.connector_configs:
        project.connector_configs = {
            key: value for key, value in project.connector_configs.items()
            if key != connector_type
        }
        await db.commit()
        
        return {"message": f"Connector '{connector_type}' removed successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_type}' not found in project configuration") 