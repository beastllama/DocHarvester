from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel
import os

from backend.database import get_db
from backend.models import User, Project, Document
from backend.api.auth import get_current_user, get_password_hash
from backend.config import settings
from backend.services import runtime_settings
from backend.services.knowledge_graph.local_llm import LocalLLMService

router = APIRouter()


async def update_env_file(env_updates: dict):
    """Update .env file with new environment variables"""
    env_file_path = ".env"
    
    # Read current .env file content
    env_lines = []
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            env_lines = f.readlines()
    
    # Convert to dict for easier manipulation
    env_dict = {}
    for line in env_lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env_dict[key.strip()] = value.strip()
    
    # Update with new values
    env_dict.update(env_updates)
    
    # Write back to file
    with open(env_file_path, 'w') as f:
        for key, value in env_dict.items():
            f.write(f"{key}={value}\n")


# Admin authentication dependency
async def get_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# Dashboard models
class DashboardStats(BaseModel):
    total_users: int
    total_projects: int
    total_documents: int
    recent_users: List[dict]
    recent_projects: List[dict]


# User management models
class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None


class UserCreate(BaseModel):
    email: str
    full_name: str
    password: str
    is_admin: bool = False


# Settings models
class PlatformSettings(BaseModel):
    app_name: str
    debug_mode: bool
    max_file_size_mb: int
    chunk_size: int
    chunk_overlap: int
    worker_batch_size: int
    worker_timeout_seconds: int
    llm_provider: str
    llm_model: str
    embedding_model: str
    llm_temperature: float
    llm_max_tokens: int
    # New LLM-related fields
    use_local_llm: bool
    current_llm_provider: str
    openai_api_key_configured: bool
    openai_organization_id: Optional[str] = None
    local_llm_model: str
    # Model selection fields
    available_openai_models: List[str]
    available_local_models: List[str]


# LLM Management models
class LLMProviderSwitch(BaseModel):
    provider: str  # "LOCAL" or "OPENAI"


class LLMStatus(BaseModel):
    current_provider: str
    openai_configured: bool
    ollama_configured: bool
    openai_status: dict
    ollama_status: dict
    available_models: dict


class OpenAISettings(BaseModel):
    api_key: Optional[str] = None
    organization_id: Optional[str] = None


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user)
):
    """Get admin dashboard statistics"""
    # Get counts
    total_users = await db.scalar(select(func.count(User.id)))
    total_projects = await db.scalar(select(func.count(Project.id)))
    total_documents = await db.scalar(select(func.count(Document.id)))
    
    # Get recent users (last 5)
    recent_users_result = await db.execute(
        select(User)
        .order_by(User.created_at.desc())
        .limit(5)
    )
    recent_users = [
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in recent_users_result.scalars().all()
    ]
    
    # Get recent projects (last 5)
    recent_projects_result = await db.execute(
        select(Project)
        .order_by(Project.created_at.desc())
        .limit(5)
    )
    recent_projects = [
        {
            "id": project.id,
            "name": project.name,
            "created_at": project.created_at.isoformat() if project.created_at else None
        }
        for project in recent_projects_result.scalars().all()
    ]
    
    return DashboardStats(
        total_users=total_users or 0,
        total_projects=total_projects or 0,
        total_documents=total_documents or 0,
        recent_users=recent_users,
        recent_projects=recent_projects
    )


@router.get("/users")
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user)
):
    """Get all users"""
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()
    
    return [
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in users
    ]


@router.post("/users")
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user)
):
    """Create a new user"""
    # Check if user exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        is_admin=user_data.is_admin,
        is_active=True
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_admin": user.is_admin
    }


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user)
):
    """Update a user"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields
    if user_data.email is not None:
        user.email = user_data.email
    if user_data.full_name is not None:
        user.full_name = user_data.full_name
    if user_data.password is not None:
        user.hashed_password = get_password_hash(user_data.password)
    if user_data.is_admin is not None:
        user.is_admin = user_data.is_admin
    if user_data.is_active is not None:
        user.is_active = user_data.is_active
    
    await db.commit()
    await db.refresh(user)
    
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_admin": user.is_admin
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user)
):
    """Delete a user"""
    # Don't allow deleting self
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself"
        )
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    await db.delete(user)
    await db.commit()
    
    return {"message": "User deleted successfully"}


@router.get("/settings", response_model=PlatformSettings)
async def get_settings(
    admin: User = Depends(get_admin_user)
):
    """Get platform settings"""
    return PlatformSettings(
        app_name=settings.app_name,
        debug_mode=settings.debug,
        max_file_size_mb=settings.max_file_size_mb,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        worker_batch_size=settings.worker_batch_size,
        worker_timeout_seconds=settings.worker_timeout_seconds,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        use_local_llm=settings.use_local_llm,
        current_llm_provider=runtime_settings.get_provider(runtime_settings.LLM_KEY),
        openai_api_key_configured=bool(settings.openai_api_key),
        openai_organization_id=settings.openai_organization_id,
        local_llm_model=settings.local_llm_model,
        available_openai_models=settings.available_openai_models,
        available_local_models=settings.available_local_models
    )


@router.put("/settings")
async def update_settings(
    new_settings: PlatformSettings,
    admin: User = Depends(get_admin_user)
):
    """Update platform settings"""
    try:
        # Update .env file with new settings
        env_updates = {}
        
        # Map PlatformSettings fields to environment variable names
        env_mapping = {
            "app_name": "APP_NAME",
            "debug_mode": "DEBUG",
            "max_file_size_mb": "MAX_FILE_SIZE_MB",
            "chunk_size": "CHUNK_SIZE", 
            "chunk_overlap": "CHUNK_OVERLAP",
            "worker_batch_size": "WORKER_BATCH_SIZE",
            "worker_timeout_seconds": "WORKER_TIMEOUT_SECONDS",
            "llm_provider": "LLM_PROVIDER",
            "llm_model": "LLM_MODEL",
            "embedding_model": "EMBEDDING_MODEL",
            "llm_temperature": "LLM_TEMPERATURE",
            "llm_max_tokens": "LLM_MAX_TOKENS",
            "use_local_llm": "USE_LOCAL_LLM",
            "local_llm_model": "LOCAL_LLM_MODEL",
            "openai_organization_id": "OPENAI_ORGANIZATION_ID"
        }
        
        # Fields that should not be saved to environment (computed or dynamic)
        non_mappable_fields = {
            "openai_api_key_configured",  # Computed from OPENAI_API_KEY existence
            "available_openai_models",    # Dynamic list from API
            "available_local_models"      # Dynamic list from Ollama
        }
        
        # Prepare environment updates
        for field_name, env_var in env_mapping.items():
            if hasattr(new_settings, field_name):
                value = getattr(new_settings, field_name)
                # Handle None values for optional fields
                if value is not None:
                    if isinstance(value, bool):
                        env_updates[env_var] = "true" if value else "false"
                    else:
                        env_updates[env_var] = str(value)
        
        # Update .env file
        await update_env_file(env_updates)
        
        # Update current environment variables for immediate effect
        for env_var, value in env_updates.items():
            os.environ[env_var] = value
        
        return {
            "message": "Settings updated successfully",
            "updated_count": len(env_updates),
            "settings": new_settings,
            "skipped_fields": list(non_mappable_fields)
        }
        
    except Exception as e:
        print(f"Error updating settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )


# === NEW LLM MANAGEMENT ENDPOINTS ===

@router.get("/llm/status", response_model=LLMStatus)
async def get_llm_status(
    admin: User = Depends(get_admin_user)
):
    """Get comprehensive LLM status information"""
    llm_service = LocalLLMService()
    
    try:
        # Validate both providers
        openai_status = await llm_service.validate_openai_connection()
        ollama_status = await llm_service.validate_ollama_connection()
        
        # Get available models
        available_models = {
            "openai": settings.available_openai_models if openai_status["valid"] else [],
            "ollama": ollama_status.get("available_models", []) if ollama_status["valid"] else []
        }
        
        return LLMStatus(
            current_provider=llm_service.current_provider,
            openai_configured=bool(settings.openai_api_key),
            ollama_configured=ollama_status["valid"],
            openai_status=openai_status,
            ollama_status=ollama_status,
            available_models=available_models
        )
    finally:
        await llm_service.close()


@router.post("/llm/switch-provider")
async def switch_llm_provider(
    provider_data: LLMProviderSwitch,
    admin: User = Depends(get_admin_user)
):
    """Switch between LOCAL and OPENAI LLM providers"""
    llm_service = LocalLLMService()
    
    try:
        success = llm_service.switch_provider(provider_data.provider)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to switch to {provider_data.provider}. Check configuration."
            )
        
        return {
            "message": f"Successfully switched to {provider_data.provider}",
            "current_provider": llm_service.current_provider,
            "provider_ready": True
        }
    finally:
        await llm_service.close()


@router.post("/llm/test-connection")
async def test_llm_connection(
    admin: User = Depends(get_admin_user)
):
    """Test the current LLM connection with a simple query"""
    llm_service = LocalLLMService()
    
    try:
        test_prompt = "Respond with exactly: 'LLM connection test successful'"
        
        response = await llm_service.query_llm(
            prompt=test_prompt,
            temperature=0.1,
            max_tokens=50,
            json_mode=False
        )
        
        success = "successful" in response.lower()
        
        return {
            "provider": llm_service.current_provider,
            "success": success,
            "response": response,
            "message": "Connection test completed"
        }
    except Exception as e:
        return {
            "provider": llm_service.current_provider,
            "success": False,
            "error": str(e),
            "message": "Connection test failed"
        }
    finally:
        await llm_service.close()


@router.post("/llm/update-openai-settings")
async def update_openai_settings(
    openai_settings: OpenAISettings,
    admin: User = Depends(get_admin_user)
):
    """Update OpenAI API settings"""
    try:
        # Update environment variables
        if openai_settings.api_key:
            os.environ["OPENAI_API_KEY"] = openai_settings.api_key
        
        if openai_settings.organization_id:
            os.environ["OPENAI_ORGANIZATION_ID"] = openai_settings.organization_id
        
        # Test the new settings
        llm_service = LocalLLMService()
        try:
            validation_result = await llm_service.validate_openai_connection()
            
            if validation_result["valid"]:
                return {
                    "success": True,
                    "message": "OpenAI settings updated and validated successfully",
                    "organization_id": validation_result.get("organization_id"),
                    "available_models": validation_result.get("available_models", [])
                }
            else:
                return {
                    "success": False,
                    "message": f"Settings updated but validation failed: {validation_result.get('error', 'Unknown error')}",
                    "error_type": validation_result.get("error_type")
                }
        finally:
            await llm_service.close()
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to update OpenAI settings: {str(e)}"
        }


@router.get("/llm/models")
async def get_available_models(
    provider: Optional[str] = None,
    admin: User = Depends(get_admin_user)
):
    """Get available models for specified provider or current provider"""
    llm_service = LocalLLMService()
    
    try:
        if provider and provider.upper() == "OPENAI":
            status = await llm_service.validate_openai_connection()
            if status["valid"]:
                model_info = {}
                for model in settings.available_openai_models:
                    tier_info = settings.get_model_tier_info(model)
                    model_info[model] = {
                        "context_window": settings.get_model_context_window(model),
                        "large_context": settings.is_large_context_model(model),
                        "recommended_for": tier_info.get("recommended_for", []),
                        "tier": tier_info.get("tier", "unknown"),
                        "cost_per_1k": tier_info.get("cost_per_1k", 0),
                        "provider": "openai"
                    }
                
                return {
                    "provider": "OPENAI",
                    "models": status.get("available_models", settings.available_openai_models),
                    "model_info": model_info,
                    "organization_id": status.get("organization_id")
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"OpenAI not available: {status.get('error', 'Unknown error')}"
                )
        
        elif provider and provider.upper() == "LOCAL":
            status = await llm_service.validate_ollama_connection()
            if status["valid"]:
                model_info = {}
                available_models = status["available_models"]
                
                for model in available_models:
                    tier_info = settings.get_model_tier_info(model)
                    model_info[model] = {
                        "context_window": settings.get_model_context_window(model),
                        "large_context": settings.is_large_context_model(model),
                        "recommended_for": tier_info.get("recommended_for", ["general"]),
                        "tier": tier_info.get("tier", "unknown"),
                        "memory_gb": tier_info.get("memory_gb", "unknown"),
                        "provider": "local"
                    }
                
                return {
                    "provider": "LOCAL",
                    "models": available_models,
                    "model_info": model_info,
                    "recommended_models": settings.available_local_models
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ollama not available: {status.get('error', 'Unknown error')}"
                )
        
        else:
            # Return current provider info
            if llm_service.use_local_llm:
                return await get_available_models("LOCAL", admin)
            else:
                return await get_available_models("OPENAI", admin)
                
    finally:
        await llm_service.close()


@router.get("/llm/model-recommendations/{task}")
async def get_model_recommendations_for_task(
    task: str,
    admin: User = Depends(get_admin_user)
):
    """Get model recommendations for a specific task"""
    recommendations = settings.get_recommended_models_for_task(task)
    
    # Add detailed info for each recommended model
    detailed_recommendations = {}
    
    for provider, models in recommendations.items():
        detailed_recommendations[provider] = []
        for model in models:
            tier_info = settings.get_model_tier_info(model)
            detailed_recommendations[provider].append({
                "model": model,
                "tier": tier_info.get("tier"),
                "cost_per_1k": tier_info.get("cost_per_1k", 0),
                "context_window": settings.get_model_context_window(model),
                "recommended_for": tier_info.get("recommended_for", [])
            })
    
    return {
        "task": task,
        "recommendations": detailed_recommendations,
        "available_tasks": ["entity_extraction", "large_documents", "cost_effective", "production", "development"]
    } 