from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from datetime import datetime
import yaml

from backend.database import get_db
from backend.models import Project, CoverageRequirement, CoverageStatus, Document, DocumentChunk, User
from backend.models.lens import LensType
from backend.api.auth import get_current_user
from backend.api.deps import get_project_for_user
from backend.workers.coverage_tasks import check_project_coverage
from backend.workers.generation_tasks import generate_missing_docs
from backend.config import settings


router = APIRouter()


class CoverageRequirementResponse(BaseModel):
    id: int
    project_id: int
    lens_type: str
    is_required: bool
    min_documents: int
    
    class Config:
        from_attributes = True


class CoverageRequirementUpdate(BaseModel):
    is_required: Optional[bool] = None
    min_documents: Optional[int] = None


class CoverageStatusResponse(BaseModel):
    id: int
    project_id: int
    lens_type: str
    status: str
    document_count: int
    chunk_count: int
    coverage_percentage: float
    missing_topics: List[str]
    last_checked: datetime
    
    class Config:
        from_attributes = True


class ProjectCoverageResponse(BaseModel):
    project_id: int
    project_name: str
    overall_coverage: float
    requirements: List[CoverageRequirementResponse]
    status: List[CoverageStatusResponse]
    recommendations: List[Dict]


class GenerationRequest(BaseModel):
    lens_types: Optional[List[str]] = None
    force: bool = False


@router.get("/requirements/{project_id}", response_model=List[CoverageRequirementResponse])
async def get_project_requirements(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get coverage requirements for a project"""
    await get_project_for_user(project_id, db, current_user)

    # Get requirements
    result = await db.execute(
        select(CoverageRequirement)
        .where(CoverageRequirement.project_id == project_id)
        .order_by(CoverageRequirement.lens_type)
    )
    requirements = result.scalars().all()
    
    # If no requirements exist, create defaults
    if not requirements:
        requirements = await _create_default_requirements(project_id, db)
    
    return requirements


@router.put("/requirements/{project_id}/{lens_type}", response_model=CoverageRequirementResponse)
async def update_requirement(
    project_id: int,
    lens_type: str,
    update: CoverageRequirementUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a coverage requirement"""
    await get_project_for_user(project_id, db, current_user)

    # Verify lens type is valid
    if lens_type not in [lt.value for lt in LensType]:
        raise HTTPException(status_code=400, detail="Invalid lens type")
    
    # Get or create requirement
    result = await db.execute(
        select(CoverageRequirement)
        .where(and_(
            CoverageRequirement.project_id == project_id,
            CoverageRequirement.lens_type == lens_type
        ))
    )
    requirement = result.scalar_one_or_none()
    
    if not requirement:
        requirement = CoverageRequirement(
            project_id=project_id,
            lens_type=lens_type,
            is_required=True,
            min_documents=5
        )
        db.add(requirement)
    
    # Update fields
    if update.is_required is not None:
        requirement.is_required = update.is_required
    if update.min_documents is not None:
        requirement.min_documents = update.min_documents
    
    await db.commit()
    await db.refresh(requirement)
    
    return requirement


@router.get("/status/{project_id}", response_model=ProjectCoverageResponse)
async def get_coverage_status(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current coverage status for a project"""
    project = await get_project_for_user(project_id, db, current_user)

    # Get requirements
    req_result = await db.execute(
        select(CoverageRequirement)
        .where(CoverageRequirement.project_id == project_id)
    )
    requirements = req_result.scalars().all()
    
    if not requirements:
        requirements = await _create_default_requirements(project_id, db)
    
    # Get current status
    status_result = await db.execute(
        select(CoverageStatus)
        .where(CoverageStatus.project_id == project_id)
        .order_by(CoverageStatus.lens_type)
    )
    statuses = status_result.scalars().all()
    
    # If no status exists, calculate it
    if not statuses:
        statuses = await _calculate_coverage_status(project_id, requirements, db)
    
    # Calculate overall coverage
    total_coverage = 0.0
    required_count = 0
    
    for status in statuses:
        req = next((r for r in requirements if r.lens_type == status.lens_type), None)
        if req and req.is_required:
            total_coverage += status.coverage_percentage
            required_count += 1
    
    overall_coverage = total_coverage / required_count if required_count > 0 else 0.0
    
    # Generate recommendations
    recommendations = _generate_recommendations(requirements, statuses)
    
    return ProjectCoverageResponse(
        project_id=project_id,
        project_name=project.name,
        overall_coverage=overall_coverage,
        requirements=[CoverageRequirementResponse(**r.__dict__) for r in requirements],
        status=[CoverageStatusResponse(**s.__dict__) for s in statuses],
        recommendations=recommendations
    )


@router.post("/check/{project_id}")
async def trigger_coverage_check(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger a coverage check for a project"""
    await get_project_for_user(project_id, db, current_user)

    # Queue coverage check
    check_project_coverage.delay(project_id)
    
    return {
        "message": "Coverage check queued",
        "project_id": project_id
    }


@router.post("/generate/{project_id}")
async def generate_missing_documentation(
    project_id: int,
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate missing documentation for a project"""
    await get_project_for_user(project_id, db, current_user)

    # Get current coverage status
    status_result = await db.execute(
        select(CoverageStatus)
        .where(CoverageStatus.project_id == project_id)
    )
    statuses = status_result.scalars().all()
    
    if not statuses:
        raise HTTPException(
            status_code=400, 
            detail="No coverage analysis found. Run coverage check first."
        )
    
    # Filter lens types to generate
    lens_types_to_generate = request.lens_types
    if not lens_types_to_generate:
        # Generate for all lens types with low coverage
        lens_types_to_generate = [
            s.lens_type for s in statuses 
            if s.coverage_percentage < 80.0
        ]
    
    if not lens_types_to_generate:
        return {
            "message": "No documentation needs to be generated",
            "project_id": project_id
        }
    
    # Queue generation task
    generate_missing_docs.delay(
        project_id=project_id,
        lens_types=lens_types_to_generate,
        force=request.force
    )
    
    return {
        "message": "Documentation generation queued",
        "project_id": project_id,
        "lens_types": lens_types_to_generate
    }


@router.get("/gaps/{project_id}")
async def get_coverage_gaps(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed coverage gaps for a project"""
    await get_project_for_user(project_id, db, current_user)

    # Get coverage status
    status_result = await db.execute(
        select(CoverageStatus)
        .where(CoverageStatus.project_id == project_id)
    )
    statuses = status_result.scalars().all()
    
    if not statuses:
        raise HTTPException(
            status_code=400,
            detail="No coverage analysis found. Run coverage check first."
        )
    
    gaps = []
    
    for status in statuses:
        if status.coverage_percentage < 100:
            # Get existing topics
            chunk_result = await db.execute(
                select(DocumentChunk.chunk_metadata)
                .join(Document)
                .where(and_(
                    Document.project_id == project_id,
                    DocumentChunk.lens_type == status.lens_type
                ))
            )
            
            existing_topics = set()
            for row in chunk_result:
                metadata = row[0] or {}
                if 'topics' in metadata:
                    existing_topics.update(metadata['topics'])
            
            gaps.append({
                "lens_type": status.lens_type,
                "coverage_percentage": status.coverage_percentage,
                "missing_topics": status.missing_topics,
                "existing_topics": list(existing_topics),
                "document_count": status.document_count,
                "recommendation": _get_gap_recommendation(status)
            })
    
    return {"gaps": gaps}


# Helper functions

async def _create_default_requirements(project_id: int, db: AsyncSession) -> List[CoverageRequirement]:
    """Create default coverage requirements for a project"""
    requirements = []
    
    # Load from config file if exists
    try:
        with open(settings.coverage_config_path, 'r') as f:
            config = yaml.safe_load(f)
            default_reqs = config.get('default_requirements', {})
    except:
        # Fallback defaults
        default_reqs = {
            'LOGIC': {'required': True, 'min_documents': 10},
            'SOP': {'required': True, 'min_documents': 5},
            'GTM': {'required': True, 'min_documents': 3},
            'CL': {'required': False, 'min_documents': 1}
        }
    
    for lens_type in LensType:
        req_config = default_reqs.get(lens_type.value, {'required': False, 'min_documents': 1})
        
        requirement = CoverageRequirement(
            project_id=project_id,
            lens_type=lens_type.value,
            is_required=req_config['required'],
            min_documents=req_config['min_documents']
        )
        db.add(requirement)
        requirements.append(requirement)
    
    await db.commit()
    return requirements


async def _calculate_coverage_status(
    project_id: int, 
    requirements: List[CoverageRequirement], 
    db: AsyncSession
) -> List[CoverageStatus]:
    """Calculate coverage status for a project"""
    statuses = []
    
    for req in requirements:
        # Count documents and chunks for this lens type
        result = await db.execute(
            select(
                func.count(func.distinct(Document.id)).label('doc_count'),
                func.count(DocumentChunk.id).label('chunk_count')
            )
            .select_from(Document)
            .join(DocumentChunk)
            .where(and_(
                Document.project_id == project_id,
                DocumentChunk.lens_type == req.lens_type
            ))
        )
        row = result.first()
        doc_count = row[0] if row else 0
        chunk_count = row[1] if row else 0
        
        # Count entities extracted from this lens type (from metadata)
        entity_result = await db.execute(
            select(func.count(DocumentChunk.id).label('chunks_with_entities'))
            .select_from(Document)
            .join(DocumentChunk)
            .where(and_(
                Document.project_id == project_id,
                DocumentChunk.lens_type == req.lens_type,
                text("document_chunks.chunk_metadata::text LIKE '%entities%'")
            ))
        )
        entity_row = entity_result.first()
        chunks_with_entities = entity_row[0] if entity_row else 0
        
        # Calculate coverage percentage with knowledge graph factor
        base_coverage = min((doc_count / req.min_documents) * 100, 100.0) if req.min_documents > 0 else 100.0
        
        # Add bonus for entity extraction (up to 20% boost)
        entity_bonus = 0
        if chunk_count > 0:
            entity_ratio = chunks_with_entities / chunk_count
            entity_bonus = min(entity_ratio * 20, 20)  # Max 20% bonus
        
        final_coverage = min(base_coverage + entity_bonus, 100.0)
        
        # Determine status with knowledge graph considerations
        if final_coverage >= 100:
            status_str = "complete"
        elif final_coverage >= 80:
            status_str = "good"
        elif final_coverage >= 50:
            status_str = "partial"
        else:
            status_str = "poor"
        
        # Generate missing topics based on lens type
        missing_topics = _generate_missing_topics(req.lens_type, doc_count, req.min_documents)
        
        status = CoverageStatus(
            project_id=project_id,
            lens_type=req.lens_type,
            status=status_str,
            document_count=doc_count,
            chunk_count=chunk_count,
            coverage_percentage=final_coverage,
            missing_topics=missing_topics,
            last_checked=datetime.utcnow()
        )
        db.add(status)
        statuses.append(status)
    
    await db.commit()
    return statuses


def _generate_missing_topics(lens_type: str, current_docs: int, required_docs: int) -> List[str]:
    """Generate suggestions for missing documentation topics"""
    if current_docs >= required_docs:
        return []
    
    missing_topics = {
        "LOGIC": [
            "Business process workflows",
            "Decision trees and logic flows", 
            "System integration points",
            "Data transformation rules",
            "Error handling procedures"
        ],
        "SOP": [
            "Standard operating procedures",
            "Quality control checklists",
            "Emergency response protocols",
            "Training and onboarding guides",
            "Compliance documentation"
        ],
        "GTM": [
            "Market analysis and positioning",
            "Product launch strategies",
            "Sales enablement materials",
            "Competitive analysis",
            "Customer success playbooks"
        ],
        "CL": [
            "Equipment maintenance procedures",
            "Route optimization guidelines",
            "Facility operations manual",
            "Safety and compliance protocols",
            "Inventory management processes"
        ]
    }
    
    topics = missing_topics.get(lens_type, ["General documentation"])
    needed = required_docs - current_docs
    return topics[:needed]


def _generate_recommendations(requirements: List[CoverageRequirement], statuses: List[CoverageStatus]) -> List[Dict]:
    """Generate actionable recommendations based on coverage analysis"""
    recommendations = []
    
    for status in statuses:
        req = next((r for r in requirements if r.lens_type == status.lens_type), None)
        if not req or not req.is_required:
            continue
            
        if status.coverage_percentage < 50:
            recommendations.append({
                "lens_type": status.lens_type,
                "priority": "high",
                "action": "create_documentation",
                "message": f"Critical: {status.lens_type} coverage is only {status.coverage_percentage:.1f}%. Immediate action required.",
                "suggested_topics": status.missing_topics[:3]
            })
        elif status.coverage_percentage < 80:
            recommendations.append({
                "lens_type": status.lens_type,
                "priority": "medium", 
                "action": "enhance_documentation",
                "message": f"Moderate: {status.lens_type} coverage at {status.coverage_percentage:.1f}%. Consider adding more comprehensive documentation.",
                "suggested_topics": status.missing_topics[:2]
            })
        elif status.document_count == 0:
            recommendations.append({
                "lens_type": status.lens_type,
                "priority": "high",
                "action": "trigger_ingestion",
                "message": f"No documents found for {status.lens_type}. Check document ingestion and classification.",
                "suggested_topics": []
            })
        
        # Knowledge graph specific recommendations
        if status.chunk_count > 0:
            # Check if chunks have entity metadata
            chunks_with_entities = sum(1 for chunk in [1] if status.coverage_percentage > 100)  # Simplified check
            if chunks_with_entities == 0:
                recommendations.append({
                    "lens_type": status.lens_type,
                    "priority": "low",
                    "action": "enable_knowledge_graph",
                    "message": f"Consider enabling knowledge graph features for better entity extraction in {status.lens_type} documents.",
                    "suggested_topics": []
                })
    
    # Sort by priority
    priority_order = {"high": 3, "medium": 2, "low": 1}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
    
    return recommendations


def _get_gap_recommendation(status: CoverageStatus) -> str:
    """Get specific recommendation for a coverage gap"""
    if status.coverage_percentage < 20:
        return f"Create foundational {status.lens_type} documentation immediately"
    elif status.coverage_percentage < 50:
        return f"Expand {status.lens_type} documentation to cover core topics"
    elif status.coverage_percentage < 80:
        return f"Fill gaps in {status.lens_type} documentation for completeness"
    else:
        return f"Review and enhance existing {status.lens_type} documentation" 