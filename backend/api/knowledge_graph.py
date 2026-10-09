"""Knowledge Graph API endpoints"""
from typing import List, Optional, Dict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from pydantic import BaseModel

from backend.database import get_db
from backend.models import Document, DocumentChunk
from backend.api.auth import get_current_user, User
from backend.api.deps import get_project_for_user
from starlette.concurrency import run_in_threadpool
from backend.services.knowledge_graph import neo4j_store
from backend.services.knowledge_graph.local_llm import LocalLLMService
from backend.workers.ingest_tasks import discover_and_ingest_project
from backend.workers.entity_extraction_tasks import extract_entities_for_project


router = APIRouter(tags=["knowledge-graph"])


class KnowledgeGraphStats(BaseModel):
    total_entities: int
    total_relationships: int
    entities_by_type: Dict[str, int]
    last_updated: Optional[datetime]


class EntityExtractionRequest(BaseModel):
    force_reprocess: bool = False
    lens_types: Optional[List[str]] = None


class EntityExtractionFromTextRequest(BaseModel):
    text: str
    lens_type: Optional[str] = None
    use_logistics_entities: bool = False


class EntitySearchRequest(BaseModel):
    query: str
    entity_types: Optional[List[str]] = None
    limit: int = 50





@router.get("/projects/{project_id}/stats")
async def get_knowledge_graph_stats(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> KnowledgeGraphStats:
    """Get knowledge graph statistics for a project"""
    
    # Verify project access (404 if missing, 403 if not a member)
    project = await get_project_for_user(project_id, db, current_user)
    
    # Query Neo4j, scoped to this project
    try:
        stats = await run_in_threadpool(neo4j_store.project_stats, project_id)
        return KnowledgeGraphStats(
            total_entities=stats["total_entities"],
            total_relationships=stats["total_relationships"],
            entities_by_type=stats["entities_by_type"],
            last_updated=stats["last_updated"]
        )
        
    except Exception as e:
        print(f"Neo4j stats query failed: {e}")
        
        # Fallback to counting chunks with entities
        entity_result = await db.execute(
            select(func.count(DocumentChunk.id))
            .select_from(Document)
            .join(DocumentChunk)
            .where(
                Document.project_id == project_id,
                text("document_chunks.chunk_metadata::text LIKE '%entities%'")
            )
        )
        total_entities = entity_result.scalar() or 0
        
        # Get most recent chunk update
        last_updated_result = await db.execute(
            select(func.max(DocumentChunk.updated_at))
            .select_from(Document)
            .join(DocumentChunk)
            .where(Document.project_id == project_id)
        )
        last_updated = last_updated_result.scalar()
        
        return KnowledgeGraphStats(
            total_entities=total_entities,
            total_relationships=0,
            entities_by_type={"fallback_count": total_entities},
            last_updated=last_updated
        )


@router.post("/projects/{project_id}/extract-entities")
async def extract_entities_for_project_endpoint(
    project_id: int,
    request: EntityExtractionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Extract entities from project documents using knowledge graph"""
    
    # Verify project access (404 if missing, 403 if not a member)
    project = await get_project_for_user(project_id, db, current_user)
    
    # Check if documents exist
    doc_count_result = await db.execute(
        select(func.count(Document.id))
        .where(Document.project_id == project_id)
    )
    doc_count = doc_count_result.scalar() or 0
    
    if doc_count == 0:
        raise HTTPException(
            status_code=400, 
            detail="No documents found. Upload or ingest documents first."
        )
    
    # Use dedicated entity extraction task instead of full reingestion
    task = extract_entities_for_project.delay(project_id)
    
    return {
        "message": "Entity extraction started",
        "project_id": project_id,
        "task_id": task.id,
        "documents_to_process": doc_count,
        "force_reprocess": request.force_reprocess,
        "extraction_type": "dedicated_entity_extraction"
    }


@router.post("/projects/{project_id}/reingest-with-entities")
async def reingest_project_with_entities(
    project_id: int,
    request: EntityExtractionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Re-ingest project documents with entity extraction enabled"""
    
    # Verify project access (404 if missing, 403 if not a member)
    project = await get_project_for_user(project_id, db, current_user)
    
    # Check if documents exist
    doc_count_result = await db.execute(
        select(func.count(Document.id))
        .where(Document.project_id == project_id)
    )
    doc_count = doc_count_result.scalar() or 0
    
    if doc_count == 0:
        raise HTTPException(
            status_code=400, 
            detail="No documents found. Upload or ingest documents first."
        )
    
    # Queue full reingestion (when entity extraction is re-enabled in main pipeline)
    task = discover_and_ingest_project.delay(project_id)
    
    return {
        "message": "Full reingestion with entity extraction started",
        "project_id": project_id,
        "task_id": task.id,
        "documents_to_process": doc_count,
        "force_reprocess": request.force_reprocess,
        "extraction_type": "full_pipeline_with_entities"
    }


@router.get("/projects/{project_id}/entities")
async def search_entities(
    project_id: int,
    query: str = "",
    entity_types: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search entities in the knowledge graph for a project"""
    
    # Verify project access (404 if missing, 403 if not a member)
    project = await get_project_for_user(project_id, db, current_user)
    
    # Query Neo4j, scoped to this project
    try:
        entities = await run_in_threadpool(
            neo4j_store.search_entities, project_id, query, entity_types, limit
        )
        return {
            "entities": entities,
            "total_found": len(entities),
            "query": query,
            "entity_types": entity_types,
            "source": "neo4j"
        }
        
    except Exception as e:
        # Fallback to chunk metadata search if Neo4j fails
        print(f"Neo4j query failed: {e}")
        
        # Build query conditions for fallback
        conditions = [Document.project_id == project_id]
        
        if query:
            conditions.append(
                or_(
                    DocumentChunk.text.ilike(f"%{query}%"),
                    DocumentChunk.chunk_metadata.contains(query)
                )
            )
        
        # Search chunks with entities
        result = await db.execute(
            select(DocumentChunk, Document)
            .join(Document)
            .where(and_(*conditions))
            .where(text("document_chunks.chunk_metadata::text LIKE '%entities%'"))
            .limit(limit)
        )
        
        chunks_with_docs = result.all()
        
        entities = []
        for chunk, doc in chunks_with_docs:
            try:
                chunk_metadata = chunk.chunk_metadata or {}
                chunk_entities = chunk_metadata.get("entities", [])
                
                for entity in chunk_entities:
                    entities.append({
                        "name": entity.get("name", "Unknown"),
                        "type": entity.get("type", "Entity"),
                        "properties": entity.get("properties", {}),
                        "confidence": entity.get("confidence", 0.0),
                        "source_document": doc.title,
                        "source_chunk": chunk.chunk_index,
                        "lens_type": chunk.lens_type,
                        "source": "chunk_metadata"
                    })
            except Exception as e:
                continue
        
        return {
            "entities": entities[:limit],
            "total_found": len(entities),
            "query": query,
            "entity_types": entity_types,
            "source": "chunk_metadata_fallback",
            "error": f"Neo4j unavailable: {str(e)}"
        }


@router.get("/projects/{project_id}/neo4j-status")
async def check_neo4j_integration(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Check Neo4j integration status for a project"""
    
    # Verify project access before touching Neo4j (403 if not a member)
    await get_project_for_user(project_id, db, current_user)
    
    try:
        status = await run_in_threadpool(neo4j_store.project_status, project_id)
        return {
            "neo4j_connected": True,
            "project_documents": status["project_documents"],
            "total_entities": status["total_entities"],
            "total_relationships": status["total_relationships"],
            "status": "operational"
        }
    except Exception as e:
        return {
            "neo4j_connected": False,
            "error": str(e),
            "status": "unavailable"
        }


@router.post("/projects/{project_id}/refresh-knowledge-graph")
async def refresh_knowledge_graph(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Refresh the knowledge graph by reprocessing all documents"""
    
    # Verify project access (404 if missing, 403 if not a member)
    project = await get_project_for_user(project_id, db, current_user)
    
    # Clear existing knowledge graph data for this project
    try:
        await run_in_threadpool(neo4j_store.delete_project_graph, project_id)
    except Exception as e:
        print(f"Warning: Could not clear Neo4j data: {e}")
    
    # Trigger reingestion with fresh entity extraction
    task = discover_and_ingest_project.delay(project_id)
    
    return {
        "message": "Knowledge graph refresh started",
        "project_id": project_id,
        "task_id": task.id,
        "status": "processing"
    }


@router.get("/projects/{project_id}/graph")
async def get_project_graph(
    project_id: int,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Documents and entities for a project, as graph nodes and edges"""
    # Verify project access (404 if missing, 403 if not a member)
    await get_project_for_user(project_id, db, current_user)

    graph = await run_in_threadpool(neo4j_store.project_graph, project_id, limit)
    return {
        "project_id": project_id,
        "graph": graph,
        "entity_count": sum(1 for node in graph["nodes"] if node["kind"] == "entity"),
        "relationship_count": len(graph["edges"]),
    }


@router.post("/projects/{project_id}/graph/search")
async def search_knowledge_graph(
    project_id: int,
    query: str,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Keyword search over entity names and properties, within one project"""
    # Verify project access (404 if missing, 403 if not a member)
    await get_project_for_user(project_id, db, current_user)

    results = await run_in_threadpool(neo4j_store.search_entities, project_id, query, None, limit)
    return {
        "query": query,
        "search_type": "keyword",
        "results": results
    }


@router.get("/projects/{project_id}/entities/{entity_name}/related")
async def get_related_entities(
    project_id: int,
    entity_name: str,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Entities that share documents with the named entity, within one project"""
    # Verify project access (404 if missing, 403 if not a member)
    await get_project_for_user(project_id, db, current_user)

    related = await run_in_threadpool(neo4j_store.related_entities, project_id, entity_name, limit)
    return {
        "project_id": project_id,
        "entity": entity_name,
        "related": related
    }


@router.post("/entities/extract")
async def extract_entities_from_text(
    request: EntityExtractionFromTextRequest,
    current_user: User = Depends(get_current_user)
):
    """Extract entities from text using local LLM"""
    llm_service = LocalLLMService()
    
    try:
        # Get entity types
        if request.use_logistics_entities:
            from backend.services.knowledge_graph.labels import LOGISTICS_ENTITIES
            entity_types = LOGISTICS_ENTITIES
        else:
            # Entity templates live in labels.py
            from backend.workers.ingest_tasks import _get_entity_types_for_lens
            entity_types = _get_entity_types_for_lens(request.lens_type or "GENERAL")
        
        # Extract entities
        result = await llm_service.extract_entities(
            text=request.text,
            entity_types=entity_types,
            lens_type=request.lens_type
        )
        
        return result
    finally:
        await llm_service.close()


@router.get("/llm/models")
async def get_available_models(
    current_user: User = Depends(get_current_user)
):
    """Get available LLM models"""
    llm_service = LocalLLMService()
    
    try:
        if llm_service.use_local_llm:
            # Get Ollama models
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{llm_service.ollama_url}/api/tags")
                models = response.json().get("models", [])
                
            return {
                "provider": "ollama",
                "models": [
                    {
                        "name": m["name"],
                        "size": m.get("size", "Unknown"),
                        "recommended_for": _get_model_recommendation(m["name"])
                    }
                    for m in models
                ],
                "recommended_models": llm_service.RECOMMENDED_MODELS
            }
        else:
            return {
                "provider": "openai",
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "recommended_models": {
                    "entity_extraction": "gpt-4",
                    "relationship_mapping": "gpt-3.5-turbo",
                    "summarization": "gpt-3.5-turbo"
                }
            }
    finally:
        await llm_service.close()


@router.post("/llm/pull-model")
async def pull_llm_model(
    model_name: str,
    current_user: User = Depends(get_current_user)
):
    """Pull a specific LLM model for local use"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    llm_service = LocalLLMService()
    
    try:
        success = await llm_service.ensure_model_available(model_name)
        
        return {
            "model": model_name,
            "success": success,
            "message": f"Model {model_name} {'is now available' if success else 'failed to download'}"
        }
    finally:
        await llm_service.close()


def _get_model_recommendation(model_name: str) -> str:
    """Get recommendation for what a model is good at"""
    recommendations = {
        "llama3": "General purpose, excellent for entity extraction",
        "mistral": "Fast and efficient, good for structured output",
        "phi3": "Compact model, efficient for summarization",
        "codellama": "Code understanding and technical documentation",
        "neural-chat": "Conversational AI and natural language understanding"
    }
    
    for key, rec in recommendations.items():
        if key in model_name.lower():
            return rec
    
    return "General purpose model" 