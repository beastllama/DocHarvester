"""Entity extraction tasks for Celery"""
import asyncio
from typing import Dict, List, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import settings
from backend.models import Project, Document, DocumentChunk
from backend.workers.celery_app import celery_app
from backend.services.knowledge_graph.local_llm import LocalLLMService
from backend.workers.ingest_tasks import _get_entity_types_for_lens, _store_entities_in_neo4j


# Create database session
engine = create_engine(settings.database_url.replace("+asyncpg", ""))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@celery_app.task(name="backend.workers.entity_extraction_tasks.extract_entities_for_project")
def extract_entities_for_project(project_id: int) -> Dict:
    """
    Extract entities for all chunks in a project
    
    Args:
        project_id: The ID of the project to process
        
    Returns:
        Dictionary with extraction results
    """
    return _extract_entities_for_project_sync(project_id)


def _extract_entities_for_project_sync(project_id: int) -> Dict:
    """Synchronous implementation of entity extraction"""
    from backend.database import SessionLocal
    from backend.models import Project, Document, DocumentChunk
    
    db = SessionLocal()
    
    try:
        # Get project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"Project {project_id} not found"}
        
        print(f"🧠 Starting entity extraction for project: {project.name} (ID: {project_id})")
        
        # Get all chunks for the project
        chunks = db.query(DocumentChunk).join(Document).filter(
            Document.project_id == project_id
        ).all()
        
        print(f"📄 Found {len(chunks)} chunks to process")
        
        if not chunks:
            return {"message": "No chunks found to process", "project_id": project_id}
        
        # Initialize LLM service for entity extraction
        llm_service = LocalLLMService()
        llm_service.default_model = "gemma:2b"
        
        entities_extracted = []
        chunks_processed = 0
        chunks_updated = 0
        
        for chunk in chunks:
            try:
                print(f"🔍 Processing chunk {chunk.id} (lens: {chunk.lens_type})")
                
                # Get entity types for this lens
                entity_types = _get_entity_types_for_lens(chunk.lens_type)
                
                # Extract entities using a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    entities_result = loop.run_until_complete(
                        llm_service.extract_entities(chunk.text, entity_types, chunk.lens_type)
                    )
                finally:
                    loop.close()
                
                if isinstance(entities_result, dict) and "entities" in entities_result:
                    entities = entities_result["entities"]
                    print(f"   ✅ Extracted {len(entities)} entities")
                    
                    # Update chunk metadata
                    if chunk.chunk_metadata is None:
                        chunk.chunk_metadata = {}
                    
                    chunk.chunk_metadata["entities"] = entities
                    entities_extracted.extend(entities)
                    chunks_updated += 1
                else:
                    print(f"   ⚠️ No entities found")
                
                chunks_processed += 1
                
            except Exception as e:
                print(f"❌ Failed to extract entities for chunk {chunk.id}: {e}")
                chunks_processed += 1
                continue
        
        # Commit changes
        db.commit()
        
        # Store entities in Neo4j if available
        if entities_extracted:
            try:
                # Group entities by document
                doc_entities = {}
                for chunk in chunks:
                    if chunk.chunk_metadata and "entities" in chunk.chunk_metadata:
                        doc_id = chunk.document_id
                        if doc_id not in doc_entities:
                            doc_entities[doc_id] = []
                        doc_entities[doc_id].extend(chunk.chunk_metadata["entities"])
                
                # Store in Neo4j for each document
                for doc_id, entities in doc_entities.items():
                    doc = db.query(Document).filter(Document.id == doc_id).first()
                    if doc and entities:
                        _store_entities_in_neo4j(doc, entities, project.id)
                
                print(f"✅ Stored {len(entities_extracted)} entities in knowledge graph")
            except Exception as e:
                print(f"Warning: Failed to store entities in Neo4j: {e}")
        
        return {
            "project_id": project_id,
            "project_name": project.name,
            "chunks_processed": chunks_processed,
            "chunks_updated": chunks_updated,
            "entities_extracted": len(entities_extracted),
            "success": True
        }
        
    except Exception as e:
        db.rollback()
        print(f"💥 Fatal error during entity extraction: {e}")
        return {
            "error": str(e),
            "project_id": project_id,
            "success": False
        }
    finally:
        db.close()


@celery_app.task(name="backend.workers.entity_extraction_tasks.extract_entities_for_chunk")
def extract_entities_for_chunk(chunk_id: int) -> Dict:
    """
    Extract entities for a single chunk
    
    Args:
        chunk_id: The ID of the chunk to process
        
    Returns:
        Dictionary with extraction results
    """
    db = SessionLocal()
    
    try:
        # Get chunk
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            return {"error": f"Chunk {chunk_id} not found"}
        
        print(f"🧠 Extracting entities for chunk {chunk_id} (lens: {chunk.lens_type})")
        
        # Initialize LLM service
        llm_service = LocalLLMService()
        llm_service.default_model = "gemma:2b"
        
        # Get entity types
        entity_types = _get_entity_types_for_lens(chunk.lens_type)
        
        # Extract entities
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            entities_result = loop.run_until_complete(
                llm_service.extract_entities(chunk.text, entity_types, chunk.lens_type)
            )
        finally:
            loop.close()
        
        if isinstance(entities_result, dict) and "entities" in entities_result:
            entities = entities_result["entities"]
            
            # Update chunk metadata
            if chunk.chunk_metadata is None:
                chunk.chunk_metadata = {}
            
            chunk.chunk_metadata["entities"] = entities
            db.commit()
            
            print(f"✅ Extracted {len(entities)} entities for chunk {chunk_id}")
            
            return {
                "chunk_id": chunk_id,
                "entities_extracted": len(entities),
                "entities": entities,
                "success": True
            }
        else:
            print(f"⚠️ No entities found for chunk {chunk_id}")
            return {
                "chunk_id": chunk_id,
                "entities_extracted": 0,
                "success": True
            }
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error extracting entities for chunk {chunk_id}: {e}")
        return {
            "error": str(e),
            "chunk_id": chunk_id,
            "success": False
        }
    finally:
        db.close() 