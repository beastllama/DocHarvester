"""Document ingestion tasks for Celery"""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select
import os

from backend.config import settings
from backend.models import Project, Document, DocumentChunk
from backend.workers.celery_app import celery_app
from backend.connectors.local_folder import LocalFolderConnector
from backend.connectors.base import SearchResult
from backend.services.text_processor import TextProcessor
from backend.services.classifier import LensClassifier
from backend.services.embeddings import EmbeddingService
from backend.services.knowledge_graph.labels import LOGISTICS_ENTITIES
from backend.services.knowledge_graph.local_llm import LocalLLMService
from backend.services.knowledge_graph.labels import safe_entity_label


# Create database session
engine = create_engine(settings.database_url.replace("+asyncpg", ""))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_default_password")


@celery_app.task(name="backend.workers.ingest_tasks.discover_and_ingest_project")
def discover_and_ingest_project(project_id: int) -> Dict:
    """
    Discover and ingest documents for a project
    
    Args:
        project_id: The ID of the project to process
        
    Returns:
        Dictionary with ingestion results
    """
    # Run the sync version directly instead of async
    return _discover_and_ingest_project_sync(project_id)


def _discover_and_ingest_project_sync(project_id: int) -> Dict:
    """Synchronous implementation of discover and ingest"""
    from backend.database import SessionLocal
    from backend.models import Project
    
    db = SessionLocal()
    
    try:
        # Get project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"Project {project_id} not found"}
        
        print(f"🔍 Starting ingestion for project: {project.name} (ID: {project_id})")
        
        # Initialize services
        text_processor = TextProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        classifier = LensClassifier()
        embedding_service = EmbeddingService()
        
        # Get connectors from project
        connectors = _get_project_connectors(project)
        if not connectors:
            print(f"⚠️ No sources are configured for project {project_id}; nothing to ingest")
            return {
                "project_id": project_id,
                "project_name": project.name,
                "documents_processed": 0,
                "errors": 0,
                "results": []
            }
        print(f"📁 Found {len(connectors)} connectors to process")
        
        all_results = []
        
        for connector in connectors:
            try:
                print(f"🔗 Processing connector: {connector.__class__.__name__}")
                # Use a broad search query to get all documents
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, create a new one
                        search_results = asyncio.run(connector.search("", limit=100))
                    else:
                        search_results = loop.run_until_complete(connector.search("", limit=100))
                except:
                    # Fallback: create new event loop
                    search_results = asyncio.run(connector.search("", limit=100))
                
                print(f"📄 Found {len(search_results)} documents to process")
                
                # Process each document
                for result in search_results:
                    print(f"⚙️  Processing: {result.title}")
                    doc_result = _process_document_sync(
                        db, project, result,
                        text_processor, classifier, embedding_service
                    )
                    all_results.append(doc_result)
                    if doc_result.get("success"):
                        print(f"✅ Successfully processed: {result.title} ({doc_result.get('chunks_created', 0)} chunks)")
                    else:
                        print(f"❌ Failed to process: {result.title} - {doc_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"Error with connector {connector.__class__.__name__}: {e}"
                print(f"❌ {error_msg}")
                all_results.append({
                    "connector": connector.__class__.__name__,
                    "error": str(e)
                })
        
        db.commit()
        
        success_count = len([r for r in all_results if r.get("success")])
        error_count = len([r for r in all_results if r.get("error")])
        
        print(f"🎉 Ingestion complete! Processed {success_count} documents, {error_count} errors")
        
        # Trigger wiki generation if documents were successfully processed
        if success_count > 0:
            from backend.workers.wiki_tasks import update_wiki_after_ingestion
            print(f"📚 Triggering wiki generation for project {project_id}...")
            wiki_task = update_wiki_after_ingestion.delay(project_id)
            print(f"📝 Wiki generation task queued: {wiki_task.id}")
        
        return {
            "project_id": project_id,
            "project_name": project.name,
            "documents_processed": success_count,
            "errors": error_count,
            "results": all_results
        }
        
    except Exception as e:
        db.rollback()
        print(f"💥 Fatal error during ingestion: {e}")
        return {
            "error": str(e),
            "project_id": project_id
        }
    finally:
        db.close()


def _process_document_sync(
    db, project: Project, search_result: SearchResult,
    text_processor: TextProcessor, classifier: LensClassifier,
    embedding_service: EmbeddingService
) -> Dict:
    """Process a single document (synchronous version)"""
    try:
        # Check if document already exists
        existing_doc = db.query(Document).filter(
            Document.doc_id == search_result.doc_id
        ).first()
        
        if existing_doc:
            # Update existing document
            existing_doc.title = search_result.title
            existing_doc.raw_text = search_result.raw_text
            existing_doc.last_modified = search_result.last_modified or datetime.utcnow()
            
            # Delete old chunks
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == existing_doc.id
            ).delete()
            
            doc = existing_doc
        else:
            # Create new document
            doc = Document(
                project_id=project.id,
                doc_id=search_result.doc_id,
                title=search_result.title,
                source_type=search_result.source_type,
                source_url=search_result.source_url,
                source_meta=search_result.source_meta,
                raw_text=search_result.raw_text,
                file_type=search_result.file_type,
                last_modified=search_result.last_modified or datetime.utcnow()
            )
            db.add(doc)
            db.flush()
        
        # Chunk the document
        chunks = text_processor.chunk_text(search_result.raw_text)
        
        # Initialize knowledge graph service for entity extraction
        llm_service = LocalLLMService()
        llm_service.default_model = "gemma:2b"
        entities_extracted = []
        
        # Process each chunk
        chunk_records = []
        for i, chunk in enumerate(chunks):
            # Classify the chunk (synchronous call)
            import asyncio
            try:
                # Try to run async classification if possible
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, use sync fallback
                    lens_type, confidence = classifier._rule_based_classification(chunk.text)
                else:
                    lens_type, confidence = loop.run_until_complete(
                        classifier.classify_chunk(chunk.text, f"{project.name} - {project.description}")
                    )
            except:
                # Fallback to rule-based classification
                lens_type, confidence = classifier._rule_based_classification(chunk.text)
            
            # Extract entities from chunk using knowledge graph service
            entities_result = {"entities": [], "relationships": []}  # Default fallback
            try:
                print(f"🧠 Extracting entities for chunk {i} (lens: {lens_type.value})")
                entity_types = _get_entity_types_for_lens(lens_type.value)
                
                # Use a new event loop for entity extraction to avoid async conflicts
                extraction_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(extraction_loop)
                
                try:
                    entities_result = extraction_loop.run_until_complete(
                        llm_service.extract_entities(chunk.text, entity_types, lens_type.value)
                    )
                    
                    if isinstance(entities_result, dict) and "entities" in entities_result:
                        print(f"   ✅ Extracted {len(entities_result['entities'])} entities")
                    else:
                        print(f"   ⚠️ No entities found, using fallback")
                        entities_result = {"entities": [], "relationships": []}
                        
                finally:
                    extraction_loop.close()
                    
            except Exception as e:
                print(f"❌ Entity extraction failed for chunk {i}: {e}")
                print(f"   Exception type: {type(e)}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                # Ensure entities_result is properly initialized on failure
                entities_result = {"entities": [], "relationships": []}
            
            # Generate embedding
            embedding = embedding_service.get_embedding(chunk.text)
            
            # Calculate importance score
            recency_score = _calculate_recency_score(doc.last_modified)
            source_weight = _get_source_weight(doc.source_type)
            lens_weight = _get_lens_weight(lens_type)
            importance_score = (recency_score * 0.3 + 
                              source_weight * 0.3 + 
                              lens_weight * 0.4)
            
            # Create chunk record
            chunk_record = DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                text=chunk.text,
                embedding=embedding,
                lens_type=lens_type.value,
                confidence_score=confidence,
                recency_score=recency_score,
                source_weight=source_weight,
                lens_weight=lens_weight,
                importance_score=importance_score,
                tokens=chunk.tokens,
                chunk_metadata={
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "entities": entities_result.get("entities", []) if isinstance(entities_result, dict) else []
                }
            )
            db.add(chunk_record)
            chunk_records.append(chunk_record)
        
        db.flush()
        
        # Store extracted entities in Neo4j if available
        if entities_extracted:
            try:
                _store_entities_in_neo4j(doc, entities_extracted, project.id)
                print(f"✅ Stored {len(entities_extracted)} entities in knowledge graph")
            except Exception as e:
                print(f"Warning: Failed to store entities in Neo4j: {e}")
        
        return {
            "success": True,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "chunks_created": len(chunk_records),
            "entities_extracted": len(entities_extracted)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "doc_id": search_result.doc_id
        }


def _calculate_recency_score(last_modified: datetime) -> float:
    """Calculate recency score based on last modified date"""
    if not last_modified:
        return 0.5
    
    days_old = (datetime.utcnow() - last_modified).days
    
    if days_old < 7:
        return 1.0
    elif days_old < 30:
        return 0.8
    elif days_old < 90:
        return 0.6
    elif days_old < 365:
        return 0.4
    else:
        return 0.2


def _get_source_weight(source_type: str) -> float:
    """Get weight based on source type"""
    weights = {
        "git": 1.0,
        "confluence": 0.9,
        "sharepoint": 0.8,
        "local_folder": 0.7,
        "jira": 0.6,
        "auto_generated": 0.5
    }
    return weights.get(source_type, 0.5)


def _get_lens_weight(lens_type) -> float:
    """Get weight based on lens type"""
    # This could be made configurable
    weights = {
        "LOGIC": 1.0,
        "SOP": 1.0,
        "GTM": 0.8,
        "CL": 0.7
    }
    return weights.get(lens_type.value, 0.5)


def _get_project_connectors(project: Project) -> List:
    """Get configured connectors for a project"""
    connectors = []
    
    # Always include uploads folder connector for uploaded files
    uploads_path = f"/app/uploads/{project.id}"
    uploads_connector = LocalFolderConnector({
        "folder_path": uploads_path,
        "allowed_extensions": [".txt", ".md", ".pdf", ".docx", ".html", ".json", ".yml", ".yaml", ".py", ".js", ".ts"]
    })
    connectors.append(uploads_connector)
    
    # Always include local folder connector if configured
    if "local_folder" in project.connector_configs:
        config = project.connector_configs["local_folder"]
        connectors.append(LocalFolderConnector(config))
    
    # Add other connectors as they are implemented
    # if "sharepoint" in project.connector_configs:
    #     connectors.append(SharePointConnector(project.connector_configs["sharepoint"]))
    
    # No sources configured: only the uploads connector exists and its folder is missing or empty.
    # Return no connectors instead of falling back to the process working directory.
    if len(connectors) == 1:  # Only uploads connector
        if not os.path.exists(uploads_path) or not os.listdir(uploads_path):
            return []
    
    return connectors 


def _get_entity_types_for_lens(lens_type: str) -> List[Dict]:
    """Get entity type definitions based on lens type"""
    
    # Base entity types common to all lenses
    base_entities = [
        {
            "name": "Document",
            "properties": ["title", "type", "source", "date"]
        },
        {
            "name": "Section",
            "properties": ["heading", "content", "order"]
        },
        {
            "name": "Concept",
            "properties": ["name", "definition", "category"]
        }
    ]
    
    # Lens-specific entity types
    lens_entities = {
        "LOGIC": [
            {
                "name": "BusinessRule",
                "properties": ["name", "condition", "action", "priority"]
            },
            {
                "name": "Process",
                "properties": ["name", "steps", "inputs", "outputs"]
            },
            {
                "name": "Decision",
                "properties": ["criteria", "options", "outcome"]
            }
        ],
        "SOP": [
            {
                "name": "Procedure",
                "properties": ["name", "steps", "responsible_party", "frequency"]
            },
            {
                "name": "Checklist",
                "properties": ["name", "items", "completion_criteria"]
            },
            {
                "name": "Policy",
                "properties": ["name", "scope", "requirements", "exceptions"]
            }
        ],
        "GTM": [
            {
                "name": "Product",
                "properties": ["name", "features", "target_market", "pricing"]
            },
            {
                "name": "Market",
                "properties": ["name", "size", "characteristics", "trends"]
            },
            {
                "name": "Strategy",
                "properties": ["name", "objectives", "tactics", "timeline"]
            }
        ],
        "CL": [
            {
                "name": "Equipment",
                "properties": ["name", "type", "location", "status", "capacity"]
            },
            {
                "name": "Route",
                "properties": ["origin", "destination", "distance", "duration", "mode"]
            },
            {
                "name": "Facility",
                "properties": ["name", "type", "location", "capacity", "services"]
            }
        ],
        "GENERAL": [
            {
                "name": "Entity",
                "properties": ["name", "type", "description", "category"]
            },
            {
                "name": "Topic",
                "properties": ["name", "description", "keywords"]
            },
            {
                "name": "Reference",
                "properties": ["name", "source", "link", "description"]
            }
        ]
    }
    
    # Return base entities plus lens-specific ones (fallback to GENERAL if lens not found)
    specific_entities = lens_entities.get(lens_type, lens_entities.get("GENERAL", []))
    return base_entities + specific_entities


def _store_entities_in_neo4j(doc: Document, entities: List[Dict], project_id: int) -> None:
    """Store one document's entities in the knowledge graph, scoped to its project"""
    from backend.services.knowledge_graph import neo4j_store
    neo4j_store.write_document_entities(
        doc_id=doc.doc_id,
        title=doc.title or "",
        source_type=doc.source_type or "",
        project_id=project_id,
        entities=entities,
    )
