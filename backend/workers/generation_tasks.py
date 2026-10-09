from datetime import datetime
from typing import Dict, List, Optional
import openai
from sqlalchemy import create_engine, desc, select, and_
from sqlalchemy.orm import sessionmaker
import asyncio
from openai import AsyncOpenAI

from backend.config import settings
from backend.models import Project, DocumentChunk, Document, CoverageStatus
from backend.models.lens import LensType
from backend.workers.celery_app import celery_app
from backend.services.text_processor import TextProcessor
from backend.services.embeddings import EmbeddingService
from backend.services.wiki_generator import WikiGenerator


# Create database session
engine = create_engine(settings.database_url.replace("+asyncpg", ""))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize OpenAI
if settings.openai_api_key:
    openai.api_key = settings.openai_api_key


# Lens-specific prompt templates
LENS_PROMPTS = {
    LensType.LOGIC: """Create a technical documentation page explaining how {project_name} works.

Focus on:
- System architecture and design
- Key components and their interactions
- Implementation details and algorithms
- Technical decisions and trade-offs
- API specifications and data models

Use the following context from existing documentation:
{context}

Format the output as Markdown with clear sections and subsections.
Insert [TODO: <specific information needed>] markers where the context is insufficient.""",

    LensType.SOP: """Create a user guide for {project_name}.

Focus on:
- Getting started instructions
- Step-by-step procedures for common tasks
- Configuration and setup guides
- Troubleshooting tips
- Best practices for users

Use the following context from existing documentation:
{context}

Format the output as Markdown with numbered steps and clear headings.
Insert [TODO: <specific information needed>] markers where the context is insufficient.""",

    LensType.GTM: """Create a go-to-market document for {project_name}.

Focus on:
- Product positioning and value proposition
- Target audience and use cases
- Key features and benefits
- Competitive advantages
- Success stories or case studies

Use the following context from existing documentation:
{context}

Format the output as Markdown suitable for internal teams.
Insert [TODO: <specific information needed>] markers where the context is insufficient.""",

    LensType.CL: """Create a changelog and release notes document for {project_name}.

Focus on:
- Recent updates and new features
- Bug fixes and improvements
- Breaking changes
- Migration guides
- Known issues and workarounds

Use the following context from existing documentation:
{context}

Format the output as Markdown with version numbers and dates.
Insert [TODO: <specific information needed>] markers where the context is insufficient."""
}


@celery_app.task(name="backend.workers.generation_tasks.generate_missing_docs")
def generate_missing_docs(project_id: int, lens_types: List[str], force: bool = False) -> Dict:
    """
    Generate missing documentation for specified lens types
    
    Args:
        project_id: The project ID
        lens_types: List of lens types to generate docs for
        force: Force regeneration even if docs exist
        
    Returns:
        Dict with generation results
    """
    return asyncio.run(_generate_missing_docs_async(project_id, lens_types, force))


async def _generate_missing_docs_async(
    project_id: int, 
    lens_types: List[str], 
    force: bool = False
) -> Dict:
    """Async implementation of document generation"""
    db = SessionLocal()
    
    try:
        # Get project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"Project {project_id} not found"}
        
        # Initialize services
        text_processor = TextProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        embedding_service = EmbeddingService()
        
        results = {
            "project_id": project_id,
            "project_name": project.name,
            "generated": [],
            "errors": []
        }
        
        for lens_type in lens_types:
            try:
                # Get existing content for context
                existing_chunks = db.query(DocumentChunk).join(Document).filter(
                    and_(
                        Document.project_id == project_id,
                        DocumentChunk.lens_type == lens_type,
                        DocumentChunk.is_generated == False  # Only use human-written content
                    )
                ).limit(10).all()
                
                # Get coverage status to understand gaps
                coverage_status = db.query(CoverageStatus).filter(
                    and_(
                        CoverageStatus.project_id == project_id,
                        CoverageStatus.lens_type == lens_type
                    )
                ).first()
                
                missing_topics = coverage_status.missing_topics if coverage_status else []
                
                # Generate content for each missing topic
                for topic in missing_topics[:5]:  # Limit to 5 topics per run
                    doc_result = await _generate_document_for_topic(
                        db, project, lens_type, topic, existing_chunks,
                        text_processor, embedding_service
                    )
                    
                    if doc_result.get("success"):
                        results["generated"].append(doc_result)
                    else:
                        results["errors"].append(doc_result)
                
                # If no specific topics, generate general content
                if not missing_topics:
                    doc_result = await _generate_general_document(
                        db, project, lens_type, existing_chunks,
                        text_processor, embedding_service
                    )
                    
                    if doc_result.get("success"):
                        results["generated"].append(doc_result)
                    else:
                        results["errors"].append(doc_result)
                        
            except Exception as e:
                results["errors"].append({
                    "lens_type": lens_type,
                    "error": str(e)
                })
        
        db.commit()
        
        return results
        
    except Exception as e:
        db.rollback()
        return {
            "error": str(e),
            "project_id": project_id
        }
    finally:
        db.close()


async def _generate_document_for_topic(
    db, project, lens_type: str, topic: str, 
    existing_chunks: List[DocumentChunk],
    text_processor: TextProcessor,
    embedding_service: EmbeddingService
) -> Dict:
    """Generate a document for a specific topic"""
    try:
        # Build context from existing chunks
        context = "\n\n".join([
            f"[{chunk.lens_type}] {chunk.text[:200]}..." 
            for chunk in existing_chunks[:5]
        ])
        
        prompt = _build_generation_prompt(lens_type, topic, project.name, context)

        # Provider-aware: local (Ollama) or the configured cloud provider
        from backend.services.knowledge_graph.local_llm import LocalLLMService
        llm = LocalLLMService()
        full_prompt = f"{_get_system_prompt(lens_type)}\n\n{prompt}"
        content = await llm.query_llm(prompt=full_prompt, temperature=0.7, max_tokens=2000, task_type="general")
        
        # Create document
        doc = Document(
            project_id=project.id,
            doc_id=f"gen_{project.id}_{lens_type}_{topic.replace(' ', '_')}_{datetime.utcnow().timestamp()}",
            title=f"[Generated] {topic} - {lens_type}",
            source_type="generated",
            source_url=f"generated://{lens_type}/{topic}",
            source_meta={
                "generator": "DocHarvester AI",
                "lens_type": lens_type,
                "topic": topic,
                "generation_date": datetime.utcnow().isoformat()
            },
            raw_text=content,
            file_type="md"
        )
        
        db.add(doc)
        db.flush()
        
        # Process into chunks
        text_chunks = text_processor.chunk_text(content)
        
        for idx, chunk in enumerate(text_chunks):
            # Generate embedding
            embedding = await asyncio.to_thread(embedding_service.get_embedding, chunk.text)
            
            document_chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                text=chunk.text,
                embedding=embedding,
                lens_type=lens_type,
                confidence_score=0.85,  # Generated content has lower confidence
                is_generated=True,
                generation_version="1.0",
                generation_status="draft",
                tokens=chunk.tokens,
                chunk_metadata={
                    "topic": topic,
                    "generation_provider": llm.current_provider,
                    "generation_model": settings.local_llm_model if llm.current_provider == "LOCAL" else settings.llm_model
                }
            )
            
            db.add(document_chunk)
        
        return {
            "success": True,
            "lens_type": lens_type,
            "topic": topic,
            "document_id": doc.id,
            "chunks_created": len(text_chunks)
        }
        
    except Exception as e:
        return {
            "success": False,
            "lens_type": lens_type,
            "topic": topic,
            "error": str(e)
        }


async def _generate_general_document(
    db, project, lens_type: str,
    existing_chunks: List[DocumentChunk],
    text_processor: TextProcessor,
    embedding_service: EmbeddingService
) -> Dict:
    """Generate general documentation for a lens type"""
    try:
        # Determine what kind of general doc to create
        doc_templates = {
            "LOGIC": "Technical Architecture Overview",
            "SOP": "Getting Started Guide",
            "GTM": "Product Overview and Benefits",
            "CL": "Recent Updates and Improvements"
        }
        
        topic = doc_templates.get(lens_type, f"{lens_type} Overview")
        
        return await _generate_document_for_topic(
            db, project, lens_type, topic,
            existing_chunks, text_processor, embedding_service
        )
        
    except Exception as e:
        return {
            "success": False,
            "lens_type": lens_type,
            "error": str(e)
        }


def _build_generation_prompt(lens_type: str, topic: str, project_name: str, context: str) -> str:
    """Build the prompt for content generation"""
    prompts = {
        "LOGIC": f"""
Generate technical documentation for "{topic}" in the {project_name} project.

Context from existing documentation:
{context}

Please create comprehensive technical documentation that includes:
1. Overview and purpose
2. Technical implementation details
3. Architecture and design decisions
4. Code examples or configurations where relevant
5. Integration points and dependencies

Make it detailed, accurate, and developer-friendly.
""",
        "SOP": f"""
Generate a standard operating procedure for "{topic}" in the {project_name} project.

Context from existing documentation:
{context}

Please create a clear step-by-step guide that includes:
1. Purpose and scope
2. Prerequisites and requirements
3. Detailed step-by-step instructions
4. Screenshots or diagrams descriptions where helpful
5. Troubleshooting tips
6. Best practices

Make it user-friendly and easy to follow.
""",
        "GTM": f"""
Generate go-to-market documentation for "{topic}" in the {project_name} project.

Context from existing documentation:
{context}

Please create compelling marketing content that includes:
1. Value proposition
2. Target audience and use cases
3. Key features and benefits
4. Competitive advantages
5. Success stories or case studies
6. Call to action

Make it persuasive and customer-focused.
""",
        "CL": f"""
Generate changelog documentation for "{topic}" in the {project_name} project.

Context from existing documentation:
{context}

Please create a detailed changelog that includes:
1. Version information
2. New features and enhancements
3. Bug fixes and improvements
4. Breaking changes or deprecations
5. Migration guides if applicable
6. Future roadmap hints

Make it informative and well-organized.
"""
    }
    
    return prompts.get(lens_type, f"Generate documentation for {topic} in {project_name}")


def _get_system_prompt(lens_type: str) -> str:
    """Get the system prompt for the AI model"""
    return f"""You are a technical documentation expert specializing in creating {lens_type} documentation. 
Your task is to generate high-quality, accurate, and useful documentation based on the provided context and requirements.
Write in a clear, professional tone appropriate for the documentation type.
Use markdown formatting for better readability."""


@celery_app.task(name="backend.workers.generation_tasks.review_generated_docs")
def review_generated_docs(project_id: int) -> Dict:
    """
    Review and potentially approve generated documentation
    
    Args:
        project_id: The project ID
        
    Returns:
        Dict with review results
    """
    db = SessionLocal()
    
    try:
        # Get all draft generated documents
        draft_chunks = db.query(DocumentChunk).join(Document).filter(
            and_(
                Document.project_id == project_id,
                DocumentChunk.is_generated == True,
                DocumentChunk.generation_status == "draft"
            )
        ).all()
        
        reviewed = 0
        approved = 0
        
        for chunk in draft_chunks:
            # Simple auto-approval logic - in production, this would involve
            # more sophisticated checks or human review
            if chunk.confidence_score > 0.8 and len(chunk.text) > 100:
                chunk.generation_status = "final"
                chunk.confidence_score = min(chunk.confidence_score + 0.1, 1.0)
                approved += 1
            
            reviewed += 1
        
        db.commit()
        
        return {
            "project_id": project_id,
            "reviewed": reviewed,
            "approved": approved,
            "status": "completed"
        }
        
    except Exception as e:
        db.rollback()
        return {
            "error": str(e),
            "project_id": project_id
        }
    finally:
        db.close() 