from sqlalchemy import Column, String, Text, Float, Integer, ForeignKey, JSON, Boolean, DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from backend.config import settings
from .base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    doc_id = Column(String(255), unique=True, nullable=False)  # External ID from source
    title = Column(String(500), nullable=False)
    source_type = Column(String(50))  # sharepoint, git, jira, etc.
    source_url = Column(Text)
    source_meta = Column(JSON, default=dict)
    raw_text = Column(Text)
    file_type = Column(String(20))
    last_modified = Column(DateTime(timezone=True))
    
    # Relationships
    project = relationship("Project", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"
    
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimension))
    
    # Classification
    lens_type = Column(String(10), nullable=False)
    confidence_score = Column(Float)
    
    # Importance scoring
    recency_score = Column(Float, default=0.5)
    source_weight = Column(Float, default=1.0)
    lens_weight = Column(Float, default=1.0)
    importance_score = Column(Float, default=0.0)
    
    # Metadata
    tokens = Column(Integer)
    chunk_metadata = Column(JSON, default=dict)
    
    # Auto-generation tracking
    is_generated = Column(Boolean, default=False)
    generation_version = Column(String(50))
    generation_status = Column(String(20), default="manual")  # manual, draft, final
    
    # Relationships
    document = relationship("Document", back_populates="chunks") 