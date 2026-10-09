from .base import Base
from .project import Project
from .document import Document, DocumentChunk
from .lens import Lens, LensType
from .coverage import CoverageRequirement, CoverageStatus
from .user import User
from .wiki import WikiPage, WikiStructure
from .platform_setting import PlatformSetting
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

__all__ = [
    "Base",
    "Project",
    "Document",
    "DocumentChunk",
    "Lens",
    "LensType",
    "CoverageRequirement",
    "CoverageStatus",
    "User",
    "WikiPage",
    "WikiStructure",
    "PlatformSetting",
]

class ProcessingTask(Base):
    """Model for tracking long-running processing tasks"""
    __tablename__ = "processing_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(String, nullable=False)  # wiki_generation, entity_extraction, knowledge_graph_refresh
    status = Column(String, default="pending")  # pending, running, completed, failed
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String)
    total_steps = Column(Integer)
    completed_steps = Column(Integer, default=0)
    estimated_duration_seconds = Column(Integer)
    elapsed_time_seconds = Column(Float, default=0.0)
    remaining_time_seconds = Column(Integer)
    project_id = Column(Integer, ForeignKey("projects.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    result_data = Column(JSON)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="processing_tasks")
    user = relationship("User", back_populates="processing_tasks")

# Add relationship to existing Project model
Project.processing_tasks = relationship("ProcessingTask", back_populates="project")

# Add relationship to existing User model  
User.processing_tasks = relationship("ProcessingTask", back_populates="user") 