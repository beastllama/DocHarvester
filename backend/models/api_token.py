from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from backend.models.base import Base


class ApiToken(Base):
    """Personal API token for MCP clients such as Hermes Agent. Only the SHA-256 hash is stored."""
    __tablename__ = "api_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    token_hash = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
