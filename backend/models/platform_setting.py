from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String

from .base import Base


class PlatformSetting(Base):
    """Runtime settings that the API and the workers must agree on, such as the active LLM provider."""
    __tablename__ = "platform_settings"

    key = Column(String(100), primary_key=True)
    value = Column(String(500), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
