"""Runtime provider choice, shared by the API and the Celery workers.

The active provider is stored in the database, so an admin switch reaches the worker too.
Until someone switches, the environment value (LLM_PROVIDER / EMBEDDING_PROVIDER) applies.
"""
from backend.config import settings

LLM_KEY = "llm_provider"
EMBEDDING_KEY = "embedding_provider"
ALLOWED = ("LOCAL", "OPENAI", "AZURE_OPENAI")


def _env_default(key: str) -> str:
    return settings.llm_provider if key == LLM_KEY else settings.embedding_provider


def get_provider(key: str) -> str:
    """Active provider for LLM_KEY or EMBEDDING_KEY. Falls back to the env default if the DB is unavailable."""
    from backend.database import SessionLocal
    from backend.models import PlatformSetting

    try:
        with SessionLocal() as session:
            row = session.get(PlatformSetting, key)
            if row is not None and row.value in ALLOWED:
                return row.value
    except Exception as exc:
        print(f"⚠️ Could not read runtime setting '{key}', using env default: {exc}")
    return _env_default(key)


def set_provider(key: str, value: str) -> str:
    """Persist the provider so every process reads the same value. Returns the stored value."""
    from backend.database import SessionLocal
    from backend.models import PlatformSetting

    value = (value or "").strip().upper()
    if value not in ALLOWED:
        raise ValueError(f"Provider must be one of {', '.join(ALLOWED)}, got '{value}'")
    with SessionLocal() as session:
        row = session.get(PlatformSetting, key)
        if row is None:
            session.add(PlatformSetting(key=key, value=value))
        else:
            row.value = value
        session.commit()
    return value
