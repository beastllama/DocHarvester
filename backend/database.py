from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, create_engine
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.base import Base


# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
)

# Create sync engine for Celery tasks (convert async URL to sync)
sync_database_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
sync_engine = create_engine(
    sync_database_url,
    echo=settings.debug,
    future=True
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create sync session factory for Celery tasks
SessionLocal = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False
)


async def init_db():
    """Initialize database with tables and extensions"""
    async with engine.begin() as conn:
        # Create pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        # create_all never resizes a column. Warn loudly if the size no longer matches.
        size = (await conn.execute(text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'"
        ))).scalar()
        if size is not None and size != settings.embedding_dimension:
            print(f"⚠️ document_chunks.embedding is {size} dimensions but EMBEDDING_DIMENSION is "
                  f"{settings.embedding_dimension}. Run: python backend/scripts/reembed.py")

        # Initialize default lens types using the same connection
        from backend.models.lens import Lens, LensType
        result = await conn.execute(text("SELECT COUNT(*) FROM lenses"))
        count = result.scalar()
        if count == 0:
            default_lenses = [
                Lens(
                    name="Logic Documentation",
                    lens_type=LensType.LOGIC.value,
                    description="Technical documentation explaining how the product works",
                    weight=1.0,
                    prompt_template=None
                ),
                Lens(
                    name="Standard Operating Procedures",
                    lens_type=LensType.SOP.value,
                    description="User guides and step-by-step instructions",
                    weight=1.0,
                    prompt_template=None
                ),
                Lens(
                    name="Go-to-Market",
                    lens_type=LensType.GTM.value,
                    description="Marketing materials and sales documentation",
                    weight=0.8,
                    prompt_template=None
                ),
                Lens(
                    name="Changelog",
                    lens_type=LensType.CL.value,
                    description="Release notes, changelogs, and feedback",
                    weight=0.7,
                    prompt_template=None
                )
            ]
            for lens in default_lenses:
                await conn.execute(
                    text(
                        """
                        INSERT INTO lenses (name, lens_type, description, weight, prompt_template) 
                        VALUES (:name, :lens_type, :description, :weight, :prompt_template)
                        """
                    ),
                    {
                        "name": lens.name,
                        "lens_type": lens.lens_type,
                        "description": lens.description,
                        "weight": lens.weight,
                        "prompt_template": lens.prompt_template,
                    }
                )
            # No need to commit, engine.begin() will commit on exit


async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_async_session():
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        yield session 