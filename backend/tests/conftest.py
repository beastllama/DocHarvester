"""Shared test setup. Runs against a real Postgres with pgvector.

Point TEST_DATABASE_URL at a database whose name ends in _test. The suite drops and
recreates every table in that database, so it refuses any other name.

Example:
    createdb docharvester_test
    psql -d docharvester_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
    TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/docharvester_test pytest
"""
import os
import tempfile
from pathlib import Path

import pytest

TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/docharvester_test",
)
if not TEST_DB_URL.rstrip("/").split("/")[-1].endswith("_test"):
    raise RuntimeError("TEST_DATABASE_URL must name a database that ends in _test")

# Must be set before backend.config is imported
os.environ.setdefault("SECRET_KEY", "test-secret-0123456789abcdef-xyz")
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["INGEST_ROOT"] = str(Path(tempfile.gettempdir()) / "dh_ingest_root_test")
Path(os.environ["INGEST_ROOT"]).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def client():
    """App client with a freshly reset schema."""
    from fastapi.testclient import TestClient

    import backend.models  # noqa: F401  registers every table on Base.metadata
    from backend.database import sync_engine
    from backend.main import app
    from backend.models.base import Base

    Base.metadata.drop_all(sync_engine)
    with TestClient(app) as test_client:  # runs the app lifespan: create_all + lenses
        yield test_client
