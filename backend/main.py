"""Main FastAPI application module"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.config import settings
from backend.api import projects, documents, connectors, coverage, auth, admin, wiki, knowledge_graph, progress
from backend.database import init_db, engine, Base
from backend.api.mcp_server import mcp_gate, mcp_session_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    # The MCP session manager can only run once per process, so it starts here and nowhere else
    async with mcp_session_manager.run():
        yield
    # Shutdown
    pass

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["auth"])
app.include_router(admin.router, prefix=f"{settings.api_prefix}/admin", tags=["admin"])
app.include_router(projects.router, prefix=f"{settings.api_prefix}/projects", tags=["projects"])
app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["documents"])
app.include_router(connectors.router, prefix=f"{settings.api_prefix}/connectors", tags=["connectors"])
app.include_router(coverage.router, prefix=f"{settings.api_prefix}/coverage", tags=["coverage"])
app.include_router(wiki.router, prefix=f"{settings.api_prefix}/wiki", tags=["wiki"])
app.include_router(knowledge_graph.router, prefix=f"{settings.api_prefix}/knowledge-graph", tags=["knowledge-graph"])
app.include_router(progress.router, prefix=f"{settings.api_prefix}/progress", tags=["progress"])

# MCP server for Hermes Agent and other clients. Needs a personal API token (Bearer dh_...)
app.add_route("/mcp", mcp_gate)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.app_version} 