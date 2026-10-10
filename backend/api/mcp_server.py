"""Read-only MCP server at /mcp, for Hermes Agent and other MCP clients.

Clients send a personal API token as `Authorization: Bearer dh_...`. The token's user
gets the same project access as the REST API. Login tokens are refused here, and there
are no write tools.
"""
from typing import Optional, Tuple

from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from backend.api.deps import accessible_project_ids, is_project_member
from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.models import Document, Project, User, WikiPage
from backend.services.api_tokens import user_for_api_token
from backend.services.embeddings import EmbeddingError
from backend.services.knowledge_graph import neo4j_store
from backend.services.semantic_search import search_chunks

MAX_TEXT_CHARS = 8000
NOT_FOUND = "Not found, or you do not have access to it"

mcp = MCPServer(
    name="DocHarvester",
    version=settings.app_version,
    instructions="Read-only access to the projects, documents, wiki pages and entity links "
                 "that the API token's user can see.",
)


def _bearer(headers) -> Optional[str]:
    """The token from an Authorization: Bearer header. The header name is matched without case."""
    for name, value in (headers or {}).items():
        if name.lower() == "authorization":
            scheme, _, token = value.partition(" ")
            return token.strip() if scheme.lower() == "bearer" else None
    return None


async def _caller(db: AsyncSession, ctx: Context) -> User:
    user = await user_for_api_token(db, _bearer(ctx.headers))
    if user is None:
        raise ToolError("API token is missing, invalid, or revoked")
    return user


async def _readable_project(db: AsyncSession, user: User, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    # Same message for missing and forbidden, so a caller cannot tell which projects exist
    if project is None or not is_project_member(project, user):
        raise ToolError(NOT_FOUND)
    return project


def _cap(text: Optional[str], max_chars: int) -> Tuple[str, bool]:
    text = text or ""
    limit = max(1, min(max_chars, MAX_TEXT_CHARS))
    return text[:limit], len(text) > limit


@mcp.tool()
async def list_projects(ctx: Context) -> list[dict]:
    """List the projects you can read."""
    async with AsyncSessionLocal() as db:
        user = await _caller(db, ctx)
        allowed = await accessible_project_ids(db, user)
        statement = select(Project).order_by(Project.id)
        if allowed is not None:
            statement = statement.where(Project.id.in_(allowed))
        rows = (await db.execute(statement)).scalars().all()
        return [{"id": p.id, "name": p.name, "description": p.description} for p in rows]


@mcp.tool()
async def search_documents(ctx: Context, query: str, project_id: Optional[int] = None,
                           limit: int = 10) -> list[dict]:
    """Semantic search over document text. Most similar passages first. Optionally limit to one project."""
    async with AsyncSessionLocal() as db:
        user = await _caller(db, ctx)
        if project_id is not None:
            await _readable_project(db, user, project_id)
        allowed = await accessible_project_ids(db, user)
        try:
            return await search_chunks(
                db,
                query,
                limit=max(1, min(limit, 50)),
                project_id=project_id,
                allowed_project_ids=allowed,
            )
        except EmbeddingError as e:
            raise ToolError(f"Semantic search unavailable: {e}")


@mcp.tool()
async def get_document(ctx: Context, document_id: int, max_chars: int = MAX_TEXT_CHARS) -> dict:
    """Details and text of one document. Text is cut to max_chars (at most 8000)."""
    async with AsyncSessionLocal() as db:
        user = await _caller(db, ctx)
        doc = await db.get(Document, document_id)
        if doc is None:
            raise ToolError(NOT_FOUND)
        await _readable_project(db, user, doc.project_id)
        text, truncated = _cap(doc.raw_text, max_chars)
        return {
            "id": doc.id,
            "project_id": doc.project_id,
            "title": doc.title,
            "source_type": doc.source_type,
            "file_type": doc.file_type,
            "text": text,
            "truncated": truncated,
        }


@mcp.tool()
async def get_wiki_page(ctx: Context, project_id: int, slug: str, max_chars: int = MAX_TEXT_CHARS) -> dict:
    """One wiki page of a project, by its slug. Content is cut to max_chars (at most 8000)."""
    async with AsyncSessionLocal() as db:
        user = await _caller(db, ctx)
        await _readable_project(db, user, project_id)
        result = await db.execute(
            select(WikiPage).where(WikiPage.project_id == project_id, WikiPage.slug == slug).order_by(WikiPage.id)
        )
        page = result.scalars().first()
        if page is None:
            raise ToolError("Wiki page not found")
        content, truncated = _cap(page.content, max_chars)
        return {
            "project_id": project_id,
            "slug": page.slug,
            "title": page.title,
            "content": content,
            "truncated": truncated,
        }


@mcp.tool()
async def related_entities(ctx: Context, project_id: int, entity: str, limit: int = 20) -> list[dict]:
    """Entities that appear in the same documents as the named entity, within one project."""
    async with AsyncSessionLocal() as db:
        user = await _caller(db, ctx)
        await _readable_project(db, user, project_id)
    return await run_in_threadpool(neo4j_store.related_entities, project_id, entity, max(1, min(limit, 50)))


# Stateless: each request is self-contained, so no session state is kept between calls.
# The app is mounted at /mcp in main.py. Its session manager must run inside the app lifespan.
# Host checking is off so Hermes can connect by any name (a Tailscale name, an IP).
# That check guards against DNS rebinding, which works only on servers with no auth.
# Here every request needs an API token, and a browser on a rebound page cannot send one.
mcp_app = mcp.streamable_http_app(
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
mcp_session_manager = mcp.session_manager


class RequireApiToken:
    """Refuses /mcp requests without a valid personal API token, before any MCP code runs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        async with AsyncSessionLocal() as db:
            user = await user_for_api_token(db, _bearer(headers))
        if user is None:
            response = JSONResponse(
                {"detail": "A valid API token is required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


mcp_gate = RequireApiToken(mcp_app)
