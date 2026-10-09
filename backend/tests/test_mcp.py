"""Personal API tokens and the read-only MCP server.

Token rules are checked through the REST API. The MCP protocol tests talk to the real app
over HTTP in a separate uvicorn process, because the MCP session manager can only start
once per process, and the test client already runs it.
"""
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import httpx2
import pytest
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client

from backend.services.embeddings import EmbeddingService

REPO = Path(__file__).resolve().parents[2]

OWNER, OTHER, PASSWORD = "mcp-owner@team.test", "mcp-other@team.test", "mcp-password-123"
REFUND_TEXT = "Refunds are issued to the original payment method within 14 days of purchase."
PARKING_TEXT = "Parking permits are issued by the facilities desk on the first floor."
LONG_TEXT = "Refund policy. " * 600  # about 8,400 characters, so the cap applies
TOOLS = {"list_projects", "search_documents", "get_document", "get_wiki_page", "related_entities"}


def _login(client, email):
    r = client.post("/api/v1/auth/token", data={"username": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _make_user(email):
    from backend.api.auth import get_password_hash
    from backend.database import SessionLocal
    from backend.models import User

    with SessionLocal() as s:
        s.add(User(email=email, hashed_password=get_password_hash(PASSWORD), full_name=email,
                   is_active=True, is_admin=False))
        s.commit()


def _create_api_token(client, login_headers, name):
    r = client.post("/api/v1/auth/tokens", headers=login_headers, json={"name": name})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token"].startswith("dh_")
    return body


@pytest.fixture(scope="module")
def world(client):
    """Two users, the owner's project with two documents and a wiki page, and their API tokens."""
    from backend.database import SessionLocal
    from backend.models import Document, DocumentChunk, WikiPage

    _make_user(OWNER)
    _make_user(OTHER)
    owner_login = _login(client, OWNER)
    other_login = _login(client, OTHER)

    project = client.post("/api/v1/projects/", headers=owner_login,
                          json={"name": "mcp-alpha", "description": "owner project"})
    assert project.status_code == 200, project.text
    pid = project.json()["id"]

    embedder = EmbeddingService(provider="LOCAL")
    with SessionLocal() as s:
        refund = Document(project_id=pid, doc_id="mcp-refund", title="Refund policy", source_type="upload",
                          source_url="x", file_type="txt", source_meta={}, raw_text=LONG_TEXT)
        parking = Document(project_id=pid, doc_id="mcp-parking", title="Parking", source_type="upload",
                           source_url="x", file_type="txt", source_meta={}, raw_text=PARKING_TEXT)
        s.add_all([refund, parking])
        s.flush()
        s.add_all([
            DocumentChunk(document_id=refund.id, chunk_index=0, text=REFUND_TEXT, lens_type="SOP",
                          embedding=embedder.get_embedding(REFUND_TEXT)),
            DocumentChunk(document_id=parking.id, chunk_index=0, text=PARKING_TEXT, lens_type="SOP",
                          embedding=embedder.get_embedding(PARKING_TEXT)),
        ])
        s.add(WikiPage(project_id=pid, title="Refunds", slug="refunds", content="Refund steps. " * 50))
        s.commit()
        world = {"refund_id": refund.id}

    owner_token = _create_api_token(client, owner_login, "hermes")["token"]
    other_token = _create_api_token(client, other_login, "hermes")["token"]
    return {"pid": pid, "refund_id": world["refund_id"], "owner_login": owner_login,
            "other_login": other_login, "owner_token": owner_token, "other_token": other_token}


# ---------- REST: token lifecycle ----------

def test_token_is_shown_once_and_stored_only_as_a_hash(client, world):
    from backend.database import SessionLocal
    from backend.models import ApiToken

    created = _create_api_token(client, world["owner_login"], "laptop")
    listed = client.get("/api/v1/auth/tokens", headers=world["owner_login"]).json()

    row_for_this = next(t for t in listed if t["id"] == created["id"])
    assert "token" not in row_for_this
    assert row_for_this["name"] == "laptop"

    with SessionLocal() as s:
        stored = s.get(ApiToken, created["id"])
        assert stored.token_hash == hashlib.sha256(created["token"].encode()).hexdigest()
        assert stored.token_hash != created["token"]


def test_api_token_reads_only_what_its_user_can_read(client, world):
    headers = {"Authorization": f"Bearer {world['owner_token']}"}
    assert client.get(f"/api/v1/projects/{world['pid']}", headers=headers).status_code == 200

    other = {"Authorization": f"Bearer {world['other_token']}"}
    assert client.get(f"/api/v1/projects/{world['pid']}", headers=other).status_code == 403
    assert all(p["id"] != world["pid"] for p in client.get("/api/v1/projects/", headers=other).json())


def test_api_token_cannot_mint_more_tokens(client, world):
    headers = {"Authorization": f"Bearer {world['owner_token']}"}
    r = client.post("/api/v1/auth/tokens", headers=headers, json={"name": "escalate"})
    assert r.status_code == 403


def test_revoked_token_is_refused_and_other_users_cannot_revoke_it(client, world):
    created = _create_api_token(client, world["owner_login"], "to-revoke")
    headers = {"Authorization": f"Bearer {created['token']}"}
    assert client.get("/api/v1/projects/", headers=headers).status_code == 200

    assert client.delete(f"/api/v1/auth/tokens/{created['id']}", headers=world["other_login"]).status_code == 404
    assert client.get("/api/v1/projects/", headers=headers).status_code == 200

    assert client.delete(f"/api/v1/auth/tokens/{created['id']}", headers=world["owner_login"]).status_code == 200
    assert client.get("/api/v1/projects/", headers=headers).status_code == 401


def test_deactivated_user_loses_token_and_login(client):
    from backend.database import SessionLocal
    from backend.models import User

    email = "mcp-leaver@team.test"
    _make_user(email)
    login = _login(client, email)
    token = _create_api_token(client, login, "leaving")["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/projects/", headers=headers).status_code == 200

    with SessionLocal() as s:
        s.query(User).filter(User.email == email).update({"is_active": False})
        s.commit()

    assert client.get("/api/v1/projects/", headers=headers).status_code == 401
    assert client.get("/api/v1/projects/", headers=login).status_code == 401
    assert client.post("/api/v1/auth/token",
                       data={"username": email, "password": PASSWORD}).status_code == 400


# ---------- MCP gate: refuses before any MCP code runs ----------

@pytest.mark.parametrize("auth", [None, "Bearer dh_not-a-real-token", "LOGIN"])
def test_mcp_refuses_missing_unknown_or_login_tokens(client, world, auth):
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if auth == "LOGIN":
        headers.update(world["owner_login"])
    elif auth:
        headers["Authorization"] = auth
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    r = client.post("/mcp", headers=headers, json=body)
    assert r.status_code == 401, r.text


# ---------- Live MCP over HTTP (real uvicorn process) ----------

@pytest.fixture(scope="module")
def live_url(world):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO, env=os.environ.copy(), stdout=log, stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 90
        while True:
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None or time.time() > deadline:
                log.seek(0)
                pytest.fail(f"uvicorn did not start:\n{log.read().decode(errors='replace')[-3000:]}")
            time.sleep(0.5)
        yield f"{base}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=20)
        log.close()


def _run_mcp(url, token, action):
    """Open an MCP session as the token holder, run action(session), and return its result."""
    async def go():
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx2.AsyncClient(headers=headers) as http:
            async with streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await action(session)
    return asyncio.run(go())


def _data(result):
    """The tool's return value, decoded. Fails the test if the tool reported an error."""
    assert not result.is_error, result.content[0].text if result.content else result
    if result.structured_content is not None:
        sc = result.structured_content
        return sc["result"] if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


def test_mcp_exposes_exactly_the_five_read_only_tools(live_url, world):
    async def names(session):
        return {t.name for t in (await session.list_tools()).tools}
    assert _run_mcp(live_url, world["owner_token"], names) == TOOLS


def test_mcp_lists_only_projects_the_token_can_read(live_url, world):
    async def list_projects(session):
        return _data(await session.call_tool("list_projects", {}))
    owner = _run_mcp(live_url, world["owner_token"], list_projects)
    assert [p["name"] for p in owner] == ["mcp-alpha"]
    assert _run_mcp(live_url, world["other_token"], list_projects) == []


def test_mcp_refuses_another_users_project_on_every_tool(live_url, world):
    pid, doc_id = world["pid"], world["refund_id"]

    async def attempts(session):
        return {
            "get_document": await session.call_tool("get_document", {"document_id": doc_id}),
            "get_wiki_page": await session.call_tool("get_wiki_page", {"project_id": pid, "slug": "refunds"}),
            "related_entities": await session.call_tool("related_entities", {"project_id": pid, "entity": "x"}),
            "search_documents": await session.call_tool("search_documents",
                                                        {"query": "refund", "project_id": pid}),
        }

    results = _run_mcp(live_url, world["other_token"], attempts)
    for name, result in results.items():
        assert result.is_error, f"{name} returned data to a user without access"
        assert "Refund" not in result.content[0].text, f"{name} leaked document text"


def test_mcp_get_document_caps_the_text(live_url, world):
    async def read(session):
        capped = _data(await session.call_tool("get_document", {"document_id": world["refund_id"], "max_chars": 20}))
        full = _data(await session.call_tool("get_document", {"document_id": world["refund_id"]}))
        return capped, full

    capped, full = _run_mcp(live_url, world["owner_token"], read)
    assert len(capped["text"]) == 20 and capped["truncated"] is True
    assert len(full["text"]) == 8000 and full["truncated"] is True
    assert full["title"] == "Refund policy"


def test_mcp_semantic_search_ranks_the_right_document_first(live_url, world):
    async def search(session):
        return _data(await session.call_tool("search_documents",
                                             {"query": "how do I get my money back", "project_id": world["pid"]}))
    results = _run_mcp(live_url, world["owner_token"], search)
    assert results[0]["document"]["title"] == "Refund policy"


def test_mcp_related_entities_is_scoped_to_the_project(live_url, world):
    async def related(session):
        return _data(await session.call_tool("related_entities", {"project_id": world["pid"], "entity": "Nobody"}))
    assert _run_mcp(live_url, world["owner_token"], related) == []


def _leaves(error):
    """The innermost exceptions inside anyio's nested ExceptionGroups."""
    subs = getattr(error, "exceptions", None)
    if not subs:
        return [error]
    return [leaf for sub in subs for leaf in _leaves(sub)]


def test_mcp_refuses_connection_without_a_token(live_url):
    # The server answers with an error (the 401 from the gate), not a connection failure
    with pytest.raises(BaseException) as info:
        _run_mcp(live_url, None, lambda session: session.list_tools())
    leaves = _leaves(info.value)
    assert leaves and all(isinstance(leaf, MCPError) for leaf in leaves), leaves
