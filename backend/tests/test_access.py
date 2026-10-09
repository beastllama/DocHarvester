"""Access control, admin bootstrap, and connector safety, against the live app.

Two users (alice owns a project, bob does not) and one admin. Every route that
touches project data must refuse bob, and the list routes must hide alice's project.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

ADMIN_EMAIL = "admin@docharvester.com"
ADMIN_PW = "admin-test-password-123"
A_EMAIL, A_PW = "alice@team.test", "alice-password-123"
B_EMAIL, B_PW = "bob@team.test", "bob-password-123"


def _login(client, email, password):
    r = client.post("/api/v1/auth/token", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def world(client):
    """Admin, two users, alice's project, and one document in it."""
    from backend.database import SessionLocal
    from backend.models import Document

    env = dict(os.environ, ADMIN_PASSWORD=ADMIN_PW)
    first = subprocess.run([sys.executable, "backend/scripts/create_admin.py"],
                           env=env, cwd=REPO, capture_output=True, text=True)
    assert "created" in first.stdout, first.stdout + first.stderr

    adm = _login(client, ADMIN_EMAIL, ADMIN_PW)
    assert client.post("/api/v1/auth/register", headers=adm,
                       json={"email": A_EMAIL, "password": A_PW, "full_name": "A"}).status_code == 200
    assert client.post("/api/v1/auth/register", headers=adm,
                       json={"email": B_EMAIL, "password": B_PW, "full_name": "B"}).status_code == 200
    alice = _login(client, A_EMAIL, A_PW)
    bob = _login(client, B_EMAIL, B_PW)

    project = client.post("/api/v1/projects/", headers=alice,
                          json={"name": "alpha", "description": "alice project"})
    assert project.status_code == 200, project.text
    pid = project.json()["id"]

    with SessionLocal() as s:
        doc = Document(project_id=pid, doc_id="t-secret-1", title="secret", source_type="upload",
                       source_url="x", file_type="txt", source_meta={})
        s.add(doc)
        s.commit()
        doc_id = doc.id

    return {"adm": adm, "alice": alice, "bob": bob, "pid": pid, "doc_id": doc_id}


def test_admin_script_creates_once_and_never_resets(client, world):
    env = dict(os.environ, ADMIN_PASSWORD="different-password-999")
    again = subprocess.run([sys.executable, "backend/scripts/create_admin.py"],
                           env=env, cwd=REPO, capture_output=True, text=True)
    assert "already exists" in again.stdout
    assert client.post("/api/v1/auth/token",
                       data={"username": ADMIN_EMAIL, "password": ADMIN_PW}).status_code == 200
    assert client.post("/api/v1/auth/token",
                       data={"username": ADMIN_EMAIL, "password": "different-password-999"}).status_code == 401


def test_signup_is_admin_only(client, world):
    assert client.post("/api/v1/auth/register", headers=world["alice"],
                       json={"email": "x@y.test", "password": "xxxxxxxxxxxx"}).status_code == 403
    assert client.post("/api/v1/auth/register",
                       json={"email": "z@y.test", "password": "zzzzzzzzzzzz"}).status_code == 401


@pytest.mark.parametrize("method, path, kwargs", [
    ("get", "/api/v1/projects/{pid}", {}),
    ("get", "/api/v1/projects/{pid}/stats", {}),
    ("get", "/api/v1/projects/{pid}/ingestion-status", {}),
    ("put", "/api/v1/projects/{pid}", {"json": {"description": "hacked"}}),
    ("post", "/api/v1/projects/{pid}/ingest", {}),
    ("post", "/api/v1/projects/{pid}/upload", {"files": {"files": ("a.txt", b"hi", "text/plain")}}),
    ("get", "/api/v1/documents/{doc_id}", {}),
    ("get", "/api/v1/documents/{doc_id}/content", {}),
    ("get", "/api/v1/documents/{doc_id}/chunks", {}),
    ("get", "/api/v1/documents/?project_id={pid}", {}),
    ("get", "/api/v1/coverage/requirements/{pid}", {}),
    ("get", "/api/v1/wiki/structure/{pid}", {}),
    ("get", "/api/v1/progress/projects/{pid}/tasks", {}),
    ("get", "/api/v1/knowledge-graph/projects/{pid}/stats", {}),
    ("get", "/api/v1/connectors/project/{pid}/configurations", {}),
])
def test_non_member_is_refused(client, world, method, path, kwargs):
    url = path.format(pid=world["pid"], doc_id=world["doc_id"])
    r = getattr(client, method)(url, headers=world["bob"], **kwargs)
    assert r.status_code == 403, f"{method.upper()} {url} returned {r.status_code}"


def test_member_and_admin_can_read(client, world):
    assert client.get(f"/api/v1/projects/{world['pid']}", headers=world["alice"]).status_code == 200
    assert client.get(f"/api/v1/projects/{world['pid']}", headers=world["adm"]).status_code == 200
    assert client.get(f"/api/v1/documents/{world['doc_id']}", headers=world["alice"]).status_code == 200


def test_lists_hide_other_projects(client, world):
    projects = client.get("/api/v1/projects/", headers=world["bob"]).json()
    assert all(p["id"] != world["pid"] for p in projects)
    docs = client.get("/api/v1/documents/", headers=world["bob"]).json().get("documents", [])
    assert all(d.get("id") != world["doc_id"] for d in docs)


def test_only_admin_can_hand_over_owners(client, world):
    r = client.put(f"/api/v1/projects/{world['pid']}", headers=world["alice"],
                   json={"owners": ["attacker@x.test"]})
    assert r.status_code == 403
    r = client.put(f"/api/v1/projects/{world['pid']}", headers=world["alice"],
                   json={"description": "alice edited"})
    assert r.status_code == 200


def test_connector_folder_root_and_secret_masking(client, world):
    pid = world["pid"]
    outside = client.post(f"/api/v1/connectors/project/{pid}/configure", headers=world["alice"],
                          json={"connector_type": "local_folder", "config": {"folder_path": "/etc"}})
    assert outside.status_code == 400

    inside_root = os.environ["INGEST_ROOT"]
    inside = client.post(f"/api/v1/connectors/project/{pid}/configure", headers=world["alice"],
                         json={"connector_type": "local_folder",
                               "config": {"folder_path": f"{inside_root}/alpha"}})
    assert inside.status_code == 200

    client.post(f"/api/v1/connectors/project/{pid}/configure", headers=world["alice"],
                json={"connector_type": "custom", "config": {"api_key": "abc123"}})
    shown = client.get(f"/api/v1/connectors/project/{pid}/configurations",
                       headers=world["alice"]).json()["connectors"]
    assert shown["custom"]["api_key"] == "********"

    # Sending the masked value back must keep the stored secret
    client.post(f"/api/v1/connectors/project/{pid}/configure", headers=world["alice"],
                json={"connector_type": "custom", "config": {"api_key": "********"}})
    from backend.database import SessionLocal
    from backend.models import Project
    with SessionLocal() as s:
        stored = s.get(Project, pid).connector_configs
    assert stored["custom"]["api_key"] == "abc123"

    # Removal must be saved (in-place JSON edits were lost before)
    assert client.delete(f"/api/v1/connectors/project/{pid}/connector/custom",
                         headers=world["alice"]).status_code == 200
    with SessionLocal() as s:
        assert "custom" not in (s.get(Project, pid).connector_configs or {})


def test_admin_can_delete_user_and_not_self(client, world):
    users = client.get("/api/v1/admin/users", headers=world["adm"]).json()
    rows = users.get("users", users) if isinstance(users, dict) else users
    bob = next(u for u in rows if u["email"] == B_EMAIL)
    assert client.delete(f"/api/v1/admin/users/{bob['id']}", headers=world["adm"]).status_code == 200
    me = client.get("/api/v1/auth/me", headers=world["adm"]).json()
    assert client.delete(f"/api/v1/admin/users/{me['id']}", headers=world["adm"]).status_code == 400


def test_create_project_cannot_hand_over_ownership_or_connectors(client, world):
    """Non-admins always own what they create and cannot set connector config."""
    r = client.post("/api/v1/projects/", headers=world["alice"],
                    json={"name": "create-check", "owners": ["attacker@x.test"]})
    assert r.status_code == 200, r.text
    assert r.json()["owners"] == [A_EMAIL]

    r = client.post("/api/v1/projects/", headers=world["alice"],
                    json={"name": "create-check-2", "connector_configs": {"custom": {"api_key": "x"}}})
    assert r.status_code == 403
