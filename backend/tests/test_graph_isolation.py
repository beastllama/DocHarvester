"""Knowledge graph isolation, against a real Neo4j.

Two projects share an entity name. Each project must see only its own graph.
Needs Neo4j reachable at NEO4J_URI (default bolt://localhost:7687). The tests fail, not skip,
when Neo4j is down, so a missing database cannot pass silently.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.services.knowledge_graph import neo4j_store

REPO = Path(__file__).resolve().parents[2]
ADMIN_EMAIL = "admin@docharvester.com"
ADMIN_PW = "admin-test-password-123"
ALICE, ALICE_PW = "graph-alice@team.test", "alice-graph-password-123"
BOB, BOB_PW = "graph-bob@team.test", "bob-graph-password-123"


def _login(client, email, password):
    r = client.post("/api/v1/auth/token", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def graph(client):
    """Two projects owned by alice, each with its own documents. Bob is not a member."""
    from backend.database import SessionLocal  # noqa: F401  (ensures DB is configured)

    try:
        neo4j_store.project_status(0)
    except Exception as exc:  # fail loudly: an unreachable Neo4j must not pass silently
        pytest.fail(f"Neo4j is not reachable at NEO4J_URI: {exc}")

    env = dict(os.environ, ADMIN_PASSWORD=ADMIN_PW)
    subprocess.run([sys.executable, "backend/scripts/create_admin.py"],
                   env=env, cwd=REPO, capture_output=True, text=True)
    adm = _login(client, ADMIN_EMAIL, ADMIN_PW)
    assert client.post("/api/v1/auth/register", headers=adm,
                       json={"email": ALICE, "password": ALICE_PW, "full_name": "A"}).status_code == 200
    assert client.post("/api/v1/auth/register", headers=adm,
                       json={"email": BOB, "password": BOB_PW, "full_name": "B"}).status_code == 200
    alice = _login(client, ALICE, ALICE_PW)
    bob = _login(client, BOB, BOB_PW)

    pa = client.post("/api/v1/projects/", headers=alice,
                     json={"name": "graph-alpha", "description": "a"}).json()["id"]
    pb = client.post("/api/v1/projects/", headers=alice,
                     json={"name": "graph-beta", "description": "b"}).json()["id"]

    # Clear anything a previous run left under these project ids
    neo4j_store.delete_project_graph(pa)
    neo4j_store.delete_project_graph(pb)

    neo4j_store.write_document_entities("g-a-1", "A1", "upload", pa, [
        {"type": "Process", "name": "Acme", "properties": {"owner": "ops"}},
        {"type": "Procedure", "name": "Onboarding"},
    ])
    neo4j_store.write_document_entities("g-a-2", "A2", "upload", pa, [
        {"type": "Process", "name": "Acme"},
        {"type": "Concept", "name": "Billing"},
    ])
    neo4j_store.write_document_entities("g-b-1", "B1", "upload", pb, [
        {"type": "Process", "name": "Acme"},
        {"type": "Concept", "name": "Shipping"},
    ])

    yield {"alice": alice, "bob": bob, "pa": pa, "pb": pb}

    neo4j_store.delete_project_graph(pa)
    neo4j_store.delete_project_graph(pb)


def test_same_entity_name_is_separate_per_project(graph):
    with neo4j_store._get_driver().session() as session:
        count = session.run(
            "MATCH (e {name: 'Acme'}) WHERE e.project_id IN [$pa, $pb] RETURN count(e) AS n",
            pa=graph["pa"], pb=graph["pb"],
        ).single()["n"]
    assert count == 2


def test_status_counts_only_own_project(graph):
    a = neo4j_store.project_status(graph["pa"])
    b = neo4j_store.project_status(graph["pb"])
    assert (a["project_documents"], a["total_entities"]) == (2, 3)   # Acme, Onboarding, Billing
    assert (b["project_documents"], b["total_entities"]) == (1, 2)   # Acme, Shipping


def test_related_route_returns_only_own_project(client, graph):
    r = client.get(f"/api/v1/knowledge-graph/projects/{graph['pa']}/entities/Acme/related",
                   headers=graph["alice"])
    assert r.status_code == 200, r.text
    names = {item["name"] for item in r.json()["related"]}
    assert names == {"Onboarding", "Billing"}
    assert "Shipping" not in names


def test_related_route_refuses_non_member(client, graph):
    r = client.get(f"/api/v1/knowledge-graph/projects/{graph['pa']}/entities/Acme/related",
                   headers=graph["bob"])
    assert r.status_code == 403


def test_search_and_graph_routes_are_scoped(client, graph):
    hits = client.post(f"/api/v1/knowledge-graph/projects/{graph['pa']}/graph/search?query=Shipping",
                       headers=graph["alice"]).json()["results"]
    assert hits == []

    graph_view = client.get(f"/api/v1/knowledge-graph/projects/{graph['pa']}/graph",
                            headers=graph["alice"]).json()["graph"]
    labels = {node["label"] for node in graph_view["nodes"]}
    assert "Shipping" not in labels and "B1" not in labels
    assert {"Acme", "Billing", "A1", "A2"} <= labels


def test_status_route_is_scoped(client, graph):
    body = client.get(f"/api/v1/knowledge-graph/projects/{graph['pa']}/neo4j-status",
                      headers=graph["alice"]).json()
    assert body["neo4j_connected"] is True
    assert body["project_documents"] == 2


def test_unsafe_entity_type_cannot_inject_cypher(graph):
    neo4j_store.write_document_entities("g-a-evil", "evil", "upload", graph["pa"], [
        {"type": "Process) DETACH DELETE d //", "name": "Trap"},
    ])
    with neo4j_store._get_driver().session() as session:
        trap_label = session.run(
            "MATCH (e {name: 'Trap', project_id: $pa}) RETURN labels(e) AS labels", pa=graph["pa"]
        ).single()["labels"]
        docs = session.run(
            "MATCH (d:Document {project_id: $pa}) RETURN count(d) AS n", pa=graph["pa"]
        ).single()["n"]
    assert trap_label == ["Entity"]
    assert docs == 3  # the injected DETACH DELETE did not run


def test_delete_one_project_leaves_the_other(graph):
    neo4j_store.delete_project_graph(graph["pa"])
    assert neo4j_store.project_status(graph["pa"])["project_documents"] == 0
    b = neo4j_store.project_status(graph["pb"])
    assert (b["project_documents"], b["total_entities"]) == (1, 2)
