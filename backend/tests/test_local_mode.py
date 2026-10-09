"""Local-only mode, against a real Ollama.

Needs Ollama at OLLAMA_BASE_URL with the embedding model installed. The tests fail, not skip,
when Ollama or the model is missing, so local mode cannot pass silently.
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.config import settings
from backend.services import runtime_settings
from backend.services.embeddings import EmbeddingError, EmbeddingService, local_model_status

REPO = Path(__file__).resolve().parents[2]
ADMIN_EMAIL = "admin@docharvester.com"
ADMIN_PW = "admin-test-password-123"


@pytest.fixture(scope="module")
def ollama():
    status = local_model_status()
    if not status["reachable"]:
        pytest.fail(f"Ollama is not reachable at {status['ollama_url']}. Start it to run local-mode tests.")
    if not status["embedding_model_installed"]:
        pytest.fail(f"Embedding model missing. Run: ollama pull {status['embedding_model']}")
    return status


@pytest.fixture(scope="module")
def admin_headers(client):
    env = dict(os.environ, ADMIN_PASSWORD=ADMIN_PW)
    subprocess.run([sys.executable, "backend/scripts/create_admin.py"], env=env, cwd=REPO,
                   capture_output=True, text=True)
    r = client.post("/api/v1/auth/token", data={"username": ADMIN_EMAIL, "password": ADMIN_PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_local_embedding_is_768_and_stable(ollama):
    svc = EmbeddingService(provider="LOCAL")
    a = svc.get_embedding("Refund policy: customers can return items within 30 days")
    b = svc.get_embedding("Refund policy: customers can return items within 30 days")
    assert len(a) == settings.embedding_dimension == 768
    assert a == b


def test_local_embeddings_separate_meaning(ollama):
    svc = EmbeddingService(provider="LOCAL")
    refund = svc.get_embedding("Refund policy: customers can return items within 30 days")
    paraphrase = svc.get_embedding("How do I get my money back?")
    unrelated = svc.get_embedding("Kubernetes deployments use helm charts")
    assert svc.cosine_similarity(refund, paraphrase) > svc.cosine_similarity(refund, unrelated)


def test_wrong_dimension_is_refused(ollama, monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimension", 1536)
    with pytest.raises(EmbeddingError, match="dimensions"):
        EmbeddingService(provider="LOCAL").get_embedding("anything")


def test_provider_switch_is_stored_for_every_process(client):
    runtime_settings.set_provider(runtime_settings.LLM_KEY, "OPENAI")
    assert runtime_settings.get_provider(runtime_settings.LLM_KEY) == "OPENAI"
    runtime_settings.set_provider(runtime_settings.LLM_KEY, "LOCAL")
    assert runtime_settings.get_provider(runtime_settings.LLM_KEY) == "LOCAL"
    with pytest.raises(ValueError):
        runtime_settings.set_provider(runtime_settings.LLM_KEY, "NOT_A_PROVIDER")


def test_semantic_search_ranks_the_right_document_first(client, ollama, admin_headers):
    from backend.database import SessionLocal
    from backend.models import Document, DocumentChunk, Project

    svc = EmbeddingService(provider="LOCAL")
    docs = [
        ("Refund policy", "Customers can return any item within 30 days for a full refund."),
        ("Deploy guide", "Deploy the service with helm charts on the Kubernetes cluster."),
        ("Holiday schedule", "The office is closed on public holidays and the week of December 24."),
    ]
    with SessionLocal() as s:
        project = Project(name="local-search-test", description="t", owners=[])
        s.add(project)
        s.flush()
        pid = project.id
        for i, (title, body) in enumerate(docs):
            doc = Document(project_id=pid, doc_id=f"local-search-{i}", title=title,
                           source_type="upload", source_url="x", file_type="txt", source_meta={})
            s.add(doc)
            s.flush()
            s.add(DocumentChunk(document_id=doc.id, chunk_index=0, text=body,
                                embedding=svc.get_embedding(body), lens_type="LOGIC", chunk_metadata={}))
        s.commit()

    r = client.post("/api/v1/documents/search/semantic", headers=admin_headers,
                    params={"query": "How do I get my money back?", "project_id": pid, "limit": 3})
    assert r.status_code == 200, r.text
    top = r.json()["results"][0]["document"]["title"]
    assert top == "Refund policy"


def test_ingest_refuses_to_start_when_embedding_model_missing(client, monkeypatch):
    from backend.database import SessionLocal
    from backend.models import Document, Project
    from backend.workers.ingest_tasks import _discover_and_ingest_project_sync

    with SessionLocal() as s:
        project = Project(name="ingest-refusal-test", description="t", owners=[])
        s.add(project)
        s.commit()
        pid = project.id
    monkeypatch.setattr(settings, "local_embedding_model", "no-such-embedding-model")

    result = _discover_and_ingest_project_sync(pid)
    assert "ollama pull no-such-embedding-model" in result.get("error", "")
    with SessionLocal() as s:
        assert s.query(Document).filter(Document.project_id == pid).count() == 0


def test_failed_document_is_kept_visible_without_chunks(client):
    from backend.database import SessionLocal
    from backend.models import Document, DocumentChunk, Project
    from backend.workers.ingest_tasks import _record_ingest_error

    with SessionLocal() as s:
        project = Project(name="failed-doc-test", description="t", owners=[])
        s.add(project)
        s.commit()
        pid = project.id
        result = SimpleNamespace(doc_id="failed-doc-1", title="Broken file", source_type="upload",
                                 source_url="x", file_type="pdf")
        _record_ingest_error(s, pid, result, "Embedding request failed")
        s.commit()
    with SessionLocal() as s:
        doc = s.query(Document).filter(Document.doc_id == "failed-doc-1").one()
        assert doc.source_meta["ingest_status"] == "error"
        assert doc.source_meta["ingest_error"] == "Embedding request failed"
        assert s.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count() == 0


def test_local_chat_answers_with_real_text(client, ollama):
    import asyncio

    from backend.services.knowledge_graph.local_llm import LocalLLMService

    if not ollama["chat_model_installed"]:
        pytest.fail(f"Chat model missing. Run: ollama pull {ollama['chat_model']}")
    llm = LocalLLMService()
    answer = asyncio.run(llm.query_llm(prompt="In one sentence, what is a refund policy?",
                                       temperature=0.2, max_tokens=120, task_type="general"))
    assert answer.strip()
    assert not answer.startswith("Error")


def test_unreachable_llm_raises_instead_of_returning_text(client, monkeypatch):
    import asyncio

    from backend.services.knowledge_graph.local_llm import LocalLLMService

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    llm = LocalLLMService()
    with pytest.raises(Exception, match="Local LLM failed"):
        asyncio.run(llm.query_llm(prompt="hi", max_tokens=20, task_type="general"))


def test_local_model_follows_configured_setting(monkeypatch):
    from backend.services.knowledge_graph.local_llm import LocalLLMService

    monkeypatch.setattr(settings, "local_llm_model", "hermes3:8b")
    llm = LocalLLMService()
    llm.current_provider = "LOCAL"
    assert llm.get_best_model_for_task("entity_extraction") == "hermes3:8b"
    assert llm.get_best_model_for_task("wiki_generation") == "hermes3:8b"
