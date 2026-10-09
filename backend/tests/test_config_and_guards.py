"""Startup refusal, CORS, Cypher label guard, and embedding failure behaviour.

These do not touch the database.
"""
import pytest
from pydantic import ValidationError

from backend.config import Settings
from backend.services.embeddings import EmbeddingError, EmbeddingService
from backend.services.knowledge_graph.labels import safe_entity_label


def test_missing_secret_is_refused():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, secret_key="")


def test_placeholder_secret_is_refused():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, secret_key="your-secure-secret-key-change-in-production")


def test_real_secret_is_accepted():
    assert Settings(_env_file=None, secret_key="a" * 48).secret_key == "a" * 48


def test_cors_has_no_wildcard():
    assert "*" not in Settings(_env_file=None, secret_key="a" * 48).cors_origins


@pytest.mark.parametrize("raw, expected", [
    ("Process", "Process"),
    ("Equipment", "Equipment"),
    ("Carrier", "Carrier"),
    ("Process) DETACH DELETE n //", "Entity"),
    ("Bad Label", "Entity"),
    ("", "Entity"),
    (None, "Entity"),
    ("Foo", "Entity"),
])
def test_label_guard_allows_only_known_labels(raw, expected):
    assert safe_entity_label(raw) == expected


def test_embedding_never_returns_vectors_when_ollama_is_down(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:9")
    with pytest.raises(EmbeddingError):
        EmbeddingService(provider="LOCAL").get_embedding("some text")
