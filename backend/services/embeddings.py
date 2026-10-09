"""Text embeddings. LOCAL (Ollama) is the default. OPENAI and AZURE_OPENAI are optional.

Every provider must return settings.embedding_dimension floats. Anything else is refused,
because the database column has a fixed size. Failures raise EmbeddingError. No fake vectors.
"""
from typing import Dict, List, Optional

import httpx
from openai import AzureOpenAI, OpenAI

from backend.config import settings
from backend.services import runtime_settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding cannot be produced."""


def _ollama_has_model(installed: List[str], model: str) -> bool:
    return any(name == model or name == f"{model}:latest" for name in installed)


def local_model_status() -> Dict:
    """Is Ollama reachable, and are the local chat and embedding models installed?"""
    status = {
        "ollama_url": settings.ollama_base_url,
        "reachable": False,
        "embedding_model": settings.local_embedding_model,
        "embedding_model_installed": False,
        "chat_model": settings.local_llm_model,
        "chat_model_installed": False,
    }
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
        r.raise_for_status()
        installed = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return status
    status["reachable"] = True
    status["embedding_model_installed"] = _ollama_has_model(installed, settings.local_embedding_model)
    status["chat_model_installed"] = _ollama_has_model(installed, settings.local_llm_model)
    return status


class EmbeddingService:
    """Embeds text with the active embedding provider."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or runtime_settings.get_provider(runtime_settings.EMBEDDING_KEY)
        self.dimension = settings.embedding_dimension
        self._cloud = None
        if self.provider == "OPENAI":
            if not settings.openai_api_key:
                raise EmbeddingError(
                    "Embedding provider is OPENAI but OPENAI_API_KEY is not set. "
                    "Set the key, or switch the embedding provider to LOCAL."
                )
            self._cloud = OpenAI(api_key=settings.openai_api_key)
        elif self.provider == "AZURE_OPENAI":
            if not (settings.azure_openai_api_key and settings.azure_openai_endpoint):
                raise EmbeddingError("Embedding provider is AZURE_OPENAI but the Azure key or endpoint is not set.")
            self._cloud = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version="2023-05-15",
            )

    def check_ready(self) -> None:
        """Raise EmbeddingError with the exact fix if the active provider cannot embed right now."""
        if self.provider != "LOCAL":
            return
        status = local_model_status()
        if not status["reachable"]:
            raise EmbeddingError(
                f"Ollama is not reachable at {status['ollama_url']}. Start it, then try again."
            )
        if not status["embedding_model_installed"]:
            raise EmbeddingError(
                f"Embedding model '{status['embedding_model']}' is not installed. "
                f"Run: ollama pull {status['embedding_model']}"
            )

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "LOCAL":
            vectors = self._embed_ollama(texts)
        else:
            vectors = self._embed_cloud(texts)
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"Embedding model returned {len(vector)} dimensions, but EMBEDDING_DIMENSION is "
                    f"{self.dimension}. Fix the setting, or re-embed with scripts/reembed.py."
                )
        return vectors

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        model = settings.local_embedding_model
        try:
            r = httpx.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=120,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama is not reachable at {settings.ollama_base_url}: {exc}") from exc
        if r.status_code == 404:
            raise EmbeddingError(f"Embedding model '{model}' is not installed. Run: ollama pull {model}")
        if r.status_code != 200:
            raise EmbeddingError(f"Ollama embedding failed ({r.status_code}): {r.text[:200]}")
        return r.json()["embeddings"]

    def _embed_cloud(self, texts: List[str]) -> List[List[float]]:
        model = settings.azure_openai_deployment if self.provider == "AZURE_OPENAI" else settings.embedding_model
        try:
            response = self._cloud.embeddings.create(model=model, input=texts, dimensions=self.dimension)
        except Exception as exc:
            raise EmbeddingError(f"{self.provider} embedding request failed: {exc}") from exc
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        import numpy as np

        a, b = np.array(vec1), np.array(vec2)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
