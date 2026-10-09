import numpy as np
from typing import List
from openai import OpenAI, AzureOpenAI
from backend.config import settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding cannot be produced. Never returns fake vectors."""


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self):
        self.client = self._get_embedding_client()
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    def _get_embedding_client(self):
        """Initialize the embedding client from the active provider.

        The runtime switch (CURRENT_LLM_PROVIDER) is honoured as well as LLM_PROVIDER.
        """
        provider = settings.llm_provider
        if provider == "LOCAL" and settings.current_llm_provider.upper() == "OPENAI":
            provider = "OPENAI"

        if provider == "OPENAI" and settings.openai_api_key:
            return OpenAI(api_key=settings.openai_api_key)
        if provider == "AZURE_OPENAI" and settings.azure_openai_api_key:
            return AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version="2023-05-15"
            )
        return None

    def _require_client(self):
        if not self.client:
            raise EmbeddingError(
                "No embedding provider is configured. Set OPENAI_API_KEY and use the "
                "OPENAI or AZURE_OPENAI provider to create embeddings."
            )
        return self.client

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding

        Raises:
            EmbeddingError: if no provider is configured or the request fails
        """
        client = self._require_client()
        try:
            if settings.llm_provider == "AZURE_OPENAI":
                response = client.embeddings.create(
                    model=settings.azure_openai_deployment,
                    input=text
                )
            else:
                response = client.embeddings.create(
                    model=self.model,
                    input=text
                )

            return response.data[0].embedding

        except Exception as e:
            raise EmbeddingError(f"Embedding request failed: {e}") from e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: if no provider is configured or the request fails
        """
        client = self._require_client()
        try:
            if settings.llm_provider == "AZURE_OPENAI":
                response = client.embeddings.create(
                    model=settings.azure_openai_deployment,
                    input=texts
                )
            else:
                response = client.embeddings.create(
                    model=self.model,
                    input=texts
                )

            return [item.embedding for item in response.data]

        except Exception as e:
            raise EmbeddingError(f"Batch embedding request failed: {e}") from e

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query

        Args:
            query_embedding: The query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, candidate)
            if sim >= threshold:
                similarities.append((i, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
