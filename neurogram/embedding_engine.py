"""Embedding engines for Neurogram.

Provides multiple embedding backends:
- NumpyEmbeddingEngine: Zero-dependency TF-IDF style embeddings (default)
- LocalEmbeddingEngine: sentence-transformers based (optional)
- OpenAIEmbeddingEngine: OpenAI API based (optional)
"""

from __future__ import annotations

import math
import hashlib
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class EmbeddingEngine(ABC):
    """Abstract base class for embedding engines."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        ...

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score between -1.0 and 1.0.
        """
        if len(vec1) != len(vec2):
            raise ValueError(
                f"Vector dimensions must match: {len(vec1)} != {len(vec2)}"
            )

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class NumpyEmbeddingEngine(EmbeddingEngine):
    """Lightweight embedding engine using hash-based projections.

    This provides a zero-dependency embedding solution that creates
    reasonably useful embeddings using character n-grams and random
    projections via hashing. It's fast and requires no ML models.

    Good for prototyping and when sentence-transformers is not available.
    Not as semantically rich as neural embeddings, but surprisingly
    effective for basic similarity search.
    """

    def __init__(self, dimensions: int = 256, ngram_range: Tuple[int, int] = (2, 4)):
        """Initialize the numpy embedding engine.

        Args:
            dimensions: Dimensionality of output vectors.
            ngram_range: Min and max character n-gram sizes.
        """
        self._dimensions = dimensions
        self._ngram_range = ngram_range

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words and character n-grams."""
        text = text.lower().strip()
        # Word tokens
        words = re.findall(r'\b\w+\b', text)
        # Character n-grams
        ngrams = []
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])
        return words + ngrams

    def _hash_feature(self, feature: str, seed: int = 0) -> int:
        """Hash a feature string to a bucket index."""
        h = hashlib.md5(f"{seed}:{feature}".encode()).hexdigest()
        return int(h, 16)

    def embed(self, text: str) -> List[float]:
        """Generate embedding using hash-based random projection.

        Uses the hashing trick: each token is hashed to multiple
        dimensions, with the hash determining both the dimension
        and the sign (+1/-1). This creates a sparse projection
        that preserves similarity.
        """
        vector = [0.0] * self._dimensions
        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            for seed in range(3):  # Multiple hash functions for stability
                h = self._hash_feature(token, seed)
                idx = h % self._dimensions
                sign = 1.0 if (h // self._dimensions) % 2 == 0 else -1.0
                vector[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class LocalEmbeddingEngine(EmbeddingEngine):
    """Embedding engine using sentence-transformers.

    Provides high-quality semantic embeddings using pre-trained models.
    Requires: pip install neurogram[embeddings]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model.

        Args:
            model_name: HuggingFace model name. Default: all-MiniLM-L6-v2
                        (fast, good quality, 384 dimensions)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddingEngine. "
                "Install it with: pip install neurogram[embeddings]"
            )

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimensions_val = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions_val

    def embed(self, text: str) -> List[float]:
        """Generate embedding using sentence-transformers."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class OpenAIEmbeddingEngine(EmbeddingEngine):
    """Embedding engine using OpenAI's Embedding API.

    Requires: pip install neurogram[openai]
    Also requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """Initialize with OpenAI API.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbeddingEngine. "
                "Install it with: pip install neurogram[openai]"
            )

        self._model = model
        self._client = openai.OpenAI(api_key=api_key)

        # Dimensions for known models
        self._known_dims: Dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimensions_val = self._known_dims.get(model, 1536)

    @property
    def dimensions(self) -> int:
        return self._dimensions_val

    def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]


def get_default_engine() -> EmbeddingEngine:
    """Get the best available embedding engine.

    Tries in order:
    1. LocalEmbeddingEngine (sentence-transformers) — best quality
    2. NumpyEmbeddingEngine — always available fallback

    Returns:
        An initialized EmbeddingEngine instance.
    """
    try:
        return LocalEmbeddingEngine()
    except ImportError:
        pass

    return NumpyEmbeddingEngine()
