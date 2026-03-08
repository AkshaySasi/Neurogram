"""Tests for embedding engines."""

import pytest
from neurogram.embedding_engine import NumpyEmbeddingEngine, EmbeddingEngine


class TestNumpyEmbeddingEngine:
    """Test the zero-dependency numpy embedding engine."""

    def setup_method(self):
        self.engine = NumpyEmbeddingEngine(dimensions=128)

    def test_embed_produces_vector(self):
        vec = self.engine.embed("Hello world")
        assert len(vec) == 128
        assert all(isinstance(v, float) for v in vec)

    def test_embed_is_normalized(self):
        vec = self.engine.embed("Test text")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01  # Should be unit length

    def test_similar_texts_have_high_similarity(self):
        vec1 = self.engine.embed("Python programming language")
        vec2 = self.engine.embed("Python programming")
        similarity = EmbeddingEngine.cosine_similarity(vec1, vec2)
        assert similarity > 0.5  # Similar texts should score high

    def test_different_texts_have_lower_similarity(self):
        vec1 = self.engine.embed("Python programming language")
        vec2 = self.engine.embed("chocolate cake recipe")
        similarity = EmbeddingEngine.cosine_similarity(vec1, vec2)
        # Different topics should be less similar
        vec3 = self.engine.embed("Python coding tutorial")
        sim_similar = EmbeddingEngine.cosine_similarity(vec1, vec3)
        assert sim_similar > similarity

    def test_embed_empty_string(self):
        vec = self.engine.embed("")
        assert len(vec) == 128
        # Empty string should produce zero vector
        assert all(v == 0.0 for v in vec)

    def test_embed_batch(self):
        texts = ["Hello", "World", "Test"]
        vecs = self.engine.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 128 for v in vecs)

    def test_dimensions_property(self):
        assert self.engine.dimensions == 128

    def test_cosine_similarity_identical(self):
        vec = self.engine.embed("identical text")
        sim = EmbeddingEngine.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_dimension_mismatch(self):
        with pytest.raises(ValueError):
            EmbeddingEngine.cosine_similarity([1.0, 2.0], [1.0])
