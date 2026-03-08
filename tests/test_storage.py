"""Tests for SQLite storage backend."""

import os
import tempfile
import time
import pytest

from neurogram.storage.sqlite_backend import SQLiteBackend
from neurogram.types import AgentConfig, Memory, MemoryType


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        backend = SQLiteBackend(db_path=db_path)
        backend.initialize()
        yield backend
        backend.close()


@pytest.fixture
def sample_agent():
    return AgentConfig(agent_id="test_agent", name="Test Agent")


@pytest.fixture
def sample_memory():
    return Memory(
        content="Test memory content",
        agent_id="test_agent",
        memory_type=MemoryType.SEMANTIC,
        importance_score=0.5,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    )


class TestAgentStorage:
    def test_save_and_load_agent(self, storage, sample_agent):
        storage.save_agent(sample_agent)
        loaded = storage.load_agent("test_agent")
        assert loaded is not None
        assert loaded.name == "Test Agent"

    def test_load_nonexistent_agent(self, storage):
        assert storage.load_agent("nonexistent") is None

    def test_list_agents(self, storage, sample_agent):
        storage.save_agent(sample_agent)
        agents = storage.list_agents()
        assert len(agents) == 1
        assert agents[0].agent_id == "test_agent"

    def test_delete_agent(self, storage, sample_agent):
        storage.save_agent(sample_agent)
        assert storage.delete_agent("test_agent") is True
        assert storage.load_agent("test_agent") is None


class TestMemoryStorage:
    def test_save_and_load_memory(self, storage, sample_agent, sample_memory):
        storage.save_agent(sample_agent)
        storage.save_memory(sample_memory)
        loaded = storage.load_memory(sample_memory.id)
        assert loaded is not None
        assert loaded.content == "Test memory content"
        assert loaded.memory_type == MemoryType.SEMANTIC

    def test_delete_memory(self, storage, sample_agent, sample_memory):
        storage.save_agent(sample_agent)
        storage.save_memory(sample_memory)
        assert storage.delete_memory(sample_memory.id) is True
        assert storage.load_memory(sample_memory.id) is None

    def test_list_memories(self, storage, sample_agent):
        storage.save_agent(sample_agent)
        for i in range(5):
            mem = Memory(content=f"Memory {i}", agent_id="test_agent",
                         embedding=[float(i)] * 5)
            storage.save_memory(mem)

        memories = storage.list_memories("test_agent")
        assert len(memories) == 5

    def test_list_memories_by_type(self, storage, sample_agent):
        storage.save_agent(sample_agent)
        for mt in [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.SEMANTIC]:
            mem = Memory(content=f"Content", agent_id="test_agent",
                         memory_type=mt, embedding=[1.0] * 5)
            storage.save_memory(mem)

        semantic = storage.list_memories("test_agent", MemoryType.SEMANTIC)
        assert len(semantic) == 2

    def test_count_memories(self, storage, sample_agent, sample_memory):
        storage.save_agent(sample_agent)
        storage.save_memory(sample_memory)
        assert storage.count_memories("test_agent") == 1
        assert storage.count_memories("test_agent", MemoryType.SEMANTIC) == 1
        assert storage.count_memories("test_agent", MemoryType.EPISODIC) == 0

    def test_search_by_embedding(self, storage, sample_agent):
        storage.save_agent(sample_agent)

        # Store memories with different embeddings
        m1 = Memory(content="Python programming", agent_id="test_agent",
                     embedding=[1.0, 0.0, 0.0, 0.0, 0.0])
        m2 = Memory(content="Java programming", agent_id="test_agent",
                     embedding=[0.9, 0.1, 0.0, 0.0, 0.0])
        m3 = Memory(content="Cooking recipes", agent_id="test_agent",
                     embedding=[0.0, 0.0, 0.0, 0.0, 1.0])

        storage.save_memory(m1)
        storage.save_memory(m2)
        storage.save_memory(m3)

        # Search with embedding similar to m1
        results = storage.search_by_embedding(
            "test_agent", [1.0, 0.0, 0.0, 0.0, 0.0], top_k=2
        )

        assert len(results) == 2
        # First result should be most similar
        assert results[0][0].content == "Python programming"

    def test_get_memories_below_importance(self, storage, sample_agent):
        storage.save_agent(sample_agent)
        low = Memory(content="Low", agent_id="test_agent",
                      importance_score=0.1, embedding=[1.0])
        high = Memory(content="High", agent_id="test_agent",
                       importance_score=0.9, embedding=[1.0])
        storage.save_memory(low)
        storage.save_memory(high)

        below = storage.get_memories_below_importance("test_agent", 0.5)
        assert len(below) == 1
        assert below[0].content == "Low"
