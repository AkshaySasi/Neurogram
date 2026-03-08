"""Tests for the core Agent API - the primary developer interface."""

import os
import tempfile
import pytest

from neurogram import Agent, MemoryType


@pytest.fixture
def agent():
    """Create a fresh agent with temp storage for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        ag = Agent("TestAgent", storage_path=db_path)
        yield ag
        ag.close()


class TestAgentBasics:
    """Test basic agent creation and identity."""

    def test_create_agent(self, agent):
        assert agent is not None
        stats = agent.stats()
        assert stats["agent_name"] == "TestAgent"
        assert stats["total_memories"] == 0

    def test_agent_repr(self, agent):
        r = repr(agent)
        assert "TestAgent" in r
        assert "memories=" in r

    def test_agent_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            with Agent("CtxAgent", storage_path=db_path) as ag:
                ag.remember("test memory")
                stats = ag.stats()
                assert stats["total_memories"] == 1


class TestRememberRecall:
    """Test storing and retrieving memories."""

    def test_remember_and_recall(self, agent):
        agent.remember("Python is a programming language")
        agent.remember("JavaScript runs in the browser")

        results = agent.recall("programming language")
        assert len(results) > 0
        contents = [r.memory.content for r in results]
        assert any("Python" in c for c in contents)

    def test_remember_with_metadata(self, agent):
        mem = agent.remember(
            "User prefers dark mode",
            metadata={"source": "settings"},
        )
        assert mem.id is not None
        assert mem.metadata["source"] == "settings"

    def test_remember_different_types(self, agent):
        agent.remember("A fact", memory_type="semantic")
        agent.remember("An experience", memory_type="episodic")

        stats = agent.stats()
        assert stats["by_type"]["semantic"] >= 1
        assert stats["by_type"]["episodic"] >= 1

    def test_recall_empty(self, agent):
        results = agent.recall("nonexistent topic")
        assert len(results) == 0

    def test_recall_limit(self, agent):
        for i in range(10):
            agent.remember(f"Memory number {i}")

        results = agent.recall("memory", limit=3)
        assert len(results) <= 3


class TestThink:
    """Test memory-augmented context generation."""

    def test_think_with_memories(self, agent):
        agent.remember("User prefers concise responses")
        agent.remember("User is working on a FastAPI project")

        context = agent.think("How should I respond?")
        assert len(context) > 0
        assert "Relevant memories:" in context or "memories" in context.lower()

    def test_think_empty(self, agent):
        context = agent.think("any question")
        assert context == ""

    def test_think_structured_format(self, agent):
        agent.remember("Test memory for structured format")
        context = agent.think("test", format_style="structured")
        assert "Memory Context" in context


class TestLearn:
    """Test episodic learning."""

    def test_learn_episode(self, agent):
        episode = agent.learn(
            topic="Kubernetes",
            action="Explained pods",
            outcome="User understood",
            feedback="positive",
            lesson="Simple analogies work best",
        )
        assert episode.topic == "Kubernetes"
        assert episode.lesson == "Simple analogies work best"

    def test_recall_experiences(self, agent):
        agent.learn(
            topic="Docker",
            action="Explained containers",
            outcome="User confused",
            lesson="Need simpler explanation",
        )

        experiences = agent.recall_experiences("Docker")
        assert len(experiences) > 0

    def test_get_lessons(self, agent):
        agent.learn(
            topic="API Design",
            lesson="REST is preferred for simple APIs",
        )
        agent.learn(
            topic="API Design",
            lesson="Use proper HTTP status codes",
        )

        lessons = agent.get_lessons("API Design")
        assert len(lessons) >= 1


class TestProcedural:
    """Test procedural memory."""

    def test_learn_procedure(self, agent):
        proc = agent.learn_procedure(
            name="Deploy API",
            steps=["Build image", "Push to registry", "Deploy"],
            description="Standard API deployment",
        )
        assert proc.name == "Deploy API"
        assert len(proc.steps) == 3

    def test_recall_procedure(self, agent):
        agent.learn_procedure(
            name="Debug Error",
            steps=["Read logs", "Reproduce", "Fix", "Test"],
            context="When errors occur",
        )

        procs = agent.recall_procedures("debugging an error")
        assert len(procs) > 0


class TestFacts:
    """Test semantic fact storage."""

    def test_store_and_query_fact(self, agent):
        agent.store_fact("Docker is a containerization platform")
        facts = agent.query_facts("containerization")
        assert len(facts) > 0
        assert any("Docker" in f for f in facts)

    def test_store_fact_with_category(self, agent):
        mem = agent.store_fact(
            "FastAPI is fast",
            category="frameworks",
            source="docs",
        )
        assert mem.metadata["category"] == "frameworks"


class TestForgetDecay:
    """Test memory management operations."""

    def test_forget_memory(self, agent):
        mem = agent.remember("Temporary memory")
        assert agent.stats()["total_memories"] == 1

        deleted = agent.forget(mem.id)
        assert deleted is True
        assert agent.stats()["total_memories"] == 0

    def test_forget_nonexistent(self, agent):
        deleted = agent.forget("nonexistent-id")
        assert deleted is False

    def test_decay(self, agent):
        # Decay shouldn't crash even with no memories
        forgotten = agent.decay()
        assert forgotten >= 0
