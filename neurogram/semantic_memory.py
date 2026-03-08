"""Semantic Memory system for Neurogram.

Stores and retrieves factual knowledge — concepts, definitions,
relationships, and general world knowledge. This is the agent's
"encyclopedia" of learned facts.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from neurogram.types import Memory, MemoryType
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.storage.base import StorageBackend


class SemanticMemory:
    """Manages semantic memories — the agent's factual knowledge.

    Semantic memory stores facts, concepts, and relationships:
    - "Docker is a containerization platform"
    - "FastAPI is a Python web framework"
    - "User prefers concise responses"

    Facts are stored as natural language and can be retrieved
    via semantic similarity search.
    """

    def __init__(self, storage: StorageBackend, embedding_engine: EmbeddingEngine):
        """Initialize semantic memory.

        Args:
            storage: Backend for persistent storage.
            embedding_engine: Engine for generating embeddings.
        """
        self._storage = storage
        self._embedding = embedding_engine

    def store_fact(
        self,
        agent_id: str,
        fact: str,
        category: str = "",
        source: str = "",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a factual memory.

        Args:
            agent_id: The agent storing this fact.
            fact: The factual content to store.
            category: Optional category (e.g., "technology", "user_preference").
            source: Where this fact came from.
            importance: Initial importance score (0.0 - 1.0).
            metadata: Additional key-value metadata.

        Returns:
            The created Memory object.
        """
        embedding = self._embedding.embed(fact)

        mem_metadata = metadata.copy() if metadata else {}
        if category:
            mem_metadata["category"] = category
        if source:
            mem_metadata["source"] = source

        memory = Memory(
            content=fact,
            agent_id=agent_id,
            memory_type=MemoryType.SEMANTIC,
            importance_score=importance,
            embedding=embedding,
            metadata=mem_metadata,
        )

        self._storage.save_memory(memory)
        return memory

    def store_knowledge_triple(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        obj: str,
        source: str = "",
        importance: float = 0.5,
    ) -> Memory:
        """Store a knowledge triple (subject-predicate-object).

        Example: store_knowledge_triple("agent1", "Docker", "is a", "containerization platform")

        Args:
            agent_id: The agent storing this fact.
            subject: The subject of the triple.
            predicate: The relationship/predicate.
            obj: The object of the triple.
            source: Where this fact came from.
            importance: Initial importance score.

        Returns:
            The created Memory object.
        """
        fact = f"{subject} {predicate} {obj}"
        return self.store_fact(
            agent_id=agent_id,
            fact=fact,
            importance=importance,
            source=source,
            metadata={
                "triple": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                }
            },
        )

    def query(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        threshold: float = 0.1,
        category: Optional[str] = None,
    ) -> List[Tuple[Memory, float]]:
        """Query semantic memory.

        Args:
            agent_id: The agent querying its memory.
            query: Natural language query.
            limit: Maximum number of results.
            threshold: Minimum similarity threshold.
            category: Optional category filter.

        Returns:
            List of (Memory, relevance_score) tuples.
        """
        query_embedding = self._embedding.embed(query)

        results = self._storage.search_by_embedding(
            agent_id=agent_id,
            query_embedding=query_embedding,
            memory_type=MemoryType.SEMANTIC,
            top_k=limit * 2 if category else limit,  # Over-fetch if filtering
            threshold=threshold,
        )

        # Filter by category if specified
        if category:
            results = [
                (mem, score)
                for mem, score in results
                if mem.metadata.get("category") == category
            ][:limit]

        # Reinforce accessed memories
        for memory, _ in results:
            memory.access_count += 1
            memory.last_accessed = time.time()
            self._storage.save_memory(memory)

        return results

    def get_facts_about(
        self,
        agent_id: str,
        subject: str,
        limit: int = 10,
    ) -> List[str]:
        """Get all facts about a subject.

        Args:
            agent_id: The agent.
            subject: The subject to query about.
            limit: Maximum facts to return.

        Returns:
            List of fact content strings.
        """
        results = self.query(agent_id, subject, limit=limit)
        return [memory.content for memory, _ in results]

    def count(self, agent_id: str) -> int:
        """Count semantic memories for an agent."""
        return self._storage.count_memories(agent_id, MemoryType.SEMANTIC)
