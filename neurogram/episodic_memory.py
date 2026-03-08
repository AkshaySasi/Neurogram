"""Episodic Memory system for Neurogram.

Stores and retrieves structured experiences - what happened,
what the outcome was, and what was learned. This enables AI
agents to learn from past interactions and improve over time.
"""

from __future__ import annotations

import time
from typing import List, Optional

from neurogram.types import Episode, Memory, MemoryType
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.storage.base import StorageBackend


class EpisodicMemory:
    """Manages episodic memories - the agent's experiences.

    Episodic memory stores structured records of events:
    - What happened (topic, action)
    - What the result was (outcome)
    - How it was received (feedback)
    - What was learned (lesson)

    This allows agents to learn from experience and avoid
    repeating mistakes.
    """

    def __init__(self, storage: StorageBackend, embedding_engine: EmbeddingEngine):
        """Initialize episodic memory.

        Args:
            storage: Backend for persistent storage.
            embedding_engine: Engine for generating embeddings.
        """
        self._storage = storage
        self._embedding = embedding_engine

    def record(
        self,
        agent_id: str,
        topic: str,
        action: str = "",
        outcome: str = "",
        feedback: str = "",
        lesson: str = "",
        emotional_valence: float = 0.0,
        importance: float = 0.6,
    ) -> Episode:
        """Record a new episode.

        Args:
            agent_id: The agent recording this episode.
            topic: What the episode is about.
            action: What action was taken.
            outcome: What happened as a result.
            feedback: User or system feedback.
            lesson: Key takeaway or insight.
            emotional_valence: Emotional tone (-1.0 to 1.0).
            importance: Initial importance score.

        Returns:
            The created Episode.
        """
        episode = Episode(
            agent_id=agent_id,
            topic=topic,
            action=action,
            outcome=outcome,
            feedback=feedback,
            lesson=lesson,
            emotional_valence=emotional_valence,
        )

        # Store as a memory with episode metadata
        content = episode.to_content_string()
        embedding = self._embedding.embed(content)

        memory = Memory(
            content=content,
            agent_id=agent_id,
            memory_type=MemoryType.EPISODIC,
            importance_score=importance,
            embedding=embedding,
            metadata={
                "episode": episode.to_dict(),
                "emotional_valence": emotional_valence,
            },
        )

        self._storage.save_memory(memory)
        return episode

    def recall(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        threshold: float = 0.1,
    ) -> List[Episode]:
        """Recall episodes related to a query.

        Args:
            agent_id: The agent recalling episodes.
            query: Natural language query to search for.
            limit: Maximum number of episodes to return.
            threshold: Minimum relevance threshold.

        Returns:
            List of relevant episodes, sorted by relevance.
        """
        query_embedding = self._embedding.embed(query)

        results = self._storage.search_by_embedding(
            agent_id=agent_id,
            query_embedding=query_embedding,
            memory_type=MemoryType.EPISODIC,
            top_k=limit,
            threshold=threshold,
        )

        episodes = []
        for memory, score in results:
            episode_data = memory.metadata.get("episode", {})
            if episode_data:
                episodes.append(Episode.from_dict(episode_data))

                # Reinforce: update access
                memory.access_count += 1
                memory.last_accessed = time.time()
                self._storage.save_memory(memory)

        return episodes

    def get_lessons(
        self,
        agent_id: str,
        topic: str,
        limit: int = 10,
    ) -> List[str]:
        """Extract lessons learned from episodes about a topic.

        Args:
            agent_id: The agent.
            topic: Topic to find lessons for.
            limit: Maximum number of lessons.

        Returns:
            List of lesson strings extracted from episodes.
        """
        episodes = self.recall(agent_id, topic, limit=limit)
        lessons = []
        for ep in episodes:
            if ep.lesson:
                lessons.append(ep.lesson)
        return lessons

    def count(self, agent_id: str) -> int:
        """Count episodic memories for an agent."""
        return self._storage.count_memories(agent_id, MemoryType.EPISODIC)
