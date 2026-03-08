"""Agent — the primary developer-facing API for Neurogram.

Provides an elegant, high-level interface for giving AI agents memory.
This is the class most developers will interact with directly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from neurogram.types import (
    AgentConfig,
    Episode,
    Memory,
    MemoryType,
    Procedure,
    RetrievalResult,
)
from neurogram.memory_manager import MemoryManager
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.importance_engine import ImportanceConfig
from neurogram.storage.base import StorageBackend
from neurogram.storage.sqlite_backend import SQLiteBackend


class Agent:
    """An AI agent with persistent, multi-type memory.

    This is the primary developer-facing class. It provides an
    intuitive, high-level API for giving AI agents human-like memory.

    Example:
        ```python
        from neurogram import Agent

        adam = Agent("adam")

        # Store memories
        adam.remember("User prefers concise responses")
        adam.learn(topic="API design", outcome="User liked REST approach")

        # Recall memories
        context = adam.think("How should I explain this API?")
        memories = adam.recall("user preferences")
        ```
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        goals: Optional[List[str]] = None,
        personality: str = "",
        skills: Optional[List[str]] = None,
        storage: Optional[StorageBackend] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        importance_config: Optional[ImportanceConfig] = None,
        storage_path: Optional[str] = None,
    ):
        """Create or load an AI agent with persistent memory.

        If an agent with this name already exists in storage,
        its configuration and memories are loaded automatically.

        Args:
            name: Human-readable agent name (used as agent_id).
            description: What this agent does.
            goals: List of agent objectives.
            personality: Personality traits or system prompt hints.
            skills: List of agent capabilities.
            storage: Storage backend. Defaults to SQLite.
            embedding_engine: Embedding engine. Auto-selects best.
            importance_config: Importance scoring configuration.
            storage_path: Custom path for SQLite database.
        """
        self._name = name
        self._agent_id = name.lower().replace(" ", "_")

        # Set up storage
        if storage is None:
            storage = SQLiteBackend(db_path=storage_path)

        self._storage = storage
        self._storage.initialize()

        # Set up or load agent config
        existing = self._storage.load_agent(self._agent_id)
        if existing:
            self._config = existing
        else:
            self._config = AgentConfig(
                agent_id=self._agent_id,
                name=name,
                description=description,
                goals=goals or [],
                personality=personality,
                skills=skills or [],
            )
            self._storage.save_agent(self._config)

        # Initialize memory manager
        self._memory = MemoryManager(
            agent_id=self._agent_id,
            storage=self._storage,
            embedding_engine=embedding_engine,
            importance_config=importance_config,
        )

    # ── Core Memory API ────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a memory.

        This is the primary method for adding information to the
        agent's memory. The content is automatically embedded and
        stored with the given importance.

        Args:
            content: The information to remember.
            memory_type: Type of memory: "semantic", "episodic",
                        "procedural", or "short_term".
            importance: How important this memory is (0.0 - 1.0).
            metadata: Optional additional data to store.

        Returns:
            The created Memory object.

        Example:
            ```python
            adam.remember("User prefers dark mode")
            adam.remember("Completed project setup", memory_type="episodic")
            ```
        """
        mtype = MemoryType(memory_type)
        return self._memory.store(content, mtype, importance, metadata)

    def recall(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Search memory for relevant information.

        Uses semantic similarity to find memories related to the query.

        Args:
            query: What to search for (natural language).
            limit: Maximum number of results.
            memory_type: Optional filter by memory type.
            threshold: Minimum relevance score (0.0 - 1.0).

        Returns:
            List of RetrievalResult with memory and relevance score.

        Example:
            ```python
            results = adam.recall("user preferences")
            for r in results:
                print(f"{r.memory.content} (relevance: {r.relevance_score:.2f})")
            ```
        """
        mtype = MemoryType(memory_type) if memory_type else None
        return self._memory.retrieve(query, mtype, limit, threshold)

    def think(
        self,
        prompt: str,
        max_memories: int = 5,
        format_style: str = "bullet",
    ) -> str:
        """Get memory-augmented context for an LLM prompt.

        Retrieves relevant memories and formats them for injection
        into an LLM prompt. This is the key integration point —
        use this to make your LLM responses context-aware.

        Args:
            prompt: The user's question or prompt.
            max_memories: Maximum memories to retrieve.
            format_style: How to format memories:
                         "bullet" (default), "narrative", "structured".

        Returns:
            Formatted context string with relevant memories.

        Example:
            ```python
            context = adam.think("How should I deploy this?")
            llm_prompt = f"{context}\\n\\nUser: How should I deploy this?"
            ```
        """
        return self._memory.get_context(prompt, max_memories, format_style)

    # ── Episodic Memory (Learning from Experience) ─────────────────

    def learn(
        self,
        topic: str,
        action: str = "",
        outcome: str = "",
        feedback: str = "",
        lesson: str = "",
        emotional_valence: float = 0.0,
    ) -> Episode:
        """Record a learning experience.

        Stores an episodic memory that captures what happened
        and what was learned. Over time, these episodes help
        the agent improve its behavior.

        Args:
            topic: What the experience was about.
            action: What action was taken.
            outcome: What happened as a result.
            feedback: User or system feedback.
            lesson: Key takeaway or insight.
            emotional_valence: Emotional tone (-1.0 to 1.0).

        Returns:
            The recorded Episode.

        Example:
            ```python
            adam.learn(
                topic="Kubernetes explanation",
                action="Gave detailed explanation",
                outcome="User was confused",
                feedback="negative",
                lesson="User prefers simple analogies"
            )
            ```
        """
        return self._memory.episodic.record(
            agent_id=self._agent_id,
            topic=topic,
            action=action,
            outcome=outcome,
            feedback=feedback,
            lesson=lesson,
            emotional_valence=emotional_valence,
        )

    def recall_experiences(
        self,
        topic: str,
        limit: int = 5,
    ) -> List[Episode]:
        """Recall past experiences related to a topic.

        Args:
            topic: What to search experiences for.
            limit: Maximum experiences to return.

        Returns:
            List of relevant Episodes.
        """
        return self._memory.episodic.recall(self._agent_id, topic, limit)

    def get_lessons(self, topic: str, limit: int = 10) -> List[str]:
        """Get lessons learned about a topic.

        Args:
            topic: What to get lessons for.
            limit: Maximum lessons to return.

        Returns:
            List of lesson strings.
        """
        return self._memory.episodic.get_lessons(self._agent_id, topic, limit)

    # ── Procedural Memory (Skills & How-To) ────────────────────────

    def learn_procedure(
        self,
        name: str,
        steps: List[str],
        description: str = "",
        context: str = "",
    ) -> Procedure:
        """Teach the agent a procedure.

        Args:
            name: Name of the procedure.
            steps: Ordered list of steps.
            description: What the procedure accomplishes.
            context: When to use this procedure.

        Returns:
            The stored Procedure.

        Example:
            ```python
            adam.learn_procedure(
                name="Deploy API",
                steps=["Build Docker image", "Push to registry", "Deploy to K8s"],
                context="When user asks about API deployment"
            )
            ```
        """
        return self._memory.procedural.store_procedure(
            agent_id=self._agent_id,
            name=name,
            steps=steps,
            description=description,
            context=context,
        )

    def recall_procedures(
        self,
        task: str,
        limit: int = 3,
    ) -> List[Procedure]:
        """Find procedures relevant to a task.

        Args:
            task: Description of the task.
            limit: Maximum procedures to return.

        Returns:
            List of relevant Procedures.
        """
        return self._memory.procedural.recall_procedure(
            self._agent_id, task, limit
        )

    # ── Semantic Memory (Facts & Knowledge) ────────────────────────

    def store_fact(
        self,
        fact: str,
        category: str = "",
        source: str = "",
        importance: float = 0.5,
    ) -> Memory:
        """Store a factual memory.

        Args:
            fact: The fact to store.
            category: Optional category.
            source: Where this fact came from.
            importance: Importance score.

        Returns:
            The created Memory.
        """
        return self._memory.semantic.store_fact(
            agent_id=self._agent_id,
            fact=fact,
            category=category,
            source=source,
            importance=importance,
        )

    def query_facts(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> List[str]:
        """Query factual knowledge.

        Args:
            query: What to search for.
            limit: Maximum facts to return.
            category: Optional category filter.

        Returns:
            List of fact content strings.
        """
        results = self._memory.semantic.query(
            self._agent_id, query, limit, category=category
        )
        return [mem.content for mem, _ in results]

    # ── Memory Management ──────────────────────────────────────────

    def forget(self, memory_id: str) -> bool:
        """Explicitly forget a memory.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        return self._memory.forget(memory_id)

    def decay(self) -> int:
        """Run memory decay — forget unimportant memories.

        Applies time-based decay and prunes low-importance memories.
        Call this periodically to simulate natural forgetting.

        Returns:
            Number of memories forgotten.
        """
        return self._memory.decay()

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with agent info and memory counts.
        """
        memory_stats = self._memory.stats()
        return {
            **memory_stats,
            "agent_name": self._config.name,
            "agent_description": self._config.description,
        }

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        """Close the agent and release resources."""
        self._memory.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        stats = self._memory.stats()
        return (
            f"Agent(name='{self._config.name}', "
            f"memories={stats['total_memories']})"
        )
