"""Central Memory Manager for Neurogram.

Orchestrates all memory subsystems, handles storage, embedding,
importance scoring, and memory decay. This is the core engine
that ties everything together.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from neurogram.types import Memory, MemoryType, RetrievalResult
from neurogram.embedding_engine import EmbeddingEngine, get_default_engine
from neurogram.importance_engine import ImportanceConfig, ImportanceEngine
from neurogram.storage.base import StorageBackend
from neurogram.storage.sqlite_backend import SQLiteBackend
from neurogram.episodic_memory import EpisodicMemory
from neurogram.semantic_memory import SemanticMemory
from neurogram.procedural_memory import ProceduralMemory


class MemoryManager:
    """Central memory orchestrator for an agent.

    Coordinates all memory subsystems and provides a unified interface
    for storing, retrieving, updating, and decaying memories.

    Example:
        ```python
        manager = MemoryManager(agent_id="adam")
        manager.store("User prefers concise responses")
        results = manager.retrieve("What does the user prefer?")
        ```
    """

    def __init__(
        self,
        agent_id: str,
        storage: Optional[StorageBackend] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        importance_config: Optional[ImportanceConfig] = None,
    ):
        """Initialize the memory manager.

        Args:
            agent_id: Unique identifier for the agent.
            storage: Storage backend. Defaults to SQLite.
            embedding_engine: Embedding engine. Auto-selects best available.
            importance_config: Importance scoring configuration.
        """
        self.agent_id = agent_id

        # Initialize components
        self._storage = storage or SQLiteBackend()
        self._storage.initialize()

        self._embedding = embedding_engine or get_default_engine()
        self._importance = ImportanceEngine(importance_config)

        # Initialize subsystems
        self.episodic = EpisodicMemory(self._storage, self._embedding)
        self.semantic = SemanticMemory(self._storage, self._embedding)
        self.procedural = ProceduralMemory(self._storage, self._embedding)

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a new memory.

        This is the primary method for adding memories. The content
        is automatically embedded and stored with the given importance.

        Args:
            content: The textual content of the memory.
            memory_type: Type of memory (default: SEMANTIC).
            importance: Initial importance score (0.0 - 1.0).
            metadata: Optional key-value metadata.

        Returns:
            The created Memory object.
        """
        embedding = self._embedding.embed(content)

        memory = Memory(
            content=content,
            agent_id=self.agent_id,
            memory_type=memory_type,
            importance_score=importance,
            embedding=embedding,
            metadata=metadata or {},
        )

        self._storage.save_memory(memory)
        return memory

    def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve relevant memories via semantic search.

        Searches across all (or filtered) memories using embedding
        similarity. Accessed memories are automatically reinforced.

        Args:
            query: Natural language query.
            memory_type: Optional filter by memory type.
            top_k: Maximum number of results.
            threshold: Minimum similarity threshold.

        Returns:
            List of RetrievalResult objects with memory and score.
        """
        query_embedding = self._embedding.embed(query)

        results = self._storage.search_by_embedding(
            agent_id=self.agent_id,
            query_embedding=query_embedding,
            memory_type=memory_type,
            top_k=top_k,
            threshold=threshold,
        )

        retrieval_results = []
        for memory, score in results:
            # Reinforce accessed memory
            self._importance.reinforce(memory, boost=0.05)
            self._storage.save_memory(memory)

            retrieval_results.append(
                RetrievalResult(memory=memory, relevance_score=score)
            )

        return retrieval_results

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> Optional[Memory]:
        """Update an existing memory.

        Args:
            memory_id: ID of the memory to update.
            content: New content (re-embeds if changed).
            metadata: New metadata (merged with existing).
            importance: New importance score.

        Returns:
            Updated Memory if found, None otherwise.
        """
        memory = self._storage.load_memory(memory_id)
        if memory is None:
            return None

        if content is not None:
            memory.content = content
            memory.embedding = self._embedding.embed(content)

        if metadata is not None:
            memory.metadata.update(metadata)

        if importance is not None:
            memory.importance_score = importance

        self._storage.save_memory(memory)
        return memory

    def forget(self, memory_id: str) -> bool:
        """Explicitly forget (delete) a memory.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        return self._storage.delete_memory(memory_id)

    def decay(self) -> int:
        """Run a memory decay cycle.

        Applies time-based decay to all memories and deletes
        those below the forget threshold. This simulates the
        natural forgetting process.

        Returns:
            Number of memories forgotten (deleted).
        """
        memories = self._storage.list_memories(self.agent_id, limit=10000)
        forgotten = 0

        for memory in memories:
            # Apply decay
            new_importance = self._importance.apply_decay(memory)
            memory.importance_score = new_importance
            self._storage.save_memory(memory)

            # Check if should be forgotten
            if self._importance.should_forget(memory):
                self._storage.delete_memory(memory.id)
                forgotten += 1

        return forgotten

    def get_context(
        self,
        query: str,
        max_memories: int = 5,
        format_style: str = "bullet",
    ) -> str:
        """Get formatted context for LLM injection.

        Retrieves relevant memories and formats them for injecting
        into an LLM prompt. This is the primary integration point
        for LLM-based agents.

        Args:
            query: The user's query or prompt.
            max_memories: Maximum memories to include.
            format_style: Formatting style ("bullet", "narrative", "structured").

        Returns:
            Formatted context string ready for LLM injection.
        """
        results = self.retrieve(query, top_k=max_memories)

        if not results:
            return ""

        if format_style == "bullet":
            lines = ["Relevant memories:"]
            for r in results:
                lines.append(f"- {r.memory.content}")
            return "\n".join(lines)

        elif format_style == "narrative":
            memories_text = "; ".join(r.memory.content for r in results)
            return f"Based on previous interactions: {memories_text}"

        elif format_style == "structured":
            lines = ["=== Agent Memory Context ==="]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] ({r.memory.memory_type.value}) "
                    f"{r.memory.content} "
                    f"[relevance: {r.relevance_score:.2f}]"
                )
            lines.append("=== End Memory Context ===")
            return "\n".join(lines)

        return ""

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of memory statistics including counts by type.
        """
        total = self._storage.count_memories(self.agent_id)
        by_type = {}
        for mt in MemoryType:
            by_type[mt.value] = self._storage.count_memories(self.agent_id, mt)

        return {
            "agent_id": self.agent_id,
            "total_memories": total,
            "by_type": by_type,
            "embedding_engine": type(self._embedding).__name__,
            "embedding_dimensions": self._embedding.dimensions,
            "storage_backend": type(self._storage).__name__,
        }

    def close(self) -> None:
        """Close the memory manager and release resources."""
        self._storage.close()
