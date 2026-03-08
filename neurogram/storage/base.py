"""Abstract storage backend interface for Neurogram."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from neurogram.types import AgentConfig, Memory, MemoryType


class StorageBackend(ABC):
    """Abstract base class for Neurogram storage backends.

    All storage backends must implement these methods to provide
    persistent memory storage for agents.
    """

    # ── Memory Operations ──────────────────────────────────────────

    @abstractmethod
    def save_memory(self, memory: Memory) -> None:
        """Save a memory to storage.

        If a memory with the same ID exists, it should be updated.

        Args:
            memory: The memory to save.
        """
        ...

    @abstractmethod
    def load_memory(self, memory_id: str) -> Optional[Memory]:
        """Load a single memory by ID.

        Args:
            memory_id: The unique memory identifier.

        Returns:
            The memory if found, None otherwise.
        """
        ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The unique memory identifier.

        Returns:
            True if the memory was deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Memory]:
        """List memories for an agent.

        Args:
            agent_id: The agent's unique identifier.
            memory_type: Optional filter by memory type.
            limit: Maximum number of memories to return.
            offset: Number of memories to skip.

        Returns:
            List of memories matching the criteria.
        """
        ...

    @abstractmethod
    def search_by_embedding(
        self,
        agent_id: str,
        query_embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[tuple]:
        """Search memories by embedding similarity.

        Args:
            agent_id: The agent's unique identifier.
            query_embedding: The query embedding vector.
            memory_type: Optional filter by memory type.
            top_k: Maximum number of results.
            threshold: Minimum similarity threshold.

        Returns:
            List of (Memory, similarity_score) tuples, sorted by
            similarity descending.
        """
        ...

    @abstractmethod
    def count_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Count memories for an agent.

        Args:
            agent_id: The agent's unique identifier.
            memory_type: Optional filter by memory type.

        Returns:
            Number of memories matching the criteria.
        """
        ...

    @abstractmethod
    def get_memories_below_importance(
        self,
        agent_id: str,
        threshold: float,
    ) -> List[Memory]:
        """Get memories with importance below a threshold.

        Used by the decay engine to find memories to prune.

        Args:
            agent_id: The agent's unique identifier.
            threshold: Importance score threshold.

        Returns:
            List of memories below the threshold.
        """
        ...

    # ── Agent Operations ───────────────────────────────────────────

    @abstractmethod
    def save_agent(self, config: AgentConfig) -> None:
        """Save an agent configuration.

        Args:
            config: The agent configuration to save.
        """
        ...

    @abstractmethod
    def load_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Load an agent configuration.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The agent config if found, None otherwise.
        """
        ...

    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all its memories.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            True if the agent was deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_agents(self) -> List[AgentConfig]:
        """List all agents.

        Returns:
            List of all agent configurations.
        """
        ...

    # ── Lifecycle ──────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend.

        Called once when the backend is first used.
        Should create tables, indexes, etc.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources."""
        ...
