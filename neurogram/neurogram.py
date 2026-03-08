"""Neurogram — system-level manager for multiple agents.

Provides a high-level interface for managing multiple AI agents,
each with their own isolated memory space.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from neurogram.agent import Agent
from neurogram.types import AgentConfig
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.importance_engine import ImportanceConfig
from neurogram.storage.base import StorageBackend
from neurogram.storage.sqlite_backend import SQLiteBackend


class Neurogram:
    """System-level manager for Neurogram agents.

    Use this class when managing multiple AI agents with separate
    memory spaces. For single-agent use, you can use Agent directly.

    Example:
        ```python
        from neurogram import Neurogram

        brain = Neurogram()

        adam = brain.create_agent("Adam", description="Research assistant")
        nova = brain.create_agent("Nova", description="Coding assistant")

        adam.remember("User studies machine learning")
        nova.remember("User's project uses FastAPI")

        # Each agent has its own memory space
        adam_context = adam.think("What does the user study?")
        nova_context = nova.think("What framework does the user use?")
        ```
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        storage: Optional[StorageBackend] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        importance_config: Optional[ImportanceConfig] = None,
    ):
        """Initialize the Neurogram system.

        Args:
            storage_path: Path for SQLite database. Defaults to ~/.neurogram/neurogram.db
            storage: Custom storage backend. Overrides storage_path.
            embedding_engine: Custom embedding engine. Auto-selects if None.
            importance_config: Custom importance scoring config.
        """
        if storage is not None:
            self._storage = storage
        else:
            self._storage = SQLiteBackend(db_path=storage_path)

        self._storage.initialize()
        self._embedding = embedding_engine
        self._importance_config = importance_config
        self._agents: Dict[str, Agent] = {}

    def create_agent(
        self,
        name: str,
        description: str = "",
        goals: Optional[List[str]] = None,
        personality: str = "",
        skills: Optional[List[str]] = None,
    ) -> Agent:
        """Create a new agent or load an existing one.

        Args:
            name: Agent name (used as identifier).
            description: What this agent does.
            goals: Agent objectives.
            personality: Personality traits.
            skills: Agent capabilities.

        Returns:
            The created or loaded Agent.
        """
        agent = Agent(
            name=name,
            description=description,
            goals=goals,
            personality=personality,
            skills=skills,
            storage=self._storage,
            embedding_engine=self._embedding,
            importance_config=self._importance_config,
        )
        self._agents[name.lower().replace(" ", "_")] = agent
        return agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an existing agent by name.

        Args:
            name: Agent name.

        Returns:
            The Agent if found, None otherwise.
        """
        agent_id = name.lower().replace(" ", "_")

        # Check in-memory cache first
        if agent_id in self._agents:
            return self._agents[agent_id]

        # Check storage
        config = self._storage.load_agent(agent_id)
        if config:
            agent = Agent(
                name=config.name,
                storage=self._storage,
                embedding_engine=self._embedding,
                importance_config=self._importance_config,
            )
            self._agents[agent_id] = agent
            return agent

        return None

    def list_agents(self) -> List[AgentConfig]:
        """List all agents.

        Returns:
            List of AgentConfig objects.
        """
        return self._storage.list_agents()

    def delete_agent(self, name: str) -> bool:
        """Delete an agent and all its memories.

        Args:
            name: Agent name to delete.

        Returns:
            True if deleted, False if not found.
        """
        agent_id = name.lower().replace(" ", "_")
        self._agents.pop(agent_id, None)
        return self._storage.delete_agent(agent_id)

    def close(self) -> None:
        """Close all connections and release resources."""
        self._storage.close()
        self._agents.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        agents = self._storage.list_agents()
        return f"Neurogram(agents={len(agents)})"
