"""Procedural Memory system for Neurogram.

Stores and retrieves how-to knowledge - step-by-step procedures,
workflows, and skills. This enables agents to remember and improve
how they perform tasks.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from neurogram.types import Memory, MemoryType, Procedure
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.storage.base import StorageBackend


class ProceduralMemory:
    """Manages procedural memories - the agent's skills and procedures.

    Procedural memory stores how to do things:
    - "To deploy an API: 1) Build Docker image, 2) Push to registry, 3) Deploy"
    - "To handle user complaints: 1) Acknowledge, 2) Investigate, 3) Resolve"

    Procedures can be refined over time based on success/failure feedback.
    """

    def __init__(self, storage: StorageBackend, embedding_engine: EmbeddingEngine):
        """Initialize procedural memory.

        Args:
            storage: Backend for persistent storage.
            embedding_engine: Engine for generating embeddings.
        """
        self._storage = storage
        self._embedding = embedding_engine

    def store_procedure(
        self,
        agent_id: str,
        name: str,
        steps: List[str],
        description: str = "",
        context: str = "",
        importance: float = 0.6,
    ) -> Procedure:
        """Store a new procedure.

        Args:
            agent_id: The agent storing this procedure.
            name: Name of the procedure.
            steps: Ordered list of steps.
            description: What this procedure accomplishes.
            context: When to use this procedure.
            importance: Initial importance score.

        Returns:
            The created Procedure.
        """
        procedure = Procedure(
            name=name,
            agent_id=agent_id,
            description=description,
            steps=steps,
            context=context,
        )

        content = procedure.to_content_string()
        embedding = self._embedding.embed(content)

        memory = Memory(
            content=content,
            agent_id=agent_id,
            memory_type=MemoryType.PROCEDURAL,
            importance_score=importance,
            embedding=embedding,
            metadata={
                "procedure": procedure.to_dict(),
            },
        )

        self._storage.save_memory(memory)
        return procedure

    def recall_procedure(
        self,
        agent_id: str,
        task_description: str,
        limit: int = 3,
        threshold: float = 0.1,
    ) -> List[Procedure]:
        """Find procedures relevant to a task.

        Args:
            agent_id: The agent recalling procedures.
            task_description: Description of the task.
            limit: Maximum procedures to return.
            threshold: Minimum relevance threshold.

        Returns:
            List of relevant procedures, sorted by relevance.
        """
        query_embedding = self._embedding.embed(task_description)

        results = self._storage.search_by_embedding(
            agent_id=agent_id,
            query_embedding=query_embedding,
            memory_type=MemoryType.PROCEDURAL,
            top_k=limit,
            threshold=threshold,
        )

        procedures = []
        for memory, score in results:
            proc_data = memory.metadata.get("procedure", {})
            if proc_data:
                proc = Procedure.from_dict(proc_data)
                procedures.append(proc)

                # Reinforce
                memory.access_count += 1
                memory.last_accessed = time.time()
                self._storage.save_memory(memory)

        # Procedural memory prioritizes proven procedures (highest success rate first)
        procedures.sort(
            key=lambda p: p.success_count / max(1, p.success_count + p.failure_count),
            reverse=True,
        )

        return procedures

    def record_outcome(
        self,
        agent_id: str,
        procedure_name: str,
        success: bool,
    ) -> Optional[Procedure]:
        """Record the outcome of following a procedure.

        Updates the procedure's success/failure counts, which
        influences its importance score over time.

        Args:
            agent_id: The agent.
            procedure_name: Name of the procedure.
            success: Whether the procedure worked.

        Returns:
            Updated Procedure if found, None otherwise.
        """
        # Find the procedure
        procedures = self.recall_procedure(
            agent_id, f"Procedure: {procedure_name}", limit=5
        )

        for proc in procedures:
            if proc.name.lower() == procedure_name.lower():
                if success:
                    proc.success_count += 1
                else:
                    proc.failure_count += 1

                # Re-store with updated counts
                self.store_procedure(
                    agent_id=agent_id,
                    name=proc.name,
                    steps=proc.steps,
                    description=proc.description,
                    context=proc.context,
                    importance=0.6 + (0.1 * proc.success_count)
                    - (0.05 * proc.failure_count),
                )
                return proc

        return None

    def count(self, agent_id: str) -> int:
        """Count procedural memories for an agent."""
        return self._storage.count_memories(agent_id, MemoryType.PROCEDURAL)
