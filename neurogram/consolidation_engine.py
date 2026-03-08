"""Memory Consolidation Engine for Neurogram.

Simulates human memory consolidation - the process where the brain
transfers short-term memories into long-term storage during sleep.

This module:
1. Groups similar memories together
2. Merges them into stronger, consolidated memories
3. Removes the weak originals
4. Boosts importance of consolidated memories

This is a truly unique feature - no competitor does this.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from neurogram.types import Memory, MemoryType
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.storage.base import StorageBackend


class ConsolidationEngine:
    """Engine for consolidating fragmented memories into stronger ones.

    Mimics the human brain's memory consolidation during sleep:
    - Groups similar memories by embedding proximity
    - Merges clusters into single, stronger memories
    - Preserves important details while reducing redundancy
    - Boosts the importance of consolidated memories

    This makes agents more efficient over time - instead of 100
    fragmented memories about "user preferences", they get 5
    strong, comprehensive ones.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedding_engine: EmbeddingEngine,
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 2,
        max_cluster_size: int = 10,
    ):
        """Initialize the consolidation engine.

        Args:
            storage: Storage backend.
            embedding_engine: Embedding engine for similarity.
            similarity_threshold: Minimum similarity to group memories.
            min_cluster_size: Minimum memories to trigger consolidation.
            max_cluster_size: Maximum memories to merge at once.
        """
        self._storage = storage
        self._embedding = embedding_engine
        self._sim_threshold = similarity_threshold
        self._min_cluster = min_cluster_size
        self._max_cluster = max_cluster_size

    def consolidate(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run memory consolidation for an agent.

        Groups similar memories, merges clusters into consolidated
        memories, and removes weak originals.

        Args:
            agent_id: The agent to consolidate memories for.
            memory_type: Optional filter - only consolidate this type.
            dry_run: If True, show what would happen without changes.

        Returns:
            Dictionary with consolidation stats:
            - clusters_found: Number of memory clusters identified
            - memories_merged: Number of individual memories merged
            - memories_created: Number of consolidated memories created
            - memories_removed: Number of weak originals removed
        """
        # Load all memories with embeddings
        memories = self._storage.list_memories(
            agent_id, memory_type=memory_type, limit=10000
        )
        memories = [m for m in memories if m.embedding is not None]

        if len(memories) < self._min_cluster:
            return {
                "clusters_found": 0,
                "memories_merged": 0,
                "memories_created": 0,
                "memories_removed": 0,
            }

        # Find clusters of similar memories
        clusters = self._find_clusters(memories)

        stats = {
            "clusters_found": len(clusters),
            "memories_merged": 0,
            "memories_created": 0,
            "memories_removed": 0,
        }

        if dry_run:
            stats["memories_merged"] = sum(len(c) for c in clusters)
            return stats

        # Merge each cluster
        for cluster in clusters:
            merged = self._merge_cluster(agent_id, cluster)
            if merged:
                stats["memories_created"] += 1
                stats["memories_merged"] += len(cluster)

                # Remove originals (keep the merged version)
                for mem in cluster:
                    self._storage.delete_memory(mem.id)
                    stats["memories_removed"] += 1

        return stats

    def _find_clusters(self, memories: List[Memory]) -> List[List[Memory]]:
        """Find clusters of similar memories using greedy clustering.

        Uses a simple greedy approach: for each unvisited memory,
        find all similar memories and form a cluster.
        """
        used = set()
        clusters: List[List[Memory]] = []

        for i, mem_i in enumerate(memories):
            if i in used:
                continue

            cluster = [mem_i]
            used.add(i)

            for j, mem_j in enumerate(memories):
                if j in used or j <= i:
                    continue

                if mem_i.embedding and mem_j.embedding:
                    sim = self._cosine_similarity(mem_i.embedding, mem_j.embedding)
                    if sim >= self._sim_threshold:
                        cluster.append(mem_j)
                        used.add(j)

                        if len(cluster) >= self._max_cluster:
                            break

            if len(cluster) >= self._min_cluster:
                clusters.append(cluster)

        return clusters

    def _merge_cluster(
        self, agent_id: str, cluster: List[Memory]
    ) -> Optional[Memory]:
        """Merge a cluster of similar memories into one consolidated memory.

        The consolidated memory:
        - Combines content from all memories (deduplicated)
        - Gets the maximum importance from the cluster
        - Has a boosted importance score (consolidation bonus)
        - Gets a fresh embedding of the combined content
        - Preserves metadata from all originals
        """
        if not cluster:
            return None

        # Combine content - deduplicate while preserving order
        seen_content = set()
        content_parts = []
        for mem in cluster:
            normalized = mem.content.strip().lower()
            if normalized not in seen_content:
                seen_content.add(normalized)
                content_parts.append(mem.content.strip())

        consolidated_content = " | ".join(content_parts)

        # If everything collapsed to one unique item, skip consolidation
        if len(content_parts) <= 1:
            return None

        # Take the best properties from the cluster
        max_importance = max(m.importance_score for m in cluster)
        total_access = sum(m.access_count for m in cluster)
        primary_type = cluster[0].memory_type

        # Merge metadata
        merged_metadata: Dict[str, Any] = {
            "consolidated": True,
            "source_count": len(cluster),
            "source_ids": [m.id for m in cluster],
            "consolidated_at": time.time(),
        }
        for mem in cluster:
            for k, v in mem.metadata.items():
                if k not in merged_metadata:
                    merged_metadata[k] = v

        # Generate fresh embedding for combined content
        embedding = self._embedding.embed(consolidated_content)

        # Create consolidated memory with boosted importance
        consolidated = Memory(
            content=consolidated_content,
            agent_id=agent_id,
            memory_type=primary_type,
            importance_score=min(1.0, max_importance * 1.2),  # 20% consolidation bonus
            embedding=embedding,
            metadata=merged_metadata,
            access_count=total_access,
            decay_rate=cluster[0].decay_rate * 0.5,  # Consolidated memories decay slower
        )

        self._storage.save_memory(consolidated)
        return consolidated

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        import math

        if len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        n1 = math.sqrt(sum(a * a for a in vec1))
        n2 = math.sqrt(sum(b * b for b in vec2))

        if n1 == 0 or n2 == 0:
            return 0.0

        return dot / (n1 * n2)
