"""SQLite storage backend for Neurogram.

Default storage backend using SQLite - zero external dependencies.
Stores memories, embeddings, and agent configs in a local database file.
Cosine similarity search is done in Python (fast enough for <100k memories).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurogram.storage.base import StorageBackend
from neurogram.types import AgentConfig, Memory, MemoryType


class SQLiteBackend(StorageBackend):
    """SQLite-based storage backend for Neurogram.

    Stores all data in a single SQLite database file. Embedding
    similarity search is performed in Python using cosine similarity.

    This backend is ideal for:
    - Development and prototyping
    - Single-agent applications
    - Small to medium memory stores (<100k memories)
    - Environments where external databases aren't available

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to ~/.neurogram/neurogram.db
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            home = Path.home() / ".neurogram"
            home.mkdir(parents=True, exist_ok=True)
            db_path = str(home / "neurogram.db")

        self._db_path = db_path
        self._local = threading.local()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._conn

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                goals TEXT DEFAULT '[]',
                personality TEXT DEFAULT '',
                skills TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                importance_score REAL DEFAULT 0.5,
                embedding TEXT,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.01,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memories_agent
                ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(agent_id, memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(agent_id, importance_score);
        """)
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Memory Operations ──────────────────────────────────────────

    def save_memory(self, memory: Memory) -> None:
        """Save or update a memory."""
        conn = self._conn
        embedding_json = (
            json.dumps(memory.embedding) if memory.embedding else None
        )
        metadata_json = json.dumps(memory.metadata)

        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, agent_id, memory_type, content, metadata,
                 importance_score, embedding, created_at,
                 last_accessed, access_count, decay_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.agent_id,
                memory.memory_type.value,
                memory.content,
                metadata_json,
                memory.importance_score,
                embedding_json,
                memory.created_at,
                memory.last_accessed,
                memory.access_count,
                memory.decay_rate,
            ),
        )
        conn.commit()

    def load_memory(self, memory_id: str) -> Optional[Memory]:
        """Load a memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Memory]:
        """List memories for an agent."""
        if memory_type:
            rows = self._conn.execute(
                """SELECT * FROM memories
                   WHERE agent_id = ? AND memory_type = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (agent_id, memory_type.value, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM memories
                   WHERE agent_id = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (agent_id, limit, offset),
            ).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def search_by_embedding(
        self,
        agent_id: str,
        query_embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Memory, float]]:
        """Search memories by embedding cosine similarity."""
        # Fetch all memories with embeddings for this agent
        if memory_type:
            rows = self._conn.execute(
                """SELECT * FROM memories
                   WHERE agent_id = ? AND memory_type = ?
                   AND embedding IS NOT NULL""",
                (agent_id, memory_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM memories
                   WHERE agent_id = ? AND embedding IS NOT NULL""",
                (agent_id,),
            ).fetchall()

        # Calculate similarities
        results: List[Tuple[Memory, float]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                similarity = self._cosine_similarity(
                    query_embedding, memory.embedding
                )
                if similarity >= threshold:
                    results.append((memory, similarity))

        # Sort by similarity descending and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Count memories for an agent."""
        if memory_type:
            row = self._conn.execute(
                """SELECT COUNT(*) as cnt FROM memories
                   WHERE agent_id = ? AND memory_type = ?""",
                (agent_id, memory_type.value),
            ).fetchone()
        else:
            row = self._conn.execute(
                """SELECT COUNT(*) as cnt FROM memories
                   WHERE agent_id = ?""",
                (agent_id,),
            ).fetchone()

        return row["cnt"]

    def get_memories_below_importance(
        self,
        agent_id: str,
        threshold: float,
    ) -> List[Memory]:
        """Get memories below an importance threshold."""
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE agent_id = ? AND importance_score < ?""",
            (agent_id, threshold),
        ).fetchall()

        return [self._row_to_memory(row) for row in rows]

    # ── Agent Operations ───────────────────────────────────────────

    def save_agent(self, config: AgentConfig) -> None:
        """Save or update an agent configuration."""
        conn = self._conn
        conn.execute(
            """
            INSERT OR REPLACE INTO agents
                (agent_id, name, description, goals,
                 personality, skills, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.agent_id,
                config.name,
                config.description,
                json.dumps(config.goals),
                config.personality,
                json.dumps(config.skills),
                json.dumps(config.metadata),
            ),
        )
        conn.commit()

    def load_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Load an agent configuration."""
        row = self._conn.execute(
            "SELECT * FROM agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()

        if row is None:
            return None

        return AgentConfig(
            agent_id=row["agent_id"],
            name=row["name"],
            description=row["description"],
            goals=json.loads(row["goals"]),
            personality=row["personality"],
            skills=json.loads(row["skills"]),
            metadata=json.loads(row["metadata"]),
        )

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all its memories."""
        conn = self._conn
        # Memories are cascade-deleted due to FK
        cursor = conn.execute(
            "DELETE FROM agents WHERE agent_id = ?", (agent_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_agents(self) -> List[AgentConfig]:
        """List all agents."""
        rows = self._conn.execute("SELECT * FROM agents").fetchall()
        return [
            AgentConfig(
                agent_id=row["agent_id"],
                name=row["name"],
                description=row["description"],
                goals=json.loads(row["goals"]),
                personality=row["personality"],
                skills=json.loads(row["skills"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])

        metadata = {}
        if row["metadata"]:
            metadata = json.loads(row["metadata"])

        return Memory(
            id=row["id"],
            agent_id=row["agent_id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            metadata=metadata,
            importance_score=row["importance_score"],
            embedding=embedding,
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            decay_rate=row["decay_rate"],
        )

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)
