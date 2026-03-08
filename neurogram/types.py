"""Core type definitions for Neurogram."""

from __future__ import annotations

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """Types of memory in the Neurogram cognitive architecture."""

    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class Memory:
    """A single memory unit stored in Neurogram.

    Attributes:
        id: Unique identifier for this memory.
        agent_id: The agent this memory belongs to.
        memory_type: Classification of memory type.
        content: The textual content of the memory.
        metadata: Arbitrary key-value metadata.
        importance_score: Current importance (0.0 - 1.0).
        embedding: Vector embedding of the content.
        created_at: Unix timestamp of creation.
        last_accessed: Unix timestamp of last access.
        access_count: Number of times this memory was retrieved.
        decay_rate: How quickly this memory loses importance (0.0 - 1.0).
    """

    content: str
    agent_id: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    decay_rate: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "importance_score": self.importance_score,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Deserialize memory from dictionary."""
        data = data.copy()
        if isinstance(data.get("memory_type"), str):
            data["memory_type"] = MemoryType(data["memory_type"])
        return cls(**data)


@dataclass
class AgentConfig:
    """Configuration for an AI agent's identity.

    Attributes:
        agent_id: Unique identifier for the agent.
        name: Human-readable agent name.
        description: What this agent does.
        goals: List of agent objectives.
        personality: Personality traits or system prompt hints.
        skills: List of agent capabilities.
        metadata: Additional configuration data.
    """

    agent_id: str
    name: str
    description: str = ""
    goals: List[str] = field(default_factory=list)
    personality: str = ""
    skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "goals": self.goals,
            "personality": self.personality,
            "skills": self.skills,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Deserialize config from dictionary."""
        return cls(**data)


@dataclass
class RetrievalResult:
    """A memory retrieval result with relevance score.

    Attributes:
        memory: The retrieved memory.
        relevance_score: Cosine similarity or relevance score (0.0 - 1.0).
    """

    memory: Memory
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memory": self.memory.to_dict(),
            "relevance_score": self.relevance_score,
        }


@dataclass
class Episode:
    """A structured episodic memory record.

    Attributes:
        id: Unique episode identifier.
        agent_id: The agent this episode belongs to.
        topic: What the episode is about.
        action: What action was taken.
        outcome: The result of the action.
        feedback: User or system feedback.
        lesson: Extracted lesson or insight.
        emotional_valence: Emotional tone (-1.0 negative to 1.0 positive).
        timestamp: When the episode occurred.
        metadata: Additional episode data.
    """

    topic: str
    agent_id: str
    action: str = ""
    outcome: str = ""
    feedback: str = ""
    lesson: str = ""
    emotional_valence: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_content_string(self) -> str:
        """Convert episode to a searchable content string."""
        parts = [f"Topic: {self.topic}"]
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
        if self.feedback:
            parts.append(f"Feedback: {self.feedback}")
        if self.lesson:
            parts.append(f"Lesson: {self.lesson}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "topic": self.topic,
            "action": self.action,
            "outcome": self.outcome,
            "feedback": self.feedback,
            "lesson": self.lesson,
            "emotional_valence": self.emotional_valence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Deserialize episode from dictionary."""
        return cls(**data)


@dataclass
class Procedure:
    """A procedural memory record — how to do something.

    Attributes:
        id: Unique procedure identifier.
        agent_id: The agent this procedure belongs to.
        name: Name of the procedure.
        description: What this procedure accomplishes.
        steps: Ordered list of steps.
        context: When to use this procedure.
        success_count: How many times this procedure succeeded.
        failure_count: How many times this procedure failed.
        metadata: Additional procedure data.
    """

    name: str
    agent_id: str
    description: str = ""
    steps: List[str] = field(default_factory=list)
    context: str = ""
    success_count: int = 0
    failure_count: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_content_string(self) -> str:
        """Convert procedure to a searchable content string."""
        parts = [f"Procedure: {self.name}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.steps:
            parts.append(f"Steps: {' -> '.join(self.steps)}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize procedure to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "context": self.context,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Procedure":
        """Deserialize procedure from dictionary."""
        return cls(**data)
