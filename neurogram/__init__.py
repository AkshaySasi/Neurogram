"""Neurogram — Memory infrastructure for AI agents.

Give your AI a brain. Neurogram provides persistent, multi-type
memory for AI agents with episodic learning, semantic knowledge,
procedural skills, importance scoring, and natural forgetting.

Quick Start:
    ```python
    from neurogram import Agent

    agent = Agent("adam")
    agent.remember("User prefers concise responses")
    context = agent.think("How should I respond?")
    ```

Multi-Agent:
    ```python
    from neurogram import Neurogram

    brain = Neurogram()
    adam = brain.create_agent("Adam")
    nova = brain.create_agent("Nova")
    ```
"""

__version__ = "0.2.0"

from neurogram.agent import Agent
from neurogram.neurogram import Neurogram
from neurogram.types import (
    AgentConfig,
    Episode,
    Memory,
    MemoryType,
    Procedure,
    RetrievalResult,
)
from neurogram.memory_manager import MemoryManager
from neurogram.embedding_engine import (
    EmbeddingEngine,
    NumpyEmbeddingEngine,
    LocalEmbeddingEngine,
    OpenAIEmbeddingEngine,
)
from neurogram.importance_engine import ImportanceConfig, ImportanceEngine
from neurogram.consolidation_engine import ConsolidationEngine

__all__ = [
    # Primary API
    "Agent",
    "Neurogram",
    # Types
    "Memory",
    "MemoryType",
    "Episode",
    "Procedure",
    "AgentConfig",
    "RetrievalResult",
    # Engines
    "MemoryManager",
    "EmbeddingEngine",
    "NumpyEmbeddingEngine",
    "LocalEmbeddingEngine",
    "OpenAIEmbeddingEngine",
    "ImportanceEngine",
    "ImportanceConfig",
    "ConsolidationEngine",
]
