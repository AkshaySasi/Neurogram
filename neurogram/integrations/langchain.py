"""LangChain integration for Neurogram.

Provides a `NeurogramMemory` class that plugs directly into
LangChain's memory system, giving any LangChain agent persistent,
multi-type memory powered by Neurogram.

Usage:
    ```python
    from neurogram.integrations.langchain import NeurogramMemory
    from langchain.chains import ConversationChain
    from langchain.llms import OpenAI

    memory = NeurogramMemory(agent_name="assistant")

    chain = ConversationChain(
        llm=OpenAI(),
        memory=memory,
    )

    chain.predict(input="My name is Alice")
    chain.predict(input="What's my name?")  # Remembers!
    ```

Requires: pip install langchain
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.schema.messages import get_buffer_string

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Create stub base class so module can be imported even without langchain
    class BaseChatMemory:  # type: ignore
        pass


from neurogram.agent import Agent
from neurogram.embedding_engine import EmbeddingEngine
from neurogram.importance_engine import ImportanceConfig
from neurogram.storage.base import StorageBackend


class NeurogramMemory(BaseChatMemory):
    """LangChain-compatible memory powered by Neurogram.

    Drop-in replacement for LangChain's memory classes that gives
    your chain persistent, semantic memory with importance scoring,
    episodic learning, and memory consolidation.

    Features over standard LangChain memory:
    - Persistent across sessions (SQLite storage)
    - Semantic search (not just recent history)
    - Importance-based retrieval
    - Memory decay (forgets unimportant things)
    - Episodic learning from conversations
    """

    # LangChain memory interface attributes
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False

    # Neurogram config
    agent_name: str = "langchain_agent"
    max_memories: int = 5
    store_conversations: bool = True
    auto_learn: bool = True

    # Internal
    _agent: Optional[Agent] = None

    class Config:
        """Pydantic config for LangChain compatibility."""
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(
        self,
        agent_name: str = "langchain_agent",
        max_memories: int = 5,
        store_conversations: bool = True,
        auto_learn: bool = True,
        storage: Optional[StorageBackend] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        importance_config: Optional[ImportanceConfig] = None,
        storage_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize NeurogramMemory.

        Args:
            agent_name: Name for the Neurogram agent.
            max_memories: Max memories to include in context.
            store_conversations: Whether to store each exchange.
            auto_learn: Whether to create episodic memories.
            storage: Custom storage backend.
            embedding_engine: Custom embedding engine.
            importance_config: Custom importance config.
            storage_path: Custom SQLite path.
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for NeurogramMemory. "
                "Install it with: pip install langchain"
            )

        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.max_memories = max_memories
        self.store_conversations = store_conversations
        self.auto_learn = auto_learn

        self._agent = Agent(
            name=agent_name,
            description="LangChain agent with Neurogram memory",
            storage=storage,
            embedding_engine=embedding_engine,
            importance_config=importance_config,
            storage_path=storage_path,
        )

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables (LangChain interface)."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memories for the current query.

        Retrieves memories relevant to the input query using
        semantic similarity search.
        """
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: "" if not self.return_messages else []}

        # Get memory context
        context = self._agent.think(
            prompt=str(query),
            max_memories=self.max_memories,
            format_style="bullet",
        )

        if self.return_messages:
            if context:
                return {self.memory_key: [AIMessage(content=context)]}
            return {self.memory_key: []}

        return {self.memory_key: context}

    def save_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save a conversation exchange to memory.

        Stores both the user input and AI response as memories,
        and creates an episodic learning record.
        """
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        if self.store_conversations and user_input:
            # Store user input as a memory
            self._agent.remember(
                f"User said: {user_input}",
                memory_type="semantic",
                importance=0.4,
                metadata={"role": "user", "type": "conversation"},
            )

        if self.store_conversations and ai_output:
            # Store AI response as a memory
            self._agent.remember(
                f"Assistant responded: {ai_output[:200]}",
                memory_type="semantic",
                importance=0.3,
                metadata={"role": "assistant", "type": "conversation"},
            )

        if self.auto_learn and user_input:
            # Create episodic learning record
            self._agent.learn(
                topic=user_input[:100],
                action="conversation",
                outcome=ai_output[:100] if ai_output else "",
            )

    def clear(self) -> None:
        """Clear all memories (LangChain interface)."""
        # We don't actually clear - Neurogram memories are persistent
        # Instead, run decay to remove unimportant ones
        self._agent.decay()

    # ── Neurogram-specific methods ─────────────────────────────────

    def remember(self, content: str, importance: float = 0.5) -> None:
        """Explicitly store a memory (beyond automatic conversation storage)."""
        self._agent.remember(content, importance=importance)

    def consolidate(self) -> Dict[str, Any]:
        """Run memory consolidation."""
        return self._agent.consolidate()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self._agent.stats()

    def close(self) -> None:
        """Close the underlying agent."""
        if self._agent:
            self._agent.close()
