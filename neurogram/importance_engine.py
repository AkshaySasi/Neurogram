"""Memory importance scoring and decay engine for Neurogram.

Implements a biologically-inspired memory importance model:
- Frequency: Memories accessed more often are more important
- Recency: Recently accessed memories are more important
- Emotional valence: Emotionally charged memories persist longer
- Feedback: User feedback boosts or reduces importance
- Decay: All memories gradually lose importance over time
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from neurogram.types import Memory


@dataclass
class ImportanceConfig:
    """Configuration for the importance scoring algorithm.

    Attributes:
        frequency_weight: Weight for access frequency (α).
        recency_weight: Weight for recency (β).
        emotional_weight: Weight for emotional salience (γ).
        feedback_weight: Weight for user feedback (δ).
        base_importance: Default importance for new memories.
        decay_rate: Global decay multiplier (higher = faster forgetting).
        forget_threshold: Memories below this importance get deleted.
        recency_halflife: Half-life for recency in seconds (default: 24h).
    """

    frequency_weight: float = 0.2
    recency_weight: float = 0.35
    emotional_weight: float = 0.15
    feedback_weight: float = 0.3
    base_importance: float = 0.5
    decay_rate: float = 0.01
    forget_threshold: float = 0.05
    recency_halflife: float = 86400.0  # 24 hours


class ImportanceEngine:
    """Engine for calculating memory importance and applying decay.

    Models human-like memory retention where:
    - Frequently accessed memories strengthen (like studying)
    - Recent memories are naturally more vivid
    - Emotionally charged memories persist longer
    - Unused memories gradually fade (forgetting curve)
    """

    def __init__(self, config: Optional[ImportanceConfig] = None):
        """Initialize the importance engine.

        Args:
            config: Importance scoring configuration.
                    Uses defaults if not provided.
        """
        self.config = config or ImportanceConfig()

    def score(
        self,
        memory: Memory,
        emotional_score: float = 0.0,
        feedback_score: float = 0.0,
    ) -> float:
        """Calculate the current importance score of a memory.

        The score is a weighted combination of four factors:
        - Frequency: log-scaled access count
        - Recency: exponential decay from last access time
        - Emotional: absolute emotional valence
        - Feedback: explicit user feedback signal

        Args:
            memory: The memory to score.
            emotional_score: Emotional valence (-1.0 to 1.0).
            feedback_score: User feedback signal (-1.0 to 1.0).

        Returns:
            Importance score between 0.0 and 1.0.
        """
        cfg = self.config

        # Frequency score: logarithmic scaling (diminishing returns)
        freq_score = min(1.0, math.log1p(memory.access_count) / math.log1p(100))

        # Recency score: exponential decay from last access
        time_delta = time.time() - memory.last_accessed
        recency_score = math.exp(-0.693 * time_delta / cfg.recency_halflife)

        # Emotional score: absolute valence (both positive and negative persist)
        emo_score = abs(emotional_score)

        # Feedback score: normalize to 0-1 range
        fb_score = (feedback_score + 1.0) / 2.0

        # Weighted combination
        importance = (
            cfg.frequency_weight * freq_score
            + cfg.recency_weight * recency_score
            + cfg.emotional_weight * emo_score
            + cfg.feedback_weight * fb_score
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, importance))

    def apply_decay(self, memory: Memory, time_delta: Optional[float] = None) -> float:
        """Apply time-based decay to a memory's importance score.

        Uses exponential decay: importance *= e^(-decay_rate * time)

        This simulates the Ebbinghaus forgetting curve - memories
        exponentially lose strength without reinforcement.

        Args:
            memory: The memory to decay.
            time_delta: Time elapsed in seconds. If None, calculates
                       from memory's last_accessed timestamp.

        Returns:
            New importance score after decay.
        """
        if time_delta is None:
            time_delta = time.time() - memory.last_accessed

        # Exponential decay
        rate = memory.decay_rate * self.config.decay_rate
        decay_factor = math.exp(-rate * time_delta / 3600.0)  # Normalize to hours

        new_importance = memory.importance_score * decay_factor
        return max(0.0, new_importance)

    def should_forget(self, memory: Memory) -> bool:
        """Determine if a memory should be forgotten (deleted).

        Args:
            memory: The memory to evaluate.

        Returns:
            True if the memory's importance is below the forget threshold.
        """
        current_importance = self.apply_decay(memory)
        return current_importance < self.config.forget_threshold

    def reinforce(self, memory: Memory, boost: float = 0.1) -> float:
        """Reinforce a memory (increase its importance).

        Called when a memory is accessed or explicitly reinforced.
        Simulates the spacing effect in learning.

        Args:
            memory: The memory to reinforce.
            boost: Amount to boost importance by.

        Returns:
            New importance score after reinforcement.
        """
        memory.access_count += 1
        memory.last_accessed = time.time()

        # Boost with diminishing returns
        new_importance = memory.importance_score + boost * (
            1.0 - memory.importance_score
        )
        memory.importance_score = min(1.0, new_importance)

        # Reduce decay rate for frequently accessed memories
        memory.decay_rate *= 0.95  # Memories that are used more decay slower

        return memory.importance_score
