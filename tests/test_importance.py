"""Tests for importance scoring and memory decay."""

import time
import pytest

from neurogram.importance_engine import ImportanceConfig, ImportanceEngine
from neurogram.types import Memory, MemoryType


@pytest.fixture
def engine():
    return ImportanceEngine()


@pytest.fixture
def memory():
    return Memory(
        content="Test memory",
        agent_id="test",
        memory_type=MemoryType.SEMANTIC,
        importance_score=0.5,
        access_count=5,
        last_accessed=time.time(),
    )


class TestImportanceScoring:
    def test_score_fresh_memory(self, engine, memory):
        score = engine.score(memory)
        assert 0.0 <= score <= 1.0

    def test_higher_frequency_increases_score(self, engine):
        low_freq = Memory(content="low", agent_id="t", access_count=1,
                          last_accessed=time.time())
        high_freq = Memory(content="high", agent_id="t", access_count=50,
                           last_accessed=time.time())

        s_low = engine.score(low_freq)
        s_high = engine.score(high_freq)
        assert s_high > s_low

    def test_emotional_score_increases_importance(self, engine, memory):
        neutral = engine.score(memory, emotional_score=0.0)
        emotional = engine.score(memory, emotional_score=0.9)
        assert emotional > neutral

    def test_positive_feedback_increases_importance(self, engine, memory):
        negative = engine.score(memory, feedback_score=-0.5)
        positive = engine.score(memory, feedback_score=0.5)
        assert positive > negative


class TestDecay:
    def test_decay_reduces_importance(self, engine, memory):
        original = memory.importance_score
        decayed = engine.apply_decay(memory, time_delta=86400)  # 24 hours
        assert decayed < original

    def test_no_decay_at_zero_time(self, engine, memory):
        decayed = engine.apply_decay(memory, time_delta=0)
        assert abs(decayed - memory.importance_score) < 0.001

    def test_should_forget_low_importance(self, engine):
        mem = Memory(content="forgettable", agent_id="t",
                     importance_score=0.01,
                     last_accessed=time.time() - 86400 * 30)  # 30 days old
        assert engine.should_forget(mem)

    def test_should_not_forget_important(self, engine):
        mem = Memory(content="important", agent_id="t",
                     importance_score=0.9,
                     last_accessed=time.time())
        assert not engine.should_forget(mem)


class TestReinforce:
    def test_reinforce_increases_importance(self, engine, memory):
        original = memory.importance_score
        engine.reinforce(memory)
        assert memory.importance_score > original

    def test_reinforce_increments_access_count(self, engine, memory):
        original_count = memory.access_count
        engine.reinforce(memory)
        assert memory.access_count == original_count + 1

    def test_reinforce_reduces_decay_rate(self, engine, memory):
        original_rate = memory.decay_rate
        engine.reinforce(memory)
        assert memory.decay_rate < original_rate
