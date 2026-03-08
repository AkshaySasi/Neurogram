"""
Neurogram - Quick Start Example

Demonstrates the basic API in 10 lines of code.
"""

from neurogram import Agent

# Create an agent with persistent memory
adam = Agent("Adam", description="AI research assistant")

# Store memories
adam.remember("User prefers concise, technical responses")
adam.remember("User is working on a machine learning project")
adam.remember("User's favorite framework is PyTorch")

# Retrieve context for an LLM prompt
context = adam.think("What framework should I recommend?")
print("📝 Context for LLM:")
print(context)
print()

# Search specific memories
results = adam.recall("user preferences")
print("🔍 Relevant memories:")
for r in results:
    print(f"  [{r.relevance_score:.2f}] {r.memory.content}")
print()

# View stats
stats = adam.stats()
print(f"🧠 {stats['agent_name']} has {stats['total_memories']} memories")

adam.close()
