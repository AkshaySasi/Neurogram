"""
Neurogram - Multi-Agent Memory Example

Demonstrates multiple agents with separate memory spaces,
each learning and remembering independently.
"""

from neurogram import Neurogram


def main():
    # Create a Neurogram system with shared storage
    brain = Neurogram()

    # Create specialized agents
    adam = brain.create_agent(
        "Adam",
        description="Research assistant",
        skills=["literature review", "summarization", "analysis"],
    )

    nova = brain.create_agent(
        "Nova",
        description="Coding assistant",
        skills=["Python", "FastAPI", "Docker", "debugging"],
    )

    atlas = brain.create_agent(
        "Atlas",
        description="Data analyst",
        skills=["SQL", "pandas", "visualization", "statistics"],
    )

    # Each agent learns different things
    print("📚 Teaching agents...\n")

    # Adam learns research facts
    adam.remember("Transformers are the dominant architecture in NLP")
    adam.remember("RAG improves factual accuracy of LLMs")
    adam.learn(
        topic="Literature Review",
        lesson="Start with survey papers, then follow citations",
    )

    # Nova learns coding procedures
    nova.remember("User's project uses FastAPI with PostgreSQL")
    nova.learn_procedure(
        name="Debug Python Error",
        steps=["Read traceback", "Check imports", "Add logging", "Test fix"],
    )
    nova.learn(
        topic="Code Review",
        action="Suggested type hints",
        outcome="User appreciated it",
        feedback="positive",
        lesson="Always suggest type hints for Python code",
    )

    # Atlas learns data patterns
    atlas.remember("Sales data has seasonal patterns - Q4 is strongest")
    atlas.store_fact(
        "Average customer lifetime value is $2,400",
        category="metrics",
        source="analytics dashboard",
    )

    # Query each agent's memory
    print("🔍 Querying agents...\n")

    for agent, query in [
        (adam, "NLP architectures"),
        (nova, "debugging errors"),
        (atlas, "customer metrics"),
    ]:
        results = agent.recall(query, limit=2)
        stats = agent.stats()
        print(f"  🤖 {stats['agent_name']} ({stats['total_memories']} memories):")
        for r in results:
            print(f"     [{r.relevance_score:.2f}] {r.memory.content}")
        print()

    # Show all agents
    print("📋 All agents:")
    for config in brain.list_agents():
        print(f"  • {config.name}: {config.description}")

    brain.close()


if __name__ == "__main__":
    main()
