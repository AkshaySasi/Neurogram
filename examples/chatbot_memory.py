"""
Neurogram — Chatbot with Persistent Memory

Shows how to build a chatbot that remembers conversations
across sessions and personalizes responses.
"""

from neurogram import Agent


def main():
    # Create a persistent agent (memories survive restarts)
    assistant = Agent(
        "ChatAssistant",
        description="Helpful AI assistant with memory",
        personality="Friendly, concise, and proactive",
    )

    print("🧠 Neurogram Chatbot (type 'quit' to exit)")
    print("   Memories persist between sessions!\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Get memory context for the user's message
        context = assistant.think(user_input, max_memories=3)

        if context:
            print(f"\n  💭 Memory context:\n  {context}\n")

        # Store this interaction as an experience
        assistant.learn(
            topic=user_input[:50],
            action="responded to user query",
            outcome="conversation continues",
        )

        # Auto-remember if user shares preferences
        preference_keywords = ["prefer", "like", "want", "favorite", "always"]
        if any(kw in user_input.lower() for kw in preference_keywords):
            assistant.remember(
                f"User said: {user_input}",
                importance=0.7,
                metadata={"type": "user_preference"},
            )
            print("  📌 Noted as a preference!\n")

        # Show stats periodically
        stats = assistant.stats()
        print(f"  [{stats['total_memories']} memories stored]\n")

    # Run memory decay before closing
    forgotten = assistant.decay()
    if forgotten > 0:
        print(f"\n  🧹 Forgot {forgotten} unimportant memories")

    print(f"\n  Final stats: {assistant.stats()['total_memories']} memories")
    assistant.close()


if __name__ == "__main__":
    main()
