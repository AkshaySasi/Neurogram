import os
import sys

# Try to import google.generativeai to run a real LLM test
try:
    import google.generativeai as genai
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    if has_gemini:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except ImportError:
    has_gemini = False

from neurogram import Agent

def run_interactive_test():
    """Runs a real interactive test using Gemini API."""
    print("==================================================")
    print("🤖 REAL LLM TEST (Using Gemini-2.5-Flash)")
    print("==================================================")
    
    if not has_gemini:
        print("\n[ERROR] Gemini API not configured.")
        print("To run this test, please:")
        print("1. pip install google-generativeai")
        print("2. Set your API key in the terminal:")
        print("   $env:GEMINI_API_KEY=\"your_key_here\"")
        return

    # Use a real-world scenario name
    agent = Agent("personal_assistant")
    
    # 1. Gather memories from the user
    print("\nLet's teach the AI some things about you.")
    print("Type 3 facts about yourself (e.g., dietary restrictions, preferences, location).")
    
    fact1 = input("Fact 1: ")
    if not fact1: fact1 = "I am highly allergic to peanuts."
    agent.remember(fact1)
    
    fact2 = input("Fact 2: ")
    if not fact2: fact2 = "I live in downtown London."
    agent.remember(fact2)
    
    fact3 = input("Fact 3: ")
    if not fact3: fact3 = "I work as a Python developer."
    agent.remember(fact3)
    
    print("\n✅ Memories saved to local SQLite database.")
    
    # 2. Ask a question
    print("\nNow, ask a question where the AI needs to use those facts.")
    print("Example: 'Can you recommend a good lunch spot for me today?'")
    query = input("\nYour question: ")
    if not query: query = "Can you recommend a good lunch spot for me today?"
    
    print("\nCalling Gemini...")
    
    # 3. Test Without Memory
    print("\n" + "-" * 50)
    print("❌ SYSTEM 1: STANDARD GEMINI (No Memory)")
    print("-" * 50)
    model_without_memory = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="You are a helpful assistant."
    )
    response1 = model_without_memory.generate_content(query)
    print(f"\nAI: {response1.text.strip()}")
    
    # 4. Test With Memory
    print("\n\n" + "-" * 50)
    print("✅ SYSTEM 2: GEMINI WITH NEUROGRAM")
    print("-" * 50)
    
    # Neurogram automatically fetches relevant context based on the query
    print("Fetching relevant memories...")
    context = agent.think(query)
    print(f"(Injected context: {context})\n")
    
    model_with_memory = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=f"You are a helpful assistant. Here is what you know about the user:\n{context}"
    )
    response2 = model_with_memory.generate_content(query)
    print(f"AI: {response2.text.strip()}")
    
    print("\n==================================================")
    print("Compare the two responses above.")
    print("Notice how Neurogram intercepts the prompt, finds the relevant facts, and gives the AI context it otherwise wouldn't have.")

if __name__ == "__main__":
    run_interactive_test()
