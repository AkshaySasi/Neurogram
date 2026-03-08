import os
import sys

# Optional: Try to import google.generativeai to run a real LLM test
try:
    import google.generativeai as genai
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    if has_gemini:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except ImportError:
    has_gemini = False

from neurogram import Agent

def run_simulation_test():
    """Runs a simulated test showing the structural difference in prompts."""
    print("==================================================")
    print("🧠 NEUROGRAM VS STANDARD AI: CONCEPTUAL TEST")
    print("==================================================")
    
    print("\n[Scenario]")
    print("Over the past month, you've told the AI:")
    print("1. 'I am highly allergic to peanuts.'")
    print("2. 'I live in downtown London.'")
    print("3. 'I work as a Python developer.'")
    
    print("\nToday, you ask a brand new question (new chat session):")
    query = "Can you recommend a good lunch spot?"
    print(f"User: '{query}'")
    
    print("\n" + "-" * 50)
    print("❌ SYSTEM 1: STANDARD AI (No Memory)")
    print("-" * 50)
    print("System Prompt: 'You are a helpful assistant.'")
    print(f"User Prompt: '{query}'")
    print("Result: The AI gives generic London restaurants, or asks where you live, and crucially, forgets to check for peanut allergies because you didn't mention it in this specific prompt.")
    
    print("\n" + "-" * 50)
    print("✅ SYSTEM 2: AI WITH NEUROGRAM")
    print("-" * 50)
    
    # Initialize Neurogram Agent
    agent = Agent("test_agent_1")
    
    # Simulate past memories
    agent.remember("User is highly allergic to peanuts", importance=0.9)
    agent.remember("User lives in downtown London", importance=0.7)
    agent.remember("User works as a Python developer", importance=0.6)
    
    # Neurogram automatically fetches relevant context based on the query
    context = agent.think(query, format_style="structured")
    
    print(f"System Prompt: 'You are a helpful assistant. You have the following context:\n{context}'")
    print(f"User Prompt: '{query}'")
    print("Result: The AI recommends a peanut-free restaurant in downtown London, perhaps with a tech/developer vibe.")

def run_real_llm_test():
    """Runs a real test using Gemini API."""
    print("==================================================")
    print("🤖 REAL LLM TEST (Using Gemini-2.5-Flash)")
    print("==================================================")
    
    agent = Agent("test_agent_2")
    
    # Seed memories
    agent.remember("User is highly allergic to peanuts")
    agent.remember("User lives in downtown London")
    
    query = "Can you recommend a good lunch spot for me today?"
    print(f"\nUser: '{query}'")
    
    # 1. Without Memory
    print("\n[Without Neurogram]")
    model_without_memory = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="You are a helpful assistant."
    )
    response1 = model_without_memory.generate_content(query)
    print(f"AI: {response1.text.strip()}")
    
    # 2. With Memory
    print("\n[With Neurogram]")
    context = agent.think(query)
    model_with_memory = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=f"You are a helpful assistant. Here is what you know about the user:\n{context}"
    )
    response2 = model_with_memory.generate_content(query)
    print(f"AI: {response2.text.strip()}")


if __name__ == "__main__":
    run_simulation_test()
    
    print("\n\n")
    if has_gemini:
        run_real_llm_test()
    else:
        print("-" * 50)
        print("Note: To run the real comparative test using Gemini, please run:")
        print("  pip install google-generativeai")
        print("And set your API key in the terminal before running the script:")
        print("  $env:GEMINI_API_KEY=\"your_key_here\"")
        print("-" * 50)
