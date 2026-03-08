<p align="center">
  <img src="assets/banner.svg" alt="Neurogram Banner" width="100%">
</p>

<p align="center">
  <strong>Memory infrastructure for AI agents.</strong><br>
  <em>Where intelligence meets remembrance.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/neurogram/"><img src="https://img.shields.io/pypi/v/neurogram?color=7C3AED&style=for-the-badge" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@centientspace/neurogram"><img src="https://img.shields.io/npm/v/@centientspace/neurogram?color=06B6D4&style=for-the-badge" alt="npm"></a>
  <a href="https://github.com/AkshaySasi/Neurogram/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/AkshaySasi/Neurogram"><img src="https://img.shields.io/github/stars/AkshaySasi/Neurogram?style=for-the-badge&color=FFD700" alt="Stars"></a>
</p>

---

## ⚡ Give your AI agents real memory in 3 lines

```python
from neurogram import Agent

adam = Agent("adam")
adam.remember("User prefers concise, technical responses")
context = adam.think("How should I explain this API?")
```

That's it. Your agent now has **persistent, searchable, evolving memory**.

---

## 🧠 What is Neurogram?

Most AI agents today are **stateless** — they forget everything after each conversation. Neurogram changes that.

Neurogram is an **AI Memory OS** that gives agents human-like memory:

| Memory Type | Human Analogy | Neurogram Feature |
|-------------|--------------|-------------------|
| **Semantic** | Knowing what a router is | `agent.remember("Docker is a containerization platform")` |
| **Episodic** | Remembering yesterday's meeting | `agent.learn(topic="API design", outcome="User liked REST")` |
| **Procedural** | Knowing how to ride a bicycle | `agent.learn_procedure("Deploy", steps=[...])` |
| **Importance** | Important memories persist | Automatic scoring + decay |
| **Forgetting** | Unused memories fade | `agent.decay()` — Ebbinghaus curve |

Unlike simple vector stores, Neurogram implements a **complete cognitive memory architecture**.

---

## 🚀 Installation

```bash
# Core (zero dependencies beyond numpy)
pip install neurogram

# With sentence-transformers (better embeddings)
pip install neurogram[embeddings]

# With FastAPI server
pip install neurogram[server]

# Everything
pip install neurogram[all]
```

**JavaScript/TypeScript:**
```bash
npm install @centientspace/neurogram
```

---

## 📖 Quick Start

### Single Agent

```python
from neurogram import Agent

# Create an agent — memories persist across sessions
agent = Agent("Nova", description="Coding assistant")

# Store memories
agent.remember("User's project uses FastAPI + PostgreSQL")
agent.remember("User prefers type hints in Python code")

# Learn from experience
agent.learn(
    topic="Code review",
    action="Suggested type hints",
    outcome="User appreciated it",
    lesson="Always suggest type hints"
)

# Get context for LLM prompts
context = agent.think("How should I help with this Python code?")
# → "Relevant memories:
#    - User's project uses FastAPI + PostgreSQL
#    - User prefers type hints in Python code"

# Search specific memories
results = agent.recall("user preferences")
for r in results:
    print(f"[{r.relevance_score:.2f}] {r.memory.content}")
```

### Multi-Agent System

```python
from neurogram import Neurogram

brain = Neurogram()

# Each agent has isolated memory
adam = brain.create_agent("Adam", description="Researcher")
nova = brain.create_agent("Nova", description="Coder")

adam.remember("Transformers are the dominant NLP architecture")
nova.remember("User's stack: FastAPI, Docker, K8s")

# Memories don't leak between agents
adam.think("NLP")   # → finds transformer memory
nova.think("NLP")   # → empty (Nova doesn't know about NLP)
```

### Memory-Augmented LLM

```python
from neurogram import Agent
from openai import OpenAI

agent = Agent("assistant")
client = OpenAI()

def chat(user_message: str) -> str:
    # Get relevant memories
    context = agent.think(user_message, format_style="structured")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You have memory:\n{context}"},
            {"role": "user", "content": user_message},
        ]
    )

    answer = response.choices[0].message.content

    # Learn from this interaction
    agent.learn(topic=user_message[:50], action="answered", outcome="responded")
    return answer
```

---

## 🏗️ Architecture

```
User Input
    ↓
AI Agent
    ↓
┌─────────────────────────────────────┐
│         NEUROGRAM MEMORY OS         │
│                                     │
│  ┌───────────┐  ┌───────────────┐   │
│  │ Episodic  │  │   Semantic    │   │
│  │  Memory   │  │   Memory     │   │
│  └───────────┘  └───────────────┘   │
│  ┌───────────┐  ┌───────────────┐   │
│  │Procedural │  │  Importance   │   │
│  │  Memory   │  │   Engine     │   │
│  └───────────┘  └───────────────┘   │
│  ┌───────────┐  ┌───────────────┐   │
│  │ Embedding │  │   Storage     │   │
│  │  Engine   │  │   Backend    │   │
│  └───────────┘  └───────────────┘   │
└─────────────────────────────────────┘
    ↓
LLM Reasoning
    ↓
Response
```

---

## 🔌 Pluggable Backends

### Embedding Engines

| Engine | Quality | Speed | Dependencies |
|--------|---------|-------|-------------|
| `NumpyEmbeddingEngine` | ⭐⭐ | ⚡⚡⚡ | None (default) |
| `LocalEmbeddingEngine` | ⭐⭐⭐⭐ | ⚡⚡ | `sentence-transformers` |
| `OpenAIEmbeddingEngine` | ⭐⭐⭐⭐⭐ | ⚡ | `openai` + API key |

```python
from neurogram import Agent
from neurogram import LocalEmbeddingEngine

# Use sentence-transformers for better quality
agent = Agent("nova", embedding_engine=LocalEmbeddingEngine())
```

### Storage Backends

| Backend | Best For |
|---------|----------|
| `SQLiteBackend` | Development, single-agent (default) |
| Custom backends | Implement `StorageBackend` interface |

---

## 🧪 Memory Server (REST API)

Run the Neurogram server for multi-language access:

```bash
pip install neurogram[server]
neurogram server --port 8000
```

API docs at `http://localhost:8000/docs`

```bash
# Create agent
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "adam"}'

# Store memory
curl -X POST http://localhost:8000/agents/adam/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode"}'

# Search memory
curl -X POST http://localhost:8000/agents/adam/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "UI preferences"}'
```

### JavaScript/TypeScript Client

```typescript
import { Neurogram } from "@centientspace/neurogram";

const brain = new Neurogram("adam", { serverUrl: "http://localhost:8000" });

await brain.create("Research assistant");
await brain.remember("User studies machine learning");

const memories = await brain.recall("user interests");
const context = await brain.think("What should I recommend?");
```

---

## 🧬 Importance & Forgetting

Neurogram uses biologically-inspired memory dynamics:

**Importance Score** = α×Frequency + β×Recency + γ×Emotion + δ×Feedback

- **Frequency**: Memories accessed more often strengthen
- **Recency**: Recent memories are naturally more vivid
- **Emotion**: Emotionally charged memories persist longer
- **Forgetting**: Ebbinghaus exponential decay curve

```python
agent = Agent("nova")

# Frequently accessed memories become stronger
for _ in range(10):
    agent.recall("important topic")

# Run decay — low-importance memories get pruned
forgotten = agent.decay()
print(f"Forgot {forgotten} faded memories")
```

---

## 📁 Project Structure

```
neurogram/
├── neurogram/                  # Python SDK
│   ├── __init__.py
│   ├── agent.py               # Primary developer API
│   ├── neurogram.py            # Multi-agent manager
│   ├── memory_manager.py       # Central orchestrator
│   ├── embedding_engine.py     # Pluggable embeddings
│   ├── importance_engine.py    # Scoring & decay
│   ├── episodic_memory.py      # Experience memory
│   ├── semantic_memory.py      # Factual knowledge
│   ├── procedural_memory.py    # Skills & procedures
│   ├── cli.py                  # CLI interface
│   ├── types.py                # Core data types
│   └── storage/
│       ├── base.py             # Storage interface
│       └── sqlite_backend.py   # Default SQLite storage
├── server/
│   └── app.py                  # FastAPI REST server
├── neurogram-js/               # NPM package
│   └── src/index.ts            # TypeScript SDK
├── examples/
│   ├── quickstart.py
│   ├── chatbot_memory.py
│   └── multi_agent.py
├── tests/
├── pyproject.toml
└── README.md
```

---

## 🛣️ Roadmap

- [x] Core memory engine (semantic, episodic, procedural)
- [x] Importance scoring & decay
- [x] SQLite storage backend
- [x] FastAPI REST server
- [x] TypeScript/JS SDK
- [x] Memory consolidation (short-term → long-term)
- [x] Memory visualization dashboard
- [x] LangChain integration
- [ ] Knowledge graph connections
- [ ] PostgreSQL + pgvector backend
- [ ] Hosted cloud platform

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
git clone https://github.com/AkshaySasi/Neurogram.git
cd Neurogram
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Neurogram</strong> — Where intelligence meets remembrance. 🧠
</p>
