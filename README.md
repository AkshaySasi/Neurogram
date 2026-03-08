<p align="center">
  <img src="assets/banner.svg" alt="Neurogram Banner" width="100%">
</p>

<p align="center">
  <strong>A cognitive memory architecture for AI agents.</strong><br>
  <em>Not another vector store. Structured memory types, biological forgetting, and memory consolidation.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/neurogram/"><img src="https://img.shields.io/pypi/v/neurogram?color=7C3AED&style=for-the-badge" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@centientspace/neurogram"><img src="https://img.shields.io/npm/v/@centientspace/neurogram?color=06B6D4&style=for-the-badge" alt="npm"></a>
  <a href="https://github.com/AkshaySasi/Neurogram/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/AkshaySasi/Neurogram"><img src="https://img.shields.io/github/stars/AkshaySasi/Neurogram?style=for-the-badge&color=FFD700" alt="Stars"></a>
</p>

---

## The problem with AI memory today

Most memory tools for AI agents work the same way: dump everything into one bucket, search by similarity. That's basically a vector database with extra steps.

But the human brain organizes memory very differently:

- **Semantic memory** for facts ("Python is a programming language")
- **Episodic memory** for experiences ("Last time I explained recursion, the user got confused")
- **Procedural memory** for skills ("Steps to deploy to production: 1, 2, 3...")
- **Forgetting** for efficiency (unimportant things naturally fade)

Neurogram brings this structure to AI agents. It separates memory into types, each with its own retrieval logic, and adds features like importance scoring and time-based decay.

---

## What Neurogram offers

| Feature | How it works |
|---------|-------------|
| **Three memory types** | Semantic stores facts (sorted by importance). Episodic stores experiences (sorted by recency). Procedural stores skills (sorted by success rate). |
| **Importance scoring** | Each memory gets a score based on access frequency, recency, and emotional weight. Frequently recalled memories get stronger. |
| **Biological forgetting** | Time-based decay using the Ebbinghaus curve. Unimportant memories fade, keeping the memory space efficient. |
| **Memory consolidation** | Groups similar memories and merges them into stronger, deduplicated versions. Reduces memory count, boosts quality. |
| **Zero config** | `pip install neurogram` and it works. SQLite storage, numpy-based embeddings - no external services needed. |

> **Note on default embeddings:** Out of the box, Neurogram uses a lightweight hash-based embedding engine. This works for basic matching but is not truly semantic. For proper semantic similarity, install `neurogram[embeddings]` which uses sentence-transformers.

---

## Quick start

```bash
pip install neurogram

# For much better search quality (recommended):
pip install neurogram[embeddings]
```

```python
from neurogram import Agent

agent = Agent("my_agent")

# Store different types of memory
agent.remember("User prefers dark mode and concise responses")

agent.learn(
    topic="UI design",
    outcome="User loved minimal approach",
    lesson="Keep interfaces simple"
)

agent.learn_procedure(
    name="Deploy API",
    steps=["Run tests", "Build image", "Push to registry", "Deploy"],
)

# Retrieve - each type has its own retrieval logic
context = agent.think("How should I design the next feature?")
```

---

## How each memory type works differently

This is what separates Neurogram from flat memory stores. Each memory type has distinct storage, retrieval, and data structures:

### Semantic Memory (facts)
Stores factual knowledge. Retrieval sorts by **importance score** - the most reinforced facts surface first.

```python
agent.remember("Company uses microservices architecture")
agent.store_fact("FastAPI is a Python web framework", category="tech")

# Retrieval prioritizes the most important/accessed facts
facts = agent.query_facts("web framework")
```

### Episodic Memory (experiences)
Stores structured experiences with topic, action, outcome, and lesson fields. Retrieval sorts by **recency** - recent experiences surface first, like human memory.

```python
agent.learn(
    topic="Code review",
    action="Gave complex explanation",
    outcome="User was confused",
    lesson="Use simple analogies instead"
)

# Returns recent episodes first
lessons = agent.get_lessons("code review")
```

### Procedural Memory (skills)
Stores step-by-step procedures. Retrieval sorts by **success rate** - proven procedures surface first. Tracks success/failure counts per procedure.

```python
agent.learn_procedure(
    name="Deploy to production",
    steps=["Run tests", "Build Docker image", "Push to registry", "Deploy to K8s"],
)

# Returns procedures with highest success rate first
procedures = agent.recall_procedures("deployment")
```

---

## Memory consolidation

Groups similar memories by embedding proximity and merges them into fewer, deduplicated versions. The merged memories get a 20% importance boost and decay 50% slower.

```python
# Over time, agent accumulates similar memories
agent.remember("User likes Python")
agent.remember("User prefers Python over JavaScript")
agent.remember("User uses Python for backend work")

# Consolidation merges similar ones, removes originals
stats = agent.consolidate()  # or agent.sleep()
print(stats)
# {'clusters_found': 1, 'memories_merged': 3, 'memories_created': 1, 'memories_removed': 3}
```

> **Current limitation:** Consolidation merges content by concatenation (e.g., "User likes Python | User prefers Python over JS"). A future version will use LLM-powered summarization to produce natural merged content like "User strongly prefers Python for backend development."

---

## Forgetting and decay

Importance scoring is based on:

**Score** = frequency x recency x emotional_valence x user_feedback

Memories accessed often get reinforced. Memories ignored gradually decay following the Ebbinghaus forgetting curve.

```python
# Frequently accessed memories get stronger
for _ in range(10):
    agent.recall("deployment process")  # reinforced each time

# Run decay - low-importance memories get pruned
forgotten = agent.decay()
print(f"Forgot {forgotten} faded memories")
```

---

## LangChain integration

Drop-in memory replacement for LangChain chains:

```python
from neurogram.integrations.langchain import NeurogramMemory

memory = NeurogramMemory(agent_name="assistant")
chain = ConversationChain(llm=OpenAI(), memory=memory)
```

---

## Dashboard

```bash
pip install neurogram[server]
neurogram dashboard
```

A web dashboard showing memory type breakdown, importance heatmap, and memory table.

---

## Architecture

```
                     NEUROGRAM
  ┌──────────────────────────────────────┐
  │  Episodic    Semantic    Procedural  │  <- Memory Types
  │  (recency)   (importance) (success)  │  <- Retrieval Logic
  ├──────────────────────────────────────┤
  │  Consolidation Engine                │  <- Merge similar memories
  │  Importance Scoring + Decay          │  <- Ebbinghaus curve
  ├──────────────────────────────────────┤
  │  Embedding Engine    Storage Backend │  <- Pluggable
  │  (numpy/st/openai)  (sqlite/custom) │
  └──────────────────────────────────────┘
```

### Embedding engines

| Engine | Quality | Setup |
|--------|---------|-------|
| `NumpyEmbeddingEngine` | Approximate (hash-based) | Default, zero dependencies |
| `LocalEmbeddingEngine` | High (neural) | `pip install neurogram[embeddings]` |
| `OpenAIEmbeddingEngine` | Highest | `pip install neurogram[openai]` + API key |

---

## REST API server

```bash
pip install neurogram[server]
neurogram server --port 8000
```

Full REST API with Swagger docs at `http://localhost:8000/docs`.

```typescript
import { Neurogram } from "@centientspace/neurogram";

const brain = new Neurogram("adam", { serverUrl: "http://localhost:8000" });
await brain.remember("User studies machine learning");
const context = await brain.think("What should I recommend?");
```

---

## Roadmap

**Completed:**
- [x] Three memory types with distinct retrieval logic
- [x] Importance scoring and Ebbinghaus decay
- [x] Memory consolidation (concatenation-based)
- [x] LangChain integration
- [x] Memory visualization dashboard
- [x] FastAPI REST server + JS/TS SDK
- [x] Zero-config setup (pip install and run)

**Future directions:**
- [ ] LLM-powered consolidation (intelligent summarization instead of concatenation)
- [ ] Behavior adaptation (agent actually changes behavior based on episodic lessons)
- [ ] Knowledge graph connections between memories
- [ ] PostgreSQL + pgvector backend for production scale
- [ ] Hosted cloud API

---

## Contributing

```bash
git clone https://github.com/AkshaySasi/Neurogram.git
cd Neurogram
pip install -e ".[dev]"
python -m pytest tests/ -v  # 52 tests passing
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Neurogram</strong> - structured memory for AI agents.
</p>
