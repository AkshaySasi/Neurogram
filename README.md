<p align="center">
  <img src="assets/banner.svg" alt="Neurogram Banner" width="100%">
</p>

<p align="center">
  <strong>The cognitive memory engine for AI agents.</strong><br>
  <em>Not another vector store. A complete memory architecture — modeled after the human brain.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/neurogram/"><img src="https://img.shields.io/pypi/v/neurogram?color=7C3AED&style=for-the-badge" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@centientspace/neurogram"><img src="https://img.shields.io/npm/v/@centientspace/neurogram?color=06B6D4&style=for-the-badge" alt="npm"></a>
  <a href="https://github.com/AkshaySasi/Neurogram/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/AkshaySasi/Neurogram"><img src="https://img.shields.io/github/stars/AkshaySasi/Neurogram?style=for-the-badge&color=FFD700" alt="Stars"></a>
</p>

---

## 🧠 Every AI has memory now. So what's different?

Yes — ChatGPT, Claude, Gemini all "remember" things now. And tools like mem0, Zep, and Letta exist. **But they all do the same thing**: store text, retrieve text. A glorified search engine over your past conversations.

**Humans don't work that way.** Your brain doesn't dump all memories into one bucket. It has:
- **Semantic memory** (facts: "Python is a programming language")
- **Episodic memory** (experiences: "Last time I tried recursion, the user got confused")
- **Procedural memory** (skills: "How to deploy to Kubernetes, step by step")
- **Forgetting** (unimportant things naturally fade away)

**Neurogram gives AI agents that same architecture.** Not just storage — a cognitive memory system.

---

## ⚡ What Neurogram does that others don't

| Feature | mem0 / Zep / Letta | Vector DBs | **Neurogram** |
|---------|-------------------|------------|---------------|
| Store & search text | ✅ | ✅ | ✅ |
| Separate memory types (semantic, episodic, procedural) | ❌ | ❌ | ✅ |
| Memory consolidation (merge fragmented → strong) | ❌ | ❌ | ✅ |
| Biological forgetting (Ebbinghaus decay) | ❌ | ❌ | ✅ |
| Importance scoring (frequency × recency × emotion) | ❌ | ❌ | ✅ |
| Learn from experience (not just store text) | ❌ | ❌ | ✅ |
| Remember skills & procedures | ❌ | ❌ | ✅ |
| Zero-config (`pip install` and done) | Needs setup | Needs server | ✅ |

---

## 🔥 The killer feature: Memory Consolidation

No competitor has this. Inspired by how the human brain consolidates memories during sleep:

```python
agent = Agent("nova")

# Throughout the day, agent accumulates fragmented memories
agent.remember("User likes Python")
agent.remember("User prefers Python over JavaScript")
agent.remember("User uses Python for backend work")
agent.remember("User's stack is Python + FastAPI")

# Consolidation merges similar fragments into fewer, stronger memories
stats = agent.consolidate()  # or agent.sleep()
# → 4 fragmented memories become 1 strong memory:
#   "User strongly prefers Python, uses it for backend with FastAPI"
```

This makes agents **more efficient over time** — fewer memories, better recall, lower token costs.

---

## 🧬 Three types of memory, not one bucket

```python
from neurogram import Agent

agent = Agent("nova")

# Semantic Memory — facts and knowledge
agent.remember("Company uses microservices architecture")
agent.store_fact("Kubernetes orchestrates containers", category="DevOps")

# Episodic Memory — learning from experience
agent.learn(
    topic="Code review",
    action="Gave complex explanation",
    outcome="User was confused",
    lesson="Use simple analogies instead"
)
# Next time, agent remembers to simplify!

# Procedural Memory — skills and step-by-step knowledge
agent.learn_procedure(
    name="Deploy to production",
    steps=["Run tests", "Build Docker image", "Push to registry", "Deploy to K8s"],
    context="When user asks about deployment"
)
```

---

## 📉 Memories that fade — just like yours

Most AI memory tools keep everything forever. That's not how brains work, and it's not efficient.

Neurogram uses the **Ebbinghaus forgetting curve** — memories naturally decay unless reinforced:

```python
# Important memories get accessed often → stay strong
agent.recall("deployment process")  # reinforced → importance goes UP

# Unimportant memories fade over time
agent.decay()  # applies time-based decay, prunes weak memories
```

**Importance Score** = frequency × recency × emotional valence × user feedback

---

## 🚀 Get started in 30 seconds

```bash
pip install neurogram
```

```python
from neurogram import Agent

agent = Agent("my_agent")

# It just works. No database server. No API keys. No Docker.
agent.remember("User prefers dark mode and concise responses")
agent.learn(topic="UI design", outcome="User loved minimal approach", lesson="Keep it simple")

# Later — agent recalls what matters
context = agent.think("How should I design the next feature?")
# → Relevant memories injected for LLM context
```

**JavaScript/TypeScript:**
```bash
npm install @centientspace/neurogram
```

---

## 🔌 Drop into LangChain in 2 lines

```python
from neurogram.integrations.langchain import NeurogramMemory

# Replaces LangChain's built-in memory with cognitive memory
memory = NeurogramMemory(agent_name="assistant")
chain = ConversationChain(llm=OpenAI(), memory=memory)
```

---

## 📊 Visualize your agent's brain

```bash
pip install neurogram[server]
neurogram dashboard
```

A live dashboard showing memory type breakdown, importance heatmaps, and memory decay over time.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           NEUROGRAM MEMORY OS           │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Episodic   │  │    Semantic     │   │
│  │  Memory     │  │    Memory       │   │
│  └─────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Procedural  │  │  Consolidation  │   │
│  │  Memory     │  │    Engine       │   │
│  └─────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Importance  │  │    Forgetting   │   │
│  │  Scoring    │  │    (Decay)      │   │
│  └─────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Embedding  │  │    Storage      │   │
│  │  Engine     │  │    Backend      │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
```

### Embedding Engines

| Engine | Quality | Speed | Dependencies |
|--------|---------|-------|-------------|
| `NumpyEmbeddingEngine` | ⭐⭐ | ⚡⚡⚡ | None (default) |
| `LocalEmbeddingEngine` | ⭐⭐⭐⭐ | ⚡⚡ | `sentence-transformers` |
| `OpenAIEmbeddingEngine` | ⭐⭐⭐⭐⭐ | ⚡ | `openai` + API key |

---

## 🧪 Memory Server (REST API)

```bash
pip install neurogram[server]
neurogram server --port 8000
```

Full REST API at `http://localhost:8000/docs` — use from any language.

```typescript
import { Neurogram } from "@centientspace/neurogram";

const brain = new Neurogram("adam", { serverUrl: "http://localhost:8000" });
await brain.remember("User studies machine learning");
const context = await brain.think("What should I recommend?");
```

---

## 🛣️ Roadmap

- [x] Cognitive memory types (semantic, episodic, procedural)
- [x] Importance scoring & Ebbinghaus decay
- [x] Memory consolidation (`agent.sleep()`)
- [x] LangChain integration
- [x] Memory visualization dashboard
- [x] FastAPI REST server + JS/TS SDK
- [ ] Knowledge graph connections
- [ ] PostgreSQL + pgvector backend
- [ ] Hosted cloud platform

---

## 🤝 Contributing

```bash
git clone https://github.com/AkshaySasi/Neurogram.git
cd Neurogram
pip install -e ".[dev]"
python -m pytest tests/ -v  # 52 tests passing
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Neurogram</strong> — not another vector store. A cognitive architecture for AI memory. 🧠
</p>
