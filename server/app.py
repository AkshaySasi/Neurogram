"""Neurogram Memory Server — FastAPI REST API.

Provides HTTP endpoints for managing agents and their memories.
This enables multi-language access to the Neurogram memory system.

Start the server:
    neurogram server --host 0.0.0.0 --port 8000

Or programmatically:
    from neurogram.server.app import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI and Pydantic are required for the Neurogram server. "
        "Install them with: pip install neurogram[server]"
    )

from neurogram import __version__
from neurogram.neurogram import Neurogram


# ── Pydantic Models ────────────────────────────────────────────────

class CreateAgentRequest(BaseModel):
    name: str
    description: str = ""
    goals: List[str] = []
    personality: str = ""
    skills: List[str] = []


class RememberRequest(BaseModel):
    content: str
    memory_type: str = "semantic"
    importance: float = 0.5
    metadata: Dict[str, Any] = {}


class RecallRequest(BaseModel):
    query: str
    limit: int = 5
    memory_type: Optional[str] = None
    threshold: float = 0.0


class ThinkRequest(BaseModel):
    prompt: str
    max_memories: int = 5
    format_style: str = "bullet"


class LearnRequest(BaseModel):
    topic: str
    action: str = ""
    outcome: str = ""
    feedback: str = ""
    lesson: str = ""
    emotional_valence: float = 0.0


class LearnProcedureRequest(BaseModel):
    name: str
    steps: List[str]
    description: str = ""
    context: str = ""


class StoreFactRequest(BaseModel):
    fact: str
    category: str = ""
    source: str = ""
    importance: float = 0.5


# ── App Setup ──────────────────────────────────────────────────────

app = FastAPI(
    title="Neurogram Memory Server",
    description="Memory infrastructure for AI agents. REST API for the Neurogram memory system.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Neurogram instance
brain = Neurogram()


# ── Utility ────────────────────────────────────────────────────────

def _get_agent(name: str):
    """Get an agent, raising 404 if not found."""
    agent = brain.get_agent(name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return agent


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check and system info."""
    return {
        "service": "Neurogram Memory Server",
        "version": __version__,
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Agent Management ──────────────────────────────────────────────

@app.post("/agents")
async def create_agent(request: CreateAgentRequest):
    """Create a new agent."""
    agent = brain.create_agent(
        name=request.name,
        description=request.description,
        goals=request.goals,
        personality=request.personality,
        skills=request.skills,
    )
    return {"status": "created", "agent": agent.stats()}


@app.get("/agents")
async def list_agents():
    """List all agents."""
    agents = brain.list_agents()
    return {
        "agents": [a.to_dict() for a in agents],
        "count": len(agents),
    }


@app.get("/agents/{name}")
async def get_agent(name: str):
    """Get agent info and stats."""
    agent = _get_agent(name)
    return agent.stats()


@app.delete("/agents/{name}")
async def delete_agent(name: str):
    """Delete an agent and all its memories."""
    deleted = brain.delete_agent(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return {"status": "deleted", "agent": name}


# ── Memory Operations ─────────────────────────────────────────────

@app.post("/agents/{name}/remember")
async def remember(name: str, request: RememberRequest):
    """Store a memory for an agent."""
    agent = _get_agent(name)
    memory = agent.remember(
        content=request.content,
        memory_type=request.memory_type,
        importance=request.importance,
        metadata=request.metadata,
    )
    return {"status": "stored", "memory_id": memory.id}


@app.post("/agents/{name}/recall")
async def recall(name: str, request: RecallRequest):
    """Search an agent's memory."""
    agent = _get_agent(name)
    results = agent.recall(
        query=request.query,
        limit=request.limit,
        memory_type=request.memory_type,
        threshold=request.threshold,
    )
    return {
        "results": [r.to_dict() for r in results],
        "count": len(results),
    }


@app.post("/agents/{name}/think")
async def think(name: str, request: ThinkRequest):
    """Get memory-augmented context for an LLM prompt."""
    agent = _get_agent(name)
    context = agent.think(
        prompt=request.prompt,
        max_memories=request.max_memories,
        format_style=request.format_style,
    )
    return {"context": context}


@app.post("/agents/{name}/forget/{memory_id}")
async def forget(name: str, memory_id: str):
    """Delete a specific memory."""
    agent = _get_agent(name)
    deleted = agent.forget(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "forgotten", "memory_id": memory_id}


@app.post("/agents/{name}/decay")
async def decay(name: str):
    """Run memory decay — forget unimportant memories."""
    agent = _get_agent(name)
    forgotten = agent.decay()
    return {"status": "decayed", "memories_forgotten": forgotten}


@app.get("/agents/{name}/stats")
async def stats(name: str):
    """Get agent memory statistics."""
    agent = _get_agent(name)
    return agent.stats()


# ── Episodic Memory ───────────────────────────────────────────────

@app.post("/agents/{name}/learn")
async def learn(name: str, request: LearnRequest):
    """Record a learning experience."""
    agent = _get_agent(name)
    episode = agent.learn(
        topic=request.topic,
        action=request.action,
        outcome=request.outcome,
        feedback=request.feedback,
        lesson=request.lesson,
        emotional_valence=request.emotional_valence,
    )
    return {"status": "learned", "episode": episode.to_dict()}


# ── Procedural Memory ─────────────────────────────────────────────

@app.post("/agents/{name}/procedures")
async def learn_procedure(name: str, request: LearnProcedureRequest):
    """Teach the agent a procedure."""
    agent = _get_agent(name)
    procedure = agent.learn_procedure(
        name=request.name,
        steps=request.steps,
        description=request.description,
        context=request.context,
    )
    return {"status": "stored", "procedure": procedure.to_dict()}


# ── Semantic Memory ────────────────────────────────────────────────

@app.post("/agents/{name}/facts")
async def store_fact(name: str, request: StoreFactRequest):
    """Store a factual memory."""
    agent = _get_agent(name)
    memory = agent.store_fact(
        fact=request.fact,
        category=request.category,
        source=request.source,
        importance=request.importance,
    )
    return {"status": "stored", "memory_id": memory.id}
