/**
 * Neurogram.js — Memory infrastructure for AI agents.
 *
 * TypeScript/JavaScript client SDK for the Neurogram Memory Server.
 * Requires a running Neurogram server.
 *
 * @example
 * ```typescript
 * import { Neurogram } from "neurogram-js";
 *
 * const brain = new Neurogram("adam", { serverUrl: "http://localhost:8000" });
 *
 * await brain.remember("User prefers dark mode");
 * const memories = await brain.recall("UI preferences");
 * const context = await brain.think("How should I design the UI?");
 * ```
 */

// ── Types ─────────────────────────────────────────────────────────

export interface NeurogramConfig {
    /** URL of the Neurogram Memory Server */
    serverUrl?: string;
}

export interface Memory {
    id: string;
    agent_id: string;
    memory_type: string;
    content: string;
    metadata: Record<string, any>;
    importance_score: number;
    created_at: number;
    last_accessed: number;
    access_count: number;
}

export interface RetrievalResult {
    memory: Memory;
    relevance_score: number;
}

export interface Episode {
    id: string;
    agent_id: string;
    topic: string;
    action: string;
    outcome: string;
    feedback: string;
    lesson: string;
    emotional_valence: number;
    timestamp: number;
}

export interface Procedure {
    id: string;
    agent_id: string;
    name: string;
    description: string;
    steps: string[];
    context: string;
    success_count: number;
    failure_count: number;
}

export interface AgentStats {
    agent_id: string;
    agent_name: string;
    total_memories: number;
    by_type: Record<string, number>;
    embedding_engine: string;
    embedding_dimensions: number;
    storage_backend: string;
}

export interface AgentConfig {
    agent_id: string;
    name: string;
    description: string;
    goals: string[];
    personality: string;
    skills: string[];
}

// ── Client ────────────────────────────────────────────────────────

export class Neurogram {
    private agentName: string;
    private serverUrl: string;

    /**
     * Create a Neurogram client for an agent.
     *
     * @param agentName - Name of the agent
     * @param config - Configuration options
     */
    constructor(agentName: string, config: NeurogramConfig = {}) {
        this.agentName = agentName;
        this.serverUrl = config.serverUrl || "http://localhost:8000";
    }

    // ── HTTP Helper ───────────────────────────────────────────────

    private async request<T>(
        method: string,
        path: string,
        body?: any
    ): Promise<T> {
        const url = `${this.serverUrl}${path}`;
        const options: RequestInit = {
            method,
            headers: { "Content-Type": "application/json" },
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Neurogram API error (${response.status}): ${error}`);
        }

        return response.json() as Promise<T>;
    }

    // ── Agent Management ──────────────────────────────────────────

    /**
     * Create this agent on the server (if not exists).
     */
    async create(
        description: string = "",
        goals: string[] = [],
        personality: string = "",
        skills: string[] = []
    ): Promise<AgentStats> {
        const result = await this.request<{ agent: AgentStats }>(
            "POST",
            "/agents",
            {
                name: this.agentName,
                description,
                goals,
                personality,
                skills,
            }
        );
        return result.agent;
    }

    /**
     * Get agent stats.
     */
    async stats(): Promise<AgentStats> {
        return this.request<AgentStats>(
            "GET",
            `/agents/${encodeURIComponent(this.agentName)}`
        );
    }

    /**
     * Delete this agent and all its memories.
     */
    async delete(): Promise<void> {
        await this.request("DELETE", `/agents/${encodeURIComponent(this.agentName)}`);
    }

    // ── Memory Operations ─────────────────────────────────────────

    /**
     * Store a memory.
     *
     * @param content - The information to remember
     * @param memoryType - Type of memory (default: "semantic")
     * @param importance - Importance score (0.0 - 1.0)
     * @param metadata - Additional metadata
     */
    async remember(
        content: string,
        memoryType: string = "semantic",
        importance: number = 0.5,
        metadata: Record<string, any> = {}
    ): Promise<string> {
        const result = await this.request<{ memory_id: string }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/remember`,
            {
                content,
                memory_type: memoryType,
                importance,
                metadata,
            }
        );
        return result.memory_id;
    }

    /**
     * Search memory for relevant information.
     *
     * @param query - What to search for
     * @param limit - Maximum results
     * @param memoryType - Optional type filter
     * @param threshold - Minimum relevance score
     */
    async recall(
        query: string,
        limit: number = 5,
        memoryType?: string,
        threshold: number = 0.0
    ): Promise<RetrievalResult[]> {
        const result = await this.request<{ results: RetrievalResult[] }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/recall`,
            {
                query,
                limit,
                memory_type: memoryType,
                threshold,
            }
        );
        return result.results;
    }

    /**
     * Get memory-augmented context for an LLM prompt.
     *
     * @param prompt - The user's prompt
     * @param maxMemories - Maximum memories to include
     * @param formatStyle - How to format context
     */
    async think(
        prompt: string,
        maxMemories: number = 5,
        formatStyle: string = "bullet"
    ): Promise<string> {
        const result = await this.request<{ context: string }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/think`,
            {
                prompt,
                max_memories: maxMemories,
                format_style: formatStyle,
            }
        );
        return result.context;
    }

    /**
     * Delete a specific memory.
     */
    async forget(memoryId: string): Promise<void> {
        await this.request(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/forget/${memoryId}`
        );
    }

    /**
     * Run memory decay.
     */
    async decay(): Promise<number> {
        const result = await this.request<{ memories_forgotten: number }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/decay`
        );
        return result.memories_forgotten;
    }

    // ── Episodic Memory ───────────────────────────────────────────

    /**
     * Record a learning experience.
     */
    async learn(
        topic: string,
        action: string = "",
        outcome: string = "",
        feedback: string = "",
        lesson: string = "",
        emotionalValence: number = 0.0
    ): Promise<Episode> {
        const result = await this.request<{ episode: Episode }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/learn`,
            {
                topic,
                action,
                outcome,
                feedback,
                lesson,
                emotional_valence: emotionalValence,
            }
        );
        return result.episode;
    }

    // ── Procedural Memory ─────────────────────────────────────────

    /**
     * Teach the agent a procedure.
     */
    async learnProcedure(
        name: string,
        steps: string[],
        description: string = "",
        context: string = ""
    ): Promise<Procedure> {
        const result = await this.request<{ procedure: Procedure }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/procedures`,
            { name, steps, description, context }
        );
        return result.procedure;
    }

    // ── Semantic Memory ───────────────────────────────────────────

    /**
     * Store a factual memory.
     */
    async storeFact(
        fact: string,
        category: string = "",
        source: string = "",
        importance: number = 0.5
    ): Promise<string> {
        const result = await this.request<{ memory_id: string }>(
            "POST",
            `/agents/${encodeURIComponent(this.agentName)}/facts`,
            { fact, category, source, importance }
        );
        return result.memory_id;
    }
}

// ── Multi-Agent Manager ─────────────────────────────────────────

export class NeurogramManager {
    private serverUrl: string;

    constructor(config: NeurogramConfig = {}) {
        this.serverUrl = config.serverUrl || "http://localhost:8000";
    }

    /**
     * Create a Neurogram client for an agent.
     */
    agent(name: string): Neurogram {
        return new Neurogram(name, { serverUrl: this.serverUrl });
    }

    /**
     * List all agents on the server.
     */
    async listAgents(): Promise<AgentConfig[]> {
        const response = await fetch(`${this.serverUrl}/agents`);
        const data = await response.json();
        return data.agents;
    }
}

export default Neurogram;
