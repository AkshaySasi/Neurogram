"""Neurogram Memory Visualization Dashboard.

A self-contained web dashboard for visualizing agent memories,
importance scores, decay over time, and memory type breakdowns.

Usage:
    neurogram dashboard --port 8080

Or programmatically:
    from neurogram.dashboard.server import start_dashboard
    start_dashboard(port=8080)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise ImportError(
        "FastAPI is required for the dashboard. "
        "Install with: pip install neurogram[server]"
    )

from neurogram.neurogram import Neurogram
from neurogram.types import MemoryType


def create_dashboard_app(storage_path: Optional[str] = None) -> FastAPI:
    """Create the dashboard FastAPI application."""

    app = FastAPI(title="Neurogram Dashboard", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    brain = Neurogram(storage_path=storage_path)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard HTML."""
        return get_dashboard_html()

    @app.get("/api/agents")
    async def api_agents():
        """List all agents with their stats."""
        agents = brain.list_agents()
        result = []
        for config in agents:
            agent = brain.get_agent(config.name)
            if agent:
                stats = agent.stats()
                result.append(stats)
        return {"agents": result}

    @app.get("/api/agents/{name}/memories")
    async def api_memories(name: str, limit: int = 200):
        """Get all memories for an agent."""
        agent = brain.get_agent(name)
        if not agent:
            return {"memories": []}

        # Use internal storage directly for full access
        memories = agent._memory._storage.list_memories(
            agent._agent_id, limit=limit
        )
        return {
            "memories": [
                {
                    "id": m.id,
                    "content": m.content[:200],
                    "memory_type": m.memory_type.value,
                    "importance_score": round(m.importance_score, 3),
                    "created_at": m.created_at,
                    "last_accessed": m.last_accessed,
                    "access_count": m.access_count,
                    "decay_rate": round(m.decay_rate, 4),
                    "metadata": {
                        k: v for k, v in m.metadata.items()
                        if k not in ("episode", "procedure", "embedding")
                    },
                }
                for m in memories
            ]
        }

    @app.get("/api/agents/{name}/timeline")
    async def api_timeline(name: str):
        """Get memory creation timeline data."""
        agent = brain.get_agent(name)
        if not agent:
            return {"timeline": []}

        memories = agent._memory._storage.list_memories(
            agent._agent_id, limit=1000
        )

        # Group by hour
        from collections import defaultdict
        import time

        hourly = defaultdict(lambda: {"semantic": 0, "episodic": 0, "procedural": 0, "short_term": 0})
        for m in memories:
            hour_key = int(m.created_at // 3600) * 3600
            hourly[hour_key][m.memory_type.value] += 1

        timeline = [
            {"timestamp": ts, **counts}
            for ts, counts in sorted(hourly.items())
        ]
        return {"timeline": timeline}

    return app


def get_dashboard_html() -> str:
    """Return the self-contained dashboard HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neurogram Dashboard</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-primary: #0a0a1a;
            --bg-secondary: #12122a;
            --bg-card: #1a1a3a;
            --bg-card-hover: #222250;
            --text-primary: #e8e8f0;
            --text-secondary: #8888aa;
            --text-dim: #555577;
            --accent-purple: #7C3AED;
            --accent-cyan: #06B6D4;
            --accent-pink: #EC4899;
            --accent-green: #10B981;
            --accent-yellow: #F59E0B;
            --accent-red: #EF4444;
            --border: #2a2a4a;
            --glow-purple: rgba(124, 58, 237, 0.3);
            --glow-cyan: rgba(6, 182, 212, 0.3);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }

        /* Header */
        .header {
            padding: 24px 32px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-badge {
            padding: 4px 12px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* Agent Selector */
        .agent-selector {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .agent-selector select {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            cursor: pointer;
        }

        .agent-selector select:focus {
            outline: none;
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px var(--glow-purple);
        }

        /* Main Grid */
        .dashboard {
            padding: 24px 32px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }

        /* Stat Cards */
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            border-color: var(--accent-purple);
            box-shadow: 0 4px 20px var(--glow-purple);
            transform: translateY(-2px);
        }

        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 32px;
            font-weight: 700;
        }

        .stat-value.purple { color: var(--accent-purple); }
        .stat-value.cyan { color: var(--accent-cyan); }
        .stat-value.pink { color: var(--accent-pink); }
        .stat-value.green { color: var(--accent-green); }

        /* Charts Section */
        .chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            grid-column: span 2;
        }

        .chart-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Memory Type Donut */
        .donut-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 32px;
        }

        .donut-svg { width: 180px; height: 180px; }

        .donut-legend {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        /* Importance Heatmap */
        .heatmap {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(28px, 1fr));
            gap: 4px;
        }

        .heat-cell {
            aspect-ratio: 1;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
            position: relative;
        }

        .heat-cell:hover {
            transform: scale(1.3);
            z-index: 10;
        }

        .heat-cell .tooltip {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 11px;
            white-space: nowrap;
            z-index: 100;
            pointer-events: none;
        }

        .heat-cell:hover .tooltip { display: block; }

        /* Memory List */
        .memory-list-card {
            grid-column: span 4;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
        }

        .memory-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .memory-table th {
            text-align: left;
            padding: 12px 16px;
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--border);
        }

        .memory-table td {
            padding: 12px 16px;
            font-size: 13px;
            border-bottom: 1px solid rgba(42, 42, 74, 0.5);
        }

        .memory-table tr:hover td {
            background: var(--bg-card-hover);
        }

        .type-badge {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .type-semantic { background: rgba(124, 58, 237, 0.2); color: var(--accent-purple); }
        .type-episodic { background: rgba(6, 182, 212, 0.2); color: var(--accent-cyan); }
        .type-procedural { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
        .type-short_term { background: rgba(245, 158, 11, 0.2); color: var(--accent-yellow); }

        .importance-bar {
            height: 6px;
            border-radius: 3px;
            background: var(--bg-primary);
            width: 80px;
            overflow: hidden;
        }

        .importance-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }

        /* Loading */
        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }

        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--accent-purple);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Refresh button */
        .refresh-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            transition: all 0.2s;
        }

        .refresh-btn:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 3px var(--glow-cyan);
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
            grid-column: span 4;
        }

        .empty-state-icon { font-size: 48px; margin-bottom: 16px; }

        @media (max-width: 1200px) {
            .dashboard { grid-template-columns: repeat(2, 1fr); }
            .memory-list-card { grid-column: span 2; }
        }

        @media (max-width: 768px) {
            .dashboard { grid-template-columns: 1fr; padding: 16px; }
            .chart-card, .memory-list-card { grid-column: span 1; }
            .header { flex-direction: column; gap: 16px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <div class="logo">🧠 Neurogram</div>
            <div class="header-badge">Memory Dashboard</div>
        </div>
        <div class="agent-selector">
            <select id="agentSelect" onchange="loadAgent(this.value)">
                <option value="">Select an agent...</option>
            </select>
            <button class="refresh-btn" onclick="refresh()">↻ Refresh</button>
        </div>
    </div>

    <div class="dashboard" id="dashboard">
        <div class="loading" id="loadingState">
            <div class="loading-spinner"></div>
            <div>Loading agents...</div>
        </div>
    </div>

    <script>
        let currentAgent = null;
        let agentsData = [];

        async function init() {
            try {
                const res = await fetch('/api/agents');
                const data = await res.json();
                agentsData = data.agents;

                const select = document.getElementById('agentSelect');
                agentsData.forEach(agent => {
                    const opt = document.createElement('option');
                    opt.value = agent.agent_id;
                    opt.textContent = `${agent.agent_name || agent.agent_id} (${agent.total_memories} memories)`;
                    select.appendChild(opt);
                });

                if (agentsData.length > 0) {
                    select.value = agentsData[0].agent_id;
                    loadAgent(agentsData[0].agent_id);
                } else {
                    showEmpty();
                }
            } catch (err) {
                document.getElementById('loadingState').innerHTML =
                    '<div class="empty-state-icon">⚠️</div><div>Could not connect to Neurogram</div>';
            }
        }

        function showEmpty() {
            document.getElementById('dashboard').innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🧠</div>
                    <h3>No agents found</h3>
                    <p style="margin-top: 8px;">Create an agent to see its memory visualization</p>
                    <code style="margin-top: 16px; display: block; color: var(--accent-cyan);">
                        from neurogram import Agent<br>
                        agent = Agent("my_agent")<br>
                        agent.remember("Hello world!")
                    </code>
                </div>
            `;
        }

        async function loadAgent(agentId) {
            if (!agentId) return;
            currentAgent = agentId;

            const agentStats = agentsData.find(a => a.agent_id === agentId);
            if (!agentStats) return;

            const memRes = await fetch(`/api/agents/${agentId}/memories?limit=200`);
            const memData = await memRes.json();
            const memories = memData.memories;

            renderDashboard(agentStats, memories);
        }

        function renderDashboard(stats, memories) {
            const byType = stats.by_type || {};
            const total = stats.total_memories || 0;

            const dashboard = document.getElementById('dashboard');
            dashboard.innerHTML = `
                <!-- Stat Cards -->
                <div class="stat-card">
                    <div class="stat-label">Total Memories</div>
                    <div class="stat-value purple">${total}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Semantic</div>
                    <div class="stat-value cyan">${byType.semantic || 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Episodic</div>
                    <div class="stat-value pink">${byType.episodic || 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Procedural</div>
                    <div class="stat-value green">${byType.procedural || 0}</div>
                </div>

                <!-- Memory Type Breakdown -->
                <div class="chart-card">
                    <div class="chart-title">📊 Memory Type Distribution</div>
                    <div class="donut-container">
                        <svg class="donut-svg" viewBox="0 0 42 42" id="donutChart"></svg>
                        <div class="donut-legend" id="donutLegend"></div>
                    </div>
                </div>

                <!-- Importance Heatmap -->
                <div class="chart-card">
                    <div class="chart-title">🔥 Importance Heatmap</div>
                    <div class="heatmap" id="heatmap"></div>
                </div>

                <!-- Memory List -->
                <div class="memory-list-card">
                    <div class="chart-title">📝 Recent Memories</div>
                    <table class="memory-table">
                        <thead>
                            <tr>
                                <th>Content</th>
                                <th>Type</th>
                                <th>Importance</th>
                                <th>Accessed</th>
                                <th>Created</th>
                            </tr>
                        </thead>
                        <tbody id="memoryTableBody"></tbody>
                    </table>
                </div>
            `;

            renderDonut(byType, total);
            renderHeatmap(memories);
            renderMemoryTable(memories);
        }

        function renderDonut(byType, total) {
            const types = [
                { key: 'semantic', label: 'Semantic', color: '#7C3AED' },
                { key: 'episodic', label: 'Episodic', color: '#06B6D4' },
                { key: 'procedural', label: 'Procedural', color: '#10B981' },
                { key: 'short_term', label: 'Short Term', color: '#F59E0B' },
            ];

            const svg = document.getElementById('donutChart');
            const legend = document.getElementById('donutLegend');

            if (total === 0) {
                svg.innerHTML = `<circle cx="21" cy="21" r="15.9" fill="none" stroke="#2a2a4a" stroke-width="3"/>
                    <text x="21" y="22" text-anchor="middle" fill="#8888aa" font-size="4" font-family="Inter">Empty</text>`;
                return;
            }

            let offset = 0;
            types.forEach(type => {
                const count = byType[type.key] || 0;
                const pct = (count / total) * 100;

                if (pct > 0) {
                    svg.innerHTML += `<circle cx="21" cy="21" r="15.9" fill="none"
                        stroke="${type.color}" stroke-width="3"
                        stroke-dasharray="${pct} ${100 - pct}"
                        stroke-dashoffset="${-offset}"
                        transform="rotate(-90 21 21)"/>`;
                    offset += pct;
                }

                legend.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-dot" style="background: ${type.color}"></div>
                        <span>${type.label}: ${count} (${Math.round(pct)}%)</span>
                    </div>`;
            });

            // Center text
            svg.innerHTML += `<text x="21" y="20" text-anchor="middle" fill="#e8e8f0"
                font-size="6" font-weight="700" font-family="Inter">${total}</text>
                <text x="21" y="24" text-anchor="middle" fill="#8888aa"
                font-size="2.5" font-family="Inter">memories</text>`;
        }

        function renderHeatmap(memories) {
            const container = document.getElementById('heatmap');
            if (memories.length === 0) {
                container.innerHTML = '<div style="color: var(--text-dim); padding: 20px;">No memories yet</div>';
                return;
            }

            memories.forEach(mem => {
                const intensity = mem.importance_score;
                const hue = intensity > 0.6 ? 270 : intensity > 0.3 ? 200 : 220;
                const sat = 70;
                const light = Math.max(15, intensity * 55 + 10);

                const cell = document.createElement('div');
                cell.className = 'heat-cell';
                cell.style.background = `hsl(${hue}, ${sat}%, ${light}%)`;
                cell.innerHTML = `<div class="tooltip">${mem.content.substring(0, 40)}...<br>
                    Importance: ${(intensity * 100).toFixed(0)}%</div>`;
                container.appendChild(cell);
            });
        }

        function renderMemoryTable(memories) {
            const tbody = document.getElementById('memoryTableBody');

            memories.slice(0, 50).forEach(mem => {
                const age = timeAgo(mem.created_at);
                const lastAccess = timeAgo(mem.last_accessed);
                const impColor = mem.importance_score > 0.7 ? '#10B981' :
                    mem.importance_score > 0.4 ? '#F59E0B' : '#EF4444';

                tbody.innerHTML += `
                    <tr>
                        <td style="max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            ${escapeHtml(mem.content)}
                        </td>
                        <td><span class="type-badge type-${mem.memory_type}">${mem.memory_type}</span></td>
                        <td>
                            <div class="importance-bar">
                                <div class="importance-fill" style="width: ${mem.importance_score * 100}%; background: ${impColor};"></div>
                            </div>
                            <span style="font-size: 11px; color: var(--text-dim);">${(mem.importance_score * 100).toFixed(0)}%</span>
                        </td>
                        <td style="color: var(--text-secondary); font-size: 12px;">${mem.access_count}× · ${lastAccess}</td>
                        <td style="color: var(--text-dim); font-size: 12px;">${age}</td>
                    </tr>`;
            });
        }

        function timeAgo(timestamp) {
            const diff = Date.now() / 1000 - timestamp;
            if (diff < 60) return 'just now';
            if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
            return `${Math.floor(diff / 86400)}d ago`;
        }

        function escapeHtml(str) {
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function refresh() {
            if (currentAgent) loadAgent(currentAgent);
        }

        init();
    </script>
</body>
</html>'''


def start_dashboard(
    port: int = 8080,
    host: str = "0.0.0.0",
    storage_path: Optional[str] = None,
):
    """Start the Neurogram dashboard server.

    Args:
        port: Port to serve on.
        host: Host to bind to.
        storage_path: Custom SQLite database path.
    """
    import uvicorn

    app = create_dashboard_app(storage_path)
    print(f"\n  🧠 Neurogram Memory Dashboard")
    print(f"  → http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
