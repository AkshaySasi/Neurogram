"""Neurogram CLI — command line interface for Neurogram.

Provides basic management commands for Neurogram agents.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="neurogram",
        description="Neurogram — Memory infrastructure for AI agents",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── info command ───────────────────────────────────────────────
    info_parser = subparsers.add_parser("info", help="Show system info")

    # ── agents command ─────────────────────────────────────────────
    agents_parser = subparsers.add_parser("agents", help="List all agents")
    agents_parser.add_argument(
        "--db", type=str, default=None, help="Path to database file"
    )

    # ── stats command ──────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        "stats", help="Show agent memory statistics"
    )
    stats_parser.add_argument("agent", type=str, help="Agent name")
    stats_parser.add_argument(
        "--db", type=str, default=None, help="Path to database file"
    )

    # ── server command ─────────────────────────────────────────────
    server_parser = subparsers.add_parser(
        "server", help="Start the Neurogram memory server"
    )
    server_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Server port"
    )
    server_parser.add_argument(
        "--db", type=str, default=None, help="Path to database file"
    )

    args = parser.parse_args()

    if args.version:
        from neurogram import __version__
        print(f"neurogram {__version__}")
        return

    if args.command == "info":
        _cmd_info()
    elif args.command == "agents":
        _cmd_agents(args.db)
    elif args.command == "stats":
        _cmd_stats(args.agent, args.db)
    elif args.command == "server":
        _cmd_server(args.host, args.port, args.db)
    else:
        parser.print_help()


def _cmd_info():
    """Show system information."""
    from neurogram import __version__
    from neurogram.embedding_engine import get_default_engine

    engine = get_default_engine()
    print(f"  Neurogram v{__version__}")
    print(f"  🧠 Memory infrastructure for AI agents")
    print(f"  ")
    print(f"  Embedding engine: {type(engine).__name__}")
    print(f"  Embedding dimensions: {engine.dimensions}")
    print()

    # Check optional dependencies
    deps = {
        "sentence-transformers": False,
        "openai": False,
        "fastapi": False,
        "uvicorn": False,
    }
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            deps[dep] = True
        except ImportError:
            pass

    print("  Optional dependencies:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"    {status} {dep}")


def _cmd_agents(db_path):
    """List all agents."""
    from neurogram import Neurogram

    with Neurogram(storage_path=db_path) as brain:
        agents = brain.list_agents()
        if not agents:
            print("  No agents found.")
            return

        print(f"\n  Found {len(agents)} agent(s):\n")
        for agent in agents:
            print(f"  🤖 {agent.name} ({agent.agent_id})")
            if agent.description:
                print(f"     {agent.description}")
            print()


def _cmd_stats(agent_name, db_path):
    """Show agent memory statistics."""
    from neurogram import Agent

    agent = Agent(agent_name, storage_path=db_path)
    stats = agent.stats()

    print(f"\n  Agent: {stats['agent_name']}")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Embedding engine: {stats['embedding_engine']}")
    print(f"  Dimensions: {stats['embedding_dimensions']}")
    print()
    print("  Memory breakdown:")
    for mtype, count in stats["by_type"].items():
        print(f"    {mtype}: {count}")

    agent.close()


def _cmd_server(host, port, db_path):
    """Start the memory server."""
    try:
        import uvicorn
    except ImportError:
        print("  Error: uvicorn is required for the server.")
        print("  Install it with: pip install neurogram[server]")
        sys.exit(1)

    print(f"  🧠 Starting Neurogram Memory Server on {host}:{port}")
    uvicorn.run(
        "neurogram.server.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
