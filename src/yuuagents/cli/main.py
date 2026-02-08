"""CLI entry point — ``yagents`` command."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from yuuagents.config import load as load_config
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yuuagents.cli.client import YAgentsClient

_DEFAULT_SOCKET = "~/.local/run/yagents.sock"


def _socket(ctx: click.Context) -> str:
    return str(Path(ctx.obj or _DEFAULT_SOCKET).expanduser())


def _client(ctx: click.Context)->YAgentsClient:
    from yuuagents.cli.client import YAgentsClient

    return YAgentsClient(_socket(ctx))


@click.group()
@click.option(
    "--socket",
    default=None,
    help="Path to daemon Unix socket (default: from config)",
)
@click.pass_context
def cli(ctx: click.Context, socket: str | None) -> None:
    """yuuagents — minimal agent framework."""
    if socket:
        ctx.obj = socket
    else:
        cfg = load_config()
        ctx.obj = str(cfg.socket_path)


# ── Daemon management ──


@cli.command()
@click.option("--config", "config_path", default=None, help="Config file path")
@click.pass_context
def start(ctx: click.Context, config_path: str | None) -> None:
    """Start the yagents daemon (foreground)."""
    from yuuagents.daemon.server import serve

    cfg = load_config(config_path)
    click.echo(f"Starting daemon on {cfg.socket_path} ...")
    asyncio.run(serve(cfg))


@cli.command()
@click.pass_context
def stop(ctx: click.Context) -> None:
    """Stop the running daemon."""
    import signal

    # Send SIGTERM to the daemon via the socket health check
    c = _client(ctx)
    try:
        c.health()
        click.echo("Sending shutdown signal...")
        # The daemon handles SIGTERM; for CLI we just confirm it's alive
        click.echo("Use systemctl stop yagents or kill the daemon process.")
    except Exception:
        click.echo("Daemon is not running.")
    finally:
        c.close()


# ── Agent management ──


@cli.command()
@click.option("--persona", required=True, help="Persona template name or full system prompt")
@click.option("--task", required=True, help="Task description")
@click.option("--tools", default=None, help="Comma-separated tool names")
@click.option("--skills", "skill_names", default=None, help="Comma-separated skill names")
@click.option("--model", default="", help="LLM model override")
@click.option("--container", default="", help="Existing Docker container ID to use")
@click.option("--image", default="", help="Docker image to create a new container from")
@click.pass_context
def run(
    ctx: click.Context,
    persona: str,
    task: str,
    tools: str | None,
    skill_names: str | None,
    model: str,
    container: str,
    image: str,
) -> None:
    """Submit a new agent task."""
    c = _client(ctx)
    payload = {
        "persona": persona,
        "task": task,
        "model": model,
        "container": container,
        "image": image,
    }
    if tools:
        payload["tools"] = [t.strip() for t in tools.split(",")]
    if skill_names:
        payload["skills"] = [s.strip() for s in skill_names.split(",")]
    try:
        result = c.submit(payload)
        click.echo(f"Agent started: {result['agent_id']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command("list")
@click.pass_context
def list_agents(ctx: click.Context) -> None:
    """List all agents."""
    c = _client(ctx)
    try:
        agents = c.list_agents()
        if not agents:
            click.echo("No agents.")
            return
        for a in agents:
            status = a.get("status", "?")
            aid = a.get("agent_id", "?")
            task = a.get("task", "")[:60]
            steps = a.get("steps", 0)
            cost = a.get("total_cost_usd", 0)
            click.echo(f"  [{status:>8}] {aid}  steps={steps}  ${cost:.4f}  {task}")
    finally:
        c.close()


@cli.command()
@click.argument("agent_id")
@click.pass_context
def status(ctx: click.Context, agent_id: str) -> None:
    """Show agent status."""
    c = _client(ctx)
    try:
        info = c.status(agent_id)
        click.echo(json.dumps(info, indent=2, ensure_ascii=False))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command()
@click.argument("agent_id")
@click.pass_context
def logs(ctx: click.Context, agent_id: str) -> None:
    """Show agent conversation history."""
    c = _client(ctx)
    try:
        hist = c.history(agent_id)
        for msg in hist:
            role = msg.get("role", "?")
            items = msg.get("items", [])
            click.echo(f"\n--- {role} ---")
            for item in items:
                if isinstance(item, str):
                    click.echo(item)
                else:
                    click.echo(json.dumps(item, indent=2, ensure_ascii=False))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command("stop-agent")
@click.argument("agent_id")
@click.pass_context
def stop_agent(ctx: click.Context, agent_id: str) -> None:
    """Cancel a running agent."""
    c = _client(ctx)
    try:
        c.cancel(agent_id)
        click.echo(f"Agent {agent_id} cancelled.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command()
@click.argument("agent_id")
@click.argument("message")
@click.pass_context
def input(ctx: click.Context, agent_id: str, message: str) -> None:
    """Reply to an agent's user_input request."""
    c = _client(ctx)
    try:
        c.respond(agent_id, message)
        click.echo("Input sent.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


# ── Skills ──


@cli.group()
def skills() -> None:
    """Manage Agent Skills."""


@skills.command("list")
@click.pass_context
def skills_list(ctx: click.Context) -> None:
    """List discovered skills."""
    c = _client(ctx)
    try:
        sk = c.skills()
        if not sk:
            click.echo("No skills found.")
            return
        for s in sk:
            click.echo(f"  {s['name']:20s}  {s.get('description', '')}")
    finally:
        c.close()


@skills.command("scan")
@click.pass_context
def skills_scan(ctx: click.Context) -> None:
    """Rescan skill directories."""
    c = _client(ctx)
    try:
        sk = c.scan_skills()
        click.echo(f"Found {len(sk)} skill(s).")
        for s in sk:
            click.echo(f"  {s['name']:20s}  {s.get('description', '')}")
    finally:
        c.close()
