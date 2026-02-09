"""CLI entry point — ``yagents`` command."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import click
import yaml

from yuuagents.config import (
    DEFAULT_CONFIG_PATH,
    YAGENTS_HOME,
    _PROJECT_CONFIG_NAME,
    _PROJECT_OVERRIDES_NAME,
    _deep_merge,
    find_project_root,
    load as load_config,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yuuagents.cli.client import YAgentsClient


def _socket(ctx: click.Context) -> str:
    return str(Path(ctx.obj).expanduser())


def _client(ctx: click.Context) -> YAgentsClient:
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
    """yagents — minimal agent framework."""
    if socket:
        ctx.obj = socket
    else:
        cfg = load_config()
        ctx.obj = str(cfg.socket_path)


# ── Setup ──


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml (fully replaces defaults — requires confirmation)",
)
@click.option(
    "--overrides",
    "overrides_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.overrides.yaml (deep-merged on top)",
)
@click.option(
    "--project-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Project root containing config.example.yaml (default: auto-detect)",
)
@click.pass_context
def setup(
    ctx: click.Context,
    config_path: str | None,
    overrides_path: str | None,
    project_dir: str | None,
) -> None:
    """One-time setup: install config, create directories, pull Docker, register service.

    \b
    Config resolution order:
      1. Start with defaults (config.example.yaml from project root).
      2. If --overrides is given, deep-merge it on top.
      3. If --config is given, it FULLY REPLACES the result (with confirmation).

    \b
    This command will:
      1. Write merged config to ~/.yagents/config.yaml.
      2. Create required directories (~/.yagents/skills, ~/.yagents/dockers).
      3. Install Docker if needed (prompts for sudo).
      4. Pull the Docker image specified in the config.
      5. Register yagents as a systemd user service.
    """
    # -- Step 0: resolve config sources --
    if config_path:
        # User provided a full config — requires confirmation
        click.echo(f"Config file:   {config_path}")
        click.echo("WARNING: --config fully replaces the default configuration.")
        if not click.confirm("Are you sure you want to proceed?"):
            click.echo("Aborted.")
            sys.exit(0)

        user_config_p = Path(config_path)
        user_data = yaml.safe_load(user_config_p.read_text(encoding="utf-8")) or {}

        # Apply overrides on top if given
        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            user_data = _deep_merge(user_data, over_data)
            click.echo(f"Overrides:     {overrides_path}")

        merged_data = user_data
    else:
        # Auto-detect project root for config.example.yaml
        root: Path | None
        if project_dir:
            root = Path(project_dir)
        else:
            root = find_project_root()

        if root is None or not (root / _PROJECT_CONFIG_NAME).exists():
            click.echo(
                f"Error: cannot find {_PROJECT_CONFIG_NAME}. "
                "Run this command from the project directory or pass --project-dir.",
                err=True,
            )
            sys.exit(1)

        base_path = root / _PROJECT_CONFIG_NAME
        click.echo(f"Project root:  {root}")
        click.echo(f"Base config:   {base_path}")

        base_data = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

        # Check for project-level overrides
        proj_overrides = root / _PROJECT_OVERRIDES_NAME
        if proj_overrides.exists():
            over_data = yaml.safe_load(proj_overrides.read_text(encoding="utf-8")) or {}
            base_data = _deep_merge(base_data, over_data)
            click.echo(f"Overrides:     {proj_overrides}")
        else:
            click.echo(f"Overrides:     (none — {_PROJECT_OVERRIDES_NAME} not found)")

        # Apply CLI --overrides on top
        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            base_data = _deep_merge(base_data, over_data)
            click.echo(f"CLI overrides: {overrides_path}")

        merged_data = base_data

    # -- Step 1: write config --
    click.echo()
    click.echo("[1/4] Installing configuration ...")

    YAGENTS_HOME.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(
        yaml.dump(merged_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    click.echo(f"  -> {DEFAULT_CONFIG_PATH}")

    # Load the config for subsequent steps
    cfg = load_config()

    # Validate
    errors = cfg.validate()
    if errors:
        click.echo("  WARNING: configuration has validation errors:", err=True)
        for err in errors:
            click.echo(f"    - {err}", err=True)

    # -- Step 2: create directories --
    click.echo()
    click.echo("[2/4] Creating directories ...")

    dirs_to_create = [
        YAGENTS_HOME / "skills",
        YAGENTS_HOME / "dockers",
    ]
    for sp in cfg.skills.paths:
        dirs_to_create.append(Path(sp).expanduser())

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        click.echo(f"  -> {d}")

    # -- Step 3: Docker --
    click.echo()
    click.echo("[3/4] Setting up Docker ...")

    image = cfg.docker.image
    click.echo(f"  Image: {image}")

    if not _docker_available():
        click.echo(
            "  WARNING: Docker is not installed or not running.\n"
            "  yagents requires Docker to function. Please install Docker:\n"
            "    https://docs.docker.com/engine/install/\n"
            "  You may need to run: sudo apt install docker.io && sudo usermod -aG docker $USER\n"
            f"  Then run `docker pull {image}` manually.",
            err=True,
        )
    else:
        if _image_exists(image):
            click.echo(f"  Image {image} already available locally.")
        else:
            click.echo(f"  Pulling {image} (this may take a minute) ...")
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=False,
            )
            if result.returncode != 0:
                click.echo(
                    f"  WARNING: Failed to pull {image}. "
                    f"You can retry with `docker pull {image}`.",
                    err=True,
                )
            else:
                click.echo(f"  Image {image} pulled successfully.")

    # -- Step 4: systemd service --
    click.echo()
    click.echo("[4/4] Registering systemd user service ...")

    try:
        from yuuagents.cli.service import install as install_service

        unit_path = install_service()
        click.echo(f"  Service installed: {unit_path}")
        click.echo("  Service enabled and started.")
    except FileNotFoundError as e:
        click.echo(f"  WARNING: {e}", err=True)
        click.echo("  You can start the daemon manually: yagents start", err=True)
    except RuntimeError as e:
        click.echo(f"  WARNING: {e}", err=True)
        click.echo("  You can start the daemon manually: yagents start", err=True)

    # -- Done --
    click.echo()
    click.echo("Setup complete!")
    click.echo()

    # Show next steps based on first provider
    if cfg.providers:
        first_provider = next(iter(cfg.providers.values()))
        click.echo("Next steps:")
        click.echo(
            f"  1. Set your LLM API key:  export {first_provider.api_key_env}=sk-..."
        )
        click.echo(
            '  2. Run an agent:          yagents run --agent main --task "hello world"'
        )
    else:
        click.echo("Next steps:")
        click.echo("  1. Configure at least one provider in ~/.yagents/config.yaml")
        click.echo("  2. Set your LLM API key")
        click.echo(
            '  3. Run an agent:          yagents run --agent main --task "hello world"'
        )


def _docker_available() -> bool:
    """Check if Docker CLI is available and daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError, subprocess.TimeoutExpired:
        return False


def _image_exists(image: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError, subprocess.TimeoutExpired:
        return False


# ── Configuration management ──


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml (fully replaces current config)",
)
@click.option(
    "--overrides",
    "overrides_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.overrides.yaml (deep-merged on top)",
)
@click.pass_context
def config(
    ctx: click.Context,
    config_path: str | None,
    overrides_path: str | None,
) -> None:
    """View or update configuration (hot-reload without daemon restart).

    \b
    Usage:
      yagents config                    # Print current config (from memory)
      yagents config --overrides FILE   # Update config with overrides
      yagents config --config FILE      # Replace config entirely

    \b
    This command updates both disk (~/.yagents/config.yaml) and the running
    daemon's in-memory configuration without requiring a restart.
    """
    # No arguments: print current config from daemon
    if not config_path and not overrides_path:
        try:
            cfg = load_config()
            config_dict = {
                "daemon": {
                    "socket": cfg.daemon.socket,
                    "log_level": cfg.daemon.log_level,
                },
                "docker": {
                    "image": cfg.docker.image,
                },
                "skills": {
                    "paths": cfg.skills.paths,
                },
                "tavily": {
                    "api_key_env": cfg.tavily.api_key_env,
                },
                "providers": {
                    name: {
                        "kind": p.kind,
                        "api_key_env": p.api_key_env,
                        "default_model": p.default_model,
                        "base_url": p.base_url,
                        "organization": p.organization,
                    }
                    for name, p in cfg.providers.items()
                },
                "agents": {
                    name: {
                        "provider": a.provider,
                        "model": a.model,
                        "persona": a.persona,
                        "tools": a.tools,
                        "skills": a.skills,
                    }
                    for name, a in cfg.agents.items()
                },
            }
            click.echo(
                yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
            )
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
        return

    # Update config: similar logic to setup command
    if config_path:
        click.echo(f"Config file:   {config_path}")
        click.echo("WARNING: --config fully replaces the current configuration.")
        if not click.confirm("Are you sure you want to proceed?"):
            click.echo("Aborted.")
            sys.exit(0)

        user_config_p = Path(config_path)
        user_data = yaml.safe_load(user_config_p.read_text(encoding="utf-8")) or {}

        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            user_data = _deep_merge(user_data, over_data)
            click.echo(f"Overrides:     {overrides_path}")

        merged_data = user_data
    else:
        # Load current config and apply overrides
        if not DEFAULT_CONFIG_PATH.exists():
            click.echo(
                f"Error: {DEFAULT_CONFIG_PATH} not found. Run `yagents setup` first.",
                err=True,
            )
            sys.exit(1)

        base_data = (
            yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        )
        click.echo(f"Base config:   {DEFAULT_CONFIG_PATH}")

        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            base_data = _deep_merge(base_data, over_data)
            click.echo(f"Overrides:     {overrides_path}")

        merged_data = base_data

    # Write to disk
    click.echo()
    click.echo("Updating configuration ...")
    DEFAULT_CONFIG_PATH.write_text(
        yaml.dump(merged_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    click.echo(f"  -> Written to {DEFAULT_CONFIG_PATH}")

    # Validate
    cfg = load_config()
    errors = cfg.validate()
    if errors:
        click.echo("  WARNING: configuration has validation errors:", err=True)
        for err in errors:
            click.echo(f"    - {err}", err=True)

    # Hot-reload daemon if running
    click.echo()
    click.echo("Reloading daemon configuration ...")
    c = _client(ctx)
    try:
        c.health()
        result = c.reload_config()
        if result.get("ok"):
            click.echo("  -> Daemon configuration reloaded successfully")
        else:
            click.echo(f"  WARNING: {result.get('error', 'Unknown error')}", err=True)
    except Exception as e:
        click.echo(
            f"  WARNING: Could not reload daemon (is it running?): {e}", err=True
        )
        click.echo(
            "  The config file has been updated, but you may need to restart the daemon.",
            err=True,
        )
    finally:
        c.close()

    click.echo()
    click.echo("Configuration update complete!")


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
    c = _client(ctx)
    try:
        c.health()
        click.echo("Sending shutdown signal...")
        click.echo("Use `yagents down` to stop and unregister the service,")
        click.echo("or `systemctl --user stop yagents` to stop the daemon.")
    except Exception:
        click.echo("Daemon is not running.")
    finally:
        c.close()


@cli.command()
def down() -> None:
    """Stop the daemon and unregister the systemd service."""
    from yuuagents.cli.service import uninstall

    click.echo("Stopping yagents service ...")
    try:
        uninstall()
        click.echo("Service stopped and unregistered.")
    except RuntimeError as e:
        click.echo(f"WARNING: {e}", err=True)
        click.echo("You may need to stop the daemon manually.", err=True)


# ── Agent management ──


@cli.command()
@click.option(
    "--agent",
    "agent_name",
    default="main",
    help="Agent config name (default: main)",
)
@click.option(
    "--persona",
    default="",
    help="Override the system prompt (persona) from agent config",
)
@click.option("--task", required=True, help="Task description")
@click.option("--tools", default=None, help="Comma-separated tool names")
@click.option(
    "--skills", "skill_names", default=None, help="Comma-separated skill names"
)
@click.option("--model", default="", help="LLM model override")
@click.option("--container", default="", help="Existing Docker container ID to use")
@click.option("--image", default="", help="Docker image to create a new container from")
@click.pass_context
def run(
    ctx: click.Context,
    agent_name: str,
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
    payload: dict[str, object] = {
        "agent": agent_name,
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
