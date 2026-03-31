# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is yuuagents

A minimal agent framework (`Agent = Persona + Tools + LLM`) packaged as a Linux service. It glues together yuutools, yuullm, and yuutrace into a CLI + daemon architecture communicating over a Unix domain socket.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_integration.py
uv run pytest tests/test_integration.py::test_name

# Lint
uv run ruff check src/
uv run ruff format src/

# Type check
uv run mypy src/

# Start daemon / submit task / stop
uv run yagents up -d
uv run yagents run --agent main --task "..."
uv run yagents down
```

## Architecture

```
CLI (click) ──HTTP/Unix socket──▶ Daemon (Starlette/uvicorn)
                                    ├── AgentManager (lifecycle)
                                    ├── REST API (/api/agents/...)
                                    └── DockerManager (containers)
                                            │
                                    AgentPool (pool.py)
                                    ├── run() / spawn() / stop()
                                    ├── inspect / cancel / defer / wait
                                    └── persistence (snapshots)
                                            │
                                    Agent Runtime
                                    ├── core/flow.py (Flow + Agent)
                                    └── Tools (DI via yuutools)
```

**Key modules:**
- `agent.py` — `AgentConfig` (frozen attrs) — immutable agent configuration
- `core/flow.py` — `Flow` (observable, addressable, interruptible execution unit) + `Agent` (composes Flow with LLM behaviour). Everything that runs is a Flow: LLM, tool, bash, sub-agent. A Flow has stem (append-only event log), mailbox (async queue), and cancel.
- `runtime_session.py` — `Session` — thin wrapper over Flow/Agent
- `pool.py` — `AgentPool` — manages running sessions; `run()` launches tasks, `spawn()` creates child agents via a `session_builder` callable; also hosts background-control methods (inspect/cancel/defer/send_input/wait). Single pool shared by SDK and daemon.
- `context.py` — `AgentContext` for dependency injection into tools
- `persistence.py` — SQLite task log; snapshots `AgentState` after each turn; supports restore-on-start
- `tools/` — Builtins (bash, file ops, web_search, delegate, etc.) using `yuutools` DI
- `daemon/` — Server, API routes, AgentManager, DockerManager
- `cli/` — Click commands as thin HTTP client
- `design/` — **Source of truth** for architecture decisions

## Patterns

- **attrs `@define`** for mutable classes, `@define(frozen=True)` for configs
- **`msgspec.Struct`** for serializable DTOs (API types in `types.py`)
- **Tool DI**: tools declare dependencies via `yt.depends(lambda ctx: ctx.field)`
- **All I/O is async** (LLM, Docker, DB, HTTP)
- **Composition over inheritance** throughout; protocol-based interfaces
- **Config**: YAML files parsed in `config.py`; see `config.example.yaml`
- **System prompt**: built as a plain string by the caller, passed to `AgentConfig.system`

## Design Docs

Read `design/` before making architectural changes:
- `design/persistent.md` — Snapshot-based task persistence (AgentState snapshots to SQLite)
