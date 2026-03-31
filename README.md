# yuuagents

A minimal Python agent runtime. Write a one-liner or deploy a persistent daemon — same codebase, two paths.

```
Agent = Persona + Tools + LLM
```

> **Quick links** · [SDK path](#sdk-path-local) · [Service path](#service-path-daemon) · [Flow concept](#the-flow-abstraction) · [Built-in tools](#built-in-tools) · [Config reference](#configuration)

---

## Two Ways to Run

| | SDK Path | Service Path |
|---|---|---|
| **When to use** | Embed in your code, notebooks, pipelines | Long-running tasks, background work, multi-agent |
| **Entry point** | `from yuuagents import LocalAgent` | `yagents up` |
| **Persistence** | No (ephemeral) | Yes (SQLite snapshots) |
| **Docker tools** | No | Yes (optional) |
| **Daemon required** | No | Yes |

---

## Install

**Requirements:** Python 3.14+, and optionally Docker Engine for sandboxed tool execution.

```bash
pip install yuuagents
```

Optional extras:

```bash
pip install 'yuuagents[docker]'   # execute_bash, read_file, edit_file, delete_file
pip install 'yuuagents[web]'      # web_search (requires Tavily API key)
pip install 'yuuagents[all]'      # everything
```

---

## SDK Path (Local)

Run an agent in-process. No daemon, no Docker, no database.

### 30-second quickstart

```python
import asyncio
import yuullm
from yuuagents import run_once

async def main():
    llm = yuullm.YLLMClient(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    )
    result = await run_once("Summarise the Zen of Python.", llm=llm)
    print(result.output_text)

asyncio.run(main())
```

### Stateful agent with streaming

```python
from yuuagents import LocalAgent

agent = LocalAgent.create(llm=llm, system="You are a concise coding assistant.")

run = agent.start("List the files in the current directory.")
async for step in run.step_iter():
    print(f"round {step.rounds}  tokens={step.tokens}")

result = await run.result()
print(result.output_text)
```

### With custom tools

```python
import yuutools as yt
from yuuagents import LocalAgent

@yt.tool(description="Return the current UTC time.")
async def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

agent = LocalAgent.create(
    llm=llm,
    tools=yt.ToolManager([now]),
)
run = agent.start("What time is it?")
print((await run.result()).output_text)
```

---

## Service Path (Daemon)

The daemon manages long-running tasks over a Unix socket, persists snapshots to SQLite, and optionally runs tools inside Docker containers.

### Step 1 — Bootstrap

```bash
yagents install
```

Writes `~/.yagents/config.yaml`, initialises the task database, and pulls the Docker runtime image (if Docker-backed tools are configured).

### Step 2 — Start

```bash
yagents up -d        # background process or systemd user service
```

### Step 3 — Run tasks

```bash
# Submit a task
yagents run --agent main --task "Refactor src/util.py to use pathlib"

# Check status
yagents list
yagents status <task_id>

# Read the output
yagents logs <task_id>

# Send a follow-up message to a running task
yagents input <task_id> "Focus on the read_text calls first."

# Cancel
yagents stop <task_id>
```

### Full CLI reference

| Command | Description |
|---|---|
| `yagents install` | Bootstrap config, directories, database, Docker image |
| `yagents up [-d]` | Start daemon (`-d` = background / systemd) |
| `yagents down` | Stop daemon |
| `yagents run --agent <id> --task "..."` | Submit a task |
| `yagents list` | Human-readable task list |
| `yagents status <task_id>` | JSON status for one task |
| `yagents logs <task_id>` | Conversation history by role |
| `yagents input <task_id> "..."` | Send a message to a running task |
| `yagents stop <task_id>` | Cancel a running task |
| `yagents config` | Show current resolved config |
| `yagents config --overrides FILE` | Merge overrides and hot-reload |
| `yagents config --config FILE` | Replace config and hot-reload |
| `yagents trace ui` | Open the yuutrace observability UI |
| `yagents uninstall` | Remove all installed runtime state |

`yagents run` also accepts `--persona`, `--tools`, `--model`, `--container`, `--image`.

---

## The Flow Abstraction

Everything that executes inside yuuagents is a **Flow** — a generic container that is observable, addressable, and cancellable.

```
Flow
├── stem          append-only event log   (what happened)
├── mailbox       async message queue     (what to do next)
├── children      list[Flow]              (spawned sub-flows)
└── cancel()      propagates recursively  (stop everything)
```

An **Agent** is a specialised Flow that drives the LLM turn loop:

```
Agent (a Flow)
├── AgentConfig   llm + tools + system prompt   (frozen, immutable)
├── messages      conversation history
└── steps()       AsyncGenerator[StepResult]    (call this to run)
```

At each turn, `steps()`:
1. Calls the LLM (streaming)
2. Executes any tool calls (optionally in Docker, optionally deferred to background)
3. Emits a `StepResult` and loops until the model stops

Sub-agents spawned via the `delegate` tool become **child Flows** of the parent — inheriting cancellation and sharing the observable event tree.

### Snapshots

A Flow can be frozen into an `AgentState` at any point:

```python
state = await session.snapshot()   # messages + usage + rounds
# ... persist to disk, restart daemon, restore ...
session.resume(history=state.messages, conversation_id=state.conversation_id)
```

Snapshot-based recovery is configured under `snapshot:` in `config.yaml`.

---

## Built-in Tools

| Tool | Requires | Description |
|---|---|---|
| `sleep` | — | Pause execution |
| `view_image` | — | Decode and display an image |
| `execute_bash` | `docker` extra + Docker | Run shell commands in a container |
| `read_file` | `docker` extra + Docker | Read a file from the container workspace |
| `edit_file` | `docker` extra + Docker | Patch a file in the container workspace |
| `delete_file` | `docker` extra + Docker | Delete a file from the container workspace |
| `web_search` | `web` extra + Tavily API key | Search the web |
| `delegate` | Daemon + delegate capability | Spawn a sub-agent |
| `inspect_background` | Daemon | Inspect a deferred background task |
| `cancel_background` | Daemon | Cancel a background task |
| `input_background` | Daemon | Send input to a background task |
| `defer_background` | Daemon | Move a tool call to background |
| `wait_background` | Daemon | Block until a background task finishes |

---

## Configuration

State lives under `~/.yagents/` by default:

```
~/.yagents/
├── config.yaml         active config
├── tasks.sqlite3       task log and snapshots
├── traces.db           LLM traces (yuutrace)
├── yagents.sock        Unix socket
└── dockers/            per-container working directories
```

Key sections in `config.yaml`:

```yaml
snapshot:
  enabled: false           # write AgentState snapshots after each turn
  restore_on_start: false  # auto-resume incomplete tasks on daemon startup

daemon:
  socket: ~/.yagents/yagents.sock
  log_level: info

docker:
  image: yuuagents-runtime:latest

providers:
  openai-default:
    api_type: openai-chat-completion
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o

agents:
  main:
    description: Default general-purpose agent.
    provider: openai-default
    model: gpt-4o
    persona: "You are a careful, concise assistant."
    tools:
      - sleep
      - view_image
```

Copy `config.example.yaml` and `config.overrides.example.yaml` from the source repo for the full annotated reference.

Config resolution order (highest wins):

1. Bundled package default template
2. `config.overrides.yaml` in the current working directory
3. `config.overrides.yaml` from `--project-dir`
4. `--overrides FILE` flag
5. `--config FILE` flag (replaces the default template entirely)

---

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
  SDK path  ──────▶ │  LocalAgent / run_once()            │
                    │  (in-process, no daemon required)   │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  core/flow.py                       │
                    │  Flow  ◀──── Agent                  │
                    │  (observable · addressable ·        │
                    │   interruptible execution unit)     │
                    └──────────────┬──────────────────────┘
                                   │
  Service   ┌────────────────────┐ │ ┌──────────────────────┐
  path  ───▶│  CLI (yagents)     │─┼─│  Daemon (Starlette)  │
            │  click commands    │ │ │  AgentManager        │
            │  HTTP/Unix socket  │ │ │  DockerManager       │
            └────────────────────┘   └──────────────────────┘
```

**Package dependencies:** `yuubot → yuuagents → {yuullm, yuutools, yuutrace}`

---

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/
uv build
```

Tests marked `@pytest.mark.live` require real external services and are skipped by default.
