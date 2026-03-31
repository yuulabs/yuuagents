# yuuagents

`yuuagents` is a Python agent runtime with two supported paths:

- SDK path: import `yuuagents`, build a local agent, and run it in-process.
- Service path: install the package, run `yagents install`, then start the daemon with `yagents up`.

The base package includes:

- `yagents` CLI
- SDK helpers such as `run_once(...)`, `LocalAgent`, `Session`, `AgentConfig`, and `AgentContext`
- built-in tool registry and capability guards
- snapshot-based task persistence
- optional Docker-backed and web-backed capabilities

## Install

Requirements:

- Python 3.14 or newer
- `pip` or `uv`
- For service mode, a working `ytrace` executable on `PATH`
- For Docker-backed tools, Docker Engine must be reachable

Install the base package:

```bash
pip install yuuagents
```

Optional extras:

```bash
pip install 'yuuagents[docker]'
pip install 'yuuagents[web]'
pip install 'yuuagents[docker,web]'
pip install 'yuuagents[all]'
```

Extra meanings:

- `docker`: enables Docker-backed `execute_bash`, `read_file`, `edit_file`, and `delete_file`
- `web`: enables `web_search`
- `all`: installs both optional capability sets

## SDK Path

Use the SDK path when you want to run an agent locally inside your own process.

```python
import asyncio

import yuullm
from yuuagents import LocalAgent, run_once


async def main() -> None:
    llm = yuullm.YLLMClient(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    )

    result = await run_once(
        "Say hello and report that the runtime is working.",
        llm=llm,
        system="You are a concise coding assistant.",
    )

    print(result.output_text)

    agent = LocalAgent.create(llm=llm)
    run = agent.start("Inspect the current workspace")
    async for step in run.step_iter():
        print(step)
    print((await run.result()).output_text)


asyncio.run(main())
```

SDK mode does not require the daemon, Docker, or a local SQLite task database.

## Service Path

Service mode is the CLI-and-daemon workflow.

### `yagents install`

`yagents install` writes the installed config to `~/.yagents/config.yaml`, creates the runtime directories, initializes the task database, and prepares the configured Docker image when Docker-gated tools are present.

Current config resolution order:

1. Start from the bundled default template shipped inside the package.
2. Merge `config.overrides.yaml` from the current working directory if present.
3. Merge `config.overrides.yaml` from `--project-dir` if provided.
4. Merge `--overrides` if provided.
5. If `--config` is provided, it fully replaces the default template and requires confirmation before continuing.

Important details:

- The packaged default template is used even when you do not have a repository checkout.
- On the default install path, `docker.image` is pinned to the versioned runtime image tag for the installed package.
- If no Docker-gated tools are configured, Docker image setup is skipped.
- If Docker-gated tools are configured but Docker is unavailable, install continues with a warning and those tools will fail later at runtime.

### `yagents up`

`yagents up` starts the daemon on the Unix socket from the resolved config.

Behavior to know:

- `--config` loads a specific config file.
- Without `--config`, the CLI prefers `cwd/config.yaml`, then `~/.yagents/config.yaml`, then the bundled default template.
- `config.overrides.yaml` in the current directory is merged on top when present.
- `--dot-env` loads environment variables from a `.env` file; otherwise the command auto-discovers `.env` while walking up from the current directory toward `$HOME`.
- If the configured database path changed since the last install, `up` refuses to start rather than orphan the old SQLite file.
- `-d/--daemon` prefers a systemd user service when possible; otherwise it falls back to a background subprocess.

### Other CLI commands

| Command | What it does |
|---|---|
| `yagents run --agent main --task "..."` | Submit a task to the daemon |
| `yagents list` | Show a human-readable task list |
| `yagents status <task_id>` | Return JSON status for one task |
| `yagents logs <task_id>` | Print conversation history by role |
| `yagents input <task_id> "..."` | Send a user message to a running task |
| `yagents stop <task_id>` | Cancel a running task |
| `yagents config` | Show the current installed config |
| `yagents config --overrides FILE` | Update config and try hot reload |
| `yagents config --config FILE` | Replace config and try hot reload |
| `yagents trace ui` | Open the yuutrace UI |
| `yagents down` | Stop the daemon and try to stop the systemd user service |
| `yagents uninstall` | Remove the installed runtime state |

`yagents run` accepts `--persona`, `--tools`, `--model`, `--container`, and `--image`. `--container` and `--image` are mutually exclusive.

## Built-in Tools

Available tool names in the package:

- `sleep`
- `view_image`
- `delegate`
- `inspect_background`
- `cancel_background`
- `input_background`
- `defer_background`
- `wait_background`
- `execute_bash`
- `read_file`
- `edit_file`
- `delete_file`
- `web_search`

Capability rules:

- `sleep` and `view_image` are always available.
- `delegate` and the background control tools require delegate capability from the daemon runtime.
- Docker-backed tools require Docker capability and the `docker` extra.
- `web_search` requires web capability, the `web` extra, and a Tavily API key.

## Configuration

The repository contains source-side reference templates:

- `config.example.yaml`
- `config.overrides.example.yaml`

Those files are reference material for source checkouts. Packaged installs use the bundled default template shipped inside `yuuagents` itself.

Persistent state lives under `~/.yagents` by default:

- `~/.yagents/config.yaml`
- `~/.yagents/tasks.sqlite3`
- `~/.yagents/traces.db`
- `~/.yagents/yagents.sock`
- `~/.yagents/dockers/`

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv build
```
