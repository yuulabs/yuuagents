# yuuagents

`yuuagents` is an agent runtime with two explicit modes:

- SDK mode: local agent development with no daemon, database, Docker, or config file.
- Daemon / service mode: long-running service with CLI, persistence, HTTP API, and optional Docker / web tooling.

## Install

```bash
# Minimal SDK install
pip install yuuagents

# Service mode
pip install 'yuuagents[daemon]'

# Service mode with Docker and web tooling
pip install 'yuuagents[daemon,docker,web]'

# Everything
pip install 'yuuagents[all]'
```

`yuutrace` is part of the base install. Docker, Tavily web search, CLI, YAML config, and persistence are optional extras.

## SDK Quick Start

This is the main path if you only want to build an agent locally.

```python
import asyncio

import yuullm
import yuutools as yt

from yuuagents import run_once


@yt.tool(description="Return the current working directory", params={})
async def current_directory() -> str:
    import os

    return os.getcwd()


async def main() -> None:
    llm = yuullm.YLLMClient(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    )

    result = await run_once(
        "Say hello, call current_directory, then report the directory.",
        llm=llm,
        tools=[current_directory],
        system="You are a concise coding assistant.",
    )

    print(result.output_text)
    print(result.steps)


asyncio.run(main())
```

这个示例只依赖：

- 一个 `yuullm.YLLMClient`
- 零个或多个 `yuutools` 工具
- `run_once(...)` 或 `LocalAgent`

不需要：

- daemon
- Docker
- 数据库
- YAML 配置文件
- `config.example.yaml`

如果你要逐步消费执行结果，用 `LocalAgent`：

```python
from yuuagents import LocalAgent

agent = LocalAgent.create(llm=llm, tools=[current_directory])
run = agent.start("Inspect the workspace")

async for step in run.step_iter():
    print(step)

result = await run.result()
print(result.output_text)
```

可运行示例见 [`examples/sdk_quickstart.py`](./examples/sdk_quickstart.py)。

## Daemon / Service Mode

这条路径面向把 `yuuagents` 当服务运行的场景。这里才需要 CLI、配置文件、持久化，以及可选的 Docker / web 扩展。

### 安装

```bash
# 最小服务模式
pip install 'yuuagents[daemon]'

# 常见完整服务模式
pip install 'yuuagents[daemon,docker,web]'
```

### 配置

`config.example.yaml` 只用于 service mode，且不会随包分发。请从仓库复制一份，再按宿主机环境修改：

```bash
cp config.example.yaml config.yaml
$EDITOR config.yaml
```

默认 traces 数据库路径是当前项目下的 `./.yagents/traces.db`。service mode 的任务数据库默认仍写到 `~/.yagents/tasks.sqlite3`。

### CLI 生命周期

```bash
yagents install
yagents up -d
yagents run --agent main --task "Write a Python function"
yagents list
yagents status <task-id>
yagents logs <task-id>
yagents stop <task-id>
yagents down
```

如果你想从 Python 里做 service bootstrap，用 `yuuagents.init.setup(...)`。它是 service-mode helper，不是 SDK Quick Start。

## Tool Layers

- Core / local-safe tools: `sleep`, `update_todo`, `view_image`, and your own custom tools
- Docker tools: `execute_bash`, `read_file`, `edit_file`, `delete_file`
- Web tools: `web_search`
- Service / delegate tools: `delegate`, `inspect_background`, `cancel_background`, `input_background`, `defer_background`, `wait_background`

Docker 和 web 工具只有在对应 capability 和 extras 存在时才可用。

## Examples

- [`examples/sdk_quickstart.py`](./examples/sdk_quickstart.py): pure SDK path
- [`examples/simple_start.py`](./examples/simple_start.py): service-mode onboarding without `rich`
- [`examples/getting_started.py`](./examples/getting_started.py): service-mode onboarding with `rich`
- [`examples/README.md`](./examples/README.md): example index

## Core Types

- `AgentConfig`: immutable agent configuration
- `AgentContext`: minimal runtime context
- `Session`: host-facing runtime session
- `LocalAgent`: SDK-first local wrapper
- `run_once(...)`: one-shot SDK helper
- `AgentInput`: structured runtime input model

## Development

```bash
uv sync --extra all --group examples
cd yuuagents && uv run pytest
```
