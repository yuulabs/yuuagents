# yuuagents

`yuuagents` is an agent runtime with one base installation and two runtime paths:

- SDK path: `pip install yuuagents`, then `import yuuagents` and run agents locally.
- Daemon path: still starts from `pip install yuuagents`; when you additionally run `yagents install` with a YAML config, it writes local runtime state and brings the daemon/CLI workflow online.

## Install

```bash
# SDK, local runs, CLI, and service bootstrap
pip install yuuagents

# Add Docker-backed bash/file tools
pip install 'yuuagents[docker]'

# Add web_search
pip install 'yuuagents[web]'

# Add both optional capabilities
pip install 'yuuagents[docker,web]'

# Same as above
pip install 'yuuagents[all]'
```

安装选择可以直接按用途理解：

- `yuuagents`: SDK、本地运行、`yagents` CLI、service bootstrap
- `yuuagents[docker]`: 额外提供 Docker-backed `execute_bash` / `read_file` / `edit_file` / `delete_file`
- `yuuagents[web]`: 额外提供 `web_search`
- `yuuagents[docker,web]` 或 `yuuagents[all]`: 同时启用这两类可选能力

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

- 启用 daemon
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

这条路径面向把 `yuuagents` 当服务运行的场景。CLI 已经随基础安装提供；这里额外做的是准备配置、执行 `yagents install`、启动 daemon，以及按需开启 Docker / web capability。

### 安装

```bash
# CLI and service mode
pip install yuuagents

# Add Docker bash/file tools when your agents need them
pip install 'yuuagents[docker]'

# Add both Docker tools and web_search
pip install 'yuuagents[docker,web]'
```

`pip install yuuagents` 已经足够启动 service mode。若 agent 需要 `execute_bash` / `read_file` / `edit_file` / `delete_file`，再安装 `yuuagents[docker]` 并确保 Docker Engine 可达；若 agent 需要 `web_search`，再安装 `yuuagents[web]` 并提供 Tavily key。

### 配置

`config.example.yaml` 只用于 service mode，且不会随包分发。要按下面流程操作，你需要有一份仓库副本。`yagents install` 会优先读取当前目录的 `./config.yaml`，找不到时才会回退到仓库里的 `config.example.yaml`。

从仓库模板开始最稳妥：

```bash
cp /path/to/yuuagents/config.example.yaml ./config.yaml
$EDITOR config.yaml
```

如果你已经在仓库根目录，也可以直接用 `cp config.example.yaml ./config.yaml`。

参考配置默认把 traces 数据库写到 `~/.yagents/traces.db`。service mode 的任务数据库默认仍写到 `~/.yagents/tasks.sqlite3`。

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

如果你想从 Python 里做 service bootstrap，用 `yuuagents.init.setup(...)`。它是 daemon/bootstrap helper，不是 SDK Quick Start。
如果你想用交互式引导，直接运行 [`examples/simple_start.py`](./examples/simple_start.py) 或 [`examples/getting_started.py`](./examples/getting_started.py)。这两个脚本会自动定位仓库根目录并调用 `yagents install --project-dir ...`，不会因为当前工作目录不对而找不到模板配置。

## Tool Layers

- Core builtin tools: `sleep`, `update_todo`, `view_image`
- Daemon-gated builtin tools: `delegate`, `inspect_background`, `cancel_background`, `input_background`, `defer_background`, `wait_background`
- Docker-gated builtin tools: `execute_bash`, `read_file`, `edit_file`, `delete_file`
- Web-gated builtin tools: `web_search`

上面这些名字都是 `yuuagents` base 包里的内置工具名。是否“可调用”取决于当前 runtime 注入的 capability、是否启用了 daemon，以及相关 optional dependency 是否已安装：

- `view_image` / `sleep` / `update_todo` 是本地原生工具，不依赖 daemon。
- `delegate` 和 background control 工具也是原生内置工具，但需要 host/runtime 提供 delegate capability；启用 daemon 后会自动提供，纯 SDK 路径默认不会。
- 需要持久化状态的 daemon 能力也是同一类 gated capability；只有 daemon 启用后才成立。
- Docker 工具需要 docker capability，以及安装 `yuuagents[docker]` 并确保 Docker Engine 可达。
- `web_search` 需要 web capability，以及安装 `yuuagents[web]`。

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
