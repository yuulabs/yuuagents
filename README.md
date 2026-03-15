# yuuagents

**Minimal agent framework: System + Tools + LLM with CLI and daemon**

yuuagents 是一个轻量级 AI Agent 框架，基于 Flow 架构实现可观测、可中断、可寻址的执行模型。

---

## 核心概念

### 1. Flow（执行单元）

Flow 是框架的基础执行原语。一切运行的东西——LLM、工具、bash、子 agent——都是一个 Flow。

每个 Flow 拥有三个能力：
- **stem**：类型化的 append-only 事件日志（观测）
- **mailbox**：异步队列（通信）
- **cancel**：任务取消（中断）

### 2. Agent（LLM 执行流）

Agent 组合了 Flow 并添加 LLM 特定行为（defer、messages）。Agent 与 Flow 之间是组合关系，不是继承。

```python
from yuuagents.core.flow import Agent

agent = Agent(
    client=llm_client,
    manager=tool_manager,
    ctx=context,
    system="你是一个专业的 Python 开发助手",
    model="gpt-4o",
)

agent.start()
agent.send("帮我写一个排序算法")
await agent.wait()
```

### 3. AgentConfig（不可变配置）

```python
from yuuagents.agent import AgentConfig

config = AgentConfig(
    agent_id="my-agent",
    system="你是一个专业的 Python 开发助手",
    tools=tool_manager,
    llm=llm_client,
    max_steps=20,
)
```

`system` 是唯一的系统提示源。`persona` 和 `system_prompt` 保留为兼容别名。

### 4. Session（运行时会话）

Session 是 Flow Agent 的薄包装，面向宿主层（daemon / SDK）：

```python
from yuuagents import Session, AgentContext
from yuuagents.agent import AgentConfig

session = Session(config=config, context=ctx)
```

### 5. Tools（工具）

工具使用 `yuutools` 框架定义，支持依赖注入：

```python
import yuutools as yt

@yt.tool(
    params={"command": "要执行的 bash 命令"},
    description="在 Docker 容器中执行命令",
)
async def execute_bash(
    command: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: object = yt.depends(lambda ctx: ctx.docker),
) -> str:
    return await docker.exec(container, command)
```

**内置工具：**
- `execute_bash` — 在 Docker 容器中执行 bash 命令
- `read_file` / `edit_file` / `delete_file` — 文件操作
- `web_search` — Tavily 网页搜索
- `delegate` — 委派子 agent
- `sleep` / `view_image` / `update_todo`

### 6. Defer（后台化）

工具执行支持 defer 机制：超时或外部信号可将未完成的工具移入后台。Agent 拿到 `"Moved to background, id:xxx"` 继续思考，后台工具完成后通过 mailbox 通知。

```python
# 外部信号触发 defer
agent.send("新消息", defer_tools=True)
```

### 7. Daemon（守护进程）

yuuagents 包含 HTTP Daemon，通过 Unix socket 提供 REST API：

```bash
# 启动守护进程
uv run yagents up

# 提交任务
uv run yagents run --agent main --task "创建 Flask 应用"

# 查看状态
uv run yagents list

# 停止
uv run yagents down
```

---

## 安装

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

**环境要求：**
- Python >= 3.14
- Docker（用于容器化工具执行）
- 可选：Tavily API Key（用于网页搜索）

---

## 两种使用模式

### 模式 1：CLI 模式

```bash
# 一次性初始化（创建配置、数据库、构建 Docker 镜像）
yagents install

# 启动守护进程
yagents up

# 提交任务
yagents run --agent main --task "写一个 Python 函数"

# 查看状态
yagents list
yagents status <task-id>

# 停止
yagents down
```

### 模式 2：SDK 模式

```python
from yuuagents.init import setup

# 传入配置文件路径（等价于 yagents install + yagents up -d）
cfg = await setup("/path/to/config.yaml")
```

`setup()` 是幂等的，会创建目录、初始化数据库、构建 Docker 镜像、启动 daemon。

### 混合使用

SDK 初始化后，CLI 可直接访问同一套数据（共享配置和数据库）。SDK 进程和 daemon 各自独立运行 Agent，通过数据库共享任务状态。

---

## 示例：直接使用 Flow Agent

```python
"""直接使用 core.flow.Agent 运行一个任务"""
import asyncio
import yuullm
import yuutools as yt
from yuuagents.core.flow import Agent
from yuuagents.context import AgentContext

async def main():
    # 1. 准备工具
    tool_manager = yt.ToolManager()
    tool_manager.register(execute_bash)
    tool_manager.register(read_file)

    # 2. 创建 LLM 客户端
    llm = yuullm.YLLMClient(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
    )

    # 3. 创建上下文
    ctx = AgentContext(
        agent_id="demo",
        workdir="/tmp/work",
        docker_container=container_id,
        docker=docker,
    )

    # 4. 创建并运行 Agent
    agent = Agent(
        client=llm,
        manager=tool_manager,
        ctx=ctx,
        system="你是一个 Python 开发助手",
        model="gpt-4o",
    )
    agent.start()
    agent.send("帮我写一个 Flask 应用")
    await agent.wait()

    # 5. 查看输出
    print(agent.render())

asyncio.run(main())
```

---

## 配置

通过 YAML 文件配置（参见 `config.example.yaml`）：

```yaml
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
    provider: openai-default
    model: gpt-4o
    persona: >
      You are a senior software engineer.
    tools:
      - execute_bash
      - read_file
      - edit_file
      - delete_file
      - web_search
```

---

## 架构

```
CLI (click) ──HTTP/Unix socket──▶ Daemon (Starlette/uvicorn)
                                    ├── AgentManager (lifecycle)
                                    ├── REST API (/api/agents/...)
                                    └── DockerManager (containers)
                                            │
                                    Agent Runtime
                                    ├── core/flow.py (Flow + Agent)
                                    ├── runtime_session.py (Session)
                                    └── Tools (DI via yuutools)
```

---

## License

MIT
