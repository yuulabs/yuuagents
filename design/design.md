# yuuagents Design

## Overview

yuuagents 是一个 **Linux Service 形态的 Agent 框架**。它把 yuutools、yuullm、yuutrace 粘合成一个可运行的 Agent 运行时。

核心公式：**Agent = Persona + Tools + LLM**

设计原则：

1. **CLI 是一等公民** — 所有操作通过 `yagents` 命令完成。WebUI 是锦上添花。
2. **复用 yuutools** — 工具基础设施（`@tool`、`ToolManager`、`depends()`）全部来自 yuutools，yuuagents 只 re-export，不重新实现。
3. **Linux Service** — daemon 模式常驻运行，管理多个 Agent 的生命周期。
4. **Agent Skills 兼容** — 支持 [Agent Skills](https://agentskills.io) 开放标准，通过 SKILL.md 扩展 Agent 能力。
5. **Docker 隔离** — bash 工具在 Docker 容器中执行，安全隔离。

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  yagents daemon (Linux service, systemd)                 │
│                                                         │
│  ┌─ Agent Manager ────────────────────────────────┐     │
│  │  asyncio task pool, 管理所有 agent 生命周期      │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌─ Skill Discovery ─────────────────────────────┐     │
│  │  扫描配置目录, 解析 SKILL.md frontmatter        │     │
│  │  注入 <available_skills> 到 agent system prompt │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌─ Docker Manager ──────────────────────────────┐     │
│  │  管理 bash 工具的执行容器                        │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌─ REST API (Unix Domain Socket) ───────────────┐     │
│  │  供 CLI / yuubot / WebUI 调用                   │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌─ Dashboard WebUI (可选, 未实现) ──────────────┐     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
         ▲ Unix Domain Socket
         │
    ┌────┴─────┐
    │ yagents  │  CLI — 用户的主要交互方式
    │  CLI     │
    └──────────┘
```

外部消费者（如 yuubot QQ Bot）通过 HTTP 调用 daemon 的 REST API，与 yagents 运行在不同进程中。

## Key Concepts

### Agent

一个 Agent 是一个有状态的对话实体：

```python
from attrs import define, field
import yuutools as yt
import yuullm

@define
class Agent:
    agent_id: str                          # UUID
    persona: str                           # system prompt 正文
    tools: yt.ToolManager                  # 来自 yuutools
    llm: yuullm.YLLMClient                # 来自 yuullm
    history: list[yuullm.Message] = field(factory=list)
    skills_xml: str = ""                   # <available_skills> XML，注入 system prompt
    status: AgentStatus = AgentStatus.IDLE
```

注意 `tools` 的类型是 `yt.ToolManager`，来自 yuutools。yuuagents 不重新定义工具基础设施。

### Step Loop

Agent 的运行是一个简单的 step loop：

```python
agent.setup(task_description)
while not agent.done():
    await agent.step()
```

每一步 `step()` 内部：
1. 调用 LLM（通过 yuullm stream）
2. 如果有 tool_calls，并行执行所有工具
3. 将结果追加到 history
4. 由 yuutrace 自动埋点记录 usage/cost

Loop 的完整实现集成了 yuutrace：

```python
async def run_agent(agent: Agent, task: str) -> None:
    agent.setup(task)
    with ytrace.conversation(
        id=agent.agent_id, agent=agent.persona[:50], model=...
    ) as chat:
        chat.system(persona=agent.full_system_prompt, tools=agent.tools.specs())
        chat.user(task)
        while not agent.done():
            await _step(agent, chat)
```

其中 `agent.full_system_prompt` 是 `persona + skills_xml` 的拼接。

### AgentContext

工具通过 `yt.depends()` 注入的上下文对象：

```python
from attrs import define

@define
class AgentContext:
    agent_id: str
    workdir: str                    # 工作目录 (容器内路径)
    docker_container: str           # Docker 容器名/ID
    input_queue: asyncio.Queue      # user_input 工具使用
```

内置工具通过 `depends(lambda ctx: ctx.docker_container)` 等方式获取所需的上下文信息。

### Builtin Tools

所有内置工具使用 `@yt.tool()` 装饰器定义，通过 `yt.depends()` 注入 `AgentContext`。

| Tool           | Description                                 | Phase |
| -------------- | ------------------------------------------- | ----- |
| execute_bash   | Docker 容器中执行命令（docker exec）          | 1     |
| read_file      | 读取文件（容器内，带大小/二进制检查，图片返回多模态） | 1     |
| write_file     | 对文件应用 patch                              | 1     |
| edit_file      | 精确替换文件中的唯一字符串                     | 1     |
| delete_file    | 删除文件（容器内）                            | 1     |
| web_search     | Tavily API 搜索，返回 LLM 友好文本            | 1     |
| web_fetch      | 获取 URL 内容，提取正文转 markdown             | 2     |
| user_input     | 向用户请求输入（阻塞 loop，asyncio.Event）     | 2     |
| subagent       | 动态创建并运行子 Agent                        | 2     |

#### execute_bash

```python
@yt.tool(
    params={
        "command": "The bash command to execute",
        "timeout": "Timeout in seconds (default 120, max 600)",
    },
)
async def execute_bash(
    command: str,
    timeout: int = 120,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
) -> str:
    """Execute a bash command in the Docker container."""
```

通过 `docker exec` 在持久容器中执行。容器在 daemon 启动时创建或复用，Agent 生命周期内保持存活。

#### file 工具

```python
@yt.tool(params={"path": "Absolute file path to patch", "patch": "Unified diff patch"})
async def write_file(
    path: str,
    patch: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
) -> str:
    """Apply a patch to a file and return a short summary."""
```

`read_file` 只返回适合直接喂给 LLM 的内容：
- 普通文本直接返回文本
- 过大的文本文件拒绝读取
- 二进制文件拒绝读取
- 常见图片返回 `image_url` 多模态块

`write_file` 和 `edit_file` 都通过容器内命令修改文件，确保与 bash 工具共享同一个文件系统视图。

#### web_search

```python
@yt.tool(
    params={
        "query": "Search query string",
        "max_results": "Maximum number of results (default 5, max 10)",
    },
)
async def web_search(
    query: str,
    max_results: int = 5,
    api_key: str = yt.depends(lambda ctx: ctx.tavily_api_key),
) -> str:
    """Search the web using Tavily API. Returns LLM-friendly text results."""
```

使用 [Tavily API](https://tavily.com)，专为 AI Agent 设计的搜索 API。返回结构化的搜索结果，包含标题、URL、摘要。

#### subagent (Phase 2)

```python
@yt.tool(
    params={
        "persona": "The sub-agent's persona/system prompt",
        "task": "Task description for the sub-agent",
        "tools": "Comma-separated tool names for the sub-agent",
        "bootstrap": "Optional file path. Sub-agent reads this file at startup for context.",
    },
)
async def subagent(persona: str, task: str, tools: str, bootstrap: str = "") -> str:
    """Dynamically create and run a sub-agent."""
```

子 Agent 拥有独立的 history 和 loop，但共享父 Agent 的 tracing context（作为子 span）。费用自动向上聚合。

#### user_input (Phase 2)

```python
@yt.tool(
    params={"message": "Message to display to the user, explaining what input is needed"},
)
async def user_input(
    message: str,
    queue: asyncio.Queue = yt.depends(lambda ctx: ctx.input_queue),
) -> str:
    """Request input from the user. The agent loop will pause until the user responds."""
```

实现方式：工具内部 `await queue.get()`，Agent 状态变为 `BLOCKED_ON_INPUT`。外部通过 REST API `POST /api/agents/{id}/input` 往 queue 里塞消息，Agent loop 自然恢复。不需要快照/恢复上下文。

## Extension: Agent Skills

yuuagents 兼容 [Agent Skills](https://agentskills.io) 开放标准。

### 工作方式

采用 **Filesystem-based** 集成方式（标准推荐）：

1. **Discovery** — daemon 启动时扫描配置的 skill 目录，解析每个 `SKILL.md` 的 YAML frontmatter（name + description）。
2. **Injection** — 将所有已发现 skill 的 metadata 生成 `<available_skills>` XML，注入 Agent 的 system prompt。
3. **Activation** — Agent 在执行任务时，根据需要通过 `read_file` 或 `execute_bash(cat ...)` 自行读取 SKILL.md 的完整内容来激活 skill。
4. **Execution** — Agent 按照 SKILL.md 中的指令操作，可以执行 `scripts/` 中的脚本、读取 `references/` 中的文档。

### Skill 目录结构

遵循 Agent Skills 标准：

```
~/.yagents/skills/
├── git-worktree/
│   ├── SKILL.md              # 必须
│   ├── scripts/              # 可选：可执行脚本
│   └── references/           # 可选：参考文档
├── code-review/
│   └── SKILL.md
└── python-testing/
    ├── SKILL.md
    └── references/
        └── pytest-patterns.md
```

### System Prompt 注入

```xml
<available_skills>
  <skill>
    <name>git-worktree</name>
    <description>Manage git worktrees for parallel development...</description>
    <location>~/.yagents/skills/git-worktree/SKILL.md</location>
  </skill>
  <skill>
    <name>code-review</name>
    <description>Perform structured code reviews...</description>
    <location>~/.yagents/skills/code-review/SKILL.md</location>
  </skill>
</available_skills>
```

### 扩展工具的两层体系

| 层 | 机制 | 适用场景 |
|---|---|---|
| **Agent Skills** (SKILL.md) | 目录 + Markdown 指令 | 知识/流程扩展。从外部扩展。 |
| **Python 工具** (@yt.tool) | 代码中扩展。内置工具。 | 能力扩展：给 agent 新的可调用函数。tool是内置的，最基本工具，有严格的语义和权限检查，不从外部导入。 |

Agent Skills 是知识扩展，Python 工具是能力扩展。两者互补。

## Daemon

### 进程模型

`yagents start` 启动一个前台进程（配合 systemd 管理）。进程内部：

- 一个 asyncio event loop
- AgentManager 管理多个 Agent，每个 Agent 是一个 asyncio.Task
- Starlette ASGI app 监听 Unix Domain Socket，提供 REST API
- Docker 容器管理器，按需创建/复用容器

### AgentManager

```python
@define
class AgentManager:
    _agents: dict[str, Agent]                    # agent_id → Agent
    _tasks: dict[str, asyncio.Task]              # agent_id → running task
    _input_queues: dict[str, asyncio.Queue]      # agent_id → user_input queue
    _config: Config
    _docker: DockerManager

    async def submit(self, request: TaskRequest) -> str:
        """创建 Agent, 启动 asyncio task, 返回 agent_id"""

    async def list_agents(self) -> list[AgentInfo]:
        """列出所有 agents 的状态摘要"""

    async def get_status(self, agent_id: str) -> AgentInfo:
        """获取单个 agent 的详细状态"""

    async def get_history(self, agent_id: str) -> list[Message]:
        """获取 agent 的对话历史"""

    async def respond(self, agent_id: str, content: str) -> None:
        """回复 user_input 请求"""

    async def cancel(self, agent_id: str) -> None:
        """取消 agent (cancel asyncio task)"""
```

### Docker Manager

```python
@define
class DockerManager:
    _image: str                  # 默认镜像
    _containers: dict[str, str]  # agent_id → container_id

    async def ensure_container(self, agent_id: str) -> str:
        """为 agent 创建或复用 Docker 容器，返回 container_id"""

    async def exec(self, container_id: str, command: str, timeout: int) -> str:
        """在容器中执行命令，返回 stdout+stderr"""

    async def cleanup(self, agent_id: str) -> None:
        """清理 agent 的容器"""
```

使用 `aiodocker` 库进行异步 Docker 操作。

### REST API

通过 Unix Domain Socket 提供，路由定义：

```
POST   /api/agents              创建并启动 agent
GET    /api/agents              列出所有 agents
GET    /api/agents/{id}         获取 agent 状态 + 摘要
GET    /api/agents/{id}/history 获取 agent 完整对话历史
POST   /api/agents/{id}/input   回复 user_input 请求
DELETE /api/agents/{id}         取消 agent

GET    /api/skills              列出已发现的 skills
POST   /api/skills/scan         重新扫描 skill 目录

GET    /api/config              获取当前配置 (脱敏)
GET    /health                  健康检查
```

#### POST /api/agents

```json
{
    "persona": "coder",
    "task": "Implement feature X",
    "tools": ["execute_bash", "read_file", "write_file", "web_search"],
    "skills": ["git-worktree"],
    "model": "gpt-4o"
}
```

`persona` 可以是预配置模板名（如 `"coder"`），也可以是完整的 system prompt 文本。如果匹配到模板名，使用模板中预配置的 system_prompt 和默认 tools/skills；显式传入的 tools/skills 会 override 模板默认值。

#### GET /api/agents

```json
[
    {
        "agent_id": "550e8400-...",
        "persona": "coder",
        "task": "Implement feature X",
        "status": "running",
        "created_at": "2025-01-01T00:00:00Z",
        "steps": 5,
        "total_tokens": 12345,
        "total_cost_usd": 0.05
    }
]
```

## CLI

`yagents` 是唯一的 CLI 入口，所有子命令通过 Unix Socket 与 daemon 通信。

```bash
# Daemon 管理
yagents start [--config PATH]        # 启动 daemon (前台，配合 systemd)
yagents stop                         # 停止 daemon (发送 shutdown 信号)

# Agent 任务管理
yagents run --persona coder \
            --task "Implement feature X" \
            [--tools bash,read,write] \
            [--skills git-worktree] \
            [--model gpt-4o]

yagents list                         # 列出所有 agents
yagents status <agent-id>            # 查看 agent 状态
yagents logs <agent-id> [--follow]   # 查看对话历史 (--follow 实时跟踪, Phase 2)
yagents stop-agent <agent-id>        # 停止某个 agent
yagents input <agent-id> "message"   # 回复 user_input 请求

# Skill 管理
yagents skills list                  # 列出已发现的 skills
yagents skills scan                  # 重新扫描 skill 目录
```

### CLI 实现

CLI 是一个 thin client。它不直接操作 Agent，而是通过 httpx 向 Unix Socket 发送 HTTP 请求：

```python
# cli/client.py
import httpx

class YAgentsClient:
    def __init__(self, socket_path: str):
        self._transport = httpx.AsyncHTTPTransport(uds=socket_path)
        self._client = httpx.AsyncClient(transport=self._transport, base_url="http://yagents")

    async def submit(self, request: dict) -> dict:
        resp = await self._client.post("/api/agents", json=request)
        return resp.json()

    async def list_agents(self) -> list[dict]:
        resp = await self._client.get("/api/agents")
        return resp.json()

    # ...
```

## Configuration

配置文件路径：`~/.yagents/config.yaml`（可通过 `yagents setup --config` 安装/覆盖）。

```yaml
daemon:
  socket: ~/.yagents/yagents.sock
  log_level: info

docker:
  image: ubuntu:24.04

skills:
  paths:
    - ~/.yagents/skills

tavily:
  api_key_env: TAVILY_API_KEY

providers:
  openai-default:
    api_type: openai-chat-completion
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o
    base_url: ""
    organization: ""
    pricing: []

agents:
  main:
    provider: openai-default
    model: gpt-4o
    persona: >
      You are a senior software engineer. You write clean, well-tested code.
      You prefer simple solutions and avoid over-engineering.
    tools: [execute_bash, read_file, write_file, delete_file, web_search]
    skills: ["*"]
```

## Module Layout

```
src/yuuagents/
    __init__.py              # Public API re-exports (yuutools + Agent + types)
    py.typed

    # ── 核心 ──
    agent.py                 # Agent 类 (attrs.define)
    loop.py                  # step loop: LLM → tool calls → trace
    context.py               # AgentContext — 工具 DI 上下文
    types.py                 # AgentStatus, AgentInfo, TaskRequest 等 (msgspec.Struct)
    config.py                # YAML 配置解析

    # ── 内置工具 ──
    tools/
        __init__.py          # 导出所有内置工具 + BUILTIN_TOOLS 注册表
        bash.py              # execute_bash
        file.py              # read_file, write_file, delete_file
        web.py               # web_search (Tavily)

    # ── Agent Skills ──
    skills/
        __init__.py
        discovery.py         # 扫描目录, 解析 SKILL.md frontmatter, 生成 XML

    # ── Daemon ──
    daemon/
        __init__.py
        server.py            # Unix Socket HTTP server (Starlette + uvicorn)
        api.py               # REST API 路由
        manager.py           # AgentManager — 多 agent 生命周期管理
        docker.py            # Docker 容器管理

    # ── CLI ──
    cli/
        __init__.py
        main.py              # CLI 入口 (click group)
        client.py            # Unix Socket HTTP client (httpx)
```

Phase 2 新增：

```
    tools/
        user_input.py        # user_input
        subagent.py          # subagent
```

Dashboard WebUI 暂未实现。

## Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    # 自家基础设施
    "yuutools >=0.1.0",
    "yuullm >=0.1.0",
    "yuutrace >=0.1.0",
    # 数据 & 类型
    "attrs >=24.2.0",
    "msgspec >=0.19.0",
    # Daemon
    "starlette >=0.40.0",
    "uvicorn >=0.30.0",
    # CLI
    "click >=8.0",
    # CLI → Daemon 通信 + web_fetch
    "httpx >=0.27.0",
    # Docker
    "aiodocker >=0.23.0",
    # Web search
    "tavily-python >=0.5.0",
]
```

## Public API

`yuuagents.__init__` re-export 的符号：

```python
# 来自 yuutools (re-export)
from yuutools import tool, Tool, BoundTool, ToolSpec, ParamSpec, ToolManager, depends, DependencyMarker

# 来自 yuuagents
from .agent import Agent
from .context import AgentContext
from .types import AgentStatus, AgentInfo, TaskRequest
from .loop import run_agent
from . import tools                    # yuuagents.tools.execute_bash 等
```

用户代码示例：

```python
import yuuagents as ya

# ya.tool, ya.ToolManager, ya.depends 等全部来自 yuutools
# ya.Agent, ya.run_agent 等来自 yuuagents 自身
```

## Example Usage

### 作为库使用

```python
import yuuagents as ya
import yuullm

# 1. Setup LLM
client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIProvider(api_key="sk-..."),
    default_model="gpt-4o",
)

# 2. Create agent
agent = ya.Agent(
    agent_id="my-task-001",
    persona="You are a senior Python developer.",
    tools=ya.ToolManager([
        ya.tools.execute_bash,
        ya.tools.read_file,
        ya.tools.write_file,
        ya.tools.web_search,
    ]),
    llm=client,
)

# 3. Run
await ya.run_agent(agent, task="Refactor the database module to use async SQLAlchemy")
print(agent.history[-1].content)
```

### 通过 CLI 使用

```bash
# 启动 daemon
yagents start

# 下发任务
yagents run --persona coder --task "Implement a REST API for user management"

# 查看状态
yagents list
yagents status 550e8400-...

# 查看对话
yagents logs 550e8400-...
```

### 外部消费者 (yuubot)

```python
import httpx

# yuubot 通过 HTTP 调用 yagents daemon
transport = httpx.AsyncHTTPTransport(uds="/home/user/.local/run/yagents.sock")
client = httpx.AsyncClient(transport=transport, base_url="http://yagents")

# 下发任务
resp = await client.post("/api/agents", json={
    "persona": "coder",
    "task": "Fix the login bug reported in issue #42",
    "tools": ["execute_bash", "read_file", "write_file"],
})
agent_id = resp.json()["agent_id"]

# 轮询
resp = await client.get(f"/api/agents/{agent_id}")
status = resp.json()["status"]  # "running" | "done" | "error" | "blocked_on_input"
```

## Design Decisions

1. **复用 yuutools** — 工具基础设施是 yuutools 的职责。yuuagents 只 re-export，保持单一职责。vision.md 是 source of truth。
2. **Linux Service** — daemon 模式而非一次性 CLI。支持多 Agent 并发、外部消费者（yuubot）远程调用、长时间运行的任务。
3. **Unix Domain Socket** — 经典 Linux daemon 通信方式。安全（不暴露端口），简单（标准 HTTP over UDS）。
4. **CLI 是 thin client** — CLI 不直接操作 Agent，只是 REST API 的命令行封装。这保证了 CLI 和 HTTP 消费者（yuubot）的行为完全一致。
5. **Agent Skills 兼容** — 采用开放标准而非自定义插件系统。现有的大量 skills 可以直接复用，自己写的 skills 也能在 Claude Code、Cursor 等工具里用。Filesystem-based 集成，Agent 通过 read_file 自行激活。
6. **Docker 隔离** — bash 和 file 工具都在容器内执行。Agent 不能直接操作宿主机文件系统。
7. **attrs + msgspec** — Agent 等有状态对象用 `attrs.define`（可变）；API 数据传输类型用 `msgspec.Struct`（不可变，高性能序列化）。
8. **yuutrace 自动集成** — loop 内部自动埋点，用户无需手动调用 yuutrace API。
9. **Tavily 搜索** — 专为 AI Agent 设计的搜索 API，返回 LLM 友好的结构化结果。

## Implementation Phases

### Phase 1: 核心可运行

全部实现，使 yagents 成为一个完整可用的 Agent 服务：

- Agent 类 + Step Loop + yuutrace 集成
- 内置工具: execute_bash (Docker), read_file, write_file, delete_file, web_search (Tavily)
- Agent Skills 发现 + 加载
- 配置文件 (YAML)
- Daemon (Unix Socket server) + Agent Manager + Docker Manager
- CLI (start, run, list, status, logs, stop-agent, skills)
- REST API
- 外部 Python 工具注册

### Phase 2: 高级功能

- SubAgent 工具
- user_input 工具
- web_fetch 工具（正文提取，防止塞爆上下文）
- `yagents logs --follow` 实时跟踪 (WebSocket)

### Phase 3: Dashboard（未实现）

- React WebUI（复用 @yuulabs/ytrace-ui 组件）
- Agent 列表、状态、交互面板
- `yagents dashboard` 命令
