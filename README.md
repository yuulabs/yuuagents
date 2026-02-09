# yuuagents

**Minimal agent framework: Persona + Tools + LLM with CLI and dashboard**

yuuagents 是一个轻量级、可扩展的 AI Agent 框架，支持通过 Persona（角色设定）、Tools（工具）和 LLM（大语言模型）的组合创建自主运行的智能代理。

---

## 核心概念

### 1. Agent（代理）

Agent 是框架的核心实体，由两部分组成：

- **AgentConfig（不可变配置）**：定义了代理的身份、角色、可用工具集和 LLM 客户端
- **AgentState（可变状态）**：追踪运行时的任务状态、历史对话、步数、成本等

```python
from yuuagents import Agent, AgentContext
from yuuagents.agent import AgentConfig, SimplePromptBuilder

# 创建配置
config = AgentConfig(
    agent_id="my-agent",
    persona="你是一个专业的Python开发助手",
    tools=tool_manager,
    llm=llm_client,
    prompt_builder=prompt_builder,
)

# 创建代理
agent = Agent(config=config)
```

### 2. Persona（角色设定）

Persona 定义了代理的系统提示（system prompt），通过 `PromptBuilder` 构建：

```python
class SimplePromptBuilder:
    """简单提示构建器，通过拼接多个 section 构建完整提示"""
    
    def add_section(self, section: str) -> "SimplePromptBuilder":
        self._sections.append(section)
        return self
    
    def build(self) -> str:
        return "\n\n".join(self._sections)
```

### 3. Tools（工具）

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
- `execute_bash` - 在 Docker 容器中执行 bash 命令
- `read_file` - 读取文件内容
- `write_file` - 写入文件内容
- `delete_file` - 删除文件
- `web_search` - Tavily 网页搜索

### 4. Loop（执行循环）

代理执行遵循 "LLM → Tool → LLM" 的循环模式：

1. 调用 LLM 获取响应
2. 解析工具调用请求
3. 并行执行所有工具
4. 将结果返回给 LLM
5. 重复直到完成或出错

```python
from yuuagents.loop import run as run_agent

# 运行代理
await run_agent(agent, task="帮我写一个 Python 函数", ctx=context)
```

### 5. Skills（技能）

Skills 是可复用的工具集合，通过文件系统发现：

```
~/.yagents/skills/
├── math/
│   ├── __init__.py
│   └── calculator.py
└── web/
    ├── __init__.py
    └── scraper.py
```

### 6. Daemon（守护进程）

yuuagents 包含一个 HTTP Daemon，提供 REST API 管理代理：

```bash
# 启动守护进程
uv run yagents start

# 提交任务
uv run yagents run --persona "coder" --task "创建 Flask 应用"
```

**Agent 状态：**
- `IDLE` - 空闲
- `RUNNING` - 运行中
- `DONE` - 完成
- `ERROR` - 错误
- `BLOCKED_ON_INPUT` - 等待输入
- `CANCELLED` - 已取消

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

## 示例代码

### 基础示例：创建自定义工具

```python
from __future__ import annotations
import yuutools as yt

@yt.tool(
    params={"x": "第一个数字", "y": "第二个数字"},
    description="计算两个数字的和",
)
async def add(x: float, y: float) -> float:
    return x + y

@yt.tool(
    params={"url": "目标 URL"},
    description="获取网页内容",
)
async def fetch_webpage(
    url: str,
    http_client: httpx.AsyncClient = yt.depends(lambda ctx: ctx.http_client),
) -> str:
    response = await http_client.get(url)
    return response.text
```

### 示例：构建 Agent

```python
from __future__ import annotations
import yuullm
import yuutools as yt
from yuuagents import Agent
from yuuagents.agent import AgentConfig, SimplePromptBuilder
from yuuagents.context import AgentContext
from yuuagents.loop import run as run_agent

async def main():
    # 1. 创建工具管理器
    tool_manager = yt.ToolManager()
    tool_manager.register(add)
    tool_manager.register(fetch_webpage)
    
    # 2. 构建系统提示
    prompt_builder = SimplePromptBuilder()
    prompt_builder.add_section("你是一个数学助手，擅长计算和数据分析。")
    prompt_builder.add_section(f"可用工具: {list(tool_manager.keys())}")
    
    # 3. 创建 LLM 客户端
    llm = yuullm.YLLMClient(
        provider="openai",
        api_key="your-api-key",
        default_model="gpt-4o",
    )
    
    # 4. 创建配置
    config = AgentConfig(
        agent_id="math-agent-001",
        persona="数学助手",
        tools=tool_manager,
        llm=llm,
        prompt_builder=prompt_builder,
    )
    
    # 5. 创建代理和上下文
    agent = Agent(config=config)
    context = AgentContext(
        agent_id=agent.agent_id,
        workdir="/tmp/work",
        docker_container="my-container",
        docker=docker_manager,
    )
    
    # 6. 运行代理
    await run_agent(
        agent,
        task="计算 123 + 456，然后搜索 Python 编程的相关信息",
        ctx=context,
    )
    
    print(f"任务完成！总步数: {agent.steps}")
    print(f"总 Token 消耗: {agent.total_tokens}")
    print(f"总成本: ${agent.total_cost_usd:.4f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 配置

通过 YAML 文件配置：

```yaml
# ~/.yagents/config.yaml
daemon:
  socket: ~/.yagents/yagents.sock
  log_level: info

docker:
  image: ubuntu:24.04

providers:
  openai:
    kind: openai
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o
  
  anthropic:
    kind: anthropic
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-3-5-sonnet-20241022

agents:
  coder:
    provider: openai
    model: gpt-4o
    persona: "你是一个专业的 Python 开发者，擅长编写干净、高效的代码。"
    tools:
      - execute_bash
      - read_file
      - write_file
    skills:
      - python_dev
  
  researcher:
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    persona: "你是一个研究助理，擅长搜索和分析信息。"
    tools:
      - web_search
      - read_file
```

---

## E2E 示例

### 使用内置工具完成完整任务

```python
"""E2E 示例：创建一个 Python Web 应用"""
from __future__ import annotations
import asyncio
import yuullm
import yuutools as yt
from yuuagents import Agent, tools
from yuuagents.agent import AgentConfig, SimplePromptBuilder
from yuuagents.context import AgentContext
from yuuagents.loop import run as run_agent
from yuuagents.daemon.docker import DockerManager

async def create_web_app():
    """创建一个 Flask Web 应用的完整流程"""
    
    # 1. 初始化 Docker 管理器
    docker = DockerManager()
    container_id = await docker.create_container("web-app", image="python:3.11-slim")
    
    try:
        # 2. 创建工具管理器并注册内置工具
        tool_manager = yt.ToolManager()
        for tool in tools.get(["execute_bash", "read_file", "write_file"]):
            tool_manager.register(tool)
        
        # 3. 构建详细的系统提示
        prompt_builder = SimplePromptBuilder()
        prompt_builder.add_section("""你是一个专业的 Python Web 开发者。你的任务是：
1. 在 /app 目录下创建一个 Flask 应用
2. 包含首页路由和 API 路由
3. 添加 requirements.txt
4. 测试应用是否能正常启动""")
        
        # 4. 创建 LLM 客户端
        llm = yuullm.YLLMClient(
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-4o",
        )
        
        # 5. 配置 Agent
        config = AgentConfig(
            agent_id="web-app-builder",
            persona="Flask 应用开发者",
            tools=tool_manager,
            llm=llm,
            prompt_builder=prompt_builder,
        )
        
        # 6. 创建上下文
        agent = Agent(config=config)
        context = AgentContext(
            agent_id=agent.agent_id,
            workdir="/app",
            docker_container=container_id,
            docker=docker,
        )
        
        # 7. 运行任务
        task = """
创建一个完整的 Flask 应用，包含：
1. app.py - 主应用文件，包含：
   - 首页路由 "/" 返回 "Hello, World!"
   - API 路由 "/api/health" 返回健康状态
2. requirements.txt - 包含 Flask 依赖
3. 安装依赖并测试应用启动
"""
        
        print("开始创建 Flask 应用...")
        await run_agent(agent, task=task, ctx=context)
        
        # 8. 检查结果
        if agent.status.value == "done":
            print("✅ 任务完成！")
            
            # 读取创建的文件
            result = await docker.exec(
                container_id,
                "cat /app/app.py",
                timeout=10
            )
            print("\n生成的 app.py:")
            print(result)
            
            # 测试应用
            test_result = await docker.exec(
                container_id,
                "cd /app && python -c \"from app import app; print('应用导入成功')\"",
                timeout=30
            )
            print(f"\n应用测试: {test_result}")
            
        elif agent.status.value == "error":
            print(f"❌ 任务失败: {agent.error.message}")
        
        # 9. 输出统计信息
        print(f"\n📊 执行统计:")
        print(f"   - 总步数: {agent.steps}")
        print(f"   - Token 消耗: {agent.total_tokens}")
        print(f"   - 预估成本: ${agent.total_cost_usd:.4f}")
        
    finally:
        # 清理
        await docker.remove_container(container_id)
        print("\n🧹 已清理容器")

if __name__ == "__main__":
    asyncio.run(create_web_app())
```

### CLI 完整工作流

```bash
# 1. 配置环境
export OPENAI_API_KEY="sk-xxx"
export TAVILY_API_KEY="tvly-xxx"

# 2. 启动守护进程
uv run yagents start

# 3. 查看状态
uv run yagents status

# 4. 提交代码生成任务
uv run yagents run \
  --agent coder \
  --task "创建一个 Python 脚本，实现简单的文件加密功能，使用 Fernet 算法"

# 5. 提交研究任务
uv run yagents run \
  --agent researcher \
  --task "搜索 Python 3.14 的新特性，并总结最重要的 5 个改进"

# 6. 列出所有 Agent
uv run yagents list

# 7. 查看特定 Agent 详情
uv run yagents get <agent-id>

# 8. 停止守护进程
uv run yagents stop
```

---

## 开发

```bash
# 运行测试
uv run pytest

# 代码检查
uv run ruff check src/
uv run ruff check --fix src/
uv run ruff format src/

# 类型检查
uv run mypy src/
```

---

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                        CLI                              │
│  (uv run yagents start/run/stop/status/list/get)       │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP over Unix Socket
                        ▼
┌─────────────────────────────────────────────────────────┐
│                      Daemon                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  REST API   │  │   Manager   │  │  Docker Client  │ │
│  │  (Starlette)│  │  (Agents)   │  │   (aiodocker)   │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────┘ │
└─────────┼────────────────┼──────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────┐
│                      Agent                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Config    │  │    State    │  │  PromptBuilder  │ │
│  │  (frozen)   │  │  (mutable)  │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Execution Loop                        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │   LLM   │───▶│  Tools  │───▶│  LLM    │───▶ ...      │
│  │  Call   │    │  Exec   │    │  Again  │             │
│  └─────────┘    └─────────┘    └─────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## License

MIT
