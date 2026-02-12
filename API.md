# yuuagents SDK API 文档

本文档详细介绍如何使用 yuuagents 的 SDK 模式，特别是如何进行扩展（Tools/Skills）。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [Tools 扩展](#tools-扩展)
- [Skills 扩展](#skills-扩展)
- [完整示例](#完整示例)
- [最佳实践](#最佳实践)

## 概述

yuuagents 提供两种使用模式：

1. **CLI 模式**: 通过命令行管理 Agent，适合独立部署
2. **SDK 模式**: 通过 Python 代码集成，适合嵌入其他应用

SDK 模式的核心价值在于：**允许外部库在不修改 yuuagents 核心代码的情况下，为 Agent 提供新的能力（Tools 和 Skills）**。

## 快速开始

### 安装

```bash
pip install yuuagents
```

### 基础使用

```python
from __future__ import annotations
import asyncio
from yuuagents.init import setup

async def main():
    # 初始化 yuuagents（等价于 CLI 的 `yagents install` + `yagents up`）
    cfg = await setup("/path/to/config.yaml")
    print(f"yuuagents 已初始化，配置路径: {cfg.socket_path}")

asyncio.run(main())
```

`setup()` 是幂等的，可以安全地多次调用。

## 核心概念

### Agent 架构

```
┌─────────────────────────────────────────┐
│              Agent                      │
├─────────────────────────────────────────┤
│  AgentConfig (不可变)                    │
│  ├── agent_id: str                      │
│  ├── persona: str                       │
│  ├── tools: ToolManager                 │
│  ├── llm: YLLMClient                    │
│  └── prompt_builder: PromptBuilder      │
├─────────────────────────────────────────┤
│  AgentState (可变)                       │
│  ├── task: str                          │
│  ├── history: list[Message]             │
│  ├── status: AgentStatus                │
│  ├── steps: int                         │
│  └── total_tokens: int                  │
└─────────────────────────────────────────┘
```

### 执行流程

```
User Task → Agent.setup() → Loop (LLM → Tools → LLM) → Done/Error
```

## Tools 扩展

Tools 是 Agent 与外部世界交互的方式。yuuagents 使用 `yuutools` 框架定义工具。

### 1. 定义新工具

使用 `@yt.tool` 装饰器定义工具：

```python
from __future__ import annotations
import yuutools as yt

@yt.tool(
    params={
        "x": "第一个数字",
        "y": "第二个数字",
    },
    description="计算两个数字的和",
)
async def add(x: float, y: float) -> float:
    return x + y
```

### 2. 依赖注入

工具可以通过 `yt.depends()` 访问上下文：

```python
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

#### 可用的上下文属性

通过 `AgentContext` 可以访问：

| 属性 | 类型 | 说明 |
|------|------|------|
| `task_id` | str | 当前任务 ID |
| `agent_id` | str | Agent ID |
| `workdir` | str | 工作目录 |
| `docker_container` | str | Docker 容器 ID |
| `docker` | DockerExecutor | Docker 执行器 |
| `state` | AgentState | Agent 状态 |
| `input_queue` | asyncio.Queue | 用户输入队列 |
| `tavily_api_key` | str | Tavily API Key |

### 3. 注册到 Agent

```python
from __future__ import annotations
import yuutools as yt
from yuuagents import Agent
from yuuagents.agent import AgentConfig, SimplePromptBuilder
from yuuagents.loop import run as run_agent

async def main():
    # 1. 创建工具管理器
    tool_manager = yt.ToolManager()
    
    # 2. 注册内置工具
    from yuuagents import tools
    for tool in tools.get(["execute_bash", "read_file", "write_file"]):
        tool_manager.register(tool)
    
    # 3. 注册自定义工具
    tool_manager.register(add)
    tool_manager.register(fetch_webpage)
    
    # 4. 创建 Agent 配置
    prompt_builder = SimplePromptBuilder()
    prompt_builder.add_section("你是一个数学助手")
    prompt_builder.add_section(f"可用工具: {list(tool_manager.keys())}")
    
    config = AgentConfig(
        agent_id="math-agent",
        persona="数学助手",
        tools=tool_manager,
        llm=llm_client,  # 你的 LLM 客户端
        prompt_builder=prompt_builder,
    )
    
    # 5. 运行 Agent
    agent = Agent(config=config)
    await run_agent(agent, task="计算 1+2", ctx=context)
```

### 4. 内置 Tools 参考

#### execute_skill_cli

执行 Skill 提供的 CLI 命令（受限模式）。此工具专门用于执行 Skill 提供的命令行工具，具有严格的安全限制。

```python
from yuuagents.tools import execute_skill_cli

# 在 Agent 中使用
result = await execute_skill_cli(command="my-skill-tool --help", timeout=60)
```

**参数：**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `command` | str | 是 | 要执行的 CLI 命令 |
| `timeout` | int | 否 | 超时时间（秒），默认 300，最大 3600 |

**安全限制：**

- 禁止执行 shell 解释器（bash、sh、zsh 等）
- 禁止文件操作命令（rm、mv、dd 等）
- 禁止权限修改命令（chmod、chown、sudo 等）
- 禁止 shell 控制符（;、|、&、>、< 等）
- 禁止命令替换（$(...) 或 `...`）
- 工作目录固定为用户主目录

**返回值：**

命令输出加上退出码：`__YAGENTS_EXIT_CODE__=0`

#### delegate

将任务委托给另一个已配置的 Agent，并返回其最终的文本响应。

```python
from yuuagents.tools import delegate

# 在 Agent 中委托任务给另一个 Agent
result = await delegate(
    agent="code-reviewer",
    context="Python 代码审查",
    task="请审查以下代码...",
)
```

**参数：**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `agent` | str | 是 | 目标 Agent 名称 |
| `context` | str | 是 | 任务上下文 |
| `task` | str | 是 | 具体任务描述 |
| `tools` | list[str] | 否 | 覆盖默认工具的列表 |

**委托深度限制：**

- 最大委托深度为 3 层，防止无限递归
- 超过限制会抛出 `DelegateDepthExceededError`

**使用场景：**

- 需要不同专长 Agent 协作时
- 任务需要拆分给更专业的 Agent 时
- 实现 Agent 间的协作工作流

### 5. 外部库提供 Tools

如果你的库需要为 yuuagents 提供工具，可以创建一个 `register_tools()` 函数：

```python
# my_library/tools.py
from __future__ import annotations
import yuutools as yt

@yt.tool(
    params={"query": "搜索关键词"},
    description="在内部知识库中搜索",
)
async def search_knowledge_base(
    query: str,
    api_client = yt.depends(lambda ctx: ctx.my_api_client),
) -> str:
    return await api_client.search(query)

@yt.tool(...)
async def another_tool(...) -> str:
    ...

def register_tools(tool_manager: yt.ToolManager) -> None:
    """注册本库提供的所有工具到 yuuagents."""
    tool_manager.register(search_knowledge_base)
    tool_manager.register(another_tool)
```

使用者只需调用：

```python
from my_library.tools import register_tools
import yuutools as yt

tool_manager = yt.ToolManager()
register_tools(tool_manager)  # 一键注册所有工具
```

### 6. 工具的最佳实践

#### 参数设计

- 使用清晰的参数名和类型提示
- 提供合理的默认值
- 使用 `yt.depends()` 获取依赖，而非硬编码

#### 错误处理

```python
@yt.tool(...)
async def risky_operation(...) -> str:
    try:
        result = await do_something()
        return f"成功: {result}"
    except SpecificError as e:
        return f"错误: {e}"
```

#### 返回值

- 返回字符串，便于 LLM 理解
- 包含成功/失败状态
- 对于复杂数据，返回格式化的字符串

## Skills 扩展

Skills 是可复用的工具集合 + 文档，通过文件系统发现。

### 1. Skill 目录结构

```
~/.yagents/skills/
├── web-scraping/
│   ├── SKILL.md          # Skill 元数据
│   ├── __init__.py       # 可选：Python 实现
│   └── utils.py          # 可选：辅助模块
└── data-analysis/
    ├── SKILL.md
    └── ...
```

### 2. SKILL.md 格式

```text
---
name: web-scraping
description: 网页抓取和数据提取工具集
---

# Web Scraping Skill

## 概述

本 Skill 提供网页抓取和数据提取能力。

## 可用工具

1. **fetch_page**: 获取网页 HTML
2. **extract_data**: 从 HTML 提取结构化数据

## 使用示例

```python
# 在 Agent 中使用
fetch_page(url="https://example.com")
```// #markdown bug, don't remove this line
```

#### Frontmatter 字段

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | 是 | Skill 标识符 |
| `description` | 否 | Skill 描述 |

### 3. 在 Skill 中定义工具

Skill 可以包含 Python 代码：

```python
# ~/.yagents/skills/web-scraping/__init__.py
from __future__ import annotations
import yuutools as yt
from bs4 import BeautifulSoup

@yt.tool(
    params={"url": "网页 URL"},
    description="获取并解析网页",
)
async def fetch_and_parse(
    url: str,
    http_client = yt.depends(lambda ctx: ctx.http_client),
) -> str:
    response = await http_client.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def register(skill_manager):
    """可选：注册函数，供 discovery 使用"""
    skill_manager.register_tool(fetch_and_parse)
```

### 4. 配置 Skill 路径

在 `config.yaml` 中指定 Skill 搜索路径：

```yaml
skills:
  paths:
    - ~/.yagents/skills          # 默认路径
    - /path/to/custom/skills     # 自定义路径
    - ./my-project/skills        # 项目级 Skill
```

### 5. 外部库提供 Skills

如果你的库需要为 yuuagents 提供 Skills，可以：

**方式 A：提供 Skill 目录**

```python
# my_library/skills/__init__.py
from pathlib import Path

def get_skill_path() -> str:
    """返回本库提供的 Skill 目录路径"""
    return str(Path(__file__).parent / "yagents-skills")
```

用户在配置中添加：

```yaml
skills:
  paths:
    - ~/.yagents/skills
    - /path/to/my_library/skills/yagents-skills
```

**方式 B：自动安装**

```python
# my_library/install.py
import shutil
from pathlib import Path

def install_skills():
    """将 Skill 安装到 yuuagents 目录"""
    src = Path(__file__).parent / "skills"
    dst = Path.home() / ".yagents/skills/my-library"
    
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Skills installed to {dst}")
```

### 6. 使用 Skill

在 Agent 配置中指定使用的 Skills：

```yaml
agents:
  web-crawler:
    provider: openai
    model: gpt-4o
    persona: "你是网页抓取专家"
    skills:
      - web-scraping      # 使用 web-scraping Skill
      - data-analysis     # 同时加载多个 Skill
```

或在代码中动态使用：

```python
from yuuagents.skills import scan, render

# 扫描所有 Skill
skill_paths = ["~/.yagents/skills", "/custom/path"]
all_skills = scan(skill_paths)

# 生成系统提示片段
skills_xml = render(all_skills)
prompt_builder.add_section(skills_xml)
```

## 完整示例

### 示例 1：外部库提供 API 调用工具

```python
# my_service/yagents_plugin.py
from __future__ import annotations
import yuutools as yt
from my_service import APIClient

@yt.tool(
    params={
        "endpoint": "API 端点",
        "method": "HTTP 方法 (GET/POST/PUT/DELETE)",
        "data": "请求体数据 (JSON)",
    },
    description="调用内部服务的 REST API",
)
async def call_internal_api(
    endpoint: str,
    method: str = "GET",
    data: str = "{}",
    api_client: APIClient = yt.depends(lambda ctx: ctx.api_client),
) -> str:
    import json
    
    payload = json.loads(data) if data else None
    
    try:
        if method == "GET":
            result = await api_client.get(endpoint)
        elif method == "POST":
            result = await api_client.post(endpoint, payload)
        elif method == "PUT":
            result = await api_client.put(endpoint, payload)
        elif method == "DELETE":
            result = await api_client.delete(endpoint)
        else:
            return f"不支持的 HTTP 方法: {method}"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"API 调用失败: {e}"

def register_yagents_tools(tool_manager: yt.ToolManager) -> None:
    """供 yuuagents 使用的工具注册入口"""
    tool_manager.register(call_internal_api)
```

使用：

```python
# 用户代码
from yuuagents import Agent
from yuuagents.agent import AgentConfig
from my_service.yagents_plugin import register_yagents_tools
from my_service import APIClient

async def main():
    # 创建 API 客户端
    api_client = APIClient(base_url="https://api.example.com")
    
    # 创建工具管理器
    tool_manager = yt.ToolManager()
    
    # 注册我的服务工具
    register_yagents_tools(tool_manager)
    
    # 创建 Agent（注入 api_client 到上下文）
    context = AgentContext(
        agent_id="api-agent",
        workdir="/tmp",
        docker_container="my-container",
        api_client=api_client,  # 自定义上下文属性
    )
    
    config = AgentConfig(...)
    agent = Agent(config=config)
    
    await run_agent(
        agent,
        task="获取用户列表",
        ctx=context,
    )
```

### 示例 2：外部库提供完整 Skill

```
my-analytics-sdk/
├── pyproject.toml
├── src/
│   └── my_analytics/
│       ├── __init__.py
│       └── yagents/
│           └── skills/
│               └── analytics/
│                   ├── SKILL.md
│                   └── __init__.py
```

```markdown
<!-- SKILL.md -->
---
name: analytics
description: 数据分析与可视化工具集
---

# Analytics Skill

提供数据清洗、分析和可视化能力。
```

```python
# __init__.py
from __future__ import annotations
import yuutools as yt
import pandas as pd

@yt.tool(
    params={"csv_path": "CSV 文件路径"},
    description="加载 CSV 文件并返回统计摘要",
)
async def analyze_csv(
    csv_path: str,
    docker = yt.depends(lambda ctx: ctx.docker),
    container = yt.depends(lambda ctx: ctx.docker_container),
) -> str:
    # 读取文件
    cmd = f"cat {csv_path}"
    content = await docker.exec(container, cmd)
    
    # 分析
    import io
    df = pd.read_csv(io.StringIO(content))
    
    return f"""
行数: {len(df)}
列数: {len(df.columns)}
列名: {', '.join(df.columns)}
统计摘要:
{df.describe().to_string()}
"""
```

安装和使用：

```python
# 安装 Skill
import my_analytics
from pathlib import Path
import shutil

skill_src = Path(my_analytics.__file__).parent / "yagents/skills/analytics"
skill_dst = Path.home() / ".yagents/skills/analytics"
shutil.copytree(skill_src, skill_dst, dirs_exist_ok=True)

# 在配置中使用
# config.yaml
agents:
  data-scientist:
    skills:
      - analytics
```

## 最佳实践

### 1. 库设计原则

- **单一职责**: 一个库提供一类工具（如 `my-service-api` 只提供该服务的 API 工具）
- **可发现性**: 提供清晰的文档和入口点（如 `register_tools()` 函数）
- **依赖管理**: 明确声明 yuuagents 版本兼容性

### 2. 上下文扩展

如果工具需要自定义上下文属性：

```python
# 扩展 AgentContext（通过子类或猴子补丁）
from yuuagents.context import AgentContext

@define
class ExtendedContext(AgentContext):
    my_api_client: MyAPIClient | None = None

# 使用
context = ExtendedContext(
    agent_id="agent-1",
    workdir="/tmp",
    docker_container="container-1",
    my_api_client=api_client,
)
```

### 3. 工具命名规范

- 使用 `snake_case`
- 前缀避免冲突: `myservice_action` 而非 `action`
- 保持简洁但描述性: `call_api` 而非 `make_http_request_to_api`

### 4. 错误处理

```python
@yt.tool(...)
async def safe_tool(...):
    try:
        result = await do_something()
        return json.dumps({"success": True, "data": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
```

### 5. 测试

```python
import pytest
from my_library.tools import search_knowledge_base

@pytest.mark.asyncio
async def test_search_knowledge_base():
    # 创建 Mock 上下文
    class MockContext:
        def __init__(self):
            self.api_client = MockAPIClient()
    
    ctx = MockContext()
    
    # 绑定工具
    bound = search_knowledge_base.bind(ctx)
    
    # 调用
    result = await bound.run(query="python")
    assert "success" in result
```

## 参考

- [yuutools 文档](https://github.com/anomalyco/yuutools)
- [yuullm 文档](https://github.com/anomalyco/yuullm)
- [示例代码](../examples/)
