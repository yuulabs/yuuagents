# YAgents 入门示例

这个目录包含 YAgents 的交互式入门引导，帮助你快速上手。

## 快速开始

### 方式 1: 交互式引导 (推荐)

我们提供两个版本的入门引导：

#### A. 完整版 (需要 rich 库)

拥有漂亮的终端界面和更好的交互体验：

```bash
cd /path/to/yuuagents

# 安装 examples 依赖组
uv sync --group examples

# 运行交互式引导
uv run python examples/getting_started.py
```

#### B. 简化版 (无额外依赖)

如果不想安装 rich，可以使用简化版：

```bash
cd /path/to/yuuagents
uv run python examples/simple_start.py
```

这个脚本会：
1. 检查先决条件 (Docker, yagents 安装)
2. 引导你输入 LLM Provider 配置 (API Key, Base URL 等)
3. 运行 `yagents setup` 初始化环境
4. 启动 yagents daemon
5. 运行第一个示例任务
6. 展示如何监控任务执行

### 方式 2: 手动配置

如果你想手动配置，请按以下步骤：

#### 1. 安装依赖

```bash
# 克隆仓库
git clone <repository>
cd yuuagents

# 安装 yagents
uv sync
```

#### 2. 配置环境变量

```bash
# OpenAI (默认)
export OPENAI_API_KEY=sk-...

# 或者 Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# 可选: Tavily 用于网络搜索
export TAVILY_API_KEY=tvly-...
```

#### 3. 初始化配置

```bash
# 这会安装配置文件到 ~/.yagents/config.yaml
# 并拉取必要的 Docker 镜像
uv run yagents setup
```

#### 4. 启动 Daemon

```bash
# 方式 1: 作为 systemd 服务 (推荐)
# setup 命令已经注册服务，会自动启动

# 方式 2: 手动启动 (前台运行)
uv run yagents start
```

#### 5. 运行第一个任务

```bash
# 简单的问候任务
uv run yagents run --task "Say hello and introduce yourself"

# 文件操作任务
uv run yagents run --task "Create a file /tmp/test.txt with content 'Hello' and read it back"

# 使用特定 agent
uv run yagents run --agent researcher --task "Search for information about Python asyncio"
```

#### 6. 监控任务

```bash
# 列出所有 agents
uv run yagents list

# 查看特定 agent 状态
uv run yagents status <agent_id>

# 查看对话历史
uv run yagents logs <agent_id>
```

## 配置说明

### 配置文件位置

- 默认配置: `~/.yagents/config.yaml`
- 示例配置: `config.example.yaml` (项目根目录)

### 支持的 Provider

- **OpenAI**: `OPENAI_API_KEY`, 默认模型 `gpt-4o`
- **Anthropic**: `ANTHROPIC_API_KEY`, 默认模型 `claude-3-5-sonnet-20241022`
- **OpenRouter**: `OPENROUTER_API_KEY`, 支持多种模型
- **自定义**: 任何兼容 OpenAI API 格式的服务

### 自定义 Base URL

在 `config.yaml` 中修改 provider 配置：

```yaml
providers:
  my-provider:
    kind: openai
    api_key_env: MY_API_KEY
    default_model: gpt-4
    base_url: https://api.example.com/v1
```

## 示例任务

### 代码生成
```bash
uv run yagents run --task "Write a Python script that fetches weather data from an API"
```

### 文件操作
```bash
uv run yagents run --task "List all files in /tmp and create a summary"
```

### 网络搜索
```bash
uv run yagents run --task "Search for the latest Python 3.14 features and summarize"
```

### 使用 Docker 容器
```bash
# 使用特定容器
uv run yagents run --container <container_id> --task "Run apt update in the container"

# 使用特定镜像
uv run yagents run --image python:3.11 --task "Create a simple Flask app"
```

## 故障排除

### Docker 权限问题

如果遇到 Docker 权限错误：

```bash
# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新登录或运行
newgrp docker
```

### API Key 未设置

```bash
# 检查环境变量是否设置
echo $OPENAI_API_KEY

# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc
source ~/.bashrc
```

### Daemon 未启动

```bash
# 检查 daemon 状态
yagents stop

# 手动启动
uv run yagents start

# 或使用 systemd
systemctl --user status yagents
systemctl --user restart yagents
```

## 更多信息

- [项目文档](../README.md)
- [配置示例](../config.example.yaml)
- `yagents --help` - 查看所有命令
