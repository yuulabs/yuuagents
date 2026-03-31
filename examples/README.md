# yuuagents Examples

这个目录现在明确分成两类示例。

## SDK Mode

如果你只想本地开发 agent，用这个：

```bash
cd /path/to/yuuagents
uv sync
uv run python examples/sdk_quickstart.py
```

这个示例不需要：

- daemon
- Docker
- 数据库
- YAML 配置文件

它展示的是当前真实 SDK API：`run_once(...)`。

## Daemon / Service Mode

如果你想把 `yuuagents` 当服务运行，再看下面两个引导脚本。

### `examples/simple_start.py`

无 `rich` 依赖的 service onboarding：

```bash
cd /path/to/yuuagents
uv sync --extra daemon --extra docker --extra web
uv run python examples/simple_start.py
```

### `examples/getting_started.py`

带 `rich` 界面的 service onboarding：

```bash
cd /path/to/yuuagents
uv sync --extra daemon --extra docker --extra web --group examples
uv run python examples/getting_started.py
```

这两个脚本都会引导你：

1. 配置 provider
2. 复制并安装 service-mode 配置
3. 启动 daemon
4. 提交第一个任务

## Notes

- `config.example.yaml` 只服务模式需要，不属于 SDK Quick Start。
- Docker / web 搜索依赖都是 service-mode 扩展能力。
