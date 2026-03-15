# Flow 重构需求收集

用 `core/flow.py` 的 Flow/Agent 替换现有 `loop.py` + `step.py` + `runtime_session.py`。

## 已确认的设计决策

### 1. block/detach 不需要

Flow 架构下 delegate 就是一个普通的长时间 tool call。
子 agent 在 tool 内部 `await child.wait()`，父 agent 自然阻塞。
不需要 Session 层面的 block/detach 机制。

### 2. stem 懒合并

emit 只 append 碎片，不当场合并。读取时懒合并：
- `view()` — 全量合并视图（内部缓存 `_merged` + `_merge_cursor`）
- `delta(since)` — 增量：从 since 到当前的碎片合并后返回，同时返回新 cursor

observer 记一个 cursor，每次 `delta(cursor)` 拿增量。
通知机制用 `asyncio.Condition`，emit 时 `notify_all()`。

### 3. OutputBuffer 退场

不再需要独立的 OutputBuffer 旁路。
工具的流式输出通过子 Flow 的 `emit(chunk, merge=True)` 进入 stem，
Flow.stem 成为唯一的观测通道。

### 4. 交互式进程支持 (stdin 写入)

Flow 的 `(stem, mailbox)` 天然构成双向通道：
- stem: tool → agent（观测流）
- mailbox: agent → tool（控制流）

bash tool 的子 Flow 可以 `await mailbox.get()` 接收 agent 发来的消息，
写入 proc.stdin。agent 通过 `write_stdin(flow_id, text)` 工具与交互式进程对话。

场景：apt y/n、python debugger、ssh 登录等。

### 5. defer（工具超时/后台化）保留

现有 `_run_tools` 中的 defer 机制（asyncio.wait + defer event）在 Flow 架构下原样保留。
未完成的 tool 移入后台，agent 拿到 "moved to background, id:xxx" 继续思考。
后台 tool 完成后通过 mailbox 通知 agent。

两种触发方式：
- 超时自动 defer（timeout 参数）
- 外部信号 `flow.defer()`

defer 应独立为 Flow 的控制原语，与 `send(msg)` / `cancel()` 平级。
现有 `send(msg, defer_tools=True)` 将 defer 信号耦合在消息上，应拆开。

agent 自身无法主动 defer（self-driven 模型下它在 await 里阻塞），
但超时 + 外部信号已覆盖实际场景。

### 6. History 压缩（context compression）

`Agent.messages` 是普通 list，与 `Flow.stem` 独立。
stem 是 append-only 观测日志，messages 是 LLM 上下文窗口，互不干扰。
压缩 = 直接替换 `agent.messages`，无架构障碍。

触发方式：在 `_call_llm` 前执行 `before_llm(self)` 钩子，
钩子检查 token 数决定是否压缩。压缩是内部策略，不走 mailbox。

### 7. Agent 接受 config 对象，不接受散装参数

现有 Flow Agent 直接传 `client, manager, ctx, system, model`，太原始。
应复用现有 `AgentConfig`（`agent.py`），由 config 负责：
- build system prompt
- 选择 tools
- 决定 model
- 其他策略（max_steps, compression 等）

`Agent.messages` 的初始化（包括 system message）由 config 驱动，不硬编码。

## 改动文件清单

### 核心改造（重写/大改）

| 文件 | 动作 | 说明 |
|------|------|------|
| `core/flow.py` | 改 | 补 emit merge、defer 原语、config 接入 |
| `loop.py` | 删 | 逻辑迁入 `core/flow.py` Agent._run |
| `step.py` | 删 | StepHandle/Fork/OutputBuffer 被 Flow 替代 |
| `runtime_session.py` | 重写 | 改为 Flow 的薄包装，去掉 block/detach |
| `daemon/manager.py` | 改 | 从 session.step() 驱动改为 Flow/Agent 驱动 |
| `__init__.py` | 改 | 更新公开 API 导出 |

### 次要改动（适配接口变更）

| 文件 | 动作 | 说明 |
|------|------|------|
| `tools/delegate.py` | 改 | 去掉 session.block()，改为 await child.wait() |
| `tools/skill_cli.py` | 删 | 功能移除 |
| `tools/read_skill.py` | 删 | 功能移除 |
| `tools/bash.py` | 改 | 同上，output_buffer → flow.emit |

### 不变

| 文件 | 说明 |
|------|------|
| `agent.py` (AgentConfig) | 复用，不改 |
| `context.py` | 去掉 output_buffer 字段即可，其余不变 |
| `persistence.py` | 独立层，接口不变 |
| `tools/` 其余 | yuutools DI 解耦，不受影响 |
| `daemon/api.py` | REST 层，不直接依赖 loop/step |
| `daemon/docker.py` | output_buffer 参数类型适配，逻辑不变 |
| `cli/` | 纯 HTTP 客户端，不变 |

### 8. 持久化无障碍

`TaskRecorder` 不依赖 Session/StepHandle/Fork，纯 DTO 写入。
在 Flow Agent 的 `_stream_llm` 末尾调 `record_llm`，
`_run_tools` 末尾调 `record_tool`，即可。
recorder 作为 AgentConfig 的可选字段，不单独注入。
replay（`load_history`）重建 `list[yuullm.Message]`，直接赋给 `agent.messages`。

`persistence.py` 本身不需要改动。

## 待收集

- 持久化集成方案
- Tracing 集成方案
- Token/cost 计量
- daemon/manager 改造路径
- 公开 API 变更

## Token / cost 语义

宿主需要两类稳定数据，不能再只暴露 `total_tokens`：

1. `last_usage`：最近一次 LLM 调用的 usage delta
2. `total_usage`：当前 session 累计 usage
3. `last_cost_usd`：最近一次 LLM 调用的 cost delta
4. `total_cost_usd`：当前 session 累计 cost

约束：

1. `usage` 保留结构化字段：`input_tokens`、`output_tokens`、`cache_*_tokens`、`total_tokens`
2. `Flow Agent` 在每次 `_stream_llm()` 结束后，同时更新 trace 和内存状态
3. `Session.wait()` 必须把这些状态同步给宿主层
4. daemon/API 返回同一份语义，不能出现“trace 里有 cost，host-facing state 没有”的分叉

`total_tokens` 只保留为兼容字段；它来自 `total_usage.total_tokens`，不再是唯一真相。
