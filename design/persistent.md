## 持久化设计

yuuagents 有两套持久化系统：

1. **任务持久化**（snapshot）：储存 agent 对话历史，用于崩溃后恢复执行。
2. **Ytrace**（tracing）：可观测性，使用 OpenTelemetry SimpleSpanProcessor 即时导出，不做 batch。

两者职责分离：tracing 负责即时记录每一步的可观测数据，snapshot 只负责恢复。

## 储存

默认使用 SQLite，异步 ORM（SQLAlchemy async）。测试时使用 SQLite in-memory。

两张表：`tasks`（快查）+ `task_checkpoints`（追加日志）。

### tasks

- `task_id`：主键
- `agent_id`、`persona`、`task`、`system_prompt`、`model`、`tools_json`、`docker_container`
- `status`：`running | done | error | blocked_on_input | cancelled`
- `head_turn`：当前已持久化的最大 turn（单调递增）
- `created_at`、`updated_at`
- `error_json`：可选，终态错误摘要

### task_checkpoints

- `PRIMARY KEY (task_id, turn, phase)`
- `phase`：`"snapshot"`
- `ts`
- `payload`：msgspec JSON 序列化的 `AgentState`

## 快照（Snapshot）

### AgentState

```python
class AgentState(msgspec.Struct, frozen=True):
    messages: tuple[yuullm.Message, ...]
    total_usage: yuullm.Usage | None
    total_cost_usd: float
    rounds: int
    conversation_id: str | None
```

- 每个快照是完整状态，不是增量
- 所有 tool_call 均有配对的 tool result（no dangling calls）
- 恢复时直接 `Agent(initial_messages=list(state.messages))`

### 写入时机

由宿主（`daemon/manager.py`）在 step 边界驱动：

```python
async for _step in session.step_iter():
    if not session.has_pending_background:
        await self._persist_snapshot(session)
```

- **正常 step 完成且无后台任务**：写入快照
- **有后台任务运行时**：跳过快照（后台任务的中间态无法精确恢复，side effect 不可重放）
- **CancelledError**：`kill()`（cancel 所有 bg tasks，合成 interrupted results 到 messages）→ 写快照
- **Exception**：同 CancelledError，先 `kill()` 再写快照

### 批量写入

`TaskWriter` 后台异步收集 checkpoint，按阈值批量 flush 到数据库：

- `rows >= 200` 或 `bytes >= 1MB` 或 `elapsed >= 300ms`
- 进程退出时必须 flush
- 崩溃语义：允许丢失尚未 flush 的最后一小段 buffer

### 进程级崩溃

如果进程直接被 kill（kill -9、OOM），`_run` 的 try/except 不会执行，状态回退到最后一个已写入的快照。这是预期行为——后台任务运行期间的中间交互不可恢复。

## 加载

启动时从数据库加载未完成任务（status 为 `running` 或 `blocked_on_input`）。

1. 读取 `tasks` 表
2. 读取 `task_checkpoints`，取最新的 `phase="snapshot"` 记录
3. 反序列化 `AgentState`，用 `initial_messages` 恢复 Agent

### pending tools 恢复

加载时如果最后一个 turn 存在 `phase=llm` 的 delta checkpoint（遗留数据），且缺少对应的 `phase=tool` 记录，系统会补写一条 interrupted tool result（`ok=false, error.type="interrupted"`），然后将 status 改回 `running`。这保证 interrupted 只写入一次。
