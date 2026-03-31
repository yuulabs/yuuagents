# Snapshot 持久化设计

这份文档只描述当前实现里的 snapshot 持久化，不把 tracing、任务列表和会话历史混成一个概念。

## 范围

当前持久化分成两层：

- tracing：由 `yuutrace` 负责，可观测性用途
- snapshot persistence：由 `yuuagents.persistence` 负责，恢复用途

这里讨论的是后者。

## 配置开关

snapshot 相关开关在配置里的位置是：

```yaml
snapshot:
  enabled: false
  restore_on_start: false
```

含义如下：

- `enabled=false`：不写 snapshot，也不启动后台 writer
- `enabled=true, restore_on_start=false`：写 snapshot，但 daemon 重启后不自动恢复
- `enabled=true, restore_on_start=true`：写 snapshot，并在 daemon 启动时尝试恢复未完成任务

`restore_on_start` 依赖 `enabled=true`，否则配置验证会报错。

## 数据模型

当前 SQLite schema 由 `TaskPersistence.start()` 初始化，包含两张表。

### `tasks`

每一行代表一个任务的主记录，字段包括：

- `task_id`
- `agent_id`
- `persona`
- `input_kind`
- `input_preview`
- `input_json`
- `system_prompt`
- `model`
- `tools_json`
- `docker_container`
- `status`
- `head_turn`
- `created_at`
- `updated_at`
- `error_json`

这个表保存的是任务元数据和最新状态摘要，不保存完整的逐轮 snapshot payload。

### `task_checkpoints`

每一行代表一个 checkpoint，当前主用途是 snapshot。

字段包括：

- `task_id`
- `turn`
- `phase`
- `ts`
- `payload`

当前 `phase` 只有一个有效值：`"snapshot"`。

## 写入流程

当 `snapshot.enabled=true` 时，daemon 会创建 `TaskWriter`。

写入流程大致如下：

1. `AgentManager` 在任务运行中调用 `_persist_snapshot(...)`
2. 当前 session 通过 `Session.snapshot(...)` 生成 `AgentState`
3. `AgentState` 被 JSON 编码成 checkpoint payload
4. checkpoint 被送入 `TaskWriter` 的队列
5. `TaskWriter` 批量写入 `task_checkpoints`
6. 同时更新 `tasks.head_turn`、`tasks.status`、`tasks.updated_at`

实现上是“批量缓冲 + 异步 flush”，不是每一步都同步写盘。

## 恢复流程

当 `snapshot.restore_on_start=true` 时，daemon 启动会调用恢复逻辑。

恢复步骤是：

1. 从 `tasks` 里找状态仍处于 `running` 或 `blocked_on_input` 的任务
2. 对每个任务读取最新的 snapshot checkpoint
3. 如果没有 snapshot，就跳过这个任务
4. 如果 snapshot 反序列化失败，也视为没有可恢复状态
5. 有可恢复状态时，重建 `Session` 并恢复运行

恢复不是“重新发明一个新任务”，而是沿用原任务记录和 snapshot state。

## 当前实现的约束

- snapshot 只负责恢复，不负责 tracing
- snapshot payload 是 `AgentState` 的 JSON 编码
- restore 只看最新的 `phase="snapshot"` checkpoint
- 如果任务没有 snapshot，就不会被凭空构造出来
- `tasks` 表里的 `docker_container` 会被带回恢复流程
- 对于无效或过时的 snapshot，当前实现倾向于跳过，而不是强行继续

## 和 daemon 的关系

`AgentManager` 是 snapshot 持久化的调用方，`TaskPersistence` 只是存储层。

这意味着：

- persistence 层不决定什么时候 snapshot
- persistence 层不决定如何组装 system prompt
- persistence 层不决定 tool capability
- persistence 层只负责存取任务元数据和 snapshot 数据

## 设计动机

这样拆分的好处是：

- daemon 重启后可以恢复未完成任务
- 任务状态和 snapshot payload 分层清楚
- tracing 数据不会污染恢复逻辑
- SQLite 结构简单，便于排障和迁移

## 需要注意的点

- `snapshot.enabled=true` 并不自动意味着一定能恢复，前提是之前确实写过 snapshot
- `restore_on_start=true` 只是“尝试恢复”，不是保证所有任务都恢复成功
- 更换数据库路径会让旧任务数据留在旧文件里，所以 `up` 会阻止直接切换
- 如果你要做 schema 变更，优先考虑向后兼容，而不是重写整个任务表

