## 持久化设计

yuuagents 有两套持久化系统：

1. 任务持久化系统。对于下发的任务，它可能由于各种原因挂起（比如说网络，停机，请求用户输入，etc），在下次加载时需要能够重启。这主要是储存agent对话历史。目前的代码对于AgentState只使用了in-memory储存，这导致停机会发生数据丢失。
2. Ytrace. 这是目前代码里面已经完成的系统，它是可观测性。

## 持久化任务

### 储存

yuuagents使用任何数据库储存；默认使用sqlite. 使用异步ORM编写储存层。测试时使用sqlite的in-memory模式。

任务唯一地由task id 标定。持久化的核心原则：

- 追加写（append-only）：写入格式是 delta，永远追加新条目，不做 in-place update（除了任务头部的摘要字段）。
- 粒度：delta 的粒度只到 step 级别：LLM step 与 Tool step。
- 可回放（replayable）：从数据库读取后，通过顺序回放 delta 得到可运行的内存状态。
- 可恢复 pending tools：如果重启时发现存在未返回的工具调用，系统会写入 interrupted 结果；是否重试由 agent 自主决定。

#### 数据模型（最小）

建议拆两张表：`tasks`（快查）+ `task_checkpoints`（追加日志）。

`tasks`（任务头部，便于 list/status 查询）：

- `task_id`：主键
- `status`：`running | done | error | blocked_on_input | cancelled`
- `head_turn`：当前已持久化的最大 turn（单调递增）
- `updated_at`
- `error_json`：可选，终态错误摘要（便于 status 快查）

`task_checkpoints`（delta 主体，一条记录对应一个 step）：

- `task_id`
- `turn`：从 1 开始，单调递增
- `phase`：仅允许 `llm | tool`
- `ts`
- `payload`：建议 msgspec JSON（或 msgpack）

约束与索引：

- `PRIMARY KEY (task_id, turn, phase)`：每个 turn 最多一条 llm + 一条 tool
- `INDEX (task_id, turn)`：快速回放

#### Checkpoint（delta）格式

LLM step（`phase=llm`）表达“本轮 LLM 输出 + 本轮要调用的 tools”：

- `history_append`：新增的 assistant message（按 yuullm message 结构存）
- `tool_calls`：`[{call_id, name, args_json}]`（显式存，避免从 message 再解析）
- `status_after`：
  - 如果 `tool_calls` 非空：`blocked_on_input`（请求工具调用视作 block_on_input）
  - 如果 `tool_calls` 为空：`done`
- 可选：`usage_delta`/`cost_delta`

Tool step（`phase=tool`）表达“本轮 tools 的返回结果（批量）”：

- `results`：`[{call_id, ok, output_text | error:{type,message, interrupted?, cancelled?}}]`
- `status_after`：通常为 `running`（让下一轮 LLM 继续）

pending tools 的定义（不单独存集合）：

- 对某个 `turn`，`pending_call_ids = llm.tool_calls - tool.results`
- 正常情况下 pending 只会出现在“最后一个 turn 只有 llm 但缺 tool”的情形（崩溃或退出时）。

#### 批量写入（避免频繁 commit）

yuuagents 有一个后台 writer（任务批量收集器），负责把多个 step 的多个条目合并进一次数据库事务。

行为约束：

- agent loop 只把 checkpoint 追加到内存 buffer，不直接触库。
- writer 按阈值触发 flush，一次 flush 用一个事务写入多条记录：
  - `INSERT task_checkpoints ... VALUES (...), (...), ...`（多行）
  - `UPDATE tasks SET head_turn=?, status=?, updated_at=? ...`（按 task 聚合）

触发阈值（建议）：

- `rows_total >= 200` 或 `bytes_total >= 1MB` 或 `elapsed >= 300ms`
- 进程退出时必须 flush
- 任务进入终态（done/error/cancelled）时建议强制 flush 该任务

崩溃语义：

- 允许丢失尚未 flush 的最后一小段 buffer。
- 回放时若发现最后一个 turn 缺 tool step，则视为 pending tools，恢复逻辑会补 interrupted（见下）。

### 加载

yuuagents启动时，会从数据库加载未完成任务，特别是处于 `running` 或 `blocked_on_input` 状态的任务。

加载算法（确定性）：

1. 读取 `tasks`（status/head_turn）。
2. 读取 `task_checkpoints`：按 `(turn asc, phase asc)` 顺序回放到内存状态（history/status/pending 推导等）。

#### 重启恢复：pending tools -> interrupted（写回数据库）

当 yuuagents 从数据库读取时，如果任务处于 `blocked_on_input`，并且最后一个 turn 满足：

- 存在 `phase=llm` 的记录，且 `tool_calls` 非空
- 不存在对应 `phase=tool` 的记录（或 tool step 不包含所有 call_id）

则认为存在 pending tools。系统会为这些 pending tools 追加写入一条 `phase=tool` 的 checkpoint，内容是：

- 对每个 pending `call_id` 写入 `ok=false` 且 `error.type="interrupted"`
- `status_after` 写为 `running`

这样 interrupted 只会被记录一次（下一次重启会读到已存在的 tool step，不会重复补写）。

注意：这和 `cancelled` 不一样，`cancelled` 指任务被用户故意打断，任务应进入终态 `cancelled`，且不应被恢复逻辑自动改回 `running`。
