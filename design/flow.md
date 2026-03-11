# Flow / Ping / FlowManager

## 它是什么

Flow 是 yuuagents 的统一运行抽象。agent、tool、subagent 都是 flow。它们组成一棵树：父 flow 创建子 flow，子 flow 完成后通过 Ping 通知父 flow。

之前的 `RunningToolRegistry` 只处理"tool 超时后怎么跟踪"这一个场景，用的是 handle + collect_finished + 合成 check_running_tool 消息。silence detection 也是直接往 history 里塞消息。这些都是"运行中的东西把事件传回父节点"的特例。Flow 把它们统一了。

## 核心类型

**OutputBuffer** — 累积 subprocess 的流式输出。`write(bytes)` 写入，`tail()` / `full()` 读取。从 `running_tools.py` 搬过来，位置在 `flow.py`。

**FlowKind** — `AGENT` 或 `TOOL`。

**FlowStatus** — `RUNNING` / `WAITING_INPUT` / `DONE` / `ERROR` / `CANCELLED`。

**PingKind** — flow 之间传递的事件类型：
- `CHILD_COMPLETED` — 子 flow 完成
- `CHILD_FAILED` — 子 flow 失败
- `TOOL_OUTPUT` — 工具产生新输出
- `USER_MESSAGE` — 用户消息
- `CANCEL` — 取消请求
- `SYSTEM_NOTE` — 框架内部通知（如 silence detection）

**Ping** — `msgspec.Struct(frozen=True, kw_only=True)`，包含 `kind`、`source_flow_id`、`payload`。

**Flow** — 一个运行单元。关键字段：
- `flow_id` — 唯一标识，也是对外的 handle
- `kind` / `name` / `parent_flow_id` / `status`
- `task` — `asyncio.Task`，tool flow 用
- `result` / `error` — 完成后的结果
- `tool_call_id` — 对应 LLM 的 tool_call_id
- `_children` — 子 flow ID 列表
- `_ping_queue` — `asyncio.Queue[Ping]`
- `_output_buffer` — 流式输出缓冲

方法：`ping(Ping)` 入队，`recv(timeout)` 出队。

**FlowManager** — 管理整个 flow 树。方法：
- `create(kind, name, ...)` → 创建 flow，自动挂到父节点
- `get(flow_id)` → 查找
- `complete(flow_id, result)` → 标记完成，ping 父节点
- `fail(flow_id, error)` → 标记失败，ping 父节点
- `cancel(flow_id)` → 取消 task，标记取消，ping 父节点
- `has_running_children(flow_id)` → 是否有未完成的子 flow
- `collect_completed_children(flow_id)` → 收割已完成的子 flow 并从树上摘除

## 怎么用

### agent loop 里

`run()` 接受可选的 `root_flow` 和 `flow_manager` 参数。不传时自己创建（向后兼容）。外部传入时，调用者负责创建 flow 并持有引用，用于在 loop 运行期间 ping 用户消息进去。

主循环每一步先 `_drain_pings(root_flow)` 再经过 `_merge_user_pings()` 合并连续的 `USER_MESSAGE` / `SYSTEM_NOTE`（LLM history 不允许连续 user message），最后 `_apply_ping(agent, ping)` 翻译成 history 消息：
- `CHILD_COMPLETED` → 合成 check_running_tool 的 tool_call + tool_result 对
- `SYSTEM_NOTE` → user message
- `USER_MESSAGE` → user message

### 外部 ping 用户消息

当 agent loop 正在运行时，外部（如 dispatcher）可以通过 `root_flow.ping(Ping(kind=USER_MESSAGE, ...))` 注入用户消息。loop 在下一个 step 开头 drain 到这条 ping，经过 merge 后追加到 agent history。这使得用户不需要等当前 agent 跑完整轮就能让新消息被 LLM 看到。

### 子 flow 未完成时 loop 不退出

`_step()` 中 LLM 未发起 tool call 时，若 `flow_manager.has_running_children(root_flow)` 为 True，agent 不会被标记为 DONE。主循环改为 `await root_flow.recv()` 阻塞等待子 flow 完成的 ping（或用户追加消息）。收到 ping 后回到循环顶部正常 drain → apply → 再次调用 LLM。

这意味着：
- soft_timeout 仍然让 LLM 尽快拿到 handle 并产出文本
- 但 agent loop 不会结束，flow_manager 也不会被销毁
- 子 flow 完成时 CHILD_COMPLETED ping 自动注入，LLM 下一步能看到结果
- 不需要跨 turn 持久化 FlowManager

### soft timeout

`_step()` 的 `gather()` 传 `on_pending` 闭包。闭包里用 `flow_manager.create(TOOL, ...)` 建子 flow，启 `_monitor_tool_flow` 协程。task 完成时 monitor 调 `flow_manager.complete/fail`，自动 ping 回 root flow。

### check_running_tool / cancel_running_tool

直接从 `ctx.flow_manager` 拿 flow，用 `flow_id` 当 handle。check 等 task 完成或超时返回 tail output。cancel 调 `flow_manager.cancel()`。

## 不做

- stdin / context_query ping（未来需要时加）
- flow 持久化（运行态不可序列化）
- 跨 turn FlowManager — 不需要，因为 loop 在子 flow 未完成时不退出
- 多模态 ping（Ping.payload 是 str）
