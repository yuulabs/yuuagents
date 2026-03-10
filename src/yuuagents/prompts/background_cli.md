## Background CLI
你可以使用 background 命令在后台运行长时间任务：
- `background run '<command>'` — 启动后台任务，返回任务ID
- `background tail <id>` — 查看最近输出
- `background drain <id>` — 获取完整输出（完成时返回结果，未完成时返回当前缓冲）
- `background wait <id> [<id>...]` — 阻塞等待一个或多个任务完成，返回结果
- `background kill <id>` — 终止任务
- `background list` — 列出所有后台任务

两种检查策略，根据场景选择：
1. **轮询模式**：sleep N + background tail/drain。适合需要中间观察进展、可能提前干预的场景。
2. **等待模式**：background wait。适合确定要等结果、不需要中间干预的场景。多任务并行时用 wait 同时等多个。
