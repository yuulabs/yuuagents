# yuuagents

A minimal Python agent runtime built around a single runtime core: `Flow`.

```
Flow = observable execution + mailbox + cancellation
Agent = LLM loop on top of Flow
Basin = live-flow index
TaskHost = SDK-first host API
```

## Architecture

`Flow` is the only runtime object that matters.

- `Flow` records what happened, owns a mailbox, tracks parent/children, and can be cancelled or waited on.
- `Agent` is the LLM executor attached to a root `Flow`.
- `Deferred` is a `Flow` state, not a separate object type.
- `Basin` indexes all live flows, not just roots.
- `TaskHost` is the host-facing API for submit, status, history, inspect, wait, cancel, and persistence.
- Daemon/CLI support is an adapter layer on top of `TaskHost`, not the core model.

## Persistence

SDK-only users should be able to run the runtime without `yagents install`.

- In-process persistence should work without any daemon or socket.
- Durable local persistence should be optional and pluggable.
- The daemon adapter may use default disk locations and discovery files, but those are deployment conveniences, not runtime requirements.

## Core Model

```
TaskHost
├── Basin
│   ├── flow lookup by id
│   ├── live flow registry
│   └── flow-level inspect / wait / cancel
└── Persistence
    ├── in-memory for SDK use
    └── optional disk-backed storage

Agent
├── root Flow
├── messages / usage / rounds
└── tool execution on child Flows
```

## Flow Semantics

`Flow` is the execution topology.

- Each flow has an `id`, `kind`, `info`, `stem`, `mailbox`, `children`, and optional parent link.
- `send(content)` accepts `str`, `Item`, or `list[Item]`.
- The meaning of `send` is flow-specific and follows dependency inversion: different flow kinds interpret the same content differently.
- A deferred flow continues running and may later notify its parent through the mailbox.

## Host API

`TaskHost` is the main service boundary for both SDK and daemon usage.

```python
host = TaskHost(basin=basin, persistence=persistence, root_factory=root_factory)

task_id = await host.submit(spec)
info = await host.status(task_id)
history = await host.history(task_id)
await host.wait(task_id)
await host.cancel(task_id)
```

This is the level that should remain stable. CLI and HTTP are projections of this API.

## Built-in Tool Families

- `delegate`: create a child agent flow through the host/runtime boundary.
- `execute_bash`: create an interactive terminal flow when Docker is available.
- `inspect_background`, `wait_background`, `cancel_background`, `input_background`, `defer_background`: host-level operations over live flow ids.
- `web_search`, `read_file`, `edit_file`, `delete_file`, `sleep`, `view_image`: ordinary tools that do not require host-level control.

## Design Notes

- Root flow id should be the task id.
- The implementation should prefer one truth source for runtime state, not mirrored host-side caches.
