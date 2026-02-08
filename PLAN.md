# yuuagents Implementation Plan (Phase 1)

## Overview
Build yuuagents Phase 1 per design.md: Agent core, builtin tools, skills discovery, daemon, CLI, REST API.

## Task Breakdown

### 1. pyproject.toml — Add dependencies
Update pyproject.toml with all Phase 1 dependencies (yuutools, yuullm, yuutrace, attrs, msgspec, starlette, uvicorn, click, httpx, aiodocker, tavily-python). Add `[project.scripts]` entry for `yagents`.

### 2. Core types — `types.py`
Define `AgentStatus` (enum), `AgentInfo`, `TaskRequest` as `msgspec.Struct`. These are the API data transfer types.

### 3. Context — `context.py`
Define `AgentContext` with `attrs.define`: agent_id, workdir, docker_container, input_queue, tavily_api_key.

### 4. Agent — `agent.py`
Define `Agent` with `attrs.define`: agent_id, persona, tools (ToolManager), llm (YLLMClient), history, skills_xml, status. Add `full_system_prompt` property, `setup()`, `done()` methods.

### 5. Step loop — `loop.py`
Implement `run_agent(agent, task)` and `_step(agent, chat)`. Integrates yuullm streaming + yuutrace conversation/llm_gen/tools context managers. Collects StreamItems, executes tool calls via ToolManager, records usage/cost.

### 6. Config — `config.py`
Parse `~/.config/yagents/config.toml`. Sections: daemon, docker, llm, skills, tavily, personas. Use tomllib (stdlib 3.11+).

### 7. Builtin tools — `tools/`
- `tools/bash.py`: `execute_bash` — docker exec via aiodocker
- `tools/file.py`: `read_file`, `write_file`, `delete_file` — docker exec
- `tools/web.py`: `web_search` — Tavily API
- `tools/__init__.py`: BUILTIN_TOOLS registry, `get_tools(names)` helper

### 8. Skills discovery — `skills/discovery.py`
Scan configured directories for SKILL.md files, parse YAML frontmatter, generate `<available_skills>` XML string.

### 9. Docker manager — `daemon/docker.py`
`DockerManager` with `ensure_container()`, `exec()`, `cleanup()`. Uses aiodocker.

### 10. Agent manager — `daemon/manager.py`
`AgentManager` with `submit()`, `list_agents()`, `get_status()`, `get_history()`, `respond()`, `cancel()`. Manages asyncio.Tasks.

### 11. REST API — `daemon/api.py`
Starlette routes: POST/GET /api/agents, GET /api/agents/{id}, GET /api/agents/{id}/history, DELETE /api/agents/{id}, GET /api/skills, POST /api/skills/scan, GET /health.

### 12. Daemon server — `daemon/server.py`
`start_daemon(config)`: create Starlette app, bind to Unix Domain Socket via uvicorn.

### 13. CLI — `cli/`
- `cli/client.py`: `YAgentsClient` thin HTTP client over UDS (httpx)
- `cli/main.py`: click group with subcommands: start, stop, run, list, status, logs, stop-agent, input, skills

### 14. `__init__.py` — Public API
Re-export yuutools symbols + yuuagents own symbols per design.

## File Creation Order
1. pyproject.toml (update)
2. src/yuuagents/types.py
3. src/yuuagents/context.py
4. src/yuuagents/config.py
5. src/yuuagents/agent.py
6. src/yuuagents/tools/bash.py
7. src/yuuagents/tools/file.py
8. src/yuuagents/tools/web.py
9. src/yuuagents/tools/__init__.py
10. src/yuuagents/loop.py
11. src/yuuagents/skills/__init__.py
12. src/yuuagents/skills/discovery.py
13. src/yuuagents/daemon/docker.py
14. src/yuuagents/daemon/manager.py
15. src/yuuagents/daemon/api.py
16. src/yuuagents/daemon/server.py
17. src/yuuagents/daemon/__init__.py
18. src/yuuagents/cli/client.py
19. src/yuuagents/cli/main.py
20. src/yuuagents/cli/__init__.py
21. src/yuuagents/__init__.py (update)
