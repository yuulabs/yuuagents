# Repository Guidelines

## Project Structure & Module Organization

`yuuagents/` is a standalone Python package in the workspace. Source code lives in `src/yuuagents/`, with core execution logic in `core/`, CLI entry points in `cli/`, daemon code in `daemon/`, built-in tools in `tools/`, and runtime/config types in files such as `agent.py`, `context.py`, `types.py`, and `init.py`. Tests live under `tests/`, examples under `examples/`, and repository-specific guidance is in `README.md`, `config.example.yaml`, and `design/`.

## Build, Test, and Development Commands

Use `uv` for all local work:

```bash
uv sync
cd yuuagents && uv run pytest
cd yuuagents && uv run pytest tests/test_flow.py -v
cd yuuagents && uv run ruff check src/ tests/
cd yuuagents && uv run ruff format src/ tests/
cd yuuagents && uv run mypy src/
cd yuuagents && uv build
```

The CLI entry point is `yagents`. Common commands include `uv run yagents up`, `uv run yagents run --agent main --task "..."`, `uv run yagents list`, and `uv run yagents down`.

## Coding Style & Naming Conventions

Target Python 3.14+, use 4-space indentation, and prefer `from __future__ import annotations` in new modules. Keep imports ordered `stdlib -> third-party -> local`. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Favor `attrs` for mutable runtime objects, `msgspec.Struct` for typed payloads, and explicit async APIs for I/O and agent execution.

## Testing Guidelines

This repo uses `pytest` with `pytest-asyncio`. Name tests `tests/test_*.py` and keep coverage close to the code you change. Mock external systems such as Docker, LLM providers, and network calls in unit tests; reserve live integration checks for cases that genuinely need them. When changing CLI, daemon, or tool behavior, add a focused regression test in the matching test module.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects with conventional prefixes such as `feat:`, `fix:`, `refactor:`, and `chore:`. Keep commits scoped to one logical change. PRs should state the affected package, summarize behavior changes, list verification commands, and call out any config or migration impact. Include screenshots only when the change affects user-facing output.
