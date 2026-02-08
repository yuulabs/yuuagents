# AGENTS.md — yuuagents Development Guide

## Build & Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run the daemon
uv run yagents start

# Submit a task
uv run yagents run --persona "coder" --task "hello world"

# Lint and format code (ruff)
uv run ruff check src/
uv run ruff check --fix src/
uv run ruff format src/

# Type checking (if mypy is configured)
uv run mypy src/

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_agent.py

# Run a single test function
uv run pytest tests/test_agent.py::test_agent_setup -v

# Build package
uv build
```

## Code Style Guidelines

### Python Version & Imports
- Target Python 3.14+
- Always start with `from __future__ import annotations` for PEP 563 postponed annotations
- Import order: stdlib → third-party → local (yuuagents, yuutools, etc.)
- Use `TYPE_CHECKING` for imports that would cause circular dependencies

### Type Hints
- Use modern Python 3.10+ syntax: `str | None` not `Optional[str]`
- Use `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- Return type `-> None` for functions that don't return meaningful values
- Type all function parameters and return values

### Naming Conventions
- **Modules**: snake_case (`agent.py`, `bash.py`)
- **Classes**: PascalCase (`Agent`, `AgentConfig`, `TaskRequest`)
- **Functions/variables**: snake_case (`execute_bash`, `agent_id`)
- **Constants**: UPPER_CASE with underscores (`_DEFAULT_SOCKET`, _FRONTMATTER_RE`)
- **Private**: Leading underscore for internal use (`_step`, `_shutdown`)
- **Type variables**: PascalCase if exposed

### Data Classes
- Use `@define` from `attrs` for mutable domain objects
- Use `msgspec.Struct` with `frozen=True, kw_only=True` for API DTOs
- Use `field(factory=...)` for mutable default values

```python
# Domain model (mutable)
@define
class AgentState:
    history: list[yuullm.Message] = field(factory=list)

# API DTO (immutable)
class TaskRequest(msgspec.Struct, frozen=True, kw_only=True):
    persona: str
    task: str
```

### Tool Definitions
Tools use the `yuutools` framework:

```python
@yt.tool(
    params={"command": "Description"},
    description="Tool description",
)
async def my_tool(
    command: str,
    ctx: str = yt.depends(lambda ctx: ctx.attribute),
) -> str:
    ...
```

### Error Handling
- Use explicit try/except with specific exceptions
- Use `KeyError` for missing dict keys, not generic `Exception`
- Log errors with `loguru` (imported as needed)
- Propagate errors to caller with context

### Async Patterns
- Use `async def` for I/O operations and LLM calls
- Use `await` consistently; don't mix sync/async unnecessarily
- Use `asyncio.Queue` for inter-task communication
- Use `asyncio.create_task()` for background work

### Formatting
- 4-space indentation (no tabs)
- 88-100 character line length (ruff default)
- Docstrings in triple quotes for modules, classes, and functions
- Use `match/case` for structural pattern matching (Python 3.10+)

### API Design
- RESTful routes using Starlette
- Return JSON via msgspec encoder for performance
- Use proper HTTP status codes (201 for created, 404 for not found)
- Unix Domain Socket for CLI-daemon communication

### Testing
- Use pytest
- Test files: `tests/test_*.py`
- Use `pytest-asyncio` for async tests
- Mock external services (LLM, Docker) in unit tests

## Project Structure

```
src/yuuagents/
├── __init__.py       # Public API exports
├── agent.py          # Core Agent class
├── types.py          # msgspec DTOs
├── config.py         # TOML configuration
├── context.py        # AgentContext for DI
├── loop.py           # Main agent execution loop
├── cli/              # CLI commands
├── daemon/           # HTTP server & managers
├── tools/            # Built-in tools
└── skills/           # Skill discovery
```

## Dependencies

Key external packages:
- `attrs` — class definitions
- `msgspec` — JSON serialization
- `click` — CLI framework
- `starlette` + `uvicorn` — HTTP server
- `httpx` — HTTP client
- `aiodocker` — Docker integration
- `yuutools`, `yuullm`, `yuutrace` — Internal yuu packages

Always prefer `msgspec` over `pydantic` for performance. Prefer `httpx` over `requests` for async support.
