"""Prompt fragments for agent personas."""

from __future__ import annotations

from importlib import resources


def load(name: str) -> str:
    """Load a prompt fragment by name (without extension)."""
    ref = resources.files(__package__).joinpath(f"{name}.md")
    return ref.read_text(encoding="utf-8")


# Pre-defined variable mappings for {var} substitution in personas
PROMPT_VARS: dict[str, str] = {}


def get_vars() -> dict[str, str]:
    """Return prompt variables, lazily loading on first call."""
    return PROMPT_VARS
