"""Skills sub-package.

This package provides Agent Skills discovery functionality
using the agentskills/skills-ref standard library.
"""

from skills_ref import validate as validate  # type: ignore[import-untyped]
from skills_ref.models import SkillProperties as SkillProperties  # type: ignore[import-untyped]

from yuuagents.skills.discovery import render as render, scan as scan

__all__ = [
    "scan",
    "render",
    "validate",
    "SkillProperties",
]
