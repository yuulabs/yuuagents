"""Agent Skills discovery — scan directories for SKILL.md files.

This module uses the agentskills/skills-ref standard library for parsing
and validating Agent Skills according to the official specification.
"""

from __future__ import annotations

from pathlib import Path

from skills_ref import read_properties  # type: ignore[import-untyped]
from skills_ref.errors import ParseError, ValidationError  # type: ignore[import-untyped]

from yuuagents.types import SkillInfo


def scan(paths: list[str]) -> list[SkillInfo]:
    """Scan directories for Agent Skills.

    Scans each path for subdirectories containing SKILL.md files,
    parsing their YAML frontmatter according to the agentskills spec.

    Args:
        paths: List of directory paths to scan

    Returns:
        List of discovered SkillInfo objects
    """
    skills: list[SkillInfo] = []

    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue

        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue

            info = _parse(skill_dir)
            if info:
                skills.append(info)

    return skills


def _parse(skill_dir: Path) -> SkillInfo | None:
    """Parse a skill directory using agentskills library.

    Args:
        skill_dir: Path to the skill directory

    Returns:
        SkillInfo if valid skill found, None otherwise
    """
    try:
        props = read_properties(skill_dir)
    except ParseError, ValidationError:
        return None

    # Find SKILL.md location
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        skill_md = skill_dir / "skill.md"

    return SkillInfo(
        name=props.name,
        description=props.description,
        location=str(skill_md),
    )


def render(skills: list[SkillInfo]) -> str:
    """Generate ``<available_skills>`` XML for system prompt injection.

    Args:
        skills: List of SkillInfo objects

    Returns:
        XML string with <available_skills> block
    """
    if not skills:
        return ""

    # Build XML from SkillInfo objects directly
    lines = ["<available_skills>"]
    for s in skills:
        lines.append("<skill>")
        lines.append(f"<name>{s.name}</name>")
        lines.append(f"<description>{s.description}</description>")
        lines.append(f"<location>{s.location}</location>")
        lines.append("</skill>")
    lines.append("</available_skills>")
    lines.append("")
    lines.append(
        "使用 execute_skill_cli 工具执行上述 Skill 提供的 CLI 命令。\n"
        "⚠️ 首次调用某个 Skill 的命令前，必须先用 execute_skill_cli 执行 "
        "`cat <location>` 阅读对应的 SKILL.md，确认参数格式后再调用。不要猜测参数。"
    )
    return "\n".join(lines)
