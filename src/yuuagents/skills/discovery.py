"""Agent Skills discovery — scan directories for SKILL.md files."""

from __future__ import annotations

import re
from pathlib import Path

from yuuagents.types import SkillInfo

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def scan(paths: list[str]) -> list[SkillInfo]:
    """Scan directories for SKILL.md and parse YAML frontmatter."""
    skills: list[SkillInfo] = []
    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            md = skill_dir / "SKILL.md"
            if not md.exists():
                continue
            info = _parse(md)
            if info:
                skills.append(info)
    return skills


def _parse(path: Path) -> SkillInfo | None:
    """Extract name + description from SKILL.md YAML frontmatter."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None

    # Minimal YAML parsing — avoid pyyaml dependency for this
    fm = m.group(1)
    name = ""
    description = ""
    for line in fm.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[5:].strip().strip("\"'")
        elif line.startswith("description:"):
            description = line[12:].strip().strip("\"'")

    if not name:
        name = path.parent.name

    return SkillInfo(
        name=name,
        description=description,
        location=str(path),
    )


def render(skills: list[SkillInfo]) -> str:
    """Generate ``<available_skills>`` XML for system prompt injection."""
    if not skills:
        return ""
    lines = ["<available_skills>"]
    for s in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{s.name}</name>")
        lines.append(f"    <description>{s.description}</description>")
        lines.append(f"    <location>{s.location}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)
