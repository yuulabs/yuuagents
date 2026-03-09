"""read_skill — read a skill's documentation, stripping YAML frontmatter."""

from __future__ import annotations

import re
from pathlib import Path

import yuutools as yt

from yuuagents.context import AgentContext

_YAML_FENCE_RE = re.compile(r"^---\s*\n.*?^---\s*\n", re.DOTALL | re.MULTILINE)


def _strip_frontmatter(text: str) -> str:
    return _YAML_FENCE_RE.sub("", text, count=1).lstrip()


def _find_skill_md(name: str, paths: list[str]) -> Path | None:
    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        for candidate in (root / name / "SKILL.md", root / name / "skill.md"):
            if candidate.is_file():
                return candidate
    return None


@yt.tool(
    params={"name": "Skill name (e.g. 'mem', 'im', 'web')"},
    description="读取指定 skill 的使用文档。返回去掉 YAML header 的 Markdown 内容。",
)
def read_skill(
    name: str,
    ctx: AgentContext = yt.depends(lambda ctx: ctx),
) -> str:
    md = _find_skill_md(name, ctx.skill_paths)
    if md is None:
        available = _list_available(ctx.skill_paths)
        hint = f"可用的 skill: {', '.join(available)}" if available else "未找到任何 skill"
        return f"[ERROR] skill '{name}' 不存在。{hint}"
    text = md.read_text(encoding="utf-8")
    return _strip_frontmatter(text)


def _list_available(paths: list[str]) -> list[str]:
    names: list[str] = []
    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "SKILL.md").is_file() or (d / "skill.md").is_file():
                names.append(d.name)
    return names
