#!/usr/bin/env python3
"""YAgents service-mode 入门引导 (无 rich 依赖)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import yaml


class SetupConfig(TypedDict):
    provider_name: str
    api_key_env: str
    api_key: str
    base_url: str
    model: str
    tavily_key: str


YAGENTS_HOME = Path("~/.yagents").expanduser()
CONFIG_PATH = YAGENTS_HOME / "config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_PATH = PROJECT_ROOT / "config.overrides.yaml"


def run_yagents(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["yagents", *args],
        cwd=PROJECT_ROOT,
        **kwargs,
    )


def print_banner() -> None:
    """打印横幅"""
    print("=" * 60)
    print("  YAgents Service Mode 入门引导")
    print("  daemon / CLI onboarding")
    print("=" * 60)
    print()


def check_prerequisites() -> None:
    """检查先决条件"""
    print("步骤 0: 检查先决条件\n")

    # 检查 yagents
    try:
        result = run_yagents(["--version"], capture_output=True, text=True, check=True)
        print("  ✓ yagents 已安装")
        print(f"    {result.stdout.strip()}")
    except FileNotFoundError, subprocess.CalledProcessError:
        print("  ✗ yagents 未安装")
        print("\n请先在仓库里执行: uv sync --extra docker --extra web")
        print("或安装: pip install 'yuuagents[docker,web]'")
        sys.exit(1)

    # 检查 Docker
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, check=True
        )
        print(f"  ✓ Docker 已安装: {result.stdout.strip()}")
    except FileNotFoundError, subprocess.CalledProcessError:
        print("  ✗ Docker 未安装")
        print("\n请先安装: https://docs.docker.com/engine/install/")
        sys.exit(1)

    print()


def configure_provider() -> SetupConfig:
    """配置 Provider"""
    print("步骤 1: 配置 LLM Provider\n")

    print("选择 Provider:")
    print("  1. OpenAI")
    print("  2. Anthropic")
    print("  3. OpenRouter")
    print("  4. 自定义")

    choice = input("\n请输入选项 (1-4) [1]: ").strip() or "1"

    providers = {
        "1": ("openai-default", "OPENAI_API_KEY", "gpt-4o"),
        "2": ("anthropic-default", "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
        "3": ("openrouter-default", "OPENROUTER_API_KEY", "openai/gpt-4o"),
        "4": ("custom", "CUSTOM_API_KEY", ""),
    }

    provider_name, api_key_env, default_model = providers[choice]

    api_key = input("\n请输入 API Key: ").strip()
    os.environ[api_key_env] = api_key

    if choice == "4":
        base_url = input("请输入 Base URL: ").strip()
        model = input(f"请输入模型名称 [{default_model}]: ").strip() or default_model
    else:
        base_url = ""
        model = default_model

    # 可选 Tavily
    tavily = input("\nTavily API Key (用于搜索，留空跳过): ").strip()
    if tavily:
        os.environ["TAVILY_API_KEY"] = tavily

    print()

    return {
        "provider_name": provider_name,
        "api_key_env": api_key_env,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "tavily_key": tavily,
    }


def setup_yagents(config: SetupConfig) -> None:
    """运行 install"""
    print("步骤 2: 运行 yagents install (service mode)\n")

    # 创建 overrides
    overrides: dict[str, Any] = {
        "providers": {
            config["provider_name"]: {
                "api_type": "anthropic-messages"
                if "anthropic" in config["provider_name"].lower()
                else "openai-chat-completion",
                "api_key_env": config["api_key_env"],
                "default_model": config["model"],
                "base_url": config["base_url"],
            }
        },
        "agents": {
            "main": {
                "provider": config["provider_name"],
                "model": config["model"],
                "persona": "You are a helpful assistant.",
                "tools": ["execute_bash", "read_file", "edit_file", "web_search"],
            }
        },
    }

    if config["tavily_key"]:
        overrides["tavily"] = {"api_key_env": "TAVILY_API_KEY"}

    OVERRIDES_PATH.write_text(
        yaml.dump(overrides, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("运行 install...")
    result = run_yagents(
        ["install", "--project-dir", str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
    )

    OVERRIDES_PATH.unlink(missing_ok=True)

    if result.returncode == 0:
        print("  ✓ install 完成")
    else:
        print("  ✗ install 失败")
        print(result.stderr)
        sys.exit(1)

    print()


def start_daemon() -> None:
    """启动 daemon"""
    print("步骤 3: 启动 daemon\n")

    # 尝试停止已有的
    try:
        run_yagents(["down"], capture_output=True, timeout=5)
    except Exception:
        pass

    # 启动 daemon
    print("启动 daemon (后台)...")
    result = run_yagents(
        ["up", "-d"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print("  ✗ daemon 启动失败")
        print(result.stderr or result.stdout)
        sys.exit(1)

    time.sleep(2)
    print("  ✓ daemon 已启动")
    print()


def run_example_task(config: SetupConfig) -> None:
    """运行示例任务"""
    print("步骤 4: 运行示例任务\n")

    if input("运行示例任务? (y/n) [y]: ").strip().lower() == "n":
        return

    task = input("\n输入任务描述 (或按 Enter 使用默认): ").strip()
    if not task:
        task = "Say hello and introduce yourself briefly"

    print(f'\n执行: yagents run --task "{task}"\n')

    result = run_yagents(
        ["run", "--task", task],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode == 0:
        match = re.search(r"Task started:\s+(\S+)", result.stdout)
        task_id = match.group(1) if match else result.stdout.strip()
        print(f"  ✓ 任务已提交，Task ID: {task_id}")
        print(f"\n查看状态: yagents status {task_id}")
        print(f"查看日志: yagents logs {task_id}")
    else:
        print(f"  ✗ 失败: {result.stderr}")

    print()


def print_summary() -> None:
    """打印总结"""
    print("=" * 60)
    print("完成！")
    print("=" * 60)
    print()
    print("常用命令:")
    print('  yagents run --task "你的任务"')
    print("  yagents list")
    print("  yagents status <task_id>")
    print("  yagents logs <task_id>")
    print("  yagents stop <task_id>")
    print()
    print("配置: ~/.yagents/config.yaml")
    print(f"项目模板: {PROJECT_ROOT / 'config.example.yaml'}")
    print("说明: 这个脚本面向 daemon / service mode，不是纯 SDK quick start。")
    print()


def main() -> None:
    print_banner()
    check_prerequisites()
    config = configure_provider()
    setup_yagents(config)
    start_daemon()
    run_example_task(config)
    print_summary()


if __name__ == "__main__":
    main()
