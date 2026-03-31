#!/usr/bin/env python3
"""YAgents 交互式 service-mode 入门引导."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import yaml


class SetupConfig(TypedDict):
    provider_key: str
    api_type: str
    api_key_env: str
    api_key: str
    base_url: str
    model: str
    tavily_key: str


def _load_rich() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        console_mod = importlib.import_module("rich.console")
        panel_mod = importlib.import_module("rich.panel")
        prompt_mod = importlib.import_module("rich.prompt")
        progress_mod = importlib.import_module("rich.progress")
        table_mod = importlib.import_module("rich.table")
        rich_mod = importlib.import_module("rich")
    except ImportError:
        print("请先安装 rich: pip install rich")
        sys.exit(1)
    return (
        console_mod.Console,
        panel_mod.Panel,
        prompt_mod.Confirm,
        prompt_mod.Prompt,
        progress_mod.Progress,
        progress_mod.SpinnerColumn,
        progress_mod.TextColumn,
        table_mod.Table,
        rich_mod.box,
    )


Console, Panel, Confirm, Prompt, Progress, SpinnerColumn, TextColumn, Table, box = _load_rich()
console = Console()

YAGENTS_HOME = Path("~/.yagents").expanduser()
CONFIG_PATH = YAGENTS_HOME / "config.yaml"


def print_header() -> None:
    """打印欢迎标题"""
    console.print(
        Panel.fit(
            "[bold cyan]YAgents 交互式 Service Mode 引导[/bold cyan]\n"
            "[dim]daemon / CLI onboarding[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def check_prerequisites() -> None:
    """检查必要的先决条件"""
    console.print("[bold]步骤 0: 检查先决条件[/bold]\n")

    # 检查 yagents 是否安装
    try:
        result = subprocess.run(
            ["yagents", "--version"], capture_output=True, text=True
        )
        console.print("  [green]✓[/green] yagents 已安装")
    except FileNotFoundError:
        console.print("  [red]✗ yagents 未安装[/red]")
        console.print("\n[yellow]请先安装 service mode 依赖:[/yellow]")
        console.print("  pip install -e '.[all]'")
        sys.exit(1)

    # 检查 Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        console.print(f"  [green]✓[/green] Docker 已安装: {result.stdout.strip()}")
    except FileNotFoundError:
        console.print("  [red]✗ Docker 未安装[/red]")
        console.print("\n[yellow]请先安装 Docker:[/yellow]")
        console.print("  https://docs.docker.com/engine/install/")
        sys.exit(1)

    console.print()


def configure_provider() -> SetupConfig:
    """配置 LLM Provider"""
    console.print("[bold]步骤 1: 配置 LLM Provider[/bold]\n")

    providers = {
        "1": (
            "OpenAI",
            "openai-chat-completion",
            "OPENAI_API_KEY",
            "https://api.openai.com/v1",
            "gpt-4o",
        ),
        "2": (
            "Anthropic",
            "anthropic-messages",
            "ANTHROPIC_API_KEY",
            "https://api.anthropic.com",
            "claude-3-5-sonnet-20241022",
        ),
        "3": (
            "OpenRouter (OpenAI-compatible)",
            "openai-chat-completion",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "gpt-4o",
        ),
        "4": ("自定义", "", "", "", ""),
    }

    console.print("选择 LLM Provider:")
    for key, (name, _, _, _, _) in providers.items():
        console.print(f"  {key}. {name}")

    choice = Prompt.ask("\n请输入选项", choices=list(providers.keys()), default="1")
    provider_label, api_type, api_key_env, default_base_url, default_model = providers[
        choice
    ]

    provider_key = Prompt.ask(
        "\n请输入供应商名字（配置中的 providers.<name>）", default="main"
    )
    provider_key = provider_key.strip() or "main"

    if choice == "4":
        api_type = Prompt.ask(
            "请输入 API 类型",
            choices=[
                "openai-chat-completion",
                "openai-responses",
                "anthropic-messages",
            ],
            default="openai-chat-completion",
        )
        api_key_env = Prompt.ask("请输入 API Key 环境变量名", default="CUSTOM_API_KEY")
        api_key_env = api_key_env.strip() or "CUSTOM_API_KEY"
        default_base_url = Prompt.ask("请输入 API Base URL", default="")
        default_model = Prompt.ask(
            "请输入默认模型名称",
            default="claude-3-5-sonnet-20241022"
            if api_type == "anthropic-messages"
            else "gpt-4o",
        )

    # 获取 API Key
    api_key = Prompt.ask(f"\n请输入 {provider_label} API Key", password=True)

    # 设置环境变量
    os.environ[api_key_env] = api_key

    # 询问 Base URL
    base_url = Prompt.ask(
        "API Base URL (留空使用默认)",
        default=default_base_url if default_base_url else "",
    )
    model = Prompt.ask("请输入模型名称", default=default_model)

    # 询问 Tavily API Key (用于 web_search)
    console.print("\n[yellow]可选配置[/yellow]")
    tavily_key = Prompt.ask(
        "Tavily API Key (用于网络搜索，留空跳过)", password=True, default=""
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    console.print()

    return {
        "provider_key": provider_key,
        "api_type": api_type,
        "api_key_env": api_key_env,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "tavily_key": tavily_key,
    }


def setup_yagents(config: SetupConfig) -> None:
    """运行 yagents install"""
    console.print("[bold]步骤 2: 运行 yagents install (service mode)[/bold]\n")

    # 设置 API key 环境变量供 setup 使用
    os.environ[config["api_key_env"]] = config["api_key"]

    # 创建 overrides 文件
    overrides: dict[str, Any] = {
        "providers": {
            config["provider_key"]: {
                "api_type": config["api_type"],
                "api_key_env": config["api_key_env"],
                "default_model": config["model"],
                "base_url": config["base_url"] if config["base_url"] else "",
            }
        },
        "agents": {
            "main": {
                "provider": config["provider_key"],
                "model": config["model"],
                "persona": "You are a senior software engineer. You write clean, well-tested code. You prefer simple solutions and avoid over-engineering.",
                "tools": [
                    "execute_bash",
                    "read_file",
                    "edit_file",
                    "delete_file",
                    "web_search",
                ],
            }
        },
    }

    if config["tavily_key"]:
        overrides["tavily"] = {"api_key_env": "TAVILY_API_KEY"}

    overrides_path = Path("config.overrides.yaml")
    overrides_path.write_text(
        yaml.dump(overrides, default_flow_style=False, allow_unicode=True)
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("运行 yagents install...", total=None)

        try:
            result = subprocess.run(
                ["yagents", "install"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            progress.update(task, completed=True)

            if result.returncode == 0:
                console.print("  [green]✓[/green] install 完成")
            else:
                console.print("  [red]✗ install 失败[/red]")
                console.print(result.stderr)
                sys.exit(1)
        except subprocess.TimeoutExpired:
            console.print("  [red]✗ install 超时[/red]")
            sys.exit(1)

    # 清理临时文件
    overrides_path.unlink(missing_ok=True)

    console.print()


def start_daemon() -> None:
    """启动 daemon"""
    console.print("[bold]步骤 3: 启动 yagents daemon[/bold]\n")

    # 检查 daemon 是否已在运行
    try:
        subprocess.run(
            ["yagents", "down"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass

    # 启动 daemon (后台)
    console.print("启动 daemon...")
    result = subprocess.run(
        ["yagents", "up", "-d"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        console.print("  [red]✗[/red] daemon 启动失败")
        console.print(result.stderr or result.stdout)
        sys.exit(1)

    # 等待 daemon 启动
    time.sleep(2)

    # 检查是否成功启动
    try:
        subprocess.run(
            ["yagents", "list"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        console.print("  [green]✓[/green] daemon 已启动")
    except Exception:
        console.print("  [yellow]![/yellow] daemon 可能正在启动中，请稍后手动检查")

    console.print()


def run_first_task(config: SetupConfig) -> None:
    """运行第一个示例任务"""
    console.print("[bold]步骤 4: 运行第一个示例任务[/bold]\n")

    console.print("现在让我们运行一个简单的任务来验证配置。\n")

    # 选择示例任务
    examples = [
        (
            "简单的问候任务",
            "Say hello and introduce yourself briefly",
        ),
        (
            "文件操作",
            'Create a file named /tmp/hello.txt with content "Hello from YAgents!" and read it back',
        ),
        (
            "代码生成",
            "Write a Python function to calculate fibonacci numbers and save it to /tmp/fib.py",
        ),
    ]

    console.print("选择一个示例任务:")
    for i, (desc, _) in enumerate(examples, 1):
        console.print(f"  {i}. {desc}")
    console.print(f"  {len(examples) + 1}. 自定义任务")

    choices = [str(i) for i in range(1, len(examples) + 2)]
    choice = Prompt.ask("\n请输入选项", choices=choices, default="1")

    if int(choice) > len(examples):
        task = Prompt.ask("请输入任务描述")
    else:
        _, task = examples[int(choice) - 1]

    cmd = ["yagents", "run", "--task", task]

    console.print(f"\n[yellow]执行:[/yellow] {' '.join(cmd)}\n")

    # 执行任务
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            match = re.search(r"Task started:\s+(\S+)", result.stdout)
            task_id = match.group(1) if match else result.stdout.strip()
            console.print(
                f"  [green]✓[/green] 任务已提交，Task ID: [cyan]{task_id}[/cyan]\n"
            )

            # 显示监控提示
            console.print("你可以使用以下命令监控任务执行:")
            console.print("  [dim]yagents list              # 列出所有任务[/dim]")
            console.print(f"  [dim]yagents status {task_id}   # 查看详细状态[/dim]")
            console.print(f"  [dim]yagents logs {task_id}     # 查看对话历史[/dim]")

            # 等待几秒显示状态
            console.print("\n[bold]等待 5 秒后检查任务状态...[/bold]\n")
            time.sleep(5)

            status_result = subprocess.run(
                ["yagents", "status", task_id],
                capture_output=True,
                text=True,
            )

            if status_result.returncode == 0:
                import json

                try:
                    status = json.loads(status_result.stdout)
                    table = Table(title="Agent 状态", box=box.ROUNDED)
                    table.add_column("属性", style="cyan")
                    table.add_column("值", style="green")

                    table.add_row("Task ID", task_id)
                    table.add_row("状态", status.get("status", "unknown"))
                    table.add_row("步骤数", str(status.get("steps", 0)))
                    table.add_row("总成本", f"${status.get('total_cost_usd', 0):.4f}")

                    console.print(table)
                except json.JSONDecodeError:
                    console.print(status_result.stdout)
        else:
            console.print("  [red]✗[/red] 任务提交失败")
            console.print(result.stderr)
    except subprocess.TimeoutExpired:
        console.print("  [yellow]![/yellow] 任务提交超时")
    except Exception as e:
        console.print(f"  [red]✗[/red] 错误: {e}")

    console.print()


def print_next_steps() -> None:
    """打印后续步骤"""
    console.print("[bold]完成！[/bold]\n")

    console.print(
        Panel(
            "[bold green]🎉 YAgents 入门引导完成！[/bold green]\n\n"
            "[bold]常用命令:[/bold]\n"
            '  yagents run --task "你的任务"     提交新任务\n'
            "  yagents list                      列出所有任务\n"
            "  yagents status <task_id>          查看任务状态\n"
            "  yagents logs <task_id>            查看对话历史\n"
            "  yagents stop <task_id>            取消任务\n\n"
            "[bold]配置:[/bold]\n"
            "  配置文件位置: ~/.yagents/config.yaml\n"
            "  yagents config                    查看当前配置\n\n"
            "[bold]更多信息:[/bold]\n"
            "  yagents --help                    查看所有命令\n",
            border_style="green",
        )
    )


def main() -> None:
    """主函数"""
    print_header()

    # 检查先决条件
    check_prerequisites()

    # 配置 Provider
    config = configure_provider()

    # 运行 install
    setup_yagents(config)

    # 启动 daemon
    start_daemon()

    # 运行第一个任务
    if Confirm.ask("是否现在运行示例任务?", default=True):
        run_first_task(config)

    # 打印后续步骤
    print_next_steps()


if __name__ == "__main__":
    main()
