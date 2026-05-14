import asyncio
import datetime
from pathlib import Path

import questionary
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from chorus.graph import graph
from chorus.infrastructure.config import LLM_API_KEY, DB_PATH, TAVILY_API_KEY
from chorus.infrastructure.dao import load_value_vector, save_value_vector
from chorus.utils.constants import SCHWARTZ_DIM_LABELS, SCHWARTZ_DIMS

app = typer.Typer(name="chorus", add_completion=False)
console = Console()

_NODE_LABELS: dict[str, str] = {
    "intake_node":              "加载用户档案",
    "decision_classifier_node": "分类决策场景",
    "bias_detection_node":      "检测认知偏差",
    "reality_node":             "现实核验",
    "agent_node":               "Agent 初评",
    "entropy_monitor_node":     "熵值监测",
    "debate_node":              "辩论轮次",
    "consensus_node":           "形成共识",
    "persona_updater_node":     "更新用户档案",
}


def _check_config() -> None:
    missing = [k for k, v in [
        ("LLM_API_KEY", LLM_API_KEY),
        ("TAVILY_API_KEY",       TAVILY_API_KEY),
    ] if not v]
    if missing:
        console.print(Panel(
            f"[red]缺少必要的环境变量：{', '.join(missing)}[/red]\n"
            "[dim]请在 .env 文件中配置，或通过环境变量传入。[/dim]",
            border_style="red",
        ))
        raise typer.Exit(1)


def _collect_schwartz_scores() -> dict[str, float]:
    import json

    console.print(Panel(
        "[bold]首次使用，请为自己建立价值观档案[/bold]\n"
        "[dim]以下 10 个维度来自 Schwartz 价值观理论，每项打分 0.0–1.0[/dim]",
        border_style="blue",
    ))

    mode = questionary.select(
        "输入方式：",
        choices=["粘贴 JSON", "逐项输入"],
    ).ask()
    if mode is None:
        raise typer.Exit(1)

    if mode == "粘贴 JSON":
        console.print("[dim]粘贴后以空行（回车两次）确认[/dim]")
        while True:
            console.print("[bold yellow]JSON[/bold yellow]：", end="")
            lines: list[str] = []
            try:
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
                    # Auto-accept if accumulated text is already valid JSON
                    try:
                        json.loads("\n".join(lines))
                        break
                    except json.JSONDecodeError:
                        pass
            except (EOFError, KeyboardInterrupt):
                raise typer.Exit(1)

            if not lines:
                console.print("[yellow]未输入内容，请重新粘贴[/yellow]")
                continue
            try:
                data = json.loads("\n".join(lines))
                missing = [d for d in SCHWARTZ_DIMS if d not in data]
                if missing:
                    console.print(f"[yellow]缺少维度：{', '.join(missing)}，缺失项将默认 0.5[/yellow]")
                return {d: max(0.0, min(1.0, float(data.get(d, 0.5)))) for d in SCHWARTZ_DIMS}
            except (json.JSONDecodeError, ValueError):
                console.print("[yellow]JSON 格式有误，请重新粘贴[/yellow]")

    # 逐项输入
    scores: dict[str, float] = {}
    for dim in SCHWARTZ_DIMS:
        label = SCHWARTZ_DIM_LABELS[dim]
        while True:
            raw = questionary.text(f"  {label} ({dim})  [0.0–1.0]", default="0.5").ask()
            if raw is None:
                raise typer.Exit(1)
            try:
                val = float(raw)
                if 0.0 <= val <= 1.0:
                    scores[dim] = val
                    break
            except ValueError:
                pass
            console.print("[yellow]  请输入 0.0 到 1.0 之间的数字[/yellow]")
    return scores


def _collect_options() -> list[str]:
    options: list[str] = []
    console.print("[dim]逐行输入候选选项，输入空行结束（至少 2 个）：[/dim]")
    i = 1
    while True:
        opt = questionary.text(f"  选项 {i}").ask()
        if opt is None or opt.strip() == "":
            if len(options) < 2:
                console.print("[yellow]  至少需要 2 个选项[/yellow]")
                continue
            break
        options.append(opt.strip())
        i += 1
    return options


async def _stream_graph(username: str, narrative: str, options: list[str]) -> str:
    initial_state = {
        "username":         username,
        "user_narrative":   narrative,
        "decision_options": options,
    }
    final_report = ""
    debate_round = 0

    console.print()
    console.print(Rule("开始分析", style="bold green"))

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node, update in chunk.items():
            label = _NODE_LABELS.get(node, node)
            if node == "debate_node":
                debate_round += 1
                label = f"{label}（第 {debate_round} 轮）"
            console.print(f"  [green]✓[/green] {label}")
            if "final_report" in update:
                final_report = update["final_report"]

    return final_report


@app.command()
def analyze() -> None:
    """多 Agent 心理决策分析。"""
    console.print(Panel(
        "[bold green]Chorus — 多 Agent 心理决策系统[/bold green]",
        border_style="green",
        padding=(1, 4),
    ))

    _check_config()

    # Ensure DB parent directory exists before any DB operations
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    username = questionary.text("用户名（用于加载/保存价值观档案）：").ask()
    if not username or not username.strip():
        console.print("[red]用户名不能为空。[/red]")
        raise typer.Exit(1)
    username = username.strip()

    value_vector = load_value_vector(username)
    if value_vector is None:
        value_vector = _collect_schwartz_scores()
        save_value_vector(username, value_vector)
        console.print("[green]✓ 价值观档案已保存。[/green]\n")

    narrative = questionary.text("请描述你的决策情境：").ask()
    if not narrative or not narrative.strip():
        console.print("[red]决策描述不能为空。[/red]")
        raise typer.Exit(1)

    options = _collect_options()

    try:
        final_report = asyncio.run(_stream_graph(username, narrative.strip(), options))
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")
        raise typer.Exit(0)

    console.print()
    console.print(Rule("分析结果", style="bold cyan"))

    if final_report:
        console.print(Markdown(final_report))
    else:
        console.print("[yellow]未能生成报告，请检查日志。[/yellow]")
        raise typer.Exit(1)

    save = questionary.confirm("保存报告为 Markdown 文件？", default=True).ask()
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(DB_PATH).parent / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{username}_{timestamp}.md"
        out_file.write_text(final_report, encoding="utf-8")
        console.print(f"[green]✓ 已保存：[/green]{out_file.resolve()}")


def main() -> None:
    app()
