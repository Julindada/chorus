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
from chorus.utils.constants import SCHWARTZ_DIM_LABELS, SCHWARTZ_DIMS, SCHWARTZ_QUESTIONNAIRE_MAPPING

app = typer.Typer(name="chorus", add_completion=False)
console = Console()

# ── bilingual UI strings ──────────────────────────────────────────────────────

_S: dict[str, dict[str, str]] = {
    "zh": {
        "title":               "Chorus — 多 Agent 心理决策系统",
        "missing_env":         "[red]缺少必要的环境变量：{keys}[/red]\n[dim]请在 .env 文件中配置，或通过环境变量传入。[/dim]",
        "first_use_panel":     "[bold]首次使用，请为自己建立价值观档案[/bold]\n[dim]以下 10 个维度来自 Schwartz 价值观理论，每项打分 0.0–1.0[/dim]",
        "input_mode":          "输入方式：",
        "mode_questionnaire":  "问卷答案（推荐）",
        "mode_json":           "粘贴 JSON",
        "mode_manual":         "逐项输入",
        "q_hint":              "[dim]逐行输入每道题的答案（1–6），也可一次性粘贴空格分隔的 21 个数字：[/dim]",
        "q_paste_prompt":      "  答案（可粘贴全部 21 个，空格分隔）：",
        "q_item_prompt":       "  题 {n:>2} 答案 [1–6]：",
        "q_invalid":           "[yellow]  请输入 1 到 6 之间的整数[/yellow]",
        "q_out_of_range":      "[yellow]  题 {nums} 的答案超出范围，将以 3（中间值）代替[/yellow]",
        "q_result_header":     "\n[bold]计算结果：[/bold]",
        "json_hint":           "[dim]粘贴后以空行（回车两次）确认[/dim]",
        "json_label":          "[bold yellow]JSON[/bold yellow]：",
        "json_empty":          "[yellow]未输入内容，请重新粘贴[/yellow]",
        "json_missing_dims":   "[yellow]缺少维度：{dims}，缺失项将默认 0.5[/yellow]",
        "json_error":          "[yellow]JSON 格式有误，请重新粘贴[/yellow]",
        "manual_range_error":  "[yellow]  请输入 0.0 到 1.0 之间的数字[/yellow]",
        "options_hint":        "[dim]逐行输入候选选项，输入空行结束（至少 2 个）：[/dim]",
        "option_prompt":       "  选项 {i}",
        "options_min_error":   "[yellow]  至少需要 2 个选项[/yellow]",
        "analysis_start":      "开始分析",
        "debate_label":        "辩论轮次（第 {n} 轮）",
        "analysis_result":     "分析结果",
        "no_report":           "[yellow]未能生成报告，请检查日志。[/yellow]",
        "save_confirm":        "保存报告为 Markdown 文件？",
        "save_done":           "[green]✓ 已保存：[/green]{path}",
        "profile_saved":       "[green]✓ 价值观档案已保存。[/green]\n",
        "username_prompt":     "用户名（用于加载/保存价值观档案）：",
        "username_empty":      "[red]用户名不能为空。[/red]",
        "narrative_prompt":    "请描述你的决策情境：",
        "narrative_empty":     "[red]决策描述不能为空。[/red]",
        "interrupted":         "\n[yellow]已中断。[/yellow]",
    },
    "en": {
        "title":               "Chorus — Multi-Agent Psychological Decision System",
        "missing_env":         "[red]Missing required environment variables: {keys}[/red]\n[dim]Set them in .env or pass as environment variables.[/dim]",
        "first_use_panel":     "[bold]First run — please build your value profile[/bold]\n[dim]10 dimensions from Schwartz Basic Human Values Theory, each scored 0.0–1.0[/dim]",
        "input_mode":          "Input method:",
        "mode_questionnaire":  "Questionnaire answers (recommended)",
        "mode_json":           "Paste JSON",
        "mode_manual":         "Enter manually",
        "q_hint":              "[dim]Enter each answer (1–6) one per line, or paste all 21 space-separated on one line:[/dim]",
        "q_paste_prompt":      "  Answers (paste all 21 space-separated, or just Q1):",
        "q_item_prompt":       "  Q{n:>2} answer [1–6]:",
        "q_invalid":           "[yellow]  Please enter an integer between 1 and 6[/yellow]",
        "q_out_of_range":      "[yellow]  Q{nums} out of range — replacing with 3 (midpoint)[/yellow]",
        "q_result_header":     "\n[bold]Computed scores:[/bold]",
        "json_hint":           "[dim]End with a blank line to confirm[/dim]",
        "json_label":          "[bold yellow]JSON[/bold yellow]: ",
        "json_empty":          "[yellow]No input received — please try again[/yellow]",
        "json_missing_dims":   "[yellow]Missing dimensions: {dims} — defaulting to 0.5[/yellow]",
        "json_error":          "[yellow]Invalid JSON format — please try again[/yellow]",
        "manual_range_error":  "[yellow]  Please enter a number between 0.0 and 1.0[/yellow]",
        "options_hint":        "[dim]Enter candidate options one per line, blank line to finish (at least 2):[/dim]",
        "option_prompt":       "  Option {i}",
        "options_min_error":   "[yellow]  At least 2 options required[/yellow]",
        "analysis_start":      "Starting Analysis",
        "debate_label":        "Debate (round {n})",
        "analysis_result":     "Analysis Result",
        "no_report":           "[yellow]No report generated — check logs.[/yellow]",
        "save_confirm":        "Save report as Markdown file?",
        "save_done":           "[green]✓ Saved:[/green] {path}",
        "profile_saved":       "[green]✓ Value profile saved.[/green]\n",
        "username_prompt":     "Username (to load/save your value profile):",
        "username_empty":      "[red]Username cannot be empty.[/red]",
        "narrative_prompt":    "Describe your decision situation:",
        "narrative_empty":     "[red]Decision description cannot be empty.[/red]",
        "interrupted":         "\n[yellow]Interrupted.[/yellow]",
    },
}

_NODE_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "intake_node":              "加载用户档案",
        "decision_classifier_node": "分类决策场景",
        "bias_detection_node":      "检测认知偏差",
        "reality_node":             "现实核验",
        "agent_node":               "Agent 初评",
        "entropy_monitor_node":     "熵值监测",
        "debate_node":              "辩论轮次",
        "consensus_node":           "形成共识",
        "persona_updater_node":     "更新用户档案",
    },
    "en": {
        "intake_node":              "Load user profile",
        "decision_classifier_node": "Classify decision type",
        "bias_detection_node":      "Detect cognitive biases",
        "reality_node":             "Reality grounding",
        "agent_node":               "Agent evaluation",
        "entropy_monitor_node":     "Entropy monitor",
        "debate_node":              "Debate round",
        "consensus_node":           "Build consensus",
        "persona_updater_node":     "Update user profile",
    },
}

_SCHWARTZ_LABELS: dict[str, dict[str, str]] = {
    "zh": SCHWARTZ_DIM_LABELS,
    "en": {
        "self_direction": "Self-Direction",
        "stimulation":    "Stimulation",
        "hedonism":       "Hedonism",
        "achievement":    "Achievement",
        "power":          "Power",
        "security":       "Security",
        "conformity":     "Conformity",
        "tradition":      "Tradition",
        "benevolence":    "Benevolence",
        "universalism":   "Universalism",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _s(lang: str, key: str, **kw: object) -> str:
    template = _S[lang][key]
    return template.format(**kw) if kw else template


def _select_language() -> str:
    choice = questionary.select(
        "Select language / 选择语言",
        choices=["中文", "English"],
    ).ask()
    if choice is None:
        raise typer.Exit(1)
    return "zh" if choice == "中文" else "en"


def _check_config(lang: str) -> None:
    missing = [k for k, v in [
        ("LLM_API_KEY", LLM_API_KEY),
        ("TAVILY_API_KEY", TAVILY_API_KEY),
    ] if not v]
    if missing:
        console.print(Panel(_s(lang, "missing_env", keys=", ".join(missing)), border_style="red"))
        raise typer.Exit(1)


def _scores_from_questionnaire(lang: str) -> dict[str, float]:
    console.print(_s(lang, "q_hint"))

    answers: list[int] = []
    first = questionary.text(_s(lang, "q_paste_prompt")).ask()
    if first is None:
        raise typer.Exit(1)
    parts = first.strip().split()
    if len(parts) == 21:
        try:
            answers = [int(p) for p in parts]
        except ValueError:
            pass

    if not answers:
        if parts:
            try:
                answers.append(int(parts[0]))
            except ValueError:
                pass
        while len(answers) < 21:
            n = len(answers) + 1
            raw = questionary.text(_s(lang, "q_item_prompt", n=n)).ask()
            if raw is None:
                raise typer.Exit(1)
            try:
                val = int(raw.strip())
                if 1 <= val <= 6:
                    answers.append(val)
                    continue
            except ValueError:
                pass
            console.print(_s(lang, "q_invalid"))

    invalid = [i + 1 for i, a in enumerate(answers) if not (1 <= a <= 6)]
    if invalid:
        console.print(_s(lang, "q_out_of_range", nums=invalid))
        answers = [a if 1 <= a <= 6 else 3 for a in answers]

    dim_scores: dict[str, list[float]] = {d: [] for d in SCHWARTZ_DIMS}
    for q_idx, dims in SCHWARTZ_QUESTIONNAIRE_MAPPING.items():
        score = (6 - answers[q_idx - 1]) / 5
        for dim in dims:
            dim_scores[dim].append(score)

    result = {d: round(sum(v) / len(v), 2) for d, v in dim_scores.items()}

    console.print(_s(lang, "q_result_header"))
    labels = _SCHWARTZ_LABELS[lang]
    for d in SCHWARTZ_DIMS:
        console.print(f"  {labels[d]:<16} {d:<16}  {result[d]:.2f}")
    console.print()
    return result


def _collect_schwartz_scores(lang: str) -> dict[str, float]:
    import json

    console.print(Panel(_s(lang, "first_use_panel"), border_style="blue"))

    mode = questionary.select(
        _s(lang, "input_mode"),
        choices=[
            _s(lang, "mode_questionnaire"),
            _s(lang, "mode_json"),
            _s(lang, "mode_manual"),
        ],
    ).ask()
    if mode is None:
        raise typer.Exit(1)

    if mode == _s(lang, "mode_questionnaire"):
        return _scores_from_questionnaire(lang)

    if mode == _s(lang, "mode_json"):
        console.print(_s(lang, "json_hint"))
        while True:
            console.print(_s(lang, "json_label"), end="")
            lines: list[str] = []
            try:
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
                    try:
                        json.loads("\n".join(lines))
                        break
                    except json.JSONDecodeError:
                        pass
            except (EOFError, KeyboardInterrupt):
                raise typer.Exit(1)

            if not lines:
                console.print(_s(lang, "json_empty"))
                continue
            try:
                data = json.loads("\n".join(lines))
                missing = [d for d in SCHWARTZ_DIMS if d not in data]
                if missing:
                    console.print(_s(lang, "json_missing_dims", dims=", ".join(missing)))
                return {d: max(0.0, min(1.0, float(data.get(d, 0.5)))) for d in SCHWARTZ_DIMS}
            except (json.JSONDecodeError, ValueError):
                console.print(_s(lang, "json_error"))

    # manual entry
    scores: dict[str, float] = {}
    labels = _SCHWARTZ_LABELS[lang]
    for dim in SCHWARTZ_DIMS:
        label = labels[dim]
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
            console.print(_s(lang, "manual_range_error"))
    return scores


def _collect_options(lang: str) -> list[str]:
    options: list[str] = []
    console.print(_s(lang, "options_hint"))
    i = 1
    while True:
        opt = questionary.text(_s(lang, "option_prompt", i=i)).ask()
        if opt is None or opt.strip() == "":
            if len(options) < 2:
                console.print(_s(lang, "options_min_error"))
                continue
            break
        options.append(opt.strip())
        i += 1
    return options


async def _stream_graph(lang: str, username: str, narrative: str, options: list[str]) -> str:
    initial_state = {
        "username":         username,
        "user_narrative":   narrative,
        "decision_options": options,
    }
    final_report = ""
    debate_round = 0

    console.print()
    console.print(Rule(_s(lang, "analysis_start"), style="bold green"))

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node, update in chunk.items():
            label = _NODE_LABELS[lang].get(node, node)
            if node == "debate_node":
                debate_round += 1
                label = _s(lang, "debate_label", n=debate_round)
            console.print(f"  [green]✓[/green] {label}")
            if "final_report" in update:
                final_report = update["final_report"]

    return final_report


@app.command()
def analyze() -> None:
    """Multi-agent psychological decision analysis."""
    lang = _select_language()

    console.print(Panel(
        f"[bold green]{_s(lang, 'title')}[/bold green]",
        border_style="green",
        padding=(1, 4),
    ))

    _check_config(lang)

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    username = questionary.text(_s(lang, "username_prompt")).ask()
    if not username or not username.strip():
        console.print(_s(lang, "username_empty"))
        raise typer.Exit(1)
    username = username.strip()

    value_vector = load_value_vector(username)
    if value_vector is None:
        value_vector = _collect_schwartz_scores(lang)
        save_value_vector(username, value_vector)
        console.print(_s(lang, "profile_saved"))

    narrative = questionary.text(_s(lang, "narrative_prompt")).ask()
    if not narrative or not narrative.strip():
        console.print(_s(lang, "narrative_empty"))
        raise typer.Exit(1)

    options = _collect_options(lang)

    try:
        final_report = asyncio.run(_stream_graph(lang, username, narrative.strip(), options))
    except KeyboardInterrupt:
        console.print(_s(lang, "interrupted"))
        raise typer.Exit(0)

    console.print()
    console.print(Rule(_s(lang, "analysis_result"), style="bold cyan"))

    if final_report:
        console.print(Markdown(final_report))
    else:
        console.print(_s(lang, "no_report"))
        raise typer.Exit(1)

    save = questionary.confirm(_s(lang, "save_confirm"), default=True).ask()
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(DB_PATH).parent / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{username}_{timestamp}.md"
        out_file.write_text(final_report, encoding="utf-8")
        console.print(_s(lang, "save_done", path=out_file.resolve()))


def main() -> None:
    app()
