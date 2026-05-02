import asyncio
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, BaseLoader

from chorus.state import DecisionState
from chorus.infrastructure.dao.decision_history_dao import insert_decision_records
from chorus.infrastructure.config import DATA_DIR
from chorus.utils import AGENT_VALUE_MAPPING, SCHWARTZ_DIM_LABELS

_REPORTS_DIR = Path(DATA_DIR) / "reports"

_BIAS_NAME_TRANSLATIONS = {
    "Sunk Cost Fallacy":      "沉没成本谬误",
    "Recency Effect":         "近因效应",
    "Bandwagon Effect":       "从众效应",
    "Black-and-White Thinking": "非黑即白思维",
    "Confirmation Bias":      "确认偏误",
    "Loss Aversion":          "损失厌恶",
    "Status Quo Bias":        "现状偏误",
    "Anchoring Effect":       "锚定效应",
    "Overconfidence Bias":    "过度自信偏误",
    "Emotional Reasoning":    "情绪化推理",
    "Planning Fallacy":       "计划谬误",
    "Catastrophizing":        "灾难化思维",
    "Should Statements":      "应该陈述",
}

_AGENT_LABELS = {
    "Arbiter":    "逻辑法官",
    "Empath":     "情绪侦探",
    "Soothsayer": "躯体预言家",
    "Compass":    "意义向导",
    "Narrator":   "自我叙述者",
    "Conscience": "良知证人",
    "Guardian":   "关系守护者",
}

_DECISION_TYPE_LABELS = {
    "career":       "职业发展",
    "finance":      "财务决策",
    "relationship": "关系决策",
    "relocation":   "城市迁移",
    "health":       "健康决策",
    "identity":     "身份认同",
    "ethics":       "伦理决策",
}

_TEMPLATE_PATH = files("chorus.templates").joinpath("report.md.j2")


async def persona_updater_node(state: DecisionState) -> dict:
    decision_type  = state["decision_type"]
    timestamp      = datetime.now().isoformat()
    top_option     = state["consensus"]["top_option"]
    option_scores  = state["consensus"]["option_scores"]
    top_score      = option_scores.get(top_option, 0.0)

    records = [
        {
            "decision_type":   decision_type,
            "agent_name":      agent_name,
            "initial_stance":  state["initial_stances"][agent_name]["option_scores"].get(top_option, 0.0),
            "final_alignment": round(1.0 - abs(stance["option_scores"].get(top_option, 0.0) - top_score) / 2.0, 4),
            "timestamp":       timestamp,
        }
        for agent_name, stance in state["agent_stances"].items()
    ]
    await asyncio.to_thread(insert_decision_records, records)

    report = await asyncio.to_thread(_render_report, _prepare_data(state))
    await asyncio.to_thread(_save_report, state.get("username", "unknown"), timestamp, report)
    return {"final_report": report}


def _save_report(username: str, timestamp: str, content: str) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ts = timestamp[:19].replace(":", "-")
    path = _REPORTS_DIR / f"{username}_{safe_ts}.md"
    path.write_text(content, encoding="utf-8")


def _prepare_data(state: DecisionState) -> dict:
    stances        = state["agent_stances"]
    initial        = state.get("initial_stances", {})
    weights      = state["consensus"]["weights"]
    value_vector = state.get("value_vector", {})

    options     = state.get("decision_options", [])
    option_scores = state["consensus"]["option_scores"]
    top_option  = state["consensus"]["top_option"]
    ranked      = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)

    agents = []
    for agent, s in stances.items():
        init_scores = initial.get(agent, {}).get("option_scores", {})
        scores_fmt  = {}
        for opt in options:
            cur  = s["option_scores"].get(opt, 0.0)
            init = init_scores.get(opt, cur)
            if abs(init - cur) > 0.05:
                scores_fmt[opt] = f"{init:+.2f}→{cur:+.2f}"
            else:
                scores_fmt[opt] = f"{cur:+.2f}"
        agents.append({
            "label":      _AGENT_LABELS.get(agent, agent),
            "weight":     f"{weights.get(agent, 0):.0%}",
            "scores":     scores_fmt,
            "confidence": f"{s['confidence']:.0%}",
            "reasoning":  s["reasoning"],
        })

    # 综合得分行
    summary_scores = {opt: f"**{option_scores.get(opt, 0):+.2f}**" for opt in options}

    bias_flags = [
        {
            "bias":         _BIAS_NAME_TRANSLATIONS.get(b["bias"], b["bias"]),
            "target_agents": "、".join(_AGENT_LABELS.get(a, a) for a in b.get("target_agents", [])),
            "note":         b["note"],
        }
        for b in state.get("bias_flags", [])
    ]

    debate_rounds = []
    for rd in state.get("debate_history", []):
        entries = [
            {
                "label":     _AGENT_LABELS.get(k, k),
                "reasoning": v["reasoning"],
            }
            for k, v in rd.items() if k not in ("round", "type", "reason")
        ]
        debate_rounds.append({
            "number":  rd.get("round", 0) + 1,
            "reason":  rd.get("reason", ""),
            "entries": entries,
        })

    value_influences = []
    for agent, s in stances.items():
        dims = AGENT_VALUE_MAPPING.get(agent, [])
        avg  = sum(value_vector.get(d, 0.5) for d in dims) / len(dims) if dims else 0.5
        value_influences.append({
            "label":      _AGENT_LABELS.get(agent, agent),
            "weight":     f"{weights.get(agent, 0):.0%}",
            "dims":       "、".join(SCHWARTZ_DIM_LABELS.get(d, d) for d in dims),
            "importance": f"{avg:.0%}",
        })

    antagonism = state.get("antagonism_flags", [])

    return {
        "decision_type_label":  _DECISION_TYPE_LABELS.get(state.get("decision_type", ""), state.get("decision_type", "")),
        "options":              options,
        "user_narrative":       state.get("user_narrative", ""),
        "generated_at":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "reality_discrepancies": state.get("reality_discrepancies", []),
        "bias_flags":           bias_flags,
        "agents":               agents,
        "summary_scores":       summary_scores,
        "ranked":               ranked,
        "top_option":           top_option,
        "antagonism_flags":     "、".join(antagonism) if antagonism else "",
        "debate_rounds":        debate_rounds,
        "final_recommendation": state.get("final_recommendation", "（无）"),
        "value_influences":     value_influences,
    }


def _render_report(data: dict) -> str:
    template_str = _TEMPLATE_PATH.read_text(encoding="utf-8")
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    return env.from_string(template_str).render(**data)


