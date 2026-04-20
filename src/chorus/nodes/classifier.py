from pydantic import BaseModel
from chorus.state import DecisionState
from chorus.utils import get_model, DecisionType, DECISION_TYPES
from chorus.infrastructure.dao import find_scene_template


class ClassificationResult(BaseModel):
    decision_type: DecisionType
    reasoning: str


_TYPE_DESCRIPTIONS = "\n".join(
    f"- {dtype}：{meta['label']}（{meta['description']}）"
    for dtype, meta in DECISION_TYPES.items()
)

_PROMPT = f"""你是一位心理决策分析师。请将以下决策描述归类到最匹配的决策类型。

决策类型列表：
{_TYPE_DESCRIPTIONS}

规则：
- 只能选择上述 7 种类型之一
- 选择与叙述核心冲突最相关的类型，而非表面话题
- 在 reasoning 中简要说明分类依据（不超过 50 字）

用户决策描述：
{{narrative}}"""


def decision_classifier_node(state: DecisionState) -> dict:
    narrative = state["user_narrative"]

    # ── LLM 分类 → 固定枚举的决策类型 ────────────────────────────
    llm = get_model(temperature=0.0).with_structured_output(ClassificationResult)
    result: ClassificationResult = llm.invoke(_PROMPT.format(narrative=narrative))
    decision_type = result.decision_type

    # ── 按类型查预置权重（种子模板首次运行时自动写入 DB）────────────
    template = find_scene_template(decision_type)

    return {"decision_type": decision_type, "scene_template": template}
