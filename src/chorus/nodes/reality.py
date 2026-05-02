import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from tavily import TavilyClient

from chorus.infrastructure.config import TAVILY_API_KEY
from chorus.state import DecisionState
from chorus.utils import get_model

_EXTRACT_SYSTEM = """你是一位信息提取专家。从用户的决策叙述中提取值得联网搜索的对象。

两类情况都需要提取：
1. 用户对某个对象做出了具体陈述 → 搜索可以验证或修正该陈述
2. 用户提及了某个对象但未描述其现状 → 搜索可以补充决策所需的关键背景信息

判断方法：问自己——"搜索这个对象的最新信息，对用户的决策是否有参考价值？"
- 有 → 提取
- 只是个人感受、价值判断或完全无法搜索的抽象概念 → 不提取

规则：
- 每个实体控制在 15 字以内，保持搜索粒度
- 若叙述中无值得搜索的对象，返回空列表
- 请以 JSON 格式返回结果，字段名必须为 entities"""

_DISCREPANCY_SYSTEM = """你是一位事实核查专家。对比用户的决策叙述与联网搜索获取的现实信息，输出对决策有参考价值的发现。

两类情况都需要标记：
1. 用户的陈述与现实存在实质差异（narrative_claim 填用户原话，reality_fact 填核实结果）
2. 用户未提及但搜索补充了重要背景信息（narrative_claim 填"未提及"，reality_fact 填关键信息）

每条发现必须包含以下字段：
- entity：对应的搜索实体名称
- narrative_claim：用户叙述中的具体说法（原文引用，或"未提及"）
- reality_fact：核实后的事实，来自搜索结果
- severity：严重程度，只能为 high / medium / low

规则：
- 只标记对决策真正有参考价值的发现，不要堆砌无关信息
- 忽略个人感受和主观判断，只处理可验证的事实
- severity 评级：
  - high：会实质影响决策方向（如公司近期大规模裁员、薪资严重高估）
  - medium：值得注意但不改变大方向
  - low：细节补充，影响有限
- 若无值得标记的发现，discrepancies 返回空列表
- 请以 JSON 格式返回结果，字段名必须为 discrepancies"""


class EntityExtractionResult(BaseModel):
    entities: list[str] = Field(description="可联网核查的实体列表，无可核实实体则为空列表")


class Discrepancy(BaseModel):
    entity: str = Field(description="相关实体名称")
    narrative_claim: str = Field(description="用户叙述中的具体说法，引用原文")
    reality_fact: str = Field(description="核实后的事实，来自搜索结果")
    severity: Literal["low", "medium", "high"] = Field(description="落差严重程度")


class DiscrepancyResult(BaseModel):
    discrepancies: list[Discrepancy] = Field(description="叙述与现实的落差清单，无落差则为空列表")


def _search_entity_sync(client: TavilyClient, entity: str) -> tuple[str, str]:
    try:
        summary = client.get_search_context(
            query=entity,
            search_depth="basic",
            max_results=3,
            max_tokens=600,
        )
        return entity, summary or ""
    except Exception:
        return entity, ""


async def reality_node(state: DecisionState) -> dict:
    narrative = state["user_narrative"]

    # Step 1 — 实体提取
    extractor = get_model(temperature=0.0).with_structured_output(EntityExtractionResult)
    extraction: EntityExtractionResult = await extractor.ainvoke([
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user",   "content": narrative},
    ])

    if not extraction.entities or not TAVILY_API_KEY:
        return {"reality_discrepancies": []}

    # Step 2 — 联网搜索（同步客户端套 to_thread 避免 BlockingError）
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        pairs = await asyncio.gather(*[
            asyncio.to_thread(_search_entity_sync, client, e) for e in extraction.entities
        ])
        reality_context = {entity: summary for entity, summary in pairs if summary}
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Tavily search failed: %s", e)
        return {"reality_discrepancies": []}

    if not reality_context:
        return {"reality_discrepancies": []}

    # Step 3 — 落差检测
    context_text = "\n\n".join(f"【{e}】\n{s}" for e, s in reality_context.items())
    user_content = f"用户叙述：\n{narrative}\n\n搜索结果：\n{context_text}"

    checker = get_model(temperature=0.0).with_structured_output(DiscrepancyResult)
    result: DiscrepancyResult = await checker.ainvoke([
        {"role": "system", "content": _DISCREPANCY_SYSTEM},
        {"role": "user",   "content": user_content},
    ])

    return {"reality_discrepancies": [d.model_dump() for d in result.discrepancies]}
