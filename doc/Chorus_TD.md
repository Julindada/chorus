# Chorus — LangGraph 技术设计 (Technical Design)

---

## 1. 核心机制说明

在进入图结构之前，需要确定三个 LangGraph 关键机制的选型：

**并行评估用 `Send` API 做 fan-out**：7 个 Agent 并行运行，不顺序调用。LangGraph 的 `Send` 在运行时动态创建分支，fan-in 时用 `Annotated` + reducer 合并结果，避免顺序执行导致的锚定偏误。

**熵值判断用 `add_conditional_edges` 做路由**：Entropy Monitor 是条件路由节点，根据熵值和当前辩论轮次决定走向共识还是进入辩论。

**辩论循环是一条 back-edge**：`debate_node → entropy_monitor_node` 构成有向环，LangGraph 原生支持，不需要特殊语法。

---

## 2. 图结构

```
START
  │
  ▼
intake_node
  │  解析叙述，加载 value_vector
  ▼
decision_classifier_node
  │  查模板注册中心；未命中则 LLM 生成，用户确认后保存
  ▼
bias_detection_node
  │  规则匹配，生成 bias_flags Metadata
  ▼
fan_out_node  ── Send ──► agent_node("逻辑法官")  ──┐
              ── Send ──► agent_node("情绪侦探")  ──┤
              ── Send ──► agent_node("躯体预言家") ──┤
              ── Send ──► agent_node("意义向导")  ──┤ fan-in
              ── Send ──► agent_node("自我叙述者") ──┤ (reducer 合并)
              ── Send ──► agent_node("良知证人")  ──┤
              ── Send ──► agent_node("关系守护者") ──┘
                                                   │
                                                   ▼
                                         entropy_monitor_node
                                           │  计算散度，识别冲突对
                                           │
                        ┌──────────────────┼─────────────────┐
                        │                  │                  │
                   低熵分支           高熵分支           超出轮次
                        │            rounds < max        rounds >= max
                        │                  │                  │
                        │            debate_node              │
                        │     (只跑冲突 Agent 对，             │
                        │      互相读取上一轮输出，             │
                        │      debate_round += 1)             │
                        │                  │                  │
                        │                  └──── back-edge ───┘
                        │                        回到 entropy_monitor
                        ▼                                     │
                  consensus_node  ◄────────────────────────────┘
                    │  final_weight = value_vector × scene_template
                    │  标注拮抗 Agent 对（如超出轮次退出）
                    ▼
              persona_updater_node
                    │  写入 decision_history (SQLite)
                    ▼
                   END
```

---

## 3. State 设计

```python
from typing import TypedDict, Annotated
import operator

class AgentStance(TypedDict):
    stance: float        # 倾向分，-1.0 到 1.0
    reasoning: str       # 评估依据
    confidence: float    # 信心分，0.0 到 1.0

class DecisionState(TypedDict):
    # 输入层
    user_narrative: str
    decision_options: list[str]

    # 用户画像
    value_vector: dict[str, float]       # Schwartz 10 维权重
    scene_template: dict[str, float]     # Agent 场景权重模板

    # 偏误层
    bias_flags: list[dict]               # Bias Detection Layer 输出的 Metadata

    # Phase 1：并行评估结果
    # Annotated + operator.or_ 让多个 Send 分支的结果合并进同一个 dict
    agent_stances: Annotated[dict[str, AgentStance], operator.or_]

    # 降级处理
    critical_agents: list[str]           # 关键 Agent 列表，缺失时触发重试（常量，初始化时设定）
    failed_agents: list[str]             # fan-in 后健康检查检测到的失败 Agent

    # 熵值与辩论控制
    entropy_score: float
    last_entropy_score: float            # 上一轮熵值，用于收敛斜率与震荡检测
    conflict_type: str                   # "binary" | "multi_polar" | "outlier"
    conflicting_agents: list[str]        # 参与辩论的 Agent 名称列表
    debate_round: int
    max_debate_rounds: int               # 硬上限，防止 Deadlock

    # 震荡检测：每轮 entropy_monitor 后追加一份倾向分快照
    stance_history: Annotated[list[dict[str, float]], operator.add]

    # 辩论历史（每轮辩论后追加，保留完整原始记录）
    debate_history: Annotated[list[dict], operator.add]

    # 输出层
    consensus: dict
    antagonism_flags: list[str]          # 超轮次/震荡退出时标注的拮抗对
    final_recommendation: str
```

---

## 4. 关键节点逻辑

### intake_node

职责：初始化 State 控制字段、加载用户画像、按需触发联网搜索补充背景信息。

联网搜索采用实体识别 + 按需触发模式：从叙述中提取公司名、城市、技术栈等实体，逐类匹配搜索策略，将结果以补充背景注入 `enriched_context`，不替代原始叙述。

```python
ENTITY_SEARCH_STRATEGIES = {
    "company":     "site:glassdoor.com OR site:linkedin.com {entity} reviews 2024",
    "city":        "{entity} 生活成本 租金 2024",
    "tech_stack":  "{entity} 薪资 市场需求 2024",
    "industry":    "{entity} 行业趋势 裁员风险 2024",
}

def intake_node(state: DecisionState) -> dict:
    # ── 加载用户画像 ──────────────────────────────────────────────
    value_vector = db.load_value_vector()          # 从 SQLite 读取 Schwartz 向量

    # ── 初始化控制字段 ────────────────────────────────────────────
    init_fields = {
        "value_vector":       value_vector,
        "critical_agents":    ["逻辑法官", "情绪侦探"],  # System 2 / System 1 代表，缺失破坏熵值含义
        "failed_agents":      [],
        "debate_round":       0,
        "max_debate_rounds":  3,
        "last_entropy_score": None,
        "stance_history":     [],
        "debate_history":     [],
        "antagonism_flags":   [],
        "agent_stances":      {},
    }

    # ── 按需联网搜索 ──────────────────────────────────────────────
    entities = extract_entities(state["user_narrative"])  # 实体识别（LLM 单轮调用）
    search_results = {}
    for entity_type, entity_name in entities.items():
        if entity_type in ENTITY_SEARCH_STRATEGIES:
            query  = ENTITY_SEARCH_STRATEGIES[entity_type].format(entity=entity_name)
            result = web_search(query)
            search_results[entity_name] = {"type": entity_type, "result": result}

    enriched_context = {
        "narrative":       state["user_narrative"],
        "web_supplements": search_results,   # 补充背景，Agent 可选择性参考
    }

    return {**init_fields, "enriched_context": enriched_context}
```

### decision_classifier_node

职责：cache-aside 模式管理场景权重模板——命中直接加载，未命中由 LLM 生成候选模板后交给用户确认，确认后写入注册中心。

用户确认环节使用 LangGraph 的 `interrupt()`：图执行在此暂停，等待外部（UI 层）将用户的确认/修改结果通过 `Command(resume=...)` 传回后继续。

```python
def decision_classifier_node(state: DecisionState) -> dict:
    narrative = state["user_narrative"]

    # ── 查询模板注册中心 ──────────────────────────────────────────
    matched = db.find_scene_template(narrative)   # 语义或关键词匹配
    if matched:
        return {"scene_template": matched}

    # ── LLM 生成候选模板 ──────────────────────────────────────────
    candidate = llm.invoke([
        {"role": "system", "content": (
            "根据用户的决策描述，为以下 7 个心理维度 Agent 生成权重分配（各维度 0.0–1.0，归一化）："
            f"{AGENT_NAMES}"
        )},
        {"role": "user", "content": narrative},
    ])

    # ── 用户确认（human-in-the-loop） ─────────────────────────────
    # interrupt() 暂停图执行，UI 层展示 candidate 给用户
    # 用户可直接确认或修改后通过 Command(resume=...) 传回
    confirmed = interrupt({"action": "confirm_scene_template", "candidate": candidate})

    final_template = confirmed.get("template", candidate)
    if confirmed.get("approved", False):
        db.save_scene_template(narrative, final_template)   # 写入注册中心供复用

    return {"scene_template": final_template}
```

### bias_detection_node

职责：纯规则匹配，不调用 LLM（避免 self-referential 问题）。识别用户叙述中的认知偏误模式，生成 `bias_flags` Metadata，只呈现不裁判。

```python
import re

BIAS_RULES = [
    {
        "pattern":       r"已经投入.{0,10}(年|万|个月)",
        "bias":          "沉没成本谬误",
        "target_agents": ["逻辑法官"],
        "note":          "历史投入已发生且不可回收，请审计其在当前决策中的真实权重",
    },
    {
        "pattern":       r"(最近|上周|昨天|刚刚).{0,20}(发生|出现|经历)",
        "bias":          "近因效应",
        "target_agents": ["情绪侦探"],
        "note":          "近期事件可能放大短期情绪，请评估其是否主导了整体叙述",
    },
    {
        "pattern":       r"(所有人|大家|周围人|朋友都).{0,10}(说|觉得|认为)",
        "bias":          "从众效应",
        "target_agents": ["自我叙述者"],
        "note":          "多数意见不代表对用户本人最优，请检查是否与其价值观冲突",
    },
    {
        "pattern":       r"(要么.+要么|没有退路|只能|必须选)",
        "bias":          "全有全无思维",
        "target_agents": ["逻辑法官"],
        "note":          "请审查是否存在被忽视的中间路径或分阶段方案",
    },
]

def bias_detection_node(state: DecisionState) -> dict:
    narrative = state["user_narrative"]
    flags = []

    for rule in BIAS_RULES:
        if re.search(rule["pattern"], narrative):
            flags.append({
                "bias":          rule["bias"],
                "target_agents": rule["target_agents"],
                "note":          rule["note"],
            })

    # 确认偏误：正面描述词 >> 负面描述词
    pos = len(re.findall(r"(好|棒|优秀|喜欢|期待|兴奋|开心)", narrative))
    neg = len(re.findall(r"(不好|差|担心|害怕|紧张|压力|风险)", narrative))
    if pos > neg * 3 and pos > 2:
        flags.append({
            "bias":          "确认偏误",
            "target_agents": AGENT_NAMES,   # 所有 Agent
            "note":          "叙述中正面信息显著多于负面，请主动寻找反向证据",
        })

    return {"bias_flags": flags}
```

### fan_out_node

```python
def fan_out_node(state: DecisionState):
    # 返回 Send 列表，LangGraph 并行执行
    return [
        Send("agent_node", {**state, "agent_name": agent})
        for agent in AGENT_NAMES
    ]
```

### agent_node

7 个 Agent 共用同一个节点函数，靠 `agent_name` 区分 prompt。新增 Agent 只需在 `AGENT_NAMES` 里加一行。

Phase 1 是单轮调用，所有上下文来自 State，不需要对话历史。明确**不传入其他 Agent 的 stances**，保证独立评估不被锚定。

节点挂载 `RetryPolicy` 作为 API 瞬时失败（超时、Rate Limit）的第一道防线，健康检查是第二道（在 `entropy_monitor_node` 内联处理）。

```python
def agent_node(state: DecisionState):
    agent_name = state["agent_name"]
    stance = llm.invoke(build_prompt(agent_name, state))
    return {"agent_stances": {agent_name: stance}}

def build_prompt(agent_name: str, state: DecisionState) -> list[dict]:
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPTS[agent_name]},
        {"role": "user",   "content": {
            "user_narrative":   state["user_narrative"],
            "decision_options": state["decision_options"],
            "value_vector":     state["value_vector"],
            "bias_flags":       state["bias_flags"],
            # 不传 agent_stances —— Phase 1 各 Agent 独立评估，互相隔离
        }}
    ]
```

### entropy_monitor_node

职责扩展为三步：健康检查 → 计算熵值与聚类 → 记录快照（供震荡检测使用）。

**健康检查**：`operator.or_` 只合并到达的结果，不感知缺失。在计算熵值前先验证 `agent_stances` 完整性：关键 Agent（`critical_agents`）缺失时标记系统部分失效并跳过计算；非关键 Agent 缺失时以中立值 `0.0` 填充后继续。

```python
def entropy_monitor_node(state: DecisionState):
    stances = dict(state["agent_stances"])   # 复制，不直接修改 state

    # ── 健康检查 ──────────────────────────────────────────────────
    missing = [a for a in AGENT_NAMES if a not in stances]
    if any(a in state["critical_agents"] for a in missing):
        # 关键 Agent 缺失：标记部分失效，跳过本轮计算
        return {
            "failed_agents":    missing,
            "antagonism_flags": ["SYSTEM_PARTIAL_FAILURE"],
        }
    for agent in missing:
        # 非关键 Agent 缺失：中立值填充，避免熵值计算异常
        stances[agent] = AgentStance(stance=0.0, reasoning="[缺失，中立值填充]", confidence=0.0)

    # ── 熵值计算与聚类 ────────────────────────────────────────────
    last_entropy  = state.get("entropy_score", None)   # 保存上一轮熵值
    entropy       = compute_entropy([s["stance"] for s in stances.values()])
    conflict_type, conflicting_agents = classify_conflict(stances)

    # ── 倾向分快照（供震荡检测） ──────────────────────────────────
    stance_snapshot = {name: s["stance"] for name, s in stances.items()}

    return {
        "agent_stances":      stances,
        "entropy_score":      entropy,
        "last_entropy_score": last_entropy,
        "conflict_type":      conflict_type,
        "conflicting_agents": conflicting_agents,
        "failed_agents":      missing,
        "stance_history":     [stance_snapshot],   # operator.add 自动追加
    }

def classify_conflict(stances: dict[str, AgentStance]) -> tuple[str, list[str]]:
    scores = {name: s["stance"] for name, s in stances.items()}
    
    positive = [n for n, v in scores.items() if v >  0.3]
    negative = [n for n, v in scores.items() if v < -0.3]

    if len(positive) >= 2 and len(negative) >= 2:
        # 两个阵营都有人 → 阵营对抗
        return "binary", positive + negative

    elif len(positive) == 0 or len(negative) == 0:
        # 整体一致，只有一个声音明显偏离均值 → 单一异见
        mean = sum(scores.values()) / len(scores)
        outlier = max(scores, key=lambda n: abs(scores[n] - mean))
        return "outlier", [outlier]

    else:
        # 无法形成清晰聚类 → 多极分散，不适合辩论
        return "multi_polar", list(scores.keys())
```

### 条件路由函数

在原有的轮次上限和低熵判断基础上，增加两个震荡检测退出条件：熵值无下降（连续两轮差值低于阈值）和周期震荡（当前轮倾向分与两轮前高度相似）。两种震荡退出时均标注为不可调和冲突。

```python
OSCILLATION_EPS = 0.05   # 熵值/stance 变化低于此值视为无收敛

def route_after_entropy(state: DecisionState) -> str:
    threshold = compute_threshold(state["value_vector"])  # 阈值与 value_vector 联动

    # 退出条件 1：超出最大轮次
    if state["debate_round"] >= state["max_debate_rounds"]:
        return "consensus_node"

    # 退出条件 2：低熵，已收敛
    if state["entropy_score"] < threshold:
        return "consensus_node"

    # 退出条件 3：熵值无下降（对称震荡时熵值不变，单靠此条不够，需配合条件 4）
    last = state.get("last_entropy_score")
    if last is not None and abs(state["entropy_score"] - last) < OSCILLATION_EPS:
        return "consensus_node"   # 标注不可调和冲突

    # 退出条件 4：周期震荡（round N ≈ round N-2 的倾向分向量）
    history = state.get("stance_history", [])
    if len(history) >= 3:
        current = history[-1]
        two_ago = history[-3]
        delta = sum(abs(current.get(k, 0) - two_ago.get(k, 0)) for k in current)
        if delta < OSCILLATION_EPS * len(current):
            return "consensus_node"   # 周期震荡，提前退出

    return "debate_node"

graph.add_conditional_edges("entropy_monitor_node", route_after_entropy)
```

### debate_node

back-edge 的起点。`debate_history` 承担对话历史角色，随每轮追加后完整传入 LLM。根据 `conflict_type` 执行三种不同策略：

**binary**：每个阵营选倾向分绝对值最大的 Agent 作为代表进行辩论，对方阵营其他成员的 stances 作为"阵营立场摘要"附带传入，让代表知道自己背后有哪些声音支持。非辩论 Agent 的 stances 不传入，防止从众压力。

**outlier**：单一异见者 vs. 整体共识代表（stances 均值最近的 Agent），其余 Agent 的 stances 同样不传入。

**multi_polar**：不进入辩论，直接将所有 Agent 标注为拮抗，交由 consensus_node 做加权汇总，不强行收敛。

```python
def debate_node(state: DecisionState):
    conflict_type      = state["conflict_type"]
    conflicting_agents = state["conflicting_agents"]
    stances            = state["agent_stances"]

    if conflict_type == "multi_polar":
        # 多极分散：跳过辩论，直接标注所有拮抗 Agent
        return {"antagonism_flags": conflicting_agents}

    if conflict_type == "binary":
        scores    = {n: stances[n]["stance"] for n in conflicting_agents}
        positive  = [n for n in conflicting_agents if scores[n] >  0.3]
        negative  = [n for n in conflicting_agents if scores[n] < -0.3]
        rep_a     = max(positive, key=lambda n: abs(scores[n]))   # 正向阵营代表
        rep_b     = max(negative, key=lambda n: abs(scores[n]))   # 反向阵营代表
        allies_a  = {n: stances[n] for n in positive if n != rep_a}
        allies_b  = {n: stances[n] for n in negative if n != rep_b}

    else:  # outlier
        rep_a    = conflicting_agents[0]   # 异见者
        mean     = sum(s["stance"] for s in stances.values()) / len(stances)
        rep_b    = min(stances, key=lambda n: abs(stances[n]["stance"] - mean))  # 共识代表
        allies_a = {}
        allies_b = {n: stances[n] for n in stances if n != rep_a and n != rep_b}

    # debate_history 压缩：state 保留完整原始记录，LLM 只接收压缩视图
    debate_context = get_debate_context(state["debate_history"], llm)

    new_stance_a = llm.invoke(debate_prompt(rep_a, stances[rep_b], allies_a, state, debate_context))
    new_stance_b = llm.invoke(debate_prompt(rep_b, stances[rep_a], allies_b, state, debate_context))

    return {
        "agent_stances": {rep_a: new_stance_a, rep_b: new_stance_b},
        "debate_round":  state["debate_round"] + 1,
        "debate_history": [{         # operator.add 追加原始记录，不覆盖
            "round":   state["debate_round"],
            "type":    conflict_type,
            rep_a:     new_stance_a,
            rep_b:     new_stance_b,
        }],
    }

def get_debate_context(history: list[dict], llm) -> list[dict]:
    """为 LLM 准备辩论上下文：history ≤ 1 轮时直接返回，否则压缩旧轮次。
    state 中的 debate_history 始终保留完整原始记录，压缩只影响传入 LLM 的视图。"""
    if len(history) <= 1:
        return history
    old_rounds = history[:-1]
    latest     = history[-1]
    summary    = llm.invoke([
        {"role": "system", "content": "将以下辩论记录压缩为一段核心分歧摘要，保留关键立场和论点，不超过 200 字。"},
        {"role": "user",   "content": str(old_rounds)},
    ])
    return [{"round": "summary", "content": summary}, latest]

def debate_prompt(
    agent_name:      str,
    opponent_stance: AgentStance,
    ally_stances:    dict[str, AgentStance],  # 同阵营其他成员（可为空）
    state:           DecisionState,
    debate_context:  list[dict],              # 压缩后的辩论历史视图
) -> list[dict]:
    # bias_flags 在辩论阶段以更显著的权重呈现：
    # 作为"用户当前可能存在的认知盲点"提示 Agent 主动审视，而非作为攻击指令
    bias_note = (
        f"注意：用户叙述中检测到以下潜在认知盲点，请在评估时主动审视其是否影响了你的推理：{state['bias_flags']}"
        if state["bias_flags"] else ""
    )
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPTS[agent_name]},
        {"role": "user",   "content": {
            "user_narrative":  state["user_narrative"],
            "bias_note":       bias_note,          # 提升 bias_flags 权重，明确作为审视提示
            "my_stance":       state["agent_stances"][agent_name],
            "opponent_stance": opponent_stance,
            "ally_stances":    ally_stances,
            "debate_history":  debate_context,     # 压缩视图，非完整原始记录
            # 不传非辩论 Agent 的 stances —— 防止从众压力影响辩论
        }}
    ]
```

### consensus_node

职责：对所有 Agent 的 stance 做加权汇总，生成最终决策推荐报告。

权重计算分两层：`scene_template` 提供场景基础权重，`value_vector` 通过 Agent-Schwartz 映射表进一步修正（体现用户价值观对各维度重要性的影响），两者逐维度相乘后归一化。拮抗标注来自 State 中的 `antagonism_flags`，直接写入报告，不掩盖冲突。

```python
# 每个 Agent 主要对应的 Schwartz 价值维度
AGENT_VALUE_MAPPING = {
    "逻辑法官":   ["Achievement", "Self-Direction"],
    "情绪侦探":   ["Hedonism", "Stimulation"],
    "躯体预言家": ["Security", "Conformity"],
    "意义向导":   ["Universalism", "Self-Direction"],
    "自我叙述者": ["Self-Direction", "Benevolence"],
    "良知证人":   ["Universalism", "Benevolence", "Tradition"],
    "关系守护者": ["Benevolence", "Security"],
}

def consensus_node(state: DecisionState) -> dict:
    stances       = state["agent_stances"]
    value_vector  = state["value_vector"]
    scene_template = state["scene_template"]

    # ── 权重计算：final_weight = scene_template × value_vector 分量均值 ──
    raw_weights = {}
    for agent in stances:
        schwartz_dims   = AGENT_VALUE_MAPPING.get(agent, [])
        value_score     = (
            sum(value_vector.get(dim, 0.5) for dim in schwartz_dims) / len(schwartz_dims)
            if schwartz_dims else 0.5
        )
        raw_weights[agent] = scene_template.get(agent, 1.0) * value_score

    total   = sum(raw_weights.values()) or 1.0
    weights = {agent: w / total for agent, w in raw_weights.items()}

    # ── 加权汇总 ─────────────────────────────────────────────────
    weighted_score = sum(stances[a]["stance"] * weights[a] for a in stances)

    # ── LLM 综合生成推荐报告 ──────────────────────────────────────
    recommendation = llm.invoke([
        {"role": "system", "content": "你是决策综合分析器，基于各心理维度的评估结果生成最终决策报告。"},
        {"role": "user",   "content": {
            "user_narrative":    state["user_narrative"],
            "agent_stances":     stances,
            "weights":           weights,
            "weighted_score":    weighted_score,
            "debate_history":    state["debate_history"],
            "antagonism_flags":  state["antagonism_flags"],  # 拮抗标注直接透传，不掩盖
            "bias_flags":        state["bias_flags"],
        }},
    ])

    return {
        "consensus": {
            "weighted_score": weighted_score,
            "weights":        weights,
        },
        "final_recommendation": recommendation,
    }
```

### persona_updater_node

职责：将本次决策中各 Agent 的初始立场与最终决策的对齐度写入 SQLite `decision_history` 表，供后续 Evolving Persona 使用。纯 I/O 节点，不修改 State。

对齐度公式：`alignment = 1 - |agent_stance - weighted_score| / 2`，映射到 [0, 1]，1 表示完全一致，0 表示完全相反。

```python
from datetime import datetime

def persona_updater_node(state: DecisionState) -> dict:
    weighted_score = state["consensus"]["weighted_score"]
    decision_type  = state["scene_template"].get("decision_type", "unknown")

    records = []
    for agent_name, stance_data in state["agent_stances"].items():
        alignment = 1.0 - abs(stance_data["stance"] - weighted_score) / 2.0
        records.append({
            "decision_type":   decision_type,
            "agent_name":      agent_name,
            "initial_stance":  stance_data["stance"],
            "final_alignment": round(alignment, 4),
            "timestamp":       datetime.now().isoformat(),
        })

    db.insert_many("decision_history", records)
    return {}   # 纯副作用节点，不修改 State
```

---

## 5. 图的构建

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.pregel import RetryPolicy

graph = StateGraph(DecisionState)

# 注册节点
graph.add_node("intake_node", intake_node)
graph.add_node("decision_classifier_node", decision_classifier_node)
graph.add_node("bias_detection_node", bias_detection_node)
graph.add_node("fan_out_node", fan_out_node)
graph.add_node(                              # RetryPolicy：API 瞬时失败的第一道防线
    "agent_node", agent_node,
    retry=RetryPolicy(max_attempts=3, retry_on=Exception)
)
graph.add_node("entropy_monitor_node", entropy_monitor_node)
graph.add_node("debate_node", debate_node)
graph.add_node("consensus_node", consensus_node)
graph.add_node("persona_updater_node", persona_updater_node)

# 静态边
graph.add_edge(START, "intake_node")
graph.add_edge("intake_node", "decision_classifier_node")
graph.add_edge("decision_classifier_node", "bias_detection_node")
graph.add_edge("bias_detection_node", "fan_out_node")
graph.add_edge("agent_node", "entropy_monitor_node")   # fan-in（Send 动态创建分支，无需静态 add_edge）
graph.add_edge("debate_node", "entropy_monitor_node")  # back-edge，构成循环
graph.add_edge("consensus_node", "persona_updater_node")
graph.add_edge("persona_updater_node", END)

# 条件边
graph.add_conditional_edges("entropy_monitor_node", route_after_entropy)

# 编译，挂载 Checkpointer
app = graph.compile(checkpointer=SqliteSaver("chorus.db"))
```

---

## 6. Agent 天然张力结构

设计辩论路由时，`find_max_divergence_pair` 优先识别以下高冲突对：

| 张力对 | 心理学原因 |
|--------|-----------|
| 逻辑法官 ↔ 情绪侦探 | System 2 vs System 1，最经典的冲突 |
| 逻辑法官 ↔ 躯体预言家 | 理性收益 vs 身体成本 |
| 意义向导 ↔ 逻辑法官 | 意义无法被效用函数覆盖 |
| 自我叙述者 ↔ 关系守护者 | 自我实现 vs 对他人的影响 |

天然盟友（倾向一致，通常不进入辩论）：

| 盟友对 | 原因 |
|--------|------|
| 情绪侦探 + 躯体预言家 | 同属 System 1 驱动 |
| 意义向导 + 自我叙述者 | 同属身份层 |
| 逻辑法官 + 良知证人 | 同属分析性 |

---

## 7. 设计注意事项

**LLM Context 管理**：系统中的 LLM 调用分两类，上下文策略不同。

| 节点 | 调用类型 | 上下文来源 |
|------|----------|-----------|
| intake_node | 单轮 | user_narrative |
| decision_classifier_node | 单轮 | user_narrative + scene_templates |
| agent_node（Phase 1） | 单轮 | user_narrative + bias_flags + value_vector，隔离其他 Agent stances |
| debate_node | 多轮 | my_stance + opponent_stance + ally_stances（同阵营摘要）+ debate_context（压缩视图），隔离非辩论 Agent stances |
| consensus_node | 单轮 | 全部 agent_stances + debate_history + value_vector + scene_template |

`Checkpointer` 是跨会话的状态快照，与 LLM 对话历史是两个独立概念。`debate_history` 才是辩论阶段的对话历史，由 `operator.add` reducer 在 State 中逐轮追加。

**agent_node 复用**：7 个 Agent 共享一个节点函数，靠 `agent_name` 参数区分 prompt 模板，扩展新 Agent 不需要修改图结构。

**debate_node 三种冲突策略**：binary（阵营代表对抗）、outlier（异见者 vs 共识代表）、multi_polar（跳过辩论直接标注拮抗）。非辩论 Agent 的 stances 始终不传入辩论 prompt，防止从众压力影响辩论结果。

**Checkpointer 双用途**：`SqliteSaver` 同时承担两个职责——LangGraph 的状态持久化（支持时间旅行）和 `decision_history` 表的写入（支持 Evolving Persona），共用同一个 `chorus.db`。

**最大辩论轮次**：`max_debate_rounds` 建议默认值为 3，可由用户在初始化时配置。超出轮次时 `consensus_node` 在报告中明确标注拮抗 Agent 对，不掩盖冲突。

**Fan-in 降级处理**：`RetryPolicy` 处理 API 瞬时失败（超时、Rate Limit），是第一道防线。`entropy_monitor_node` 内联健康检查是第二道：关键 Agent（`critical_agents`，默认逻辑法官 + 情绪侦探）缺失时标记 `SYSTEM_PARTIAL_FAILURE` 并跳过计算；非关键 Agent 缺失时以中立值 `0.0` 填充后继续。

**震荡检测**：`route_after_entropy` 有四个退出条件：轮次上限、低熵收敛、熵值无下降、周期震荡（round N ≈ round N-2 倾向分向量）。后两者对应梯度下降的早停机制，触发时标注不可调和冲突而非无限循环。单靠熵值对比无法检测对称震荡，必须配合 `stance_history` 的向量差。

**debate_history 压缩**：`debate_history` 在 State 中始终保留完整原始记录（用于调试和 Persona）。传入 LLM 的是 `get_debate_context` 生成的压缩视图：超过 1 轮时将旧轮次压缩为摘要，只保留最新一轮原文，防止 context window 随轮次膨胀。

**bias_flags 在辩论阶段的权重**：`debate_prompt` 中 `bias_flags` 以 `bias_note` 的形式单独呈现，明确作为"用户潜在认知盲点"提示 Agent 主动审视，而非转化为攻击指令。保持与 Phase 1"只呈现不裁判"的一致性原则。
