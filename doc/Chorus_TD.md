# Chorus — LangGraph 技术设计 (Technical Design)

---

## 1. 核心机制说明

在进入图结构之前，需要确定三个 LangGraph 关键机制的选型：

**并行评估用 `Send` API 做并行分发**：7 个 Agent 并行运行，不顺序调用。LangGraph 的 `Send` 在运行时动态创建分支，结果汇总时用 `Annotated` + reducer 合并，避免顺序执行导致的锚定偏误。

**熵值判断用 `add_conditional_edges` 做路由**：Entropy Monitor 是条件路由节点，根据熵值和当前辩论轮次决定走向共识还是进入辩论。

**辩论循环是一条 back-edge**：`debate_node → entropy_monitor_node` 构成有向环，LangGraph 原生支持，不需要特殊语法。

---

## 2. 图结构

```
START
  │
  ▼
intake_node
  │  加载 value_vector，初始化控制字段
  ▼
decision_classifier_node
  │  LLM 分类到固定枚举，按类型加载预置权重模板
  ▼
bias_detection_node
  │  LLM 检测认知偏误，生成 bias_flags Metadata
  │
  ├─(conditional edge: dispatch_node)─►  agent_node("Arbiter")    ──┐
  │                                  ►  agent_node("Empath")     ──┤
  │                                  ►  agent_node("Soothsayer") ──┤
  │                                  ►  agent_node("Compass")    ──┤ 结果汇总
  │                                  ►  agent_node("Narrator")   ──┤ (reducer 合并)
  │                                  ►  agent_node("Conscience") ──┤
  │                                  ►  agent_node("Guardian")   ──┘
  │                                                               │
  │                                                               ▼
  │                                                     entropy_monitor_node
  │                                                       │  计算散度，识别冲突对
  │                                                       │
  │                                  ┌────────────────────┼──────────────────┐
  │                                  │                    │                  │
  │                           收敛分支              高熵分支          不可调和分支
  │                    (低熵 / 超出轮次 /        rounds < max       (multi_polar /
  │                     熵值不降 / 周期震荡)                          不可调和冲突)
  │                                  │                    │                  │
  │                                  │              debate_node              │
  │                                  │        (选分歧最大的选项，             │
  │                                  │         取对立最深的一对 Agent，        │
  │                                  │         debate_round += 1)            │
  │                                  │                    │                  │
  │                                  │                    └─── back-edge ────┤
  │                                  │                         回到 entropy   │
  │                                  ▼                                       │
  │                            consensus_node  ◄──────────────────────────────┘
  │                              │  per-option 加权汇总，输出选项排名
  │                              ▼
  │                        persona_updater_node
  │                              │  写 decision_history (SQLite)
  │                              │  生成 markdown 报告到 data/reports/
  │                              ▼
  │                             END
```

---

## 3. State 设计

| 字段 | 类型 | 说明 |
|------|------|------|
| `username` | str | 用户标识 |
| `user_narrative` | str | 用户叙述 |
| `decision_options` | list[str] | 待决策选项 |
| `value_vector` | dict[str, float] | Schwartz 10 维权重 |
| `decision_type` | str | 决策类型枚举 |
| `scene_template` | dict[str, float] | Agent 场景权重模板 |
| `bias_flags` | list[dict] | 认知偏误 Metadata |
| `agent_stances` | Annotated[dict, or_] | 各 Agent 评估结果，`operator.or_` 合并各并行分支 |
| `initial_stances` | dict | Phase 1 原始立场快照，第一次进入 entropy_monitor 时固定，辩论中只读 |
| `critical_agents` | list[str] | 关键 Agent 名单，缺失时整图崩溃 |
| `failed_agents` | list[str] | 健康检查检测到的缺失 Agent |
| `entropy_score` | float | 当前轮熵值（各选项熵值的均值） |
| `last_entropy_score` | float \| None | 上一轮熵值，用于震荡检测 |
| `conflict_type` | str | `binary` / `outlier` / `multi_polar` / `converged` |
| `conflicting_agents` | list[str] | 参与辩论的 Agent 列表 |
| `debate_round` | int | 当前辩论轮次 |
| `max_debate_rounds` | int | 硬上限，默认 3 |
| `stance_history` | Annotated[list, add] | 每轮各 Agent 的 option_scores 快照，供震荡检测 |
| `debate_history` | Annotated[list, add] | 完整辩论原始记录，逐轮追加 |
| `consensus` | dict | 加权汇总结果（option_scores + top_option + weights） |
| `antagonism_flags` | list[str] | 拮抗 Agent 标注 |
| `final_recommendation` | str | LLM 生成的最终建议文本 |
| `final_report` | str | 完整 markdown 报告（结构化数据 + LLM 建议） |

`AgentStance` 结构：`option_scores`（各选项评分，-1.0 到 1.0）、`reasoning`（评估依据，≤60 字）、`confidence`（0.0 到 1.0）。

---

## 4. 关键节点逻辑

### intake_node

从 SQLite 加载 `value_vector`；若首次使用则 `interrupt()` 暂停，等用户完成 Schwartz 10 维打分后写入 DB 再继续。初始化所有控制字段：`debate_round=0`、`max_debate_rounds=3`、`critical_agents=["Arbiter","Empath"]` 等。

### decision_classifier_node

**7 种决策类型**（由 7 个心理维度推导，每种类型有 1-2 个主导维度）：

| 类型 | 中文 | 主导 Agent |
|------|------|-----------|
| `career` | 职业与事业 | Arbiter、Compass |
| `finance` | 财务与资产 | Arbiter、Conscience |
| `relationship` | 亲密关系与家庭 | Empath、Guardian |
| `relocation` | 居住与迁移 | Guardian、Soothsayer |
| `health` | 健康与身体 | Soothsayer、Empath |
| `identity` | 身份认同与自我成长 | Narrator、Compass |
| `ethics` | 伦理与社会责任 | Conscience、Compass |

LLM 以 `temperature=0` 将 `user_narrative` 分类到以上固定枚举（Pydantic `Literal` 约束）。再按类型从 DB 加载预置 `scene_template`。

### bias_detection_node

LLM 以 `temperature=0` 识别叙述中的认知偏误（13 种），生成 `bias_flags`，每条含：偏误名称、最需警惕的 Agent、针对本叙述的描述（≤40 字）。只呈现现象，不做价值判断。

### dispatch_node（conditional edge 路由函数）

不是一个节点，而是 `bias_detection_node` 的 conditional edge 路由函数，返回 `list[Send]`，对每个 `agent_name` 动态创建并行分支：

```
返回 Send 列表：对每个 agent_name in AGENT_NAMES
    Send("agent_node", state + {agent_name})
```

### agent_node

7 个 Agent 共用同一节点函数，靠 `agent_name` 区分 prompt 模板。对**每个候选选项独立打分**（-1.0 到 +1.0），不传入其他 Agent 的 stances，保证独立评估不被锚定。

```
try:
    用 LLM structured output 生成 AgentStance
      - option_scores: {选项名: 分数} 覆盖所有候选选项
      - reasoning: ≤60 字，少用心理学术语
      - confidence: 0.0–1.0
    返回 {agent_stances: {agent_name: stance}}
except:
    若 agent_name in critical_agents → raise（整图崩溃）
    否则 → 返回 {}（静默跳过，entropy_monitor 用中立值填充）
```

节点挂载 `RetryPolicy(max_attempts=3)`。

### entropy_monitor_node

三步：健康检查 → 计算熵值与聚类 → 记录快照。

```
健康检查：
    missing = AGENT_NAMES - agent_stances.keys()
    若 missing 含 critical_agents → 标记 SYSTEM_PARTIAL_FAILURE
    若 missing 含非关键 Agent → 以 option_scores 全 0.0 填充

若 debate_round == 0 → 将当前 agent_stances 存入 initial_stances（之后只读）

计算：
    entropy = mean(per-option std_dev across agents)
    conflict_type, conflicting_agents = classify_conflict(stances)
        使用各 Agent 的 option_scores 均值作为该 Agent 的整体倾向：
        binary    : 正向阵营(均值>0.3) ≥2 且 负向阵营(均值<-0.3) ≥2
        outlier   : 一侧为空，取偏离整体均值最远的 Agent
        converged : 两侧均为空（全部接近中立）
        multi_polar: 其余情况
    若 conflict_type == multi_polar → 设置 antagonism_flags

快照：将当前 {agent: option_scores} 追加到 stance_history
```

### 条件路由函数 route_after_entropy

```
若 conflict_type in (multi_polar, converged) → consensus_node
若 debate_round >= max_debate_rounds          → consensus_node
若 entropy < threshold                        → consensus_node（已收敛）
若 |entropy - last_entropy| < ε              → consensus_node（熵值无下降）
若 flatten(stance[N]) ≈ flatten(stance[N-2]) → consensus_node（周期震荡）
否则                                          → debate_node
```

阈值 `threshold` 与 `value_vector` 联动；震荡检测将 `stance_history` 展平为 `{agent:option: score}` 向量后比较 L1 距离。

### debate_node

back-edge 起点。代表选取逻辑：

```
1. 找分歧最大的选项：max(options, key=option_std_dev across all agents)

2. binary：
   在该选项上，从 conflicting_agents 中
   - 正方：分数最高的为代表，其余为 ally
   - 负方：分数最低的为代表，其余为 ally

3. outlier：
   在该选项上
   - 离全体均值最远的 Agent 为代表 A
   - 最接近全体均值的 Agent 为代表 B
   - 不传 ally（防多数施压）
```

辩论执行：

```
debate_context = 压缩后的 debate_history（旧轮摘要 + 最新轮原文）
rep_a 新 stance = LLM(prompt + initial_stance + my_stance + opponent + allies + context)
rep_b 新 stance = LLM(prompt + initial_stance + my_stance + opponent + allies + context)
返回更新后的 agent_stances、debate_round+1、追加 debate_history（含选择原因 reason 字段）
```

`bias_flags` 以"用户潜在认知盲点"形式单独呈现，提示 Agent 主动审视，不转化为攻击指令。

### consensus_node

```
权重两层计算：
    value_score[a]  = avg(value_vector[d] for d in AGENT_VALUE_MAPPING[a])
    raw_weight[a]   = scene_template[a] × value_score[a]
    weights[a]      = raw_weight[a] / sum(raw_weight)   # 归一化，使 Σweights = 1

per-option 加权得分：
    option_scores[opt] = Σ agent_stances[a].option_scores[opt] × weights[a]

top_option = argmax(option_scores)
ranked_options = sorted(option_scores, descending)

调用 LLM 生成最终建议（≤300 字，少用心理学术语）
```

**Agent-Schwartz 映射**：

| Agent | 对应 Schwartz 维度 |
|-------|-----------------|
| Arbiter | achievement, self_direction |
| Empath | hedonism, stimulation |
| Soothsayer | security, conformity |
| Compass | universalism, self_direction |
| Narrator | self_direction, benevolence |
| Conscience | universalism, benevolence, tradition |
| Guardian | benevolence, security |

### persona_updater_node

两项职责：

**1. 写入 SQLite：** 对每个 Agent，以 `top_option` 的评分计算对齐度：
```
alignment = 1 - |agent_score[top_option] - consensus_score[top_option]| / 2
```

**2. 生成 markdown 报告：** 从 state 准备数据 dict，用 Jinja2 渲染 `templates/report.md.j2`，保存到 `data/reports/<username>_<timestamp>.md`。

报告包含：决策背景、认知偏误、各维度 × 各选项评分矩阵（含辩论前后对比）、选项排名、辩论过程（含选择依据）、最终建议、意见占比说明。

---

## 5. 图的构建

```
节点注册：
    intake, decision_classifier, bias_detection,
    agent_node (RetryPolicy max=3),
    entropy_monitor, debate, consensus, persona_updater

静态边：
    START → intake → decision_classifier → bias_detection
    agent_node → entropy_monitor
    debate → entropy_monitor（back-edge）
    consensus → persona_updater → END

条件边：
    bias_detection_node --dispatch_node--> ["agent_node"]
    entropy_monitor_node --route_after_entropy--> ["debate_node", "consensus_node"]

编译：无自定义 Checkpointer（LangGraph API 平台管理持久化）
```

---

## 6. 技术栈 (Tech Stack)

| 组件 | 选型 | 说明 |
|------|------|------|
| 框架 | LangGraph (Python) | 支持循环、Annotated State、Send API |
| LLM | Qwen（DashScope）| 通过 `langchain_openai.ChatOpenAI` + DashScope compatible endpoint 接入，模型名由 `LLM_MODEL` 环境变量配置 |
| 存储 | SQLite | 决策历史（Evolving Persona）+ 场景模板注册中心 |
| 报告模板 | Jinja2 | `templates/report.md.j2`，数据准备与格式完全分离 |
| 调试 | LangGraph Studio | 可视化节点流向与 State 变化 |

---

## 7. 设计注意事项

**LLM Context 管理**：

| 节点 | 调用类型 | 上下文来源 |
|------|----------|-----------|
| decision_classifier_node | 单轮 | user_narrative；temperature=0，只做分类 |
| bias_detection_node | 单轮 | user_narrative；temperature=0，只识别偏误 |
| agent_node（Phase 1） | 单轮 | user_narrative + bias_flags + value_vector + decision_options，隔离其他 Agent stances |
| debate_node | 多轮 | my_stance + opponent + ally_stances + debate_context（压缩视图），隔离非辩论 Agent stances |
| consensus_node | 单轮 | 全部 agent_stances + option_scores + ranked_options + debate_history |

**多选项评分设计**：每个 Agent 对所有候选选项独立打分，避免将多选题坍缩为二元立场。熵值计算为各选项跨 Agent 标准差的均值；共识加权分别对每个选项汇总，最终输出选项排名而非单一建议分。

**debate_node 代表选取**：先找跨 Agent 分歧最大的选项（标准差最高），再在该选项上选评分最对立的两个 Agent 进行辩论。binary 与 outlier 冲突类型共用同一选取逻辑，区别在于 ally 是否传入。

**agent_node 复用**：7 个 Agent 共享一个节点函数，靠 `agent_name` 参数区分 prompt 模板，扩展新 Agent 不需要修改图结构。

**dispatch_node 是路由函数而非节点**：返回 `list[Send]` 的函数不能作为普通节点使用，必须注册为 `add_conditional_edges` 的路由函数，并声明目标节点列表供 LangGraph 静态分析。

**单库双表**：`persona_updater_node` 的 DAO 负责写入 `decision_history` 表，`scene_template` 注册中心在同一 `chorus.db` 的独立表，互不干涉。

**最大辩论轮次**：`max_debate_rounds` 默认 3，可在初始化时配置。超出轮次时 consensus 报告明确标注拮抗维度，不掩盖冲突。

**并行结果汇总降级处理**：分两层。第一层：`RetryPolicy(max_attempts=3)` 处理 API 瞬时失败。第二层：关键 Agent（默认 Arbiter + Empath）重试耗尽后崩溃整图；非关键 Agent 静默跳过，`entropy_monitor_node` 以全零 option_scores 填充后继续。

**震荡检测**：`route_after_entropy` 有五个退出条件：multi_polar/converged、轮次上限、低熵收敛、熵值无下降、周期震荡。震荡检测将 `stance_history` 展平为 `{agent:option: score}` 向量，比较第 N 轮与第 N-2 轮的 L1 距离，对称震荡无法被单纯的熵值差检测，必须依赖向量比较。

**initial_stances 锚点**：`operator.or_` 每轮辩论后覆盖辩论 Agent 的 stances，Phase 1 原始结论随之丢失。`entropy_monitor_node` 在 `debate_round == 0` 时快照存入 `initial_stances`，之后只读。报告中的评分矩阵展示"辩论前→辩论后"格式。

**debate_history 压缩**：State 中始终保留完整原始记录。传入 LLM 的是压缩视图：超过 1 轮时将旧轮次压缩为摘要，只保留最新一轮原文，防止 context window 随轮次膨胀。

**报告生成**：`persona_updater_node` 的 `_prepare_data()` 负责数据准备，`_render_report()` 负责 Jinja2 渲染，两层完全分离。模板文件 `templates/report.md.j2` 可独立修改而不触碰 Python 逻辑。
