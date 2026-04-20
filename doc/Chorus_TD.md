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
  │  解析叙述，加载 value_vector
  ▼
decision_classifier_node
  │  LLM 分类到固定枚举，按类型加载预置权重模板
  ▼
bias_detection_node
  │  LLM 检测认知偏误，生成 bias_flags Metadata
  ▼
dispatch_node ── Send ──► agent_node("Arbiter")    ──┐
              ── Send ──► agent_node("Empath")     ──┤
              ── Send ──► agent_node("Soothsayer") ──┤
              ── Send ──► agent_node("Compass")    ──┤ 结果汇总
              ── Send ──► agent_node("Narrator")   ──┤ (reducer 合并)
              ── Send ──► agent_node("Conscience") ──┤
              ── Send ──► agent_node("Guardian")   ──┘
                                                   │
                                                   ▼
                                         entropy_monitor_node
                                           │  计算散度，识别冲突对
                                           │
                        ┌──────────────────┼──────────────────────────┐
                        │                  │                           │
                 收敛分支              高熵分支                   不可调和分支
          (低熵 / 超出轮次 /        rounds < max               (multi_polar /
           熵值不降 / 周期震荡)                                  不可调和冲突)
                        │                  │                           │
                        │            debate_node                       │
                        │      (只跑冲突 Agent 对，                      │
                        │      互相读取上一轮输出，                       │
                        │      debate_round += 1)                      │
                        │                  │                           │
                        │                  └──── back-edge ────────────┤
                        │                        回到 entropy_monitor   │
                        ▼                                              │
                  consensus_node  ◄─────────────────────────────────────┘
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

| 字段 | 类型 | 说明 |
|------|------|------|
| `username` | str | 用户标识 |
| `user_narrative` | str | 用户叙述 |
| `decision_options` | list[str] | 待决策选项 |
| `value_vector` | dict[str, float] | Schwartz 10 维权重 |
| `scene_template` | dict[str, float] | Agent 场景权重模板 |
| `bias_flags` | list[dict] | 认知偏误 Metadata |
| `agent_stances` | Annotated[dict, or_] | 各 Agent 评估结果，`operator.or_` 合并各并行分支 |
| `initial_stances` | dict | Phase 1 原始立场快照，第一次进入 entropy_monitor 时固定，辩论中只读 |
| `critical_agents` | list[str] | 关键 Agent 名单，缺失时整图崩溃 |
| `failed_agents` | list[str] | 健康检查检测到的缺失 Agent |
| `entropy_score` | float | 当前轮熵值 |
| `last_entropy_score` | float \| None | 上一轮熵值，用于震荡检测 |
| `conflict_type` | str | `binary` / `outlier` / `multi_polar` |
| `conflicting_agents` | list[str] | 参与辩论的 Agent 列表 |
| `debate_round` | int | 当前辩论轮次 |
| `max_debate_rounds` | int | 硬上限，默认 3 |
| `stance_history` | Annotated[list, add] | 每轮倾向分快照，供周期震荡检测 |
| `debate_history` | Annotated[list, add] | 完整辩论原始记录，逐轮追加 |
| `consensus` | dict | 加权汇总结果（weighted_score + weights） |
| `antagonism_flags` | list[str] | 拮抗 Agent 对标注 |
| `final_recommendation` | str | 最终决策报告 |

`AgentStance` 结构：`stance`（-1.0 到 1.0）、`reasoning`（评估依据）、`confidence`（0.0 到 1.0）。

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

LLM 以 `temperature=0` 将 `user_narrative` 分类到以上固定枚举（Pydantic `Literal` 约束，无法自造新 key）。再按类型从 DB 加载预置 `scene_template`（种子首次运行时 `INSERT OR IGNORE` 写入）。

### bias_detection_node

LLM 以 `temperature=0` 识别叙述中的认知偏误（13 种），生成 `bias_flags`，每条含：偏误名称、最需警惕的 Agent、针对本叙述的描述（≤40 字）。只呈现现象，不做价值判断。


### dispatch_node

```
返回 Send 列表：对每个 agent_name in AGENT_NAMES
    Send("agent_node", state + {agent_name})
```

### agent_node

7 个 Agent 共用同一节点函数，靠 `agent_name` 区分 prompt 模板。Phase 1 单轮调用，**不传入其他 Agent 的 stances**，保证独立评估不被锚定。

```
try:
    用 LLM structured output 生成 AgentStance
    返回 {agent_stances: {agent_name: stance}}
except:
    若 agent_name in critical_agents → raise（整图崩溃，拒绝缺失核心维度的决策）
    否则 → 返回 {}（静默跳过，entropy_monitor 用中立值填充）
```

节点挂载 `RetryPolicy(max_attempts=3)`，`raise` 前已穷尽重试。

### entropy_monitor_node

三步：健康检查 → 计算熵值与聚类 → 记录快照。

```
健康检查：
    missing = AGENT_NAMES - agent_stances.keys()
    若 missing 含 critical_agents → 标记 SYSTEM_PARTIAL_FAILURE（实际死代码，critical agents 失败已崩溃）
    若 missing 含非关键 Agent → 以 stance=0.0 填充

若 debate_round == 0 → 将当前 agent_stances 存入 initial_stances（之后只读）

计算：
    entropy = compute_entropy(所有 stance 值)
    conflict_type, conflicting_agents = classify_conflict(stances)
        binary    : 正向阵营(>0.3) ≥2 且 负向阵营(<-0.3) ≥2
        outlier   : 一侧为空，取偏离均值最远的 Agent
        multi_polar: 其余情况
    若 conflict_type == multi_polar → 设置 antagonism_flags = conflicting_agents

快照：将当前各 Agent stance 追加到 stance_history
```

### 条件路由函数

```
route_after_entropy:
    若 conflict_type == multi_polar   → consensus_node（辩论无法收敛）
    若 debate_round >= max_debate_rounds → consensus_node
    若 entropy < threshold            → consensus_node（已收敛）
    若 |entropy - last_entropy| < ε   → consensus_node（熵值无下降）
    若 |stance[N] - stance[N-2]| < ε  → consensus_node（周期震荡）
    否则                              → debate_node
```

阈值 `threshold` 与 `value_vector` 联动；震荡阈值 `ε = 0.05`。

### debate_node

back-edge 起点。`debate_history` 承担对话历史角色，逐轮追加后完整存储，但传入 LLM 时压缩（旧轮次摘要 + 最新一轮原文），防止 context 随轮次膨胀。

根据 `conflict_type` 执行两种策略：

**binary**：每侧取倾向分绝对值最大的 Agent 为代表，同侧其余 stances 作为"阵营摘要"传入，非辩论 Agent stances 不传（防从众压力）。

**outlier**：异见者 vs. 距均值最近的共识代表，其余 Agent stances 不传。

```
选出 rep_a, rep_b 及各自 ally_stances
debate_context = 压缩后的 debate_history
rep_a 新 stance = LLM(rep_a prompt + initial_stance=initial_stances[rep_a]
                                    + my_stance=当前stance
                                    + opponent=rep_b + allies + context)
rep_b 新 stance = LLM(rep_b prompt + initial_stance=initial_stances[rep_b]
                                    + my_stance=当前stance
                                    + opponent=rep_a + allies + context)
返回更新后的 agent_stances、debate_round+1、追加 debate_history
```

`bias_flags` 在辩论 prompt 中以"用户潜在认知盲点"形式单独呈现，提示 Agent 主动审视，不转化为攻击指令。

### consensus_node

权重两层计算：`scene_template`（场景基础权重）× `value_vector` 分量均值（Agent-Schwartz 映射修正），归一化后加权汇总各 Agent stance 得 `weighted_score`。再调用 LLM 综合生成最终报告，`antagonism_flags` 直接透传写入报告，不掩盖冲突。

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

纯 I/O 节点，不修改 State。对每个 Agent 计算对齐度：

```
alignment = 1 - |agent_stance - weighted_score| / 2
```

映射到 [0, 1]，写入 SQLite `decision_history` 表供 Evolving Persona 使用。

---

## 5. 图的构建

```
节点注册：
    intake, decision_classifier, bias_detection, dispatch,
    agent_node (RetryPolicy max=3),
    entropy_monitor, debate, consensus, persona_updater

静态边：
    START → intake → decision_classifier → bias_detection → dispatch
    agent_node → entropy_monitor   （结果汇总，Send 动态分支无需静态声明）
    debate → entropy_monitor        （back-edge，构成循环）
    consensus → persona_updater → END

条件边：
    entropy_monitor --route_after_entropy--> debate | consensus

编译：挂载 SqliteSaver(chorus.db) 作为 Checkpointer
```

---

## 6. 技术栈 (Tech Stack)

| 组件 | 选型 | 说明 |
|------|------|------|
| 框架 | LangGraph (Python) | 支持循环、Checkpointer、Annotated State |
| LLM | Qwen3.6-Plus (DashScope) | 通义千问，通过 `langchain_community.ChatTongyi` 接入，模型名由 `CHORUS_LLM_MODEL` 环境变量配置 |
| 存储 | SQLite | 状态持久化（Checkpointer）+ 决策历史（Evolving Persona）+ 场景模板注册中心 |
| 调试 | LangGraph Inspector | 可视化 Agent 节点流向与状态变化 |

---

## 7. 设计注意事项

**LLM Context 管理**：

| 节点 | 调用类型 | 上下文来源 |
|------|----------|-----------|
| decision_classifier_node | 单轮 | user_narrative；temperature=0，只做分类 |
| bias_detection_node | 单轮 | user_narrative；temperature=0，只识别偏误 |
| agent_node（Phase 1） | 单轮 | user_narrative + bias_flags + value_vector，隔离其他 Agent stances |
| debate_node | 多轮 | my_stance + opponent + ally_stances + debate_context（压缩视图），隔离非辩论 Agent stances |
| consensus_node | 单轮 | 全部 agent_stances + debate_history + value_vector + scene_template |

`Checkpointer` 是跨会话的状态快照，与 LLM 对话历史是两个独立概念。`debate_history` 才是辩论阶段的对话历史，由 `operator.add` reducer 在 State 中逐轮追加。

**agent_node 复用**：7 个 Agent 共享一个节点函数，靠 `agent_name` 参数区分 prompt 模板，扩展新 Agent 不需要修改图结构。

**debate_node 两种冲突策略**：binary（阵营代表对抗）、outlier（异见者 vs 共识代表）。multi_polar 不进入 debate_node，由路由直接送往 consensus_node 加权汇总，不强行收敛。

**单库双表**：`SqliteSaver` 负责 LangGraph 状态持久化（checkpoint 表），`persona_updater_node` 的 DAO 负责写入 `decision_history` 表，两者共用同一个 `chorus.db`，互不干涉。

**最大辩论轮次**：`max_debate_rounds` 建议默认值为 3，可由用户在初始化时配置。超出轮次时 `consensus_node` 在报告中明确标注拮抗 Agent 对，不掩盖冲突。

**并行结果汇总降级处理**：分两层，语义不同。第一层：`RetryPolicy(max_attempts=3)` 处理 API 瞬时失败（超时、Rate Limit）。第二层：`agent_node` 内部区分降级策略——关键 Agent（默认 Arbiter + Empath）重试耗尽后崩溃整图；非关键 Agent 静默跳过，`entropy_monitor_node` 以中立值 `0.0` 填充后继续。

**震荡检测**：`route_after_entropy` 有五个退出条件：multi_polar、轮次上限、低熵收敛、熵值无下降、周期震荡（round N ≈ round N-2 倾向分向量）。后两者对应梯度下降的早停机制，触发时标注不可调和冲突而非无限循环。单靠熵值对比无法检测对称震荡，必须配合 `stance_history` 的向量差。

**initial_stances 锚点**：`operator.or_` 每轮辩论后会覆盖 `agent_stances` 中辩论 Agent 的立场，Phase 1 原始结论随之丢失。`entropy_monitor_node` 在 `debate_round == 0` 时将 `agent_stances` 快照存入 `initial_stances`，此后只读。辩论 prompt 中同时传入 `initial_stance` 和 `my_stance`，让 LLM 感知自身立场漂移幅度，防止被过度说服；`consensus_node` 也可用此字段在报告中呈现"初始 → 最终"的立场演变轨迹。

**debate_history 压缩**：State 中始终保留完整原始记录。传入 LLM 的是压缩视图：超过 1 轮时将旧轮次压缩为摘要，只保留最新一轮原文，防止 context window 随轮次膨胀。

**bias_flags 在辩论阶段的权重**：辩论 prompt 中 `bias_flags` 单独呈现为"用户潜在认知盲点"，提示 Agent 主动审视，而非转化为攻击指令。保持与 Phase 1"只呈现不裁判"的一致性原则。
