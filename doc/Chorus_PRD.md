# Chorus（心声合唱）
> 一个基于 LangGraph 构建的心理学驱动多代理决策支持系统

---

## 1. 核心理念 (Core Concept)

Chorus 是一个用于辅助复杂决策的多代理系统 (Multi-Agent Decision Support System)。它不从外部给出建议，而是**从内部模拟人类心理在做决定时的冲突与整合过程**，通过多个代表不同心理维度的 Agent 进行循环博弈，最终输出一份经过共识审计的决策报告。

**核心假设：** 人类的决策失误，往往不是因为缺乏信息，而是因为某一个心理维度（如短期情绪或习惯性回避）压制了其他维度。Chorus 的目标是让每个维度都被如实"听见"，再通过逻辑推理找到最优解。

**技术类比：** 整个决策过程是一个**状态机 (State Machine)**。每一个选项（如一个 Job Offer）是输入，系统通过 7 个心理节点进行过滤和评估，触发循环直到达成共识，最终输出报告。

---

## 2. 心理学基础 (Psychological Foundations)

### 2.1 双过程理论 (Dual Process Theory)

Kahneman 的双过程理论是整个系统的底层框架：人类有两套并行运作的决策系统，它们的冲突是大多数决策困境的根源。

| 系统 | 特征 | 对应脑区 | 在 Chorus 中的映射 |
|------|------|----------|-------------------|
| System 1（快思考） | 自动、直觉、情绪驱动 | 杏仁核、边缘系统 | 情绪侦探、躯体预言家 |
| System 2（慢思考） | 刻意、分析、逻辑驱动 | 前额叶皮层 | 逻辑法官 |

其余四个 Agent（意义向导、自我叙述者、良知证人、关系守护者）代表更深层的**价值与身份驱动层**，在时间尺度上比 System 1/2 更长期，但在日常决策中经常被忽视。

### 2.2 Schwartz 基本人类价值观理论 (Theory of Basic Human Values)

Schwartz 模型是目前心理学领域验证最广泛的价值观框架，10 个基本价值观排列在一个**圆形动机连续体**上，相邻价值兼容，对角价值天然冲突。

**10 个基本价值观：**

| # | 价值观 | 英文 | 核心动机 | 典型表现 |
|---|--------|------|----------|----------|
| 1 | 自主导向 | Self-Direction | 独立思考与行动 | 创造力、自由、好奇心 |
| 2 | 刺激 | Stimulation | 兴奋感与新奇体验 | 冒险、多样性、挑战 |
| 3 | 享乐 | Hedonism | 感官愉悦与自我满足 | 享受生活、当下满足 |
| 4 | 成就 | Achievement | 个人能力的社会认可 | 雄心、成功、能力展示 |
| 5 | 权力 | Power | 控制资源与他人 | 权威、财富、地位 |
| 6 | 安全 | Security | 稳定性与规避威胁 | 秩序、安全感、可预测性 |
| 7 | 顺从 | Conformity | 抑制可能伤害他人的行为 | 服从规则、自我约束 |
| 8 | 传统 | Tradition | 尊重文化与习俗 | 谦逊、敬虔、接受命运 |
| 9 | 善意 | Benevolence | 维护身边人的福祉 | 忠诚、诚实、责任感 |
| 10 | 普世 | Universalism | 理解和保护所有人与自然 | 平等、社会正义、环保 |

**两条核心张力轴：**

```
            自我超越 (Self-Transcendence)
         Universalism ←→ Benevolence
                ↑                  ↑
开放变革  Self-Direction        Tradition  保守秩序
(Openness) Stimulation          Conformity (Conservation)
                ↓                  ↓
         Hedonism → Achievement → Power
            自我增强 (Self-Enhancement)
```

**工程实现：** 用户 Profile 里存储一个 `value_vector: dict[str, float]`，每个维度权重 0.0–1.0，由用户在初始化时填写或从笔记中提取后校准。良知证人 Agent 在评估时直接使用此向量，其他 Agent 在最终共识加权时也参考它。

---

## 3. 决策维度与角色定义 (Agent Personas)

系统共 7 个 Agent，每个对应一个独立的心理学维度，听名称即可理解其功能。

| 维度 | 角色名称 | 英文 | 心理学依据 | 核心职能 |
|------|----------|------|-----------|----------|
| 逻辑推理 | **逻辑法官** | Arbiter | System 2 / 前额叶 | 代价收益分析、概率判断、因果推导 |
| 情绪直觉 | **情绪侦探** | Empath | System 1 / 情绪评价理论 | 挖掘隐藏的真实情绪信号，识别恐惧、兴奋、厌恶 |
| 身体预警 | **躯体预言家** | Soothsayer | 躯体标记假说 (Damasio) | 预测决策对压力水平、能量、睡眠的影响 |
| 意义目的 | **意义向导** | Compass | 意义疗法 (Frankl) | 评估决策是否符合用户的生命叙事与长期目标 |
| 自我身份 | **自我叙述者** | Narrator | 自我概念理论 | 检测决策是否与"我认为自己是什么人"产生冲突 |
| 道德校准 | **良知证人** | Conscience | Schwartz 价值观向量 | 如实呈现决策与用户价值观向量之间的偏差，不裁定只陈述 |
| 关系影响 | **关系守护者** | Guardian | 关系依附理论 | 评估决策对家庭、伴侣、核心社交圈的影响 |

> **角色设计原则：** 良知证人的职能是"如实呈现"而非"裁定"——它陈述偏差，但不判断对错，最终权衡由共识机制完成。这避免了道德维度过度压制其他维度。

### 3.1 各维度心理学理论来源

**Arbiter — 双过程理论 System 2**
核心来源：Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
理论要点：人类存在两套并行决策系统。System 2 是慢速、刻意、需要认知资源的分析模式，负责概率推断、逻辑演绎和代价收益计算。Kahneman 指出 System 2 常被 System 1 的直觉劫持，导致"看起来理性"的决定实际仍被情绪驱动。Arbiter 的角色是确保 System 2 的声音被充分听见，而不是被其他维度的强信号掩盖。
参考补充：Evans, J.S.B.T. (2008). "Dual-Processing Accounts of Reasoning, Judgment, and Social Cognition." *Annual Review of Psychology*, 59, 255–278.

---

**Empath — 情绪评价理论（Appraisal Theory）**
核心来源：Lazarus, R.S. (1991). *Emotion and Adaptation*. Oxford University Press.
理论要点：情绪不是对事件的直接反应，而是个体对事件"与自身目标和资源的关联程度"进行评价后产生的结果。不同的评价维度（是否与目标相关、是否可控、谁应负责）决定产生何种情绪。Empath 的工作是还原这套评价过程——用户叙述中流露的情绪信号，往往比理性陈述更忠实地反映其真实关切。
参考补充：Frijda, N.H. (1986). *The Emotions*. Cambridge University Press. / Barrett, L.F. (2017). *How Emotions Are Made*. Houghton Mifflin Harcourt.（情绪的构建理论，强调情绪是预测性建构而非硬编码反应）

---

**Soothsayer — 躯体标记假说（Somatic Marker Hypothesis）**
核心来源：Damasio, A.R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.
理论要点：身体状态（心跳加速、肌肉紧张、肠胃反应）会在无意识层面标记过往决策的结果，并在面临类似情境时以"预感"形式浮现，影响判断。Damasio 通过腹内侧前额叶损伤患者的研究发现：切断身体信号的人反而无法做出好决策，即使逻辑推理能力完好。Soothsayer 将这些身体预警信号翻译成可供分析的维度。
参考补充：Damasio, A.R. (1996). "The somatic marker hypothesis and the possible functions of the prefrontal cortex." *Philosophical Transactions of the Royal Society B*, 351, 1413–1420. / Bechara, A., Damasio, H., Tranel, D., & Damasio, A.R. (1997). "Deciding advantageously before knowing the advantageous strategy." *Science*, 275, 1293–1295.（Iowa Gambling Task 实验验证）

---

**Compass — 意义疗法（Logotherapy）**
核心来源：Frankl, V.E. (1959). *Man's Search for Meaning*. Beacon Press.
理论要点：Frankl 在纳粹集中营的经历中发展出意义疗法：人类最基本的驱动力不是快乐或权力，而是对意义的追求（Will to Meaning）。当一个决策与个人的核心叙事和生命使命脱节时，即使在物质上合理，也会产生持续的空洞感。Compass 评估的正是这种"意义契合度"，而非短期的情绪满足或逻辑收益。
参考补充：Steger, M.F. (2009). "Meaning in Life." in *Oxford Handbook of Positive Psychology*. / Seligman, M.E.P. (2011). *Flourish*. Free Press.（PERMA 模型中 Meaning 维度的实证研究）

---

**Narrator — 叙事认同理论（Narrative Identity Theory）**
核心来源：McAdams, D.P. (1993). *The Stories We Live By: Personal Myths and the Making of the Self*. Guilford Press.
理论要点：McAdams 认为自我认同本质上是一个不断修订的"个人神话"——人们通过构建连贯的生命故事来理解自己是谁。当某个决策与这个内在故事的主角形象产生矛盾时（如"我一直是敢于冒险的人，但这次我选择了稳定"），会触发认知失调和身份焦虑。Narrator 的职责是识别这种张力，而非强制消解它。
参考补充：McAdams, D.P. (2001). "The psychology of life stories." *Review of General Psychology*, 5(2), 100–122. / Markus, H. (1977). "Self-schemata and processing information about the self." *Journal of Personality and Social Psychology*, 35(2), 63–78.（自我图式理论，描述自我概念如何过滤信息）

---

**Conscience — Schwartz 基本人类价值观理论**
核心来源：Schwartz, S.H. (1992). "Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries." *Advances in Experimental Social Psychology*, 25, 1–65.
理论要点：Schwartz 通过跨文化研究（20+ 个国家）验证了 10 个基本价值观的普遍结构，排列在圆形动机连续体上，相邻价值兼容，对角价值天然冲突（如"自主导向"与"顺从"）。个体持有的价值观优先序差异，是预测行为和决策的核心变量。Conscience 将用户的 `value_vector` 与当前决策做向量对齐分析，只呈现偏差，不裁定方向。
参考补充：Schwartz, S.H. (2012). "An Overview of the Schwartz Theory of Basic Values." *Online Readings in Psychology and Culture*, 2(1). / Schwartz, S.H., & Bilsky, W. (1987). "Toward a universal psychological structure of human values." *Journal of Personality and Social Psychology*, 53(3), 550–562.

---

**Guardian — 依附理论（Attachment Theory）**
核心来源：Bowlby, J. (1969). *Attachment and Loss, Vol. 1: Attachment*. Basic Books.
理论要点：Bowlby 提出依附系统是人类进化出的核心生存机制：与亲密他人的联结感威胁时，会激活强烈的焦虑反应。Ainsworth 后续发展出安全型、焦虑型、回避型三种依附风格，直接影响个体在人际决策中的风险评估和牺牲意愿。Guardian 评估的是决策对依附关系网络的冲击——不仅是客观影响，还包括决策者的依附风格如何放大或压缩这种感知。
参考补充：Ainsworth, M.D.S. (1978). *Patterns of Attachment*. Erlbaum. / Hazan, C., & Shaver, P. (1987). "Romantic love conceptualized as an attachment process." *Journal of Personality and Social Psychology*, 52(3), 511–524.（将依附理论延伸至成人亲密关系）

---

### 3.2 Agent 天然张力结构

高冲突对（辩论阶段优先进入对抗）：

| 张力对 | 心理学原因 |
|--------|-----------|
| Arbiter ↔ Empath | System 2 vs System 1，最经典的冲突 |
| Arbiter ↔ Soothsayer | 理性收益 vs 身体成本 |
| Compass ↔ Arbiter | 意义无法被效用函数覆盖 |
| Narrator ↔ Guardian | 自我实现 vs 对他人的影响 |

天然盟友（倾向一致，通常不进入辩论）：

| 盟友对 | 原因 |
|--------|------|
| Empath + Soothsayer | 同属 System 1 驱动 |
| Compass + Narrator | 同属身份层 |
| Arbiter + Conscience | 同属分析性 |

---

## 4. 系统架构与机制 (System Architecture)

### 4.1 循环博弈机制 (Cycling & Reflection)

与传统线性 Chain 不同，Chorus 使用 LangGraph 的**循环 (Cycles)** 特性。

**机制：** 当逻辑法官通过某个决策，但躯体预言家预测到高压力风险时，系统触发**回溯边 (Back-edge)**，强制逻辑节点重新评估"收益是否足以补偿身体成本"。循环持续直到所有 Agent 达成共识，或进入一个已知的拮抗状态（如"逻辑最优但情绪强烈抗拒"）并在报告中明确标注。

```
用户输入决策选项
       ↓
  [Intake Node] ← 解析自然语言，加载 value_vector
       ↓
  [Decision Classifier] ← LLM 归类到 7 种决策类型，加载预置场景权重模板
       ↓
  [Bias Detection Layer] ← LLM 检测 13 种认知偏误，生成 Bias Metadata
       ↓
  ┌──────────────────────────────────────────────────┐
  │  逻辑法官 → 情绪侦探 → 躯体预言家                  │
  │      ↘          ↓          ↙                    │  ← 各 Agent 输出倾向向量
  │       意义向导 ← 自我叙述者                        │  循环层
  │           ↓                                     │
  │  良知证人 → 关系守护者                             │
  └──────────────────────────────────────────────────┘
       ↓
  [Entropy Monitor] ← 计算向量散度，判断熵值
  熵值高 → 深度辩论模式（回溯边）
  熵值收敛 → 推进到共识节点
       ↓ 达成共识 / 标注拮抗
  [Consensus Node] → 输出决策报告
       ↓
  [Persona Updater] ← 存储决策偏差记录到 SQLite
```

**熵值冲突检测 (Entropy-based Conflict Detection)：** 每个 Agent 在输出观点的同时，输出一个对当前选项的"倾向向量 (Preference Vector)"（如 `+0.8` 表示强烈支持，`-0.6` 表示明显反对）。Entropy Monitor 节点计算所有向量的散度：

- 散度高（高熵）→ 各维度意见极度发散 → 强制进入深度辩论模式，触发回溯边
- 散度收敛（低熵）→ 各维度趋于一致 → 推进至 Consensus Node

防止两种失效模式：Deadlock（无限循环）和半生不熟的平庸共识（加权平均掩盖了"理智极度想去但身体极度排斥"这类深度拮抗信号）。

**熵阈值联动：** 触发深度辩论的阈值不是硬编码的，而是与 `value_vector` 联动。Security 权重高的用户阈值更低（更保守，更容易触发深度讨论）；Stimulation 权重高的用户阈值更高（对不确定性容忍度更强）。

### 4.2 状态持久化与时间旅行 (State Persistence / Time Travel)

每一次决策运行的完整状态都被持久化，支持跨会话恢复和历史回溯。

**时间旅行场景：** "如果两年前我去了另一家公司，现在会怎样？"用户用自然语言描述"两年前的自己"，系统基于历史状态快照重新运行所有 Agent，观察模拟出的演化路径。

### 4.3 自然语言上下文输入 (Narrative Context)

**不使用纯结构化数据的原因：** 结构化数据（薪资数字、通勤时间）覆盖范围窄，且会让 AI 过度锚定少量字段，不符合真实决策的权重分布。

**推荐方案：** 用户在初次使用时写一段自由叙述（类似心理咨询初次访谈），涵盖当前生活状态、过去重大决策的感受、对未来的想象。

- Intake Node 从 SQLite 加载用户的 Schwartz 向量；首次使用时通过 `interrupt()` 暂停，等待用户逐维度填写后写入 DB
- 所有 Agent 在推理时同时访问**原始叙述**，确保信息密度均匀，避免数字锚定
- 从 Obsidian/Notion 笔记提取的内容，经用户校准后进入同一流程

### 4.4 按需联网扩展 (On-Demand Web Search)

**机制：** 系统不预设固定的外部数据源，而是在 Intake Node 解析用户叙述时，**主动识别其中存在信息缺口的实体**，触发联网搜索补充上下文，再交给各 Agent 使用。

**触发逻辑：** 当叙述中出现以下类型的实体时，系统判断为可扩展信息点：

| 实体类型 | 信息缺口 | 搜索补充内容 |
|----------|----------|-------------|
| 公司名称 | 企业文化、真实评价未知 | Glassdoor 评分、近期新闻、融资状态 |
| 技术栈 / 职位 | 市场需求与薪资水平未知 | 近期招聘趋势、薪资区间 |
| 城市 / 地址 | 生活成本、通勤信息未知 | 房价租金、交通时长 |
| 行业方向 | 宏观环境未知 | 行业增长率、裁员风险信号 |

**设计原则：**

- 搜索结果以**补充背景**的形式注入 Agent 的 context，而非替代用户叙述
- 用户可以在报告中看到哪些信息来自联网扩展，保持透明
- 若搜索结果与用户叙述有出入（如用户描述的公司氛围与 Glassdoor 评价不符），系统将此作为一个**认知偏差信号**传递给情绪侦探 Agent 处理

### 4.5 认知偏误过滤器 (Cognitive Bias Middleware)

**机制：** 在 Intake Node 之后、心理节点之前，插入 Bias Detection Layer。类比 AOP 中的 Around Advice——数据进入任何 Agent 之前先经过拦截检查。

**检测方式：** 使用独立 LLM 调用（`temperature=0`，专用系统 prompt），输出结构化的 `list[BiasFlag]`，与下游 7 个 Agent 的调用完全隔离。

目前识别 13 种认知偏误：

| 偏误（英文名） | 典型表现 | 主要传递给 |
|---------------|---------|-----------|
| Sunk Cost Fallacy | 已投入大量时间/资金难以放弃 | 逻辑法官 |
| Recency Effect | 近期事件主导整体判断 | 情绪侦探 |
| Bandwagon Effect | 以他人选择为主要依据 | 自我叙述者 |
| Black-and-White Thinking | 非此即彼，忽视中间路径 | 逻辑法官 |
| Confirmation Bias | 正/负信息严重失衡 | 所有 Agent |
| Loss Aversion | 对损失的恐惧远超对等值收益的期待 | 逻辑法官、情绪侦探 |
| Status Quo Bias | 将改变本身视为风险 | 意义向导、自我叙述者 |
| Anchoring Effect | 被某个具体数字/时间点过度锚定 | 逻辑法官 |
| Overconfidence Bias | 乐观程度超出实际依据 | 躯体预言家、良知证人 |
| Emotional Reasoning | 以当下情绪作为判断事实的依据 | 逻辑法官 |
| Planning Fallacy | 低估所需时间/资源 | 躯体预言家、逻辑法官 |
| Catastrophizing | 将负面结果想象到极端最坏情形 | 情绪侦探、躯体预言家 |
| Should Statements | 用道德义务感替代对真实意愿的探索 | 良知证人、自我叙述者 |

**设计原则：** 与良知证人保持一致——只呈现 Flag，不做价值判断。偏误标注作为 Metadata 传给对应 Agent，由 Agent 自行决定如何加权，不强制干预结论。

### 4.6 场景感知动态权重 (Contextual Dynamic Weighting)

**问题：** 静态 `value_vector` 无法区分不同决策类型中各心理维度的优先级差异。"换工作"和"家庭搬迁"对各 Agent 的重要性分布截然不同。

**机制：** Decision Classifier 节点使用 cache-aside 模式管理场景权重模板。

```
Decision Classifier
       ↓
  LLM 将叙述归类到 7 种固定决策类型（Literal 枚举约束，temperature=0）
       ↓
  按类型查 SQLite scene_templates（首次运行自动写入种子数据）
       ↓
  返回对应预置权重模板
```

**7 种内置决策类型：**

| 决策类型 | 中文 | 主导 Agent | 低权重 Agent |
|----------|------|-----------|-------------|
| career | 职业与事业 | 逻辑法官、意义向导 | 关系守护者 |
| finance | 财务与资产 | 逻辑法官、良知证人 | 情绪侦探 |
| relationship | 亲密关系与家庭 | 情绪侦探、关系守护者 | 逻辑法官 |
| relocation | 居住与迁移 | 关系守护者、躯体预言家 | 情绪侦探 |
| health | 健康与身体 | 躯体预言家、情绪侦探 | 良知证人 |
| identity | 身份认同与自我成长 | 自我叙述者、意义向导 | 关系守护者 |
| ethics | 伦理与社会责任 | 良知证人、意义向导 | 躯体预言家 |

7 种类型覆盖了绝大多数人生重大决策场景，类型范围固定以保证分类一致性。

**最终权重公式：** `final_weight = value_vector × scene_weight_template`（逐维度相乘，归一化）

用户在确认模板后仍可在生成报告前手动微调各 Agent 权重，覆盖模板默认值。

### 4.7 心理画像演化 (Evolving Persona)

**问题：** 人的价值观会随时间和环境变化，单次校准的 `value_vector` 无法捕捉这种演化。

**机制：** 每次决策完成后，Persona Updater 节点将各 Agent 初始意见与最终决策的偏差存入 SQLite `decision_history` 表。

**历史模式呈现：** 下次 Intake 阶段，系统读取记录并在报告末附上统计，使用数据而非人格化表述：

> 历史参考：过往 5 次职业类决策中，最终决策与躯体预言家意见的平均对齐度为 0.31（满分 1.0）。本次建议重点参考该 Agent 的分析。

不使用"你倾向于……"的人格化表述，避免 labeling effect（被反复贴标签的自我认知容易固化为自我实现预言）。存储使用 SQLite 而非 Vector DB（历史记录结构规整，不需要语义检索）。

---

## 5. 输出格式 (Output Format)

每次决策运行结束后，Consensus Node 输出结构化报告，包含：

1. **各维度评分摘要：** 7 个 Agent 的独立评估结论与信心分
2. **共识状态：** 达成一致 / 存在拮抗（标注哪两个维度冲突）
3. **价值观偏差提示：** 良知证人输出的向量偏差报告
4. **推荐决策：** 综合加权后的最优解，及其核心依据
5. **风险提示：** 被压制的少数维度意见（防止决策盲点）
6. **认知偏误标注：** Bias Detection Layer 识别的偏误 Flag 及对应 Agent 的回应
7. **历史模式参考：** 基于 `decision_history` 的同类决策偏差统计（如有历史记录）
8. **压力测试环境：** 用户可手动调节某个 Agent 的权重或修改输入假设（如"如果薪资不是考虑因素"），系统通过 `update_state` 实时重跑图，并对比两份报告的差异

---

## 6. 为什么有意义 (Why It Matters)

**克服认知偏误：** Chorus 是对"损失厌恶"、"短期情绪劫持"和"自我叙事过度防御"的算法修正——不是替代人做决定，而是确保每个心理维度都被听见。

**可复制性：** 任何面临重大人生选择（职业转向、移民决策、关系选择）的人，通过更换 `value_vector` 和初始叙述，即可复用同一套框架。

**心理学合法性：** 系统的每个组件都有对应的学术理论支撑——双过程理论、Schwartz 模型、躯体标记假说、Frankl 意义疗法、关系依附理论——不是隐喻式的角色扮演，而是对人类决策心理机制的系统性模拟。
