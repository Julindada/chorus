# Chorus 心声合唱

一个基于 LangGraph 构建的心理学驱动多 Agent 决策支持系统。

Chorus 不从外部给出建议，而是模拟人类心理在做决定时的内在冲突与整合过程。7 个代表不同心理维度的 Agent 并行评估用户的处境，分歧激烈时进入循环博弈，最终输出一份经过共识审计的决策报告。

## 7 个心理维度

| Agent | 心理学依据 | 核心职能 |
|-------|-----------|---------|
| 逻辑法官 Arbiter | System 2 / 前额叶 | 代价收益分析、概率判断 |
| 情绪侦探 Empath | System 1 / 情绪评价理论 | 挖掘隐性情绪信号，识别恐惧与兴奋 |
| 躯体预言家 Soothsayer | 躯体标记假说 (Damasio) | 预测对压力、能量、睡眠的影响 |
| 意义向导 Compass | 意义疗法 (Frankl) | 评估与生命叙事和长期目标的契合度 |
| 自我叙述者 Narrator | 叙事认同理论 (McAdams) | 检测与"自我认知"的冲突 |
| 良知证人 Conscience | Schwartz 价值观向量 | 如实呈现决策与价值观的偏差，不裁定 |
| 关系守护者 Guardian | 关系依附理论 | 评估对家庭、伴侣、核心社交圈的影响 |

## 准备工作

**依赖安装**

```bash
# 需要 Python 3.11+，推荐使用 uv
pip install uv
uv sync
```

**所需 API Key**

| 变量名 | 用途 | 获取地址 |
|--------|------|---------|
| `DASHSCOPE_V2_API_KEY` | LLM（百炼 / Qwen） | console.aliyun.com/bailian |
| `TAVILY_API_KEY` | 现实核查联网搜索（可选） | app.tavily.com |
| `LANGSMITH_API_KEY` | LangGraph Studio 追踪 | smith.langchain.com |

在项目根目录创建 `.env` 文件：

```bash
DASHSCOPE_V2_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...
```

## 通过 LangGraph Studio 使用

LangGraph Studio 是一个可视化调试界面，可以实时观察 Agent 的每一步推理过程。

### 第一步：启动本地服务

```bash
uv run langgraph dev
```

启动成功后终端会输出类似：

```
Ready!
- API: http://127.0.0.1:2024
- Docs: http://127.0.0.1:2024/docs
- LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 第二步：打开 Studio

点击终端中的 LangGraph Studio 链接，或前往 [smith.langchain.com/studio](https://smith.langchain.com/studio)，在连接地址栏填入 `http://127.0.0.1:2024`。

> 需要登录 LangSmith 账号（免费注册）。Studio 本身在本地运行，LangSmith 仅用于 UI 渲染和追踪记录。

### 第三步：创建对话并输入问题

在 Studio 左侧选择 `chorus` 图，点击 **New Thread**，在输入框填写以下 JSON：

```json
{
  "username": "your_name",
  "user_narrative": "描述你的处境和困惑，越详细越好。例如：我在现有公司工作 3 年，薪资稳定，但最近收到一个创业公司的 offer，薪资高 30% 但风险大，不确定是否该跳槽……",
  "decision_options": ["接受新 offer", "留在现公司", "再观望三个月"]
}
```

字段说明：
- `username`：用于保存你的 Schwartz 价值观画像，再次使用时自动加载
- `user_narrative`：完整描述你的处境、顾虑和背景，越详细分析越准确
- `decision_options`：列出你正在考虑的所有选项（2–5 个为宜）

### 第四步：填写价值观画像（首次使用）

如果是第一次使用该 `username`，系统会在 `intake_node` 暂停并发出中断请求，要求为 10 个 Schwartz 价值观维度打分。

在 Studio 的中断响应框填写（0.0 为最低，1.0 为最高）：

```json
{
  "self_direction": 0.8,
  "stimulation": 0.6,
  "hedonism": 0.4,
  "achievement": 0.9,
  "power": 0.3,
  "security": 0.7,
  "conformity": 0.2,
  "tradition": 0.3,
  "benevolence": 0.8,
  "universalism": 0.7
}
```

> 价值观画像保存在本地 SQLite 数据库，下次使用同一 `username` 时直接跳过此步骤。

### 第五步：等待分析完成

系统会依次执行：

```
intake → 分类决策类型 → 认知偏误检测 → 现实情况核查
  → 7 个 Agent 并行评估 → 熵值检测
    → （分歧大时）循环辩论 → 共识生成 → 输出报告
```

完成后在 State 面板的 `final_report` 字段查看完整 Markdown 报告，报告同时保存到 `data/reports/` 目录。

## 运行测试

```bash
uv run pytest tests/
```

## 项目结构

```
src/chorus/
├── graph.py              # 图构建与编译
├── state/                # DecisionState 定义
├── nodes/                # 各节点逻辑
│   ├── intake.py         # 用户画像加载
│   ├── classifier.py     # 决策类型分类
│   ├── bias.py           # 认知偏误检测
│   ├── reality.py        # 现实情况核查（Tavily）
│   ├── agent.py          # 7 个心理维度 Agent
│   ├── entropy_monitor.py # 熵值计算与路由
│   ├── debate.py         # 循环辩论
│   ├── consensus.py      # 共识生成
│   └── persona_updater.py # 报告输出与画像更新
├── infrastructure/       # 数据库 DAO 与配置
├── templates/            # 报告 Jinja2 模板
└── utils/                # 常量、工具函数
```
