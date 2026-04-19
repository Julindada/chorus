# Chorus 心声合唱

一个基于 LangGraph 构建的心理学驱动多 Agent 决策支持系统。

## 核心理念

Chorus 不从外部给出建议，而是模拟人类心理在做决定时的内在冲突与整合过程。通过 7 个代表不同心理维度的 Agent 并行评估、循环博弈，最终输出经过共识审计的决策报告。

## 7 个心理维度

| Agent | 心理学依据 |
|-------|-----------|
| 逻辑法官 Arbiter | System 2 / 前额叶，代价收益分析 |
| 情绪侦探 Empath | System 1 / 情绪评价理论，挖掘隐性情绪信号 |
| 躯体预言家 Soothsayer | 躯体标记假说，预测压力与能量影响 |
| 意义向导 Compass | 意义疗法，评估与生命叙事的契合度 |
| 自我叙述者 Narrator | 自我概念理论，检测身份认同冲突 |
| 良知证人 Conscience | Schwartz 价值观向量，呈现价值偏差 |
| 关系守护者 Guardian | 关系依附理论，评估对社交圈的影响 |

## 技术栈

- **框架**：LangGraph — 支持循环、Checkpointer、Send API 并行
- **LLM**：Claude Sonnet
- **持久化**：SQLite（本地开发） / PostgreSQL（生产）
- **包管理**：uv

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env .env.local  # 填写 ANTHROPIC_API_KEY、TAVILY_API_KEY

# 启动 LangGraph Studio
uv run langgraph dev
```

## 项目结构

```
src/chorus/
├── agent.py      # 图构建与编译
├── state/        # DecisionState 定义
├── nodes/        # 各节点逻辑
├── tools/        # @tool 工具定义
└── utils/        # 辅助函数
```
