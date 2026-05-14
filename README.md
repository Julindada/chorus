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

**所需 API Key**

| 变量名 | 用途 | 获取地址 |
|--------|------|---------|
| `LLM_API_KEY` | LLM（百炼 / Qwen） | console.aliyun.com/bailian |
| `TAVILY_API_KEY` | 现实核查联网搜索 | app.tavily.com |
| `LANGSMITH_API_KEY` | LangGraph Studio 追踪（可选） | smith.langchain.com |

在项目根目录创建 `.env` 文件（可复制 `.env.example` 修改）：

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

## 运行方式

### 方式一：本地安装

**依赖：**
- Python 3.11+
- uv（`pip install uv` 或 `brew install uv`）

```bash
uv sync
uv run chorus
```

### 方式二：Docker

**依赖：**
- Docker

```bash
docker compose run --rm chorus
```

用户档案和报告持久化保存在 Docker volume `chorus_data` 中，容器删除后不丢失。

**从 volume 中取出报告：**

```bash
docker compose run --rm chorus cat /home/appuser/data/reports/<文件名>.md > report.md
```

## 使用流程

启动后按提示交互：

1. **输入用户名** — 用于保存和加载你的 Schwartz 价值观档案
2. **首次使用** — 选择逐项输入或粘贴 JSON 建立价值观档案（10 个维度，0.0–1.0）
3. **描述决策情境** — 越详细分析越准确
4. **输入候选选项** — 逐行输入，空行结束（至少 2 个）
5. **等待分析** — 7 个 Agent 并行评估，分歧时自动进入辩论轮次
6. **查看报告** — 分析完成后可选择保存为 Markdown 文件

**价值观档案 JSON 格式（首次使用）：**

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

