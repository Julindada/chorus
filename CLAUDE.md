# Chorus 项目规范

LangGraph 规范见 `@~/.claude/langgraph.md`。设计说明见 `doc/Chorus_TD.md`。

## 图入口

`langgraph.json` → `src/chorus/graph.py:graph`

## 新增 Agent

1. `utils/constants.py`：追加 `AGENT_NAMES`、`AGENT_PROMPTS`、`AGENT_VALUE_MAPPING`、各 `DECISION_TYPES` 的 `weights`
2. 无需改动 `nodes/` 或 `graph.py`

## State 字段约束

以下字段的 `Annotated` reducer 不可移除，移除会导致 Send 并行分支结果互相覆盖：
- `agent_stances`：`operator.or_`
- `stance_history`、`debate_history`：`operator.add`

`initial_stances` 由 `entropy_monitor_node` 在 `debate_round == 0` 时写入一次，之后只读，不要在其他节点修改。

## 运行

```bash
uv sync
uv run langgraph dev    # LangGraph Studio
uv run pytest tests/    # 单元测试
```
