根据官方文档和生产级最佳实践，构建一个标准的 LangGraph 项目应遵循以下工具选择、目录结构和核心开发规则(my_agent 更换成具体项目名称)：

### 一、 推荐使用的工具链

1.  **开发与包管理**：
    *   使用 `pip` 或 `uv` 进行包管理。
    *   **LangGraph CLI**：用于本地运行和测试，它默认使用当前目录下的 `langgraph.json`。
2.  **持久化层（Checkpointer）**：
    *   **本地开发**：推荐使用 `SqliteSaver` 或 `InMemorySaver`。
    *   **生产环境**：强烈推荐使用 **PostgreSQL** (`PostgresSaver`)，它支持高并发和大规模存储。
3.  **调试与可观测性**：
    *   **LangSmith**：用于追踪请求、调试 Agent 行为以及可视化执行路径。
    *   **LangSmith Studio**：提供可视化的 UI 界面，用于原型设计、状态检查和中断调试。
4.  **生产部署**：
    *   **LangSmith Deployment**：托管平台，负责基础设施、扩展和有状态 Agent 的运行。

### 二、 推荐的项目目录结构

一个典型的生产级项目（例如 `my-app`）应采用模块化结构，将状态定义、工具和节点逻辑分离：

```text
my-app/
├── src/my_agent/            # 所有的业务逻辑代码放在这里
│   ├── __init__.py
│   ├── agent.py             # 构建图的代码 (StateGraph)
│   ├── nodes/               # 节点函数逻辑
│   │   └── __init__.py
│   ├── state/               # State (状态) 的定义
│   │   └── __init__.py
│   ├── tools/               # 工具 (@tool) 的定义
│   │   └── __init__.py
│   ├── infrastructure/      # 基础设施
│   │   └── __init__.py
│   └── utils/               # 其他辅助工具
│       └── __init__.py
├── .env              # 环境变量 (API Keys 等)
├── langgraph.json    # LangGraph 核心配置文件
├── pyproject.toml    # 依赖管理 (或使用 requirements.txt)
└── README.md
```

### 三、 核心构建规则总结

#### 1. 配置文件规范 (`langgraph.json`)
这是项目的核心，用于指定图的入口、依赖和环境：
*   **`graphs`**：定义图的名称和对应的 Python 变量路径（格式如 `"path/file.py:variable"`）。
*   **`dependencies`**：列出项目所需的 Python 包。
*   **`env`**：指定环境变量文件路径。

#### 2. 代码编写最佳实践
*   **状态定义 (State)**：必须使用 `TypedDict` 或 `Pydantic`；对于对话历史，必须使用 `Annotated[list, add_messages]` 以确保消息是增量追加而非覆盖。
*   **节点逻辑 (Nodes)**：
    *   优先使用 **异步函数** (`async def`)。
    *   **增量更新原则**：节点应返回一个包含更新字段的字典，严禁直接修改传入的输入对象。
*   **图编排 (Graph)**：必须显式使用 `START` 和 `END` 节点；路由逻辑需通过路由函数返回明确的字符串标签。
*   **工具绑定 (Tools)**：使用标准 `@tool` 装饰器定义工具，并确保带有详尽的 Docstring 供模型理解。
*   **持久化 (Persistence)**：编译图时必须传入 `checkpointer`；调用时必须包含 `thread_id` 以支持多轮对话和断点恢复。

---

### 四、 可复制的项目初始化模版 (Markdown)

```markdown
# LangGraph 项目标准配置示例

## 1. 核心配置文件: langgraph.json
{
  "dependencies": ["langchain_openai", "langchain_anthropic"],
  "graphs": {
    "my_agent": "./my_agent/agent.py:graph"
  },
  "env": "./.env"
}

## 2. 状态定义: my_agent/state/__init__.py
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 使用 add_messages 确保消息列表是增量合并的
    messages: Annotated[list, add_messages]
    input_context: str

## 3. 工具定义: my_agent/tools/__init__.py
from langchain_core.tools import tool

@tool
def search_database(query: str):
    """在此处输入详细的工具描述，这对 LLM 决策至关重要。"""
    return "查询结果..."

## 4. 图构建: my_agent/agent.py
from langgraph.graph import StateGraph, START, END
from my_agent.state import AgentState
from my_agent.nodes import call_model, tool_node

workflow = StateGraph(AgentState)

workflow. add_node("agent", call_model)
workflow. add_node("tools", tool_node)

workflow. add_edge(START, "agent")
# 这里的逻辑通常配合条件边实现
workflow. add_edge("tools", "agent")

# 生产环境编译时需要 checkpointer
# graph = workflow.compile(checkpointer=postgres_saver)
```