from typing import Literal

DecisionType = Literal[
    "career",        # 职业与事业
    "finance",       # 财务与资产
    "relationship",  # 亲密关系与家庭
    "relocation",    # 居住与迁移
    "health",        # 健康与身体
    "identity",      # 身份认同与自我成长
    "ethics",        # 伦理与社会责任
]

DECISION_TYPES: dict[str, dict] = {
    "career": {
        "label": "职业与事业",
        "description": "职业规划、跳槽、升职、创业、事业方向选择",
        "weights": {
            "Arbiter": 0.9, "Empath": 0.4, "Soothsayer": 0.5,
            "Compass": 0.9, "Narrator": 0.7, "Conscience": 0.5, "Guardian": 0.3,
        },
    },
    "finance": {
        "label": "财务与资产",
        "description": "投资、买房、贷款、理财、创业资金、大额消费",
        "weights": {
            "Arbiter": 0.9, "Empath": 0.3, "Soothsayer": 0.6,
            "Compass": 0.5, "Narrator": 0.4, "Conscience": 0.8, "Guardian": 0.4,
        },
    },
    "relationship": {
        "label": "亲密关系与家庭",
        "description": "恋爱、分手、结婚、生育、家庭矛盾、亲子关系",
        "weights": {
            "Arbiter": 0.3, "Empath": 0.9, "Soothsayer": 0.5,
            "Compass": 0.6, "Narrator": 0.7, "Conscience": 0.5, "Guardian": 0.9,
        },
    },
    "relocation": {
        "label": "居住与迁移",
        "description": "搬家、换城市、出国、移民、异地定居",
        "weights": {
            "Arbiter": 0.7, "Empath": 0.4, "Soothsayer": 0.8,
            "Compass": 0.6, "Narrator": 0.5, "Conscience": 0.4, "Guardian": 0.9,
        },
    },
    "health": {
        "label": "健康与身体",
        "description": "治疗方案、手术、生活方式改变、心理健康、成瘾戒断",
        "weights": {
            "Arbiter": 0.6, "Empath": 0.7, "Soothsayer": 0.9,
            "Compass": 0.7, "Narrator": 0.5, "Conscience": 0.4, "Guardian": 0.5,
        },
    },
    "identity": {
        "label": "身份认同与自我成长",
        "description": "价值观冲突、人生方向、自我认知、信仰转变、角色转型",
        "weights": {
            "Arbiter": 0.4, "Empath": 0.6, "Soothsayer": 0.4,
            "Compass": 0.9, "Narrator": 0.9, "Conscience": 0.7, "Guardian": 0.4,
        },
    },
    "ethics": {
        "label": "伦理与社会责任",
        "description": "道德困境、社会责任、举报、利益冲突、公平正义",
        "weights": {
            "Arbiter": 0.5, "Empath": 0.5, "Soothsayer": 0.3,
            "Compass": 0.8, "Narrator": 0.5, "Conscience": 0.9, "Guardian": 0.7,
        },
    },
}

AGENT_PROMPTS: dict[str, str] = {
    "Arbiter": (
        "你是 Arbiter，理性仲裁者，代表「成就」与「自主导向」价值维度。\n"
        "从逻辑、效率和实际可行性出发分析决策，评估各选项的成本收益与风险边际。\n"
        "不带情绪，只看证据。"
    ),
    "Empath": (
        "你是 Empath，情感共鸣者，代表「享乐」与「刺激」价值维度。\n"
        "从情感体验和人际关系出发分析决策，关注这个选择会如何影响用户和重要他人的感受。\n"
        "感受即数据。"
    ),
    "Soothsayer": (
        "你是 Soothsayer，风险预判者，代表「安全」与「顺从」价值维度。\n"
        "从安全感和长远稳定性出发分析决策，识别潜在风险与不确定性，评估各选项的安全边际。\n"
        "未雨绸缪。"
    ),
    "Compass": (
        "你是 Compass，价值罗盘，代表「普世主义」与「自主导向」价值维度。\n"
        "从用户的核心价值观和人生方向出发分析决策，判断各选项与用户 value_vector 的对齐程度。\n"
        "方向比速度更重要。"
    ),
    "Narrator": (
        "你是 Narrator，叙事构建者，代表「自主导向」与「善意」价值维度。\n"
        "从个人成长和身份认同出发分析决策，思考这个选择在用户人生故事中的意义。\n"
        "你是谁，比你做了什么更重要。"
    ),
    "Conscience": (
        "你是 Conscience，道德良知，代表「普世主义」、「善意」与「传统」价值维度。\n"
        "从伦理和社会责任出发分析决策，评估各选项是否符合道德原则及其对他人的影响。\n"
        "公平地对待每一个受影响的人。"
    ),
    "Guardian": (
        "你是 Guardian，关系守护者，代表「善意」与「安全」价值维度。\n"
        "从保护重要关系出发分析决策，关注这个选择对家人、朋友等重要他人的影响。\n"
        "守护好对你重要的人。"
    ),
}

AGENT_NAMES: list[str] = [
    "Arbiter",     # 逻辑法官
    "Empath",      # 情绪侦探
    "Soothsayer",  # 躯体预言家
    "Compass",     # 意义向导
    "Narrator",    # 自我叙述者
    "Conscience",  # 良知证人
    "Guardian",    # 关系守护者
]

SCHWARTZ_DIMS: list[str] = [
    "self_direction",   # 自主导向
    "stimulation",      # 刺激
    "hedonism",         # 享乐
    "achievement",      # 成就
    "power",            # 权力
    "security",         # 安全
    "conformity",       # 顺从
    "tradition",        # 传统
    "benevolence",      # 善意
    "universalism",     # 普世
]
