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
