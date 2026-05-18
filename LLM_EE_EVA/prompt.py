# prompt.py

def format_schema(schema_dict, trigger_examples=None):
    """
    将事件 schema 格式化为文本行。
    如果提供 trigger_examples，则附加常见触发词。
    """
    lines = []
    for event_type in sorted(schema_dict.keys()):
        roles = schema_dict[event_type]
        roles_str = "、".join(roles) if roles else "无特定论元"
        if trigger_examples and event_type in trigger_examples:
            trg_str = "、".join(trigger_examples[event_type])
            example_part = f"；常见触发词：{trg_str}"
        else:
            example_part = ""
        lines.append(f"- {event_type}：{roles_str}{example_part}")
    return "\n".join(lines)

# ========== 批量 prompt 模板 ==========
BATCH_PROMPT_HEADER = """你是一个专业的事件抽取模型。请严格按照以下事件类型和论元角色定义，从下面每一行给出的文本中抽取所有事件。

事件类型及可用论元角色："""

BATCH_PROMPT_BODY = """

以下是一组文本，每条文本以 "文本编号: 内容" 的形式给出。请为每条文本抽取事件，并返回一个 JSON 数组，数组中的每个元素对应一条文本的事件列表。
- 输出顺序必须与输入文本的顺序完全一致。

文本列表：
"""

BATCH_PROMPT_FOOTER = """

返回格式（严格的 JSON 数组）：
[
    [   // 第一条文本的事件列表
        {"event_type": "类型", "trigger": "触发词", "arguments": [{"role": "角色", "argument": "值"}]},
        ...
    ],
    [   // 第二条文本的事件列表
        ...
    ],
    ...
]

注意：
1. 事件类型必须来自上述定义，角色必须使用该类型对应的角色名，不得编造。
2. 请尽可能抽取所有可能的事件，即使是隐含的事件也要尝试识别，不要轻易返回空数组。
3. 不要输出任何解释，只返回 JSON 数组。
4. 如果某文本没有事件，对应位置为空列表 []。
5. 事件类型必须与定义完全一致，不允许添加或省略空格。
6. 论元 argument 的值必须严格从原文复制，不得修改、添加或删除任何字符（包括空格）
"""