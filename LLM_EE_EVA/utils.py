import json
import re
from collections import defaultdict

def load_duee(path, max_samples=5000):
    """加载 DUEE 格式数据集"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "text": item["text"],
                "event_list": item.get("event_list", [])
            })
            if len(data) >= max_samples:
                break
    return data

def normalize_event(event):
    """事件规范化为元组（用于严格匹配）"""
    event_type = event.get("event_type", "")
    trigger = event.get("trigger", "")
    args = []
    for a in event.get("arguments", []):
        args.append((a.get("role", ""), a.get("argument", "")))
    return (event_type, trigger, tuple(sorted(args)))

def load_event_schema(schema_path):
    """加载事件 schema 文件，返回 {event_type: [role_name, ...]}"""
    schema = {}
    with open(schema_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            schema[item["event_type"]] = [r["role"] for r in item["role_list"]]
    return schema

def build_trigger_examples(data_path, max_samples=2000):
    """
    从数据集中统计每个事件类型各触发词的出现次数，
    返回 {event_type: [trigger1, trigger2, ...]}（按频率降序取前5个）
    """
    trigger_freq = defaultdict(lambda: defaultdict(int))
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line)
            for ev in item.get("event_list", []):
                etype = ev["event_type"]
                trigger = ev["trigger"]
                trigger_freq[etype][trigger] += 1

    examples = {}
    for etype, freq_dict in trigger_freq.items():
        sorted_triggers = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        examples[etype] = [t for t, _ in sorted_triggers[:5]]
    return examples

def format_schema(schema_dict, trigger_examples=None):
    """
    将 schema 字典格式化为文本行，每行格式：
    - 事件类型: 角色1, 角色2, ... 常见触发词：xx、xx
    """
    lines = []
    for event_type in sorted(schema_dict.keys()):
        roles = schema_dict[event_type]
        roles_str = "、".join(roles) if roles else "无特定论元"
        # 附加触发词示例
        if trigger_examples and event_type in trigger_examples:
            trg_str = "、".join(trigger_examples[event_type])
            example_part = f"；常见触发词：{trg_str}"
        else:
            example_part = ""
        lines.append(f"- {event_type}：{roles_str}{example_part}")
    return "\n".join(lines)

def extract_json(text: str):
    """
    从模型回复中提取第一个有效的 JSON 数组或对象。
    处理 Markdown 代码块、前后空格等。
    """
    text = text.strip()
    # 去掉 Markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 匹配第一个数组
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass
    # 匹配第一个对象
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("无法提取有效的 JSON")

def load_trigger_examples(path):
    """
    加载触发词示例 JSON 文件，返回 {event_type: [trigger1, trigger2, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)