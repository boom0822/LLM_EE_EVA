"""
从 train.json 前 2000 条提取每个事件类型的所有触发词（去重），
保存到 data/trigger_examples.json。
"""
import json
from collections import defaultdict

DATA_PATH = "data/DuEE/train.json"   # 训练数据路径
MAX_SAMPLES = 2000
OUTPUT_PATH = "data/trigger_examples.json"

trigger_set = defaultdict(set)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= MAX_SAMPLES:
            break
        item = json.loads(line)
        for ev in item.get("event_list", []):
            etype = ev["event_type"]
            trigger = ev["trigger"]
            trigger_set[etype].add(trigger)

# 转为列表并排序
examples = {}
for etype, triggers in trigger_set.items():
    examples[etype] = sorted(list(triggers))

# 保存为 JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print(f"Trigger examples saved to {OUTPUT_PATH}")
print(f"Total event types: {len(examples)}")