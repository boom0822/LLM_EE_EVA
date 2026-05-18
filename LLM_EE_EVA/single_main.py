import json
import os
import pandas as pd
from tqdm import tqdm

from config import EVENT_SCHEMA_PATH, TRIGGER_STATS_DATA_PATH
from llm_clients import get_kimi
from utils import (
    load_duee,
    load_event_schema,
    build_trigger_examples,
    format_schema
)
from processor import process_sample

# ================= 配置 =================
DATA_PATH = "data/DuEE/ceshi.json"       # 测试数据路径
OUTPUT_DIR = "outputs"
MODEL_NAME = "kimi"
MAX_SAMPLES = 200                        # 可调整
TRIGGER_STATS_SAMPLES = 2000             # 触发词示例所用数据量

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 1. 加载数据 =================
print("Loading test data...")
data = load_duee(DATA_PATH, max_samples=MAX_SAMPLES)
print(f"Loaded {len(data)} samples")

# ================= 2. 构建 schema 和触发词示例 =================
print("Loading event schema...")
schema = load_event_schema(EVENT_SCHEMA_PATH)
print(f"Loaded {len(schema)} event types")

print("Building trigger examples from training data...")
trigger_examples = build_trigger_examples(TRIGGER_STATS_DATA_PATH, max_samples=TRIGGER_STATS_SAMPLES)
schema_text = format_schema(schema, trigger_examples)

# ================= 3. 初始化模型 =================
print(f"\n========== Testing {MODEL_NAME} ==========\n")
llm = get_kimi()

# ================= 4. 逐条推理与评估 =================
total_tp = 0
total_fp = 0
total_fn = 0
valid_count = 0          # 有效预测（非空）的样本数
empty_pred_samples = []  # 预测为空的样本
save_results = []        # 所有样本的 gold/pred

for idx, sample in enumerate(tqdm(data, desc="Processing")):
    text = sample["text"]
    gold_events = sample["event_list"]

    res = process_sample(text, gold_events, llm, schema_text)

    pred_events = res["pred"]
    metrics = res["metrics"]
    model_output = res["model_output"]
    error = res["error"]

    # 有效预测：累加 TP/FP/FN
    if metrics is not None:
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        valid_count += 1
    else:
        # 空预测：记录详细信息
        gold_summary = [(ev["event_type"], ev["trigger"]) for ev in gold_events]
        tqdm.write(f"\n[EMPTY PRED] Sample {idx} | gold: {gold_summary}")
        tqdm.write(f"Text: {text[:150]}...")
        tqdm.write(f"Model output (first 300 chars): {model_output[:300]}")
        empty_pred_samples.append({
            "idx": idx,
            "text": text,
            "gold": gold_events,
            "gold_summary": gold_summary,
            "pred": pred_events,
            "model_output": model_output,
            "error": error
        })

    save_results.append({
        "text": text,
        "gold": gold_events,
        "pred": pred_events
    })

# ================= 5. 计算总体指标（仅基于有效预测） =================
if valid_count > 0:
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
else:
    precision = recall = f1 = 0.0

result = {
    "model": MODEL_NAME,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "valid_samples": valid_count,
    "empty_samples": len(empty_pred_samples),
    "total_samples": len(data)
}

# ================= 6. 保存结果 =================
with open(f"{OUTPUT_DIR}/{MODEL_NAME}_result.json", "w", encoding="utf-8") as f:
    json.dump(save_results, f, ensure_ascii=False, indent=2)

if empty_pred_samples:
    with open(f"{OUTPUT_DIR}/{MODEL_NAME}_empty_pred.json", "w", encoding="utf-8") as f:
        json.dump(empty_pred_samples, f, ensure_ascii=False, indent=2)

results_df = pd.DataFrame([result])
print("\n===== FINAL RESULT =====\n")
print(results_df)
results_df.to_csv(f"{OUTPUT_DIR}/{MODEL_NAME}_final_metrics.csv", index=False)
print(f"Detailed results saved to {OUTPUT_DIR}/{MODEL_NAME}_result.json")
if empty_pred_samples:
    print(f"Empty pred details saved to {OUTPUT_DIR}/{MODEL_NAME}_empty_pred.json")
print(f"Total empty predictions: {len(empty_pred_samples)} out of {len(data)} samples (excluded from metrics).")