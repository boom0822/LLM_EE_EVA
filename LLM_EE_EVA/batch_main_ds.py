import json
import os
import pandas as pd
from tqdm import tqdm
from prompt import BATCH_PROMPT_HEADER, BATCH_PROMPT_BODY, BATCH_PROMPT_FOOTER, format_schema
from config import EVENT_SCHEMA_PATH, TRIGGER_EXAMPLES_PATH
from llm_clients import get_deepseek   # 根据需要切换
from utils import (
    load_duee,
    load_event_schema,
    load_trigger_examples  # 新增
)
from batch_processor import process_batch

# ================= 配置 =================
DATA_PATH = "data/DuEE/train.json"
OUTPUT_DIR = "outputs"
MODEL_NAME = "deepseek"
MAX_SAMPLES = 5000
BATCH_SIZE = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 1. 加载数据 =================
print("Loading test data...")
data = load_duee(DATA_PATH, max_samples=MAX_SAMPLES)
print(f"Loaded {len(data)} samples")

# ================= 2. 加载事件 schema 和触发词示例 =================
print("Loading event schema...")
schema = load_event_schema(EVENT_SCHEMA_PATH)
print(f"Loaded {len(schema)} event types")

print("Loading trigger examples...")
trigger_examples = load_trigger_examples(TRIGGER_EXAMPLES_PATH)
schema_text = format_schema(schema, trigger_examples)

# ================= 3. 初始化模型 =================
print(f"\n========== Testing {MODEL_NAME} (batch size: {BATCH_SIZE}) ==========\n")
llm = get_deepseek()

# ================= 4. 批量推理与评估 =================
total_tp = 0
total_fp = 0
total_fn = 0
valid_count = 0
empty_pred_samples = []
all_preds = []

for start in tqdm(range(0, len(data), BATCH_SIZE), desc="Batches"):
    end = min(start + BATCH_SIZE, len(data))
    batch_data = data[start:end]
    batch_texts = [item["text"] for item in batch_data]
    batch_gold = [item["event_list"] for item in batch_data]

    results, model_output = process_batch(batch_texts, batch_gold, llm, schema_text)

    for i, res in enumerate(results):
        global_idx = start + i
        pred = res["pred"]
        metrics = res["metrics"]
        error = res["error"]
        text = batch_texts[i]
        gold = batch_gold[i]

        all_preds.append(pred)

        if metrics is not None:
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            valid_count += 1
        else:
            gold_summary = [(ev["event_type"], ev["trigger"]) for ev in gold]
            tqdm.write(f"\n[EMPTY PRED] Sample {global_idx} | gold: {gold_summary}")
            tqdm.write(f"Text: {text[:150]}...")
            empty_pred_samples.append({
                "idx": global_idx,
                "text": text,
                "gold": gold,
                "gold_summary": gold_summary,
                "pred": pred,
                "model_output": model_output,
                "error": error
            })

# ================= 5. 计算总体指标 =================
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
save_results = []
for item, pred in zip(data, all_preds):
    save_results.append({"text": item["text"], "gold": item["event_list"], "pred": pred})

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