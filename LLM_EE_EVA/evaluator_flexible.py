import json

# ========== 可配置参数 ==========
TRIGGER_MODE = "exact"       # 触发词匹配: "exact" 或 "substring"
ARG_VALUE_MODE = "contains"  # 论元值匹配: "exact" / "contains" / "fuzzy_0.8"
MIN_ARG_MATCH = 1            # 判定事件匹配的最少论元匹配数（设为 0 则只看事件类型+触发词）
RESULT_FILE = "outputs/qwen_result.json"   # 结果文件路径


# ========== 辅助匹配函数 ==========
def is_trigger_match(t1, t2, mode="exact"):
    if mode == "exact":
        return t1 == t2
    elif mode == "substring":
        return t1 in t2 or t2 in t1
    return False

def is_argument_value_match(v1, v2, mode="exact"):
    if mode == "exact":
        return v1 == v2
    elif mode == "contains":
        return v1 in v2 or v2 in v1
    elif mode == "fuzzy_0.8":
        if not v1 or not v2:
            return v1 == v2
        set1, set2 = set(v1), set(v2)
        overlap = len(set1 & set2)
        dice = 2 * overlap / (len(set1) + len(set2))
        return dice >= 0.8
    return False

def arg_match(pa, ga, value_mode):
    """单个论元是否匹配：角色必须完全一致，值按模式匹配"""
    return pa[0] == ga[0] and is_argument_value_match(pa[1], ga[1], value_mode)

def event_pair_score(pred_event, gold_event, trigger_mode, arg_value_mode):
    """
    计算预测事件与标准事件之间的论元匹配数。
    事件类型必须相同，触发词必须匹配，否则返回 -1。
    """
    if pred_event.get("event_type") != gold_event.get("event_type"):
        return -1
    if not is_trigger_match(pred_event.get("trigger", ""), gold_event.get("trigger", ""), trigger_mode):
        return -1

    pred_args = [(a.get("role", ""), a.get("argument", "")) for a in pred_event.get("arguments", [])]
    gold_args = [(a.get("role", ""), a.get("argument", "")) for a in gold_event.get("arguments", [])]

    matched = 0
    used = [False] * len(gold_args)
    for pa in pred_args:
        for i, ga in enumerate(gold_args):
            if not used[i] and arg_match(pa, ga, arg_value_mode):
                matched += 1
                used[i] = True
                break
    return matched

# ========== 事件级别贪心匹配 ==========
def calculate_metrics(pred_events, gold_events, trigger_mode, arg_value_mode, min_arg_match):
    tp = 0
    matched_gold = set()
    matched_pred = set()

    # 贪心：对每个 pred 找未匹配的 gold 中得分最高且 >= min_arg_match 的
    for i, pred in enumerate(pred_events):
        best_score = -1
        best_j = -1
        for j, gold in enumerate(gold_events):
            if j in matched_gold:
                continue
            score = event_pair_score(pred, gold, trigger_mode, arg_value_mode)
            if score >= min_arg_match and score > best_score:
                best_score = score
                best_j = j
        if best_j != -1:
            tp += 1
            matched_gold.add(best_j)
            matched_pred.add(i)

    fp = len(pred_events) - tp
    fn = len(gold_events) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1


# ========== 主流程 ==========
def main():
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)   # 假设是数组，每项包含 text/gold/pred

    total_tp = total_fp = total_fn = 0
    for item in data:
        tp, fp, fn, *_ = calculate_metrics(
            item["pred"], item["gold"],
            TRIGGER_MODE, ARG_VALUE_MODE, MIN_ARG_MATCH
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("========== 配置 ==========")
    print(f"触发词模式: {TRIGGER_MODE}")
    print(f"论元值模式: {ARG_VALUE_MODE}")
    print(f"最少论元匹配数: {MIN_ARG_MATCH}")
    print("\n========== 结果 ==========")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

if __name__ == "__main__":
    main()