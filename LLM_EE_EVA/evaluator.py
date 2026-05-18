# evaluator.py
from utils import normalize_event

# ================= 严格匹配（原有） =================
def calculate_metrics_strict(pred_events, gold_events):
    pred_set = set([normalize_event(e) for e in pred_events])
    gold_set = set([normalize_event(e) for e in gold_events])

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ================= 模糊匹配（新增，默认忽略空格） =================
def normalize_space(s: str) -> str:
    """移除所有空格，用于容错比较"""
    return s.replace(" ", "")

def is_trigger_match(t1, t2, mode="exact"):
    # 触发词也忽略空格差异
    t1 = normalize_space(t1)
    t2 = normalize_space(t2)
    if mode == "exact":
        return t1 == t2
    elif mode == "substring":
        return t1 in t2 or t2 in t1
    return False

def is_argument_value_match(v1, v2, mode="exact"):
    # 论元值忽略空格
    v1 = normalize_space(v1)
    v2 = normalize_space(v2)
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
    return pa[0] == ga[0] and is_argument_value_match(pa[1], ga[1], value_mode)

def event_pair_score(pred_event, gold_event, trigger_mode, arg_value_mode):
    # 事件类型也忽略空格
    if normalize_space(pred_event.get("event_type", "")) != normalize_space(gold_event.get("event_type", "")):
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

def calculate_metrics(pred_events, gold_events,
                      trigger_mode="exact",
                      arg_value_mode="contains",
                      min_arg_match=1):
    """
    灵活的事件级别评测（默认模糊匹配，并忽略所有空格差异）。
    返回 dict: tp, fp, fn, precision, recall, f1
    """
    tp = 0
    matched_gold = set()
    matched_pred = set()

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

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }