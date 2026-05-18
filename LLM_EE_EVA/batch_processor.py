from prompt import BATCH_PROMPT_HEADER, BATCH_PROMPT_BODY, BATCH_PROMPT_FOOTER
from evaluator import calculate_metrics
from utils import extract_json

def process_batch(batch_texts, batch_gold, llm, schema_text,
                  trigger_mode="exact",
                  arg_value_mode="contains",
                  min_arg_match=1):
    """
    批量处理多条文本（一次 API 调用）。
    返回：
        results: 列表，每条包含 dict:
            pred: list       预测事件列表（已清洗）
            metrics: dict or None   None 表示预测为空（应排除）
            error: str or None
        model_output: str    原始模型输出（用于调试）
    """
    # 构造文本块
    text_lines = [f"{i+1}. {txt}" for i, txt in enumerate(batch_texts)]
    text_block = "\n".join(text_lines)

    prompt = BATCH_PROMPT_HEADER + "\n" + schema_text + BATCH_PROMPT_BODY + text_block + BATCH_PROMPT_FOOTER

    content = ""
    results = [{"pred": [], "metrics": None, "error": None} for _ in batch_texts]
    try:
        response = llm.invoke(prompt)
        content = response.content
        all_preds = extract_json(content)

        if not isinstance(all_preds, list):
            raise ValueError("返回的不是数组")

        # 长度对齐或截断
        if len(all_preds) != len(batch_texts):
            # 不中止，补齐或截断
            if len(all_preds) > len(batch_texts):
                all_preds = all_preds[:len(batch_texts)]
            else:
                all_preds.extend([[]] * (len(batch_texts) - len(all_preds)))

        # 清洗每个样本的预测
        for i, pred_events in enumerate(all_preds):
            if not isinstance(pred_events, list):
                results[i]["pred"] = []
                continue
            valid = [e for e in pred_events if isinstance(e, dict) and "event_type" in e and "trigger" in e]
            results[i]["pred"] = valid

    except Exception as e:
        # 整批失败：所有样本置空，记录错误
        for i in range(len(batch_texts)):
            results[i]["error"] = str(e)
        return results, content

    # 对非空预测计算指标
    for i, gold_events in enumerate(batch_gold):
        pred_events = results[i]["pred"]
        if pred_events:
            results[i]["metrics"] = calculate_metrics(
                pred_events, gold_events,
                trigger_mode=trigger_mode,
                arg_value_mode=arg_value_mode,
                min_arg_match=min_arg_match
            )
        # 若 pred 为空，metrics 保持 None

    return results, content