from prompt import EVENT_PROMPT_HEADER, EVENT_PROMPT_RULES
from evaluator import calculate_metrics
from utils import extract_json

def process_sample(text, gold_events, llm, schema_text,
                   trigger_mode="exact",
                   arg_value_mode="contains",
                   min_arg_match=1):
    """
    处理单条样本：推理并评估。
    返回字典：
        pred: list              预测事件列表
        metrics: dict or None   None 表示预测为空（不参与指标累加）
        model_output: str       模型原始输出
        error: str or None      异常信息
    """
    prompt = EVENT_PROMPT_HEADER + "\n" + schema_text + EVENT_PROMPT_RULES + text
    content = ""
    pred_events = []
    error = None

    try:
        response = llm.invoke(prompt)
        content = response.content
        pred_events = extract_json(content)

        if isinstance(pred_events, dict):
            pred_events = [pred_events]
        elif not isinstance(pred_events, list):
            pred_events = []

        # 过滤非法事件（必须包含 event_type 和 trigger）
        pred_events = [
            e for e in pred_events
            if isinstance(e, dict) and "event_type" in e and "trigger" in e
        ]
    except Exception as e:
        error = str(e)
        pred_events = []

    # 预测为空时，不计算指标（返回 None）
    if not pred_events:
        metrics = None
    else:
        metrics = calculate_metrics(pred_events, gold_events,
                                    trigger_mode=trigger_mode,
                                    arg_value_mode=arg_value_mode,
                                    min_arg_match=min_arg_match)

    return {
        "pred": pred_events,
        "metrics": metrics,
        "model_output": content,
        "error": error
    }