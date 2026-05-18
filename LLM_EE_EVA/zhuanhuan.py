import json
import re

def compact_json_lines(input_path, output_path):
    """
    读取多行缩进的 JSON 对象文件，输出为每行一个键值对的紧凑 JSON 对象。
    """
    # 读取原始内容
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 移除尾部逗号（如果存在）
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        return

    if not isinstance(data, dict):
        print("文件内容不是 JSON 对象，无法处理")
        return

    # 写入新文件，每行一个键值对
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            # 将值转换为紧凑的 JSON 字符串（无换行，无多余缩进）
            value_str = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
            # 如果不是最后一个，加逗号
            comma = "," if i < len(items) - 1 else ""
            f.write(f'  "{key}": {value_str}{comma}\n')
        f.write("}")

    print(f"转换完成，结果已保存到 {output_path}")

# 使用示例
input_file = "data/trigger_examples.json"      # 改为你的原始文件路径
output_file = "data/trigger_examples2.json"      # 输出文件路径
compact_json_lines(input_file, output_file)